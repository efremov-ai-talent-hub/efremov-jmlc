"""Deterministic enum-conformance verifier for call-analysis output.

Code-based grader (no LLM, no transcript): checks that every enum-typed field in a
report payload holds a value from the set the report schema allows. Pipe-joined
values (``"price|deadline"``) are split token-by-token, which separates the
distinct failure shapes:

- ``multiselect`` — every pipe token IS a valid enum member, and they are a proper
  subset of the enum. The model wanted to return a list where the schema forces one
  value → the field should be a list.
- ``echo`` — the value lists the WHOLE enum: the model copied the schema's option
  string instead of choosing. Token-wise indistinguishable from ``multiselect``, but
  it carries no answer at all, so mixing the two would overstate the "this field
  should be a list" signal.
- ``novel`` — at least one token is outside the enum: an invented category, a real
  gap in the taxonomy.
- ``sentinel`` — the value is the *other* absent-marker (``not_specified`` where the
  enum only lists ``unknown``, or vice versa). A schema/prompt sloppiness, not an
  invented category — the two markers are used interchangeably by the prompt.

Conformance therefore measures how often the model is *forced off-script* — a proxy
for both format weakness and taxonomy gaps.

Allowed sets are parsed from the report schemas themselves
(``ai/reports/static/agent/call_analysis*.schema.json``, chosen by the payload's
``analyser_version``) so the check never drifts from the contract the prompt is built on: any schema leaf whose string value
contains ``|`` is an enum; splitting on ``|`` yields its members. Fields are keyed
by path (``"*"`` marks a list element), so the two different ``communication_style``
enums (psychological_profile vs client_portrait) stay distinct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "reports" / "static" / "agent"

# Which schema files define the enums of each analyser version, and where their fields sit in
# the payload. v3 splits the report in two (client + manager), and the manager schema is
# projected under the ``manager`` key — without the prefix its enums would never match and v3
# would look like a payload with almost no enum fields at all (measured: 1 instead of 20+).
_SCHEMAS_BY_VERSION: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "v2": (("call_analysis.schema.json", ()),),
    "v3": (
        ("call_analysis_v3.schema.json", ()),
        ("call_analysis_v3_manager.schema.json", ("manager",)),
    ),
}
_DEFAULT_VERSION = "v2"

# The two "absent" markers the prompt uses interchangeably (prompts.py rule "unknown
# ИЛИ not_specified"); one where the enum lists only the other is its own failure
# shape, distinct from inventing a category.
_SENTINELS = frozenset({"unknown", "not_specified"})


class EnumViolationKind(str, Enum):
    MULTISELECT = "multiselect"  # all pipe tokens valid, but >1 → wanted a list
    ECHO = "echo"  # the whole enum listed back — the schema's option string copied
    SENTINEL = "sentinel"  # wrong absent-marker (not_specified vs unknown)
    NOVEL = "novel"  # a token outside the enum (an invented category)


def build_allowed(schema: object, path: tuple[str, ...] = (), out: dict | None = None) -> dict:
    """Map every enum path in a report schema to its allowed token set.

    An enum is any schema leaf that is a string containing ``|``. List elements
    contribute a ``"*"`` path segment.
    """
    if out is None:
        out = {}
    if isinstance(schema, dict):
        for key, value in schema.items():
            build_allowed(value, path + (key,), out)
    elif isinstance(schema, list):
        if schema:
            build_allowed(schema[0], path + ("*",), out)
    elif isinstance(schema, str) and "|" in schema:
        out[path] = frozenset(token.strip() for token in schema.split("|"))
    return out


def _load_allowed(version: str) -> dict:
    allowed: dict = {}
    for filename, prefix in _SCHEMAS_BY_VERSION.get(version, ()):
        try:
            schema = json.loads((_SCHEMA_DIR / filename).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Порченый файл схемы — это пустое множество допустимых значений, а не падение
            # импорта: иначе одна кривая схема уносит сбор всех проверок разом.
            continue
        for path, tokens in build_allowed(schema).items():
            allowed[prefix + path] = tokens
    return allowed


_ALLOWED_BY_VERSION = {version: _load_allowed(version) for version in _SCHEMAS_BY_VERSION}


def allowed_for(payload: dict) -> dict:
    """Enum sets for the version that produced this payload.

    ``analyser_version`` is tagged as ``v3`` / ``v3-chain``; anything else (including an
    untagged historical payload) is read against the v2 schema.
    """
    version = str(payload.get("analyser_version") or "").split("-", 1)[0]
    return _ALLOWED_BY_VERSION.get(version, _ALLOWED_BY_VERSION[_DEFAULT_VERSION])


@dataclass(frozen=True)
class EnumViolation:
    path: str  # dotted path, e.g. "client_portrait.objections.*.type"
    value: str  # the raw value the model emitted
    kind: EnumViolationKind
    bad_tokens: tuple[str, ...]  # tokens outside the enum (empty for MULTISELECT/ECHO)


@dataclass
class EnumResult:
    total: int = 0  # enum-typed field instances seen
    violations: list[EnumViolation] = field(default_factory=list)

    @property
    def conformant(self) -> int:
        return self.total - len(self.violations)

    @property
    def conformance_rate(self) -> float:
        """Fraction of enum fields whose value is in the schema's set. Higher is better."""
        return self.conformant / self.total if self.total else 1.0

    def _count(self, kind: EnumViolationKind) -> int:
        return sum(1 for v in self.violations if v.kind is kind)

    @property
    def multiselect_count(self) -> int:
        return self._count(EnumViolationKind.MULTISELECT)

    @property
    def echo_count(self) -> int:
        return self._count(EnumViolationKind.ECHO)

    @property
    def sentinel_count(self) -> int:
        return self._count(EnumViolationKind.SENTINEL)

    @property
    def novel_count(self) -> int:
        return self._count(EnumViolationKind.NOVEL)


def _classify(
    tokens: list[str], allowed: frozenset[str]
) -> tuple[EnumViolationKind | None, tuple[str, ...]]:
    bad = tuple(t for t in tokens if t not in allowed)
    if not bad:
        if len(tokens) == 1:
            return None, ()
        # Every member listed = the schema's own option string, not a choice.
        if set(tokens) == allowed:
            return EnumViolationKind.ECHO, ()
        return EnumViolationKind.MULTISELECT, ()
    if all(t in _SENTINELS for t in bad):
        return EnumViolationKind.SENTINEL, bad
    return EnumViolationKind.NOVEL, bad


def _walk(node: object, path: tuple[str, ...], allowed: dict, result: EnumResult) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, path + (key,), allowed, result)
    elif isinstance(node, list):
        for item in node:
            _walk(item, path + ("*",), allowed, result)
    elif isinstance(node, str):
        allowed_set = allowed.get(path)
        if allowed_set is not None:
            result.total += 1
            tokens = [t.strip() for t in node.split("|")]
            kind, bad = _classify(tokens, allowed_set)
            if kind is not None:
                result.violations.append(
                    EnumViolation(path=".".join(path), value=node, kind=kind, bad_tokens=bad)
                )


def check_enum_conformance(payload: dict, allowed: dict | None = None) -> EnumResult:
    """Score a report payload for enum conformance against the report schema."""
    result = EnumResult()
    _walk(payload, (), allowed if allowed is not None else allowed_for(payload), result)
    return result

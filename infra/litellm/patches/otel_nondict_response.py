"""Build-time patch: make LiteLLM's OpenTelemetry integration tolerate a
non-dict `response_obj`.

Upstream bug: BerriAI/litellm#24516 (fix PR #26713, NOT yet released as of the
pinned v1.89.3). LiteLLM's OTEL success/failure callback calls `response_obj.get(...)`
blindly in several places. For MCP `/mcp` requests the "response" is a JSON *list*,
not a dict, so `set_attributes` / metric helpers raise

    OpenTelemetry logging error in set_attributes 'list' object has no attribute 'get'

on every MCP call. The exception is caught + logged (the request still completes),
but it spams the logs and drops the span attributes for those calls. This patch
mirrors the upstream fix: guard `response_obj` with an isinstance(dict) check.

Three edits in litellm/integrations/opentelemetry.py:
  1. set_attributes(...)         -> coerce a non-dict response_obj to {} at entry
                                     (all its later response_obj.get(...) become safe).
  2. _emit_semantic_logs(...)    -> early-return when response_obj is not a dict.
  3. metric usage-guard line     -> `if response_obj and (usage := ...)` becomes
                                     `if isinstance(response_obj, dict) and (usage := ...)`.

Patches BOTH source trees the image may load from (installed site-packages and the
/app checkout). Anchor-based: the build FAILS if the set_attributes anchor is not
found in ANY copy — that means upstream refactored and the patch needs review
(rather than silently regressing).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SENTINEL = "# otel-nondict-patch: guard non-dict response_obj (litellm#24516)"

# 1) set_attributes: inject a dict-guard as the first body statement.
_SET_ATTR = re.compile(
    r"(?P<indent>[ \t]+)def set_attributes\([^\n]*\n"
    r"[ \t]*self, span: Span, kwargs, response_obj: Optional\[Any\][^\n]*\n"
    r"[ \t]*\):[ \t]*\n",
)

# 2) _emit_semantic_logs: early-return on non-dict response_obj.
_EMIT = re.compile(
    r"(?P<indent>[ \t]+)def _emit_semantic_logs\(self, kwargs, response_obj, span: Span\):[ \t]*\n",
)

# 3) metric usage guard (covers _record_metrics + _record_time_per_output_token_metric).
_USAGE_OLD = 'if response_obj and (usage := response_obj.get("usage"))'
_USAGE_NEW = 'if isinstance(response_obj, dict) and (usage := response_obj.get("usage"))'


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    try:
        from litellm.integrations import opentelemetry as _otel  # type: ignore

        paths.append(Path(_otel.__file__))
    except Exception as exc:  # pragma: no cover
        print(f"WARN: could not import installed opentelemetry integration: {exc}", file=sys.stderr)
    app_path = Path("/app/litellm/integrations/opentelemetry.py")
    if app_path.exists() and app_path not in paths:
        paths.append(app_path)
    return paths


def _patch_file(path: Path) -> tuple[str, bool]:
    """Returns (status, set_attributes_guard_applied)."""
    src = path.read_text(encoding="utf-8")
    if SENTINEL in src:
        return "already-patched", True

    set_attr_ok = False

    # 1) set_attributes guard
    def _inject_set_attr(m: re.Match) -> str:
        indent = m.group("indent")
        body = indent + "    "
        return (
            f"{m.group(0)}"
            f"{body}{SENTINEL}\n"
            f"{body}if not isinstance(response_obj, dict):\n"
            f"{body}    response_obj = {{}}\n"
        )

    src, n_set = _SET_ATTR.subn(_inject_set_attr, src, count=1)
    if n_set == 1:
        set_attr_ok = True

    # 2) _emit_semantic_logs early-return
    def _inject_emit(m: re.Match) -> str:
        indent = m.group("indent")
        body = indent + "    "
        return (
            f"{m.group(0)}"
            f"{body}if not isinstance(response_obj, dict):\n"
            f"{body}    return\n"
        )

    src, _ = _EMIT.subn(_inject_emit, src, count=1)

    # 3) metric usage guard (may appear more than once)
    src = src.replace(_USAGE_OLD, _USAGE_NEW)

    path.write_text(src, encoding="utf-8")
    return ("patched" if set_attr_ok else "set_attributes-anchor-not-found"), set_attr_ok


def main() -> int:
    candidates = _candidate_paths()
    if not candidates:
        print("ERROR: no opentelemetry.py copies found", file=sys.stderr)
        return 2

    any_guarded = False
    for path in candidates:
        status, guarded = _patch_file(path)
        print(f"{status}: {path}")
        any_guarded = any_guarded or guarded

    if not any_guarded:
        print(
            "ERROR: set_attributes anchor not found in any opentelemetry.py — "
            "upstream likely refactored; review patches/otel_nondict_response.py",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Walk a call-analysis payload and pull out the quote-bearing items.

Two payload shapes are supported so v2 and v3 runs stay directly comparable — the same
definition of "a quote" is applied to both.

v2 (``call_analysis.schema.json`` + ``qc_schema.json``) — quotes sit inside nested objects:

- ``client_portrait.objections[].evidence`` (+ ``timecode``)
- ``client_portrait.motivation_details.reasons[].evidence`` (+ ``timecode``)
- ``client_portrait.requirements.{must_have,nice_to_have,deal_breakers}[].evidence``
- ``speech_metrics.{top_good_phrases,top_bad_phrases}[].phrase`` (+ ``timecode``)
- ``qc_scores.*.evidence`` (no ``timecode``)

v3 (``call_analysis_v3*.schema.json``) — flat pairs on scalar fields plus list items:

- ``<field>_quote`` (+ ``<field>_timecode``), e.g. ``motivation_quote``
- ``requirements[].quote`` / ``objections[].quote`` (+ ``timecode``)
- ``manager.{top_good_phrases,top_bad_phrases}[].quote`` (+ ``timecode``)
- ``qc_scores.*.evidence`` (unchanged)

Rather than hard-code every path we walk recursively and pick up any dict that
carries one of the quote keys — this mirrors how
``ai.reports.call_v2.core._fix_timecodes`` discovers timecodes.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# Priority order matches _fix_timecodes: evidence wins over phrase over reason.
# ``quote`` is the v3 spelling (list items carry ``quote`` + ``timecode``); v2 payloads never
# use that key, so both shapes flow through the same checks and stay directly comparable.
_QUOTE_KEYS = ("evidence", "phrase", "reason", "quote")

# v3 also attaches evidence to scalar fields as a flat pair — ``motivation_quote`` next to
# ``motivation_timecode``. There is no enclosing object to recognise, so the pair is matched
# by suffix.
_QUOTE_SUFFIX = "_quote"
_TIMECODE_SUFFIX = "_timecode"
# Пайплайн привязывает таймкоды к реальным сегментам и кладёт исходное утверждение модели сюда.
# Проверять надо утверждение, а не исправление: иначе точность размещения верна по построению.
_CLAIMED_SUFFIX = "_timecode_claimed"
_CLAIMED_KEY = "timecode_claimed"

# The analyser's own diagnostics, not model output about the call. ``speaker_labels`` carries a
# detector ``reason`` ("no_markers") that would otherwise be walked as a quote and scored as an
# invented one — a check must not grade its own bookkeeping.
_NOT_EVIDENCE = frozenset({"speaker_labels"})

# The root ``reason`` is pass0's rationale for the call_type it chose ("обсуждение параметров
# объекта…"). It is a judgement, never a citation, and nothing asks the model to quote there.
_ROOT_RATIONALE = "reason"

# Values that mean "no timecode" — the same set ``shared.transcript.parse_timecode_start`` uses.
_ABSENT_TIMECODES = frozenset({"", "not_specified", "unknown"})

_TC = re.compile(r"(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?")

# A single timecode token as the model emits it: bracketed like "[087.00–089.00]"
# (copied straight from the transcript's segment markers) or a bare dotted
# "087.00–089.00". The bare form requires a decimal point so a plain integer in
# prose ("3 комнаты") is never mistaken for a timecode.
_TC_TOKEN = (
    r"(?:\[\s*\d+(?:\.\d+)?\s*(?:[–-]\s*\d+(?:\.\d+)?\s*)?\]"
    r"|\d+\.\d+(?:\s*[–-]\s*\d+\.\d+)?)"
)
# The whole string is nothing but timecode token(s) and separators — the model
# pointed at a segment instead of quoting it.
_ROLE_TAG = re.compile(r"\[(?:МЕНЕДЖЕР|КЛИЕНТ)\]", re.IGNORECASE)
_TIMECODE_ONLY = re.compile(rf"^[\s,;]*(?:{_TC_TOKEN}[\s,;]*)+$")


# A quote may arrive with the transcript's own locator glued to its front:
# "[161.91–162.87] [МЕНЕДЖЕР] Ваш второй есть со своей мастер-спальней…". Those are words copied
# verbatim, with a pointer prepended — matching the whole string against a segment penalises the
# pointer and files a correct citation as a paraphrase. The prefix is therefore split off and,
# when the item has no timecode of its own, used as one.
_QUOTE_PREFIX = re.compile(
    r"^\s*\[\s*(\d+(?:\.\d+)?\s*[–-]\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?)\s*\]\s*"
    r"(?:\[(?:МЕНЕДЖЕР|КЛИЕНТ)\]\s*)?",
    re.IGNORECASE,
)


def split_locator(quote: str) -> tuple[str, str | None]:
    """Split a leading ``[timecode] [РОЛЬ]`` locator off a quote."""
    match = _QUOTE_PREFIX.match(quote or "")
    if not match:
        return quote, None
    rest = quote[match.end() :].strip()
    if not rest:  # nothing but the locator — that is the timecode-only failure, keep it intact
        return quote, None
    return rest, match.group(1).strip()


@dataclass(frozen=True)
class QuoteItem:
    quote: str
    timecode: str | None
    path: str  # JSON-ish path, e.g. client_portrait.objections[0].evidence
    # True when the timecode is the model's own answer. False when it was peeled off the front of
    # the quote: that value was copied out of the transcript together with the words, so it is
    # right by construction and is NOT a placement claim. It still locates the quote — checks may
    # search near it — but scoring it would let placement accuracy confirm itself.
    timecode_is_claim: bool = True

    @classmethod
    def build(cls, quote: str, timecode: object, path: str) -> QuoteItem:
        """Normalise a raw (quote, timecode) pair into an item the checks can compare."""
        text, locator = split_locator(quote)
        stated = timecode.strip() if isinstance(timecode, str) else None
        # The schema mandates ``not_specified`` when the model has no timecode, so an absent
        # marker must not win over one actually extracted from the quote.
        if stated in _ABSENT_TIMECODES:
            stated = None
        return cls(
            quote=text,
            timecode=stated or locator,
            path=path,
            timecode_is_claim=stated is not None,
        )


def is_na_quote(quote: str) -> bool:
    """True for placeholder/non-answer evidence that must not be scored.

    Covers the N/A markers the pipeline writes for service / connection-lost /
    repeat calls (``"н/п — …"``) and the schema's ``not_specified`` sentinel.
    """
    q = quote.strip().lower()
    if not q:
        return True
    if q.startswith("н/п"):
        return True
    # Empty quotation marks: the model's way of writing "there is no quote". Scoring them as
    # invented evidence would count an honest abstention as a hallucination.
    return q in {"not_specified", "n/a", "нет", "-", "—", "''", '""', "«»", "``"}


def is_na_after_locator(quote: str) -> bool:
    """N/A test applied to the words, not to the locator glued in front of them.

    v3's qc contract mandates the literal abstention «н/п — не прозвучало», and the model
    prefixes quotes with ``[012.34] [МЕНЕДЖЕР]``. Testing the raw string lets a prefixed
    abstention through, and grounding then scores an honest "nothing to quote" as fabrication.
    """
    return is_na_quote(split_locator(quote)[0])


def is_timecode_only(quote: str) -> bool:
    """True when the evidence is just timecode reference(s), not a spoken quote.

    The model is supposed to quote the words; sometimes it drops in a bare
    segment pointer like ``"[087.00–089.00]"`` (or several) instead. That is a
    distinct failure from inventing a quote (fabrication) or quoting something
    trivial (low substance), so the verifiers count it on its own. Judged from
    the text alone — no transcript needed.
    """
    # A trailing role tag ("[087.00–089.00] [МЕНЕДЖЕР]") is still nothing but a pointer.
    stripped = _ROLE_TAG.sub("", quote or "").strip()
    return bool(stripped and _TIMECODE_ONLY.match(stripped))


def parse_timecode(timecode: str | None) -> tuple[float, float | None] | None:
    """Parse ``"039.00"`` or ``"039.00–042.00"`` into ``(start, end|None)``.

    Returns ``None`` when there is no usable timecode.
    """
    if not timecode:
        return None
    stripped = timecode.strip()
    if stripped in {"not_specified", ""}:
        return None
    match = _TC.match(stripped)
    if not match:
        return None
    start = float(match.group(1))
    end = float(match.group(2)) if match.group(2) else None
    return (start, end)


def iter_quote_items(payload: Any, path: str = "") -> Iterator[QuoteItem]:
    """Yield every quote-bearing item in the payload, skipping N/A placeholders."""
    if isinstance(payload, dict):
        # ``evidence`` у qc-критерия — копия первой цитаты из ``quotes`` (её читают скоринг и
        # витрины). Отдать наружу обе значит посчитать одну цитату дважды: знаменатель растёт на
        # гарантированно зелёный пункт, а точность таймкода подтверждает сама себя — у копии
        # таймкод проставлен КОДОМ, а не заявлен моделью.
        has_quote_list = isinstance(payload.get("quotes"), list) and payload["quotes"]
        for key in _QUOTE_KEYS:
            if has_quote_list and key == "evidence":
                continue
            if path == "" and key == _ROOT_RATIONALE:
                continue  # pass0's classification rationale, not evidence — see below
            value = payload.get(key)
            if isinstance(value, str) and not is_na_after_locator(value):
                timecode = payload.get(_CLAIMED_KEY, payload.get("timecode"))
                yield QuoteItem.build(value, timecode, f"{path}.{key}" if path else key)
                break
        # Flat ``<field>_quote`` / ``<field>_timecode`` pairs (v3). Cannot collide with the
        # loop above: a bare ``quote`` key does not end with ``_quote``.
        for key, value in payload.items():
            if not key.endswith(_QUOTE_SUFFIX) or not isinstance(value, str):
                continue
            if is_na_after_locator(value):
                continue
            field = key[: -len(_QUOTE_SUFFIX)]
            timecode = payload.get(
                f"{field}{_CLAIMED_SUFFIX}", payload.get(f"{field}{_TIMECODE_SUFFIX}")
            )
            yield QuoteItem.build(value, timecode, f"{path}.{key}" if path else key)
        for key, value in payload.items():
            if key in _NOT_EVIDENCE:
                continue
            child = f"{path}.{key}" if path else key
            yield from iter_quote_items(value, child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from iter_quote_items(value, f"{path}[{index}]")


def iter_bad_phrases(payload: Any) -> Iterator[QuoteItem]:
    """Yield manager-attributed bad phrases (``speech_metrics`` in v2, ``manager`` in v3)."""
    yield from _iter_speech_phrases(payload, "top_bad_phrases")


def iter_good_phrases(payload: Any) -> Iterator[QuoteItem]:
    """Yield manager-attributed good phrases (``speech_metrics`` in v2, ``manager`` in v3)."""
    yield from _iter_speech_phrases(payload, "top_good_phrases")


# Manager phrases live under ``speech_metrics`` in v2 and under ``manager`` in v3, and the text
# key is ``phrase`` in v2 / ``quote`` in v3. Both are read so the speaker check applies the same
# definition to either payload.
_SPEECH_CONTAINERS = ("speech_metrics", "manager")
_PHRASE_KEYS = ("phrase", "quote")


def _iter_speech_phrases(payload: Any, key: str) -> Iterator[QuoteItem]:
    if not isinstance(payload, dict):
        return
    for container in _SPEECH_CONTAINERS:
        metrics = payload.get(container)
        if not isinstance(metrics, dict):
            continue
        phrases = metrics.get(key)
        if not isinstance(phrases, list):
            continue
        for index, item in enumerate(phrases):
            if not isinstance(item, dict):
                continue
            matched = next(
                (
                    name
                    for name in _PHRASE_KEYS
                    if isinstance(item.get(name), str) and not is_na_after_locator(item[name])
                ),
                None,
            )
            if matched is None:
                continue
            phrase = item[matched]
            timecode = item.get(_CLAIMED_KEY, item.get("timecode"))
            yield QuoteItem.build(phrase, timecode, f"{container}.{key}[{index}].{matched}")

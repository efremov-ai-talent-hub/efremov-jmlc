"""Verifier for claim 1 — evidence must be a real quote at the right timecode.

The QC/main prompts require every assessment to attach a transcript quote as
evidence. The observed failure modes:

- the model writes its own verdict / invents a phrase that is not in the
  transcript at all (``fabrication``);
- the quote is real but the attached timecode points at the wrong segment
  (``misplacement`` / timecode error).

**Anchor-first scoring.** Each quote is matched *first* against the segment(s)
the model's own timecode points at (``anchored_ratio``) — that is the primary
signal. The match against the whole transcript (``full_ratio``) is secondary:
it only decides, when the anchored match is poor, whether the quote is *real
but misplaced* (it appears verbatim elsewhere) or *fabricated* (it appears
nowhere). Scoring against the whole transcript alone is what lets a fluent
invented Russian sentence borrow coincidental letter-overlap from unrelated
segments and float up to the fabrication threshold; anchoring removes that
hiding place.

Evidence that is a bare timecode pointer (``"[087.00–089.00]"``) rather than a
quote is caught up front from the text alone and set aside as ``timecode_only``
— the model pointed at a segment instead of quoting it. That is a distinct
failure from inventing a quote, so it is **not** counted as fabrication.

The remaining quotes yield three states (plus a ``weak`` partial/paraphrase band):

- ``grounded``   — anchored match ≥ ``VERBATIM_THRESHOLD`` (real, right place);
- ``misplaced``  — anchored match poor but the exact words appear elsewhere
  (real quote, wrong timecode);
- ``fabricated`` — primary match < ``FABRICATION_THRESHOLD`` and the words
  appear nowhere (no basis in the transcript).

Matching uses the same ``difflib`` ratio the pipeline uses for timecode repair
(via :func:`ai.evals.checks.matching.partial_ratio`), at stricter thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ai.evals.checks.matching import partial_ratio
from ai.evals.checks.payload import is_timecode_only, iter_quote_items, parse_timecode
from ai.evals.transcript import Segment, parse_transcript

# Quote is treated as a faithful (near-)verbatim quote at/above this ratio.
VERBATIM_THRESHOLD = 0.85
# Below this ratio the quote has no basis where it was placed — invented.
FABRICATION_THRESHOLD = 0.40
# Tolerance (seconds) when matching a claimed timecode to real segments.
TIMECODE_TOLERANCE = 2.0


class Grounding(str, Enum):
    """How a cited quote relates to the transcript."""

    GROUNDED = "grounded"  # real quote, at the claimed timecode
    MISPLACED = "misplaced"  # real quote, but the timecode points elsewhere
    FABRICATED = "fabricated"  # no basis in the transcript at all
    WEAK = "weak"  # partial / paraphrase — present-ish but not verbatim
    TIMECODE_ONLY = "timecode_only"  # a bare timecode pointer, not a quote at all


@dataclass(frozen=True)
class GroundingItem:
    quote: str
    timecode: str | None
    path: str
    # Match against the segment(s) at the *claimed* timecode. None iff the item
    # carries no timecode (e.g. qc_scores.*.evidence); 0.0 if the timecode
    # points outside the transcript.
    anchored_ratio: float | None
    # Best match anywhere in the transcript — the "does it exist at all" signal.
    full_ratio: float
    # Where the quote actually matches best (for reporting misplacement).
    best_segment_timecode: str | None
    status: Grounding

    @property
    def grounded(self) -> bool:
        return self.status is Grounding.GROUNDED

    @property
    def misplaced(self) -> bool:
        return self.status is Grounding.MISPLACED

    @property
    def fabricated(self) -> bool:
        return self.status is Grounding.FABRICATED

    @property
    def timecode_only(self) -> bool:
        """Evidence is a bare timecode pointer, not a quote at all."""
        return self.status is Grounding.TIMECODE_ONLY

    @property
    def weak(self) -> bool:
        """Partial / paraphrase: present-ish but neither verbatim nor invented."""
        return self.status is Grounding.WEAK

    @property
    def verbatim(self) -> bool:
        """The exact words appear somewhere in the transcript (placement aside)."""
        return self.full_ratio >= VERBATIM_THRESHOLD

    @property
    def timecode_ok(self) -> bool | None:
        """Quote matches at the claimed timecode. ``None`` if it carries none."""
        if self.anchored_ratio is None:
            return None
        return self.anchored_ratio >= VERBATIM_THRESHOLD


@dataclass
class GroundingResult:
    items: list[GroundingItem] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def grounded_count(self) -> int:
        return sum(1 for i in self.items if i.grounded)

    @property
    def grounded_unplaced_count(self) -> int:
        """Grounded only because no timecode was given: the words are verbatim
        somewhere, but placement was never checked (e.g. qc_scores.*.evidence)."""
        return sum(1 for i in self.items if i.grounded and i.anchored_ratio is None)

    @property
    def grounded_placed_count(self) -> int:
        """Grounded *and* placement-verified: matched at the claimed timecode."""
        return self.grounded_count - self.grounded_unplaced_count

    @property
    def fabricated_count(self) -> int:
        return sum(1 for i in self.items if i.fabricated)

    @property
    def misplaced_count(self) -> int:
        return sum(1 for i in self.items if i.misplaced)

    @property
    def timecode_only_count(self) -> int:
        return sum(1 for i in self.items if i.timecode_only)

    @property
    def weak_count(self) -> int:
        return sum(1 for i in self.items if i.weak)

    @property
    def grounded_rate(self) -> float:
        """Fraction of evidence that is a real quote at the right place. Higher is better."""
        return self.grounded_count / self.total if self.total else 1.0

    @property
    def grounded_placed_rate(self) -> float:
        """Real quote verified at its claimed timecode — "regular" grounding."""
        return self.grounded_placed_count / self.total if self.total else 1.0

    @property
    def grounded_unplaced_rate(self) -> float:
        """Real quote, but no timecode was given to verify its placement."""
        return self.grounded_unplaced_count / self.total if self.total else 0.0

    @property
    def fabrication_rate(self) -> float:
        """Fraction of evidence with no basis in the transcript. Lower is better."""
        return self.fabricated_count / self.total if self.total else 0.0

    @property
    def misplacement_rate(self) -> float:
        """Fraction of evidence that is a real quote at the wrong timecode. Lower is better."""
        return self.misplaced_count / self.total if self.total else 0.0

    @property
    def timecode_only_rate(self) -> float:
        """Fraction of evidence that is a bare timecode pointer, not a quote. Lower is better."""
        return self.timecode_only_count / self.total if self.total else 0.0

    @property
    def weak_rate(self) -> float:
        """Fraction that is partial / paraphrase — not verbatim, not clearly invented."""
        return self.weak_count / self.total if self.total else 0.0

    @property
    def timecode_accuracy(self) -> float | None:
        """Among real quotes that carry a timecode, fraction pointing at the
        right segment. ``None`` if no such items exist."""
        placed = [
            i for i in self.items if i.anchored_ratio is not None and (i.grounded or i.misplaced)
        ]
        if not placed:
            return None
        return sum(1 for i in placed if i.grounded) / len(placed)

    @property
    def fabricated_items(self) -> list[GroundingItem]:
        return [i for i in self.items if i.fabricated]

    @property
    def misplaced_items(self) -> list[GroundingItem]:
        return [i for i in self.items if i.misplaced]

    @property
    def timecode_only_items(self) -> list[GroundingItem]:
        return [i for i in self.items if i.timecode_only]


def _best_segment_match(quote: str, segments: list[Segment]) -> tuple[float, Segment | None]:
    """Best per-segment match for the quote: ``(ratio, segment)``.

    Taken segment-by-segment rather than against the joined transcript because
    ``partial_ratio``'s matching-block heuristic grows unreliable on long text
    (it can miss the right alignment window), which would understate a quote
    that sits verbatim inside a single segment and mislabel it as invented.
    """
    best_ratio, best_seg = 0.0, None
    for segment in segments:
        ratio = partial_ratio(quote, segment.text)
        if ratio > best_ratio:
            best_ratio, best_seg = ratio, segment
    return best_ratio, best_seg


def _anchored_ratio(quote: str, timecode: str | None, segments: list[Segment]) -> float | None:
    """Match the quote against the segment(s) at the claimed timecode.

    ``None`` when the item carries no usable timecode (nothing to anchor to);
    ``0.0`` when a timecode is given but lands outside the transcript.
    """
    parsed = parse_timecode(timecode)
    if parsed is None:
        return None
    start, end = parsed
    upper = end if end is not None else start
    window = [
        seg
        for seg in segments
        if seg.end >= start - TIMECODE_TOLERANCE and seg.start <= upper + TIMECODE_TOLERANCE
    ]
    if not window:
        return 0.0
    window_text = " ".join(seg.text for seg in window)
    return partial_ratio(quote, window_text)


def _classify(anchored: float | None, full: float) -> Grounding:
    """Pick a state from the primary (anchored) and secondary (full) signals."""
    if anchored is not None and anchored >= VERBATIM_THRESHOLD:
        return Grounding.GROUNDED
    if full >= VERBATIM_THRESHOLD:
        # The exact words exist in the transcript. With a timecode that didn't
        # match, the quote is real but misplaced; without one we can't fault
        # placement, so it counts as grounded.
        return Grounding.MISPLACED if anchored is not None else Grounding.GROUNDED
    # Not verbatim anywhere. Judge fabrication by the primary signal.
    primary = anchored if anchored is not None else full
    return Grounding.FABRICATED if primary < FABRICATION_THRESHOLD else Grounding.WEAK


def check_grounding(payload: dict, transcript: str) -> GroundingResult:
    segments = parse_transcript(transcript)
    # Joined text catches a quote that legitimately spans two turns; the
    # per-segment best is the reliable signal for a quote inside one segment.
    full_text = " ".join(seg.text for seg in segments)
    result = GroundingResult()
    for item in iter_quote_items(payload):
        if is_timecode_only(item.quote):
            # A bare segment pointer, not a quote — set aside as its own failure
            # rather than scored (and mislabelled fabricated) against the text.
            result.items.append(
                GroundingItem(
                    quote=item.quote,
                    timecode=item.timecode,
                    path=item.path,
                    anchored_ratio=None,
                    full_ratio=0.0,
                    best_segment_timecode=None,
                    status=Grounding.TIMECODE_ONLY,
                )
            )
            continue
        # Таймкод, вынутый из самой цитаты, — не утверждение модели о размещении (он списан из
        # транскрипта вместе со словами). Такой пункт идёт как «дословно, но без указания места»,
        # а не как «место указано верно»: иначе точность размещения подтверждает сама себя.
        anchored = (
            _anchored_ratio(item.quote, item.timecode, segments) if item.timecode_is_claim else None
        )
        seg_best, best_seg = _best_segment_match(item.quote, segments)
        full = max(seg_best, partial_ratio(item.quote, full_text))
        best_tc = f"{best_seg.start:06.2f}–{best_seg.end:06.2f}" if best_seg is not None else None
        result.items.append(
            GroundingItem(
                quote=item.quote,
                timecode=item.timecode,
                path=item.path,
                anchored_ratio=anchored,
                full_ratio=full,
                best_segment_timecode=best_tc,
                status=_classify(anchored, full),
            )
        )
    return result

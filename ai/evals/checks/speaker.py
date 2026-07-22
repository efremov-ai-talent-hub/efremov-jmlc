"""Verifier for claim 2 — manager phrase lists must contain manager phrases.

Manager phrases live under ``speech_metrics`` (v2) or ``manager`` (v3). They are supposed
to be the *manager's* phrases; the observed failure is the model pulling *client* lines
into the manager's list.

Diarization sometimes inverts the channel labels, and an analyser may resolve roles by
meaning instead. Such a verdict is honoured **only when it came from a deterministic,
auditable detector** (``speaker_labels.source == "detector"``). A verdict produced by the
analysed model itself is ignored on purpose: trusting it would let a system nullify the
very defect this check exists to find — declare "labels are swapped" and every violation
turns into a correct attribution.

For each phrase we locate the transcript segment it refers to (by claimed
timecode when present, else by best text match) and read that segment's role
label:

- ``МЕНЕДЖЕР`` → correct attribution;
- ``КЛИЕНТ`` → violation (client phrase mis-listed as the manager's);
- no role (mono recording) → ``unknown`` (cannot be checked);
- no segment matched → ``not_found``.

Mono recordings carry no role labels, so the whole check degrades to
``unknown`` rather than producing false violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ai.evals.checks.matching import partial_ratio
from ai.evals.checks.payload import QuoteItem, iter_bad_phrases, iter_good_phrases, parse_timecode
from ai.evals.transcript import ROLE_CLIENT, ROLE_MANAGER, Segment, has_roles, parse_transcript

# Roles as written in the transcript, inverted when a deterministic detector says the
# channel labels are the wrong way round.
_FLIP_ROLE = {ROLE_MANAGER: ROLE_CLIENT, ROLE_CLIENT: ROLE_MANAGER}

# Only a deterministic detector may flip roles — see the module docstring.
_TRUSTED_SWAP_SOURCE = "detector"

# Minimum text-match ratio to consider a phrase "located" in a segment.
MATCH_THRESHOLD = 0.50
TIMECODE_TOLERANCE = 2.0


class Attribution(str, Enum):
    MANAGER = "manager"
    CLIENT = "client"
    UNKNOWN = "unknown"  # mono recording — no role labels
    NOT_FOUND = "not_found"  # phrase not located in transcript


@dataclass(frozen=True)
class SpeakerItem:
    phrase: str
    timecode: str | None
    path: str
    attribution: Attribution
    matched_segment_timecode: str | None


@dataclass
class SpeakerResult:
    items: list[SpeakerItem] = field(default_factory=list)

    @property
    def manager_count(self) -> int:
        return sum(1 for i in self.items if i.attribution is Attribution.MANAGER)

    @property
    def client_count(self) -> int:
        return sum(1 for i in self.items if i.attribution is Attribution.CLIENT)

    @property
    def attributable(self) -> int:
        """Phrases located AND role-resolved (manager or client)."""
        return self.manager_count + self.client_count

    @property
    def accuracy(self) -> float | None:
        """Fraction of attributable phrases correctly belonging to the manager.

        ``None`` when nothing is attributable (e.g. a mono recording) — the
        check could not run, which is different from a perfect score.
        """
        if self.attributable == 0:
            return None
        return self.manager_count / self.attributable

    @property
    def violations(self) -> list[SpeakerItem]:
        return [i for i in self.items if i.attribution is Attribution.CLIENT]


def _locate(phrase: str, timecode: str | None, segments: list[Segment]) -> Segment | None:
    candidates = segments
    parsed = parse_timecode(timecode)
    if parsed is not None:
        start, end = parsed
        upper = end if end is not None else start
        window = [
            seg
            for seg in segments
            if seg.end >= start - TIMECODE_TOLERANCE and seg.start <= upper + TIMECODE_TOLERANCE
        ]
        if window:
            candidates = window
    best_ratio, best_seg = 0.0, None
    for segment in candidates:
        ratio = partial_ratio(phrase, segment.text)
        if ratio > best_ratio:
            best_ratio, best_seg = ratio, segment
    if best_ratio < MATCH_THRESHOLD:
        return None
    return best_seg


def _classify(
    item: QuoteItem,
    segments: list[Segment],
    roles_present: bool,
    *,
    labels_swapped: bool = False,
) -> SpeakerItem:
    segment = _locate(item.quote, item.timecode, segments)
    # Diarization mislabels channels often enough that an analyser may resolve roles by meaning
    # and record that the transcript labels are inverted. Honouring that verdict is what keeps
    # the check symmetric: an analyser that correctly picked manager lines sitting under a
    # [КЛИЕНТ] label must not be scored as if every one of them were a violation.
    role = segment.role if segment is not None else None
    if labels_swapped and role is not None:
        role = _FLIP_ROLE.get(role, role)
    if segment is None:
        attribution = Attribution.NOT_FOUND
        matched_tc = None
    elif not roles_present or role is None:
        attribution = Attribution.UNKNOWN
        matched_tc = f"{segment.start:06.2f}–{segment.end:06.2f}"
    elif role == ROLE_MANAGER:
        attribution = Attribution.MANAGER
        matched_tc = f"{segment.start:06.2f}–{segment.end:06.2f}"
    elif role == ROLE_CLIENT:
        attribution = Attribution.CLIENT
        matched_tc = f"{segment.start:06.2f}–{segment.end:06.2f}"
    else:
        attribution = Attribution.UNKNOWN
        matched_tc = f"{segment.start:06.2f}–{segment.end:06.2f}"
    return SpeakerItem(
        phrase=item.quote,
        timecode=item.timecode,
        path=item.path,
        attribution=attribution,
        matched_segment_timecode=matched_tc,
    )


def check_speaker_attribution(
    payload: dict, transcript: str, *, include_good: bool = True
) -> SpeakerResult:
    """Check that manager phrase lists hold manager phrases.

    ``top_bad_phrases`` is always checked; ``top_good_phrases`` too when
    ``include_good`` is set (both are manager-attributed by schema).

    Segment roles are read inverted when the payload reports swapped channel labels AND that
    verdict came from a deterministic detector; a model's own verdict is ignored (see module
    docstring). Payloads without ``speaker_labels`` (v2) are unaffected.
    """
    segments = parse_transcript(transcript)
    roles_present = has_roles(segments)
    labels = payload.get("speaker_labels")
    # ``is True``, not ``bool(...)``: the field is load-bearing and the string "false" must
    # not invert every role in the report.
    labels_swapped = (
        isinstance(labels, dict)
        and labels.get("swapped") is True
        and labels.get("source") == _TRUSTED_SWAP_SOURCE
    )
    result = SpeakerResult()
    phrases = list(iter_bad_phrases(payload))
    if include_good:
        phrases += list(iter_good_phrases(payload))
    for item in phrases:
        result.items.append(_classify(item, segments, roles_present, labels_swapped=labels_swapped))
    return result

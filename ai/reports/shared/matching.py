"""Fuzzy matching helpers shared by the verifiers.

The pipeline's own timecode repair uses a *symmetric* ``SequenceMatcher.ratio``
between a quote and a whole segment. That penalises legitimate **partial**
quotes: when the model quotes only a fragment of a long utterance, the symmetric
ratio drops purely because the segment is longer, and the fragment would be
wrongly flagged as invented.

``partial_ratio`` fixes that by aligning the quote against its best-matching
sub-window of the text (the fuzzywuzzy approach), so a faithful fragment scores
~1.0 regardless of how much surrounding speech the segment contains. Pure
stdlib (``difflib``), no third-party dependency.
"""

from __future__ import annotations

from difflib import SequenceMatcher


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so formatting noise doesn't hurt match."""
    return " ".join(text.lower().split())


def partial_ratio(needle: str, haystack: str) -> float:
    """Best similarity of ``needle`` against any substring of ``haystack`` (0..1).

    A faithful partial quote scores ~1.0 even when ``haystack`` (the full segment
    or transcript) is much longer. When ``needle`` is the longer string we fall
    back to the plain symmetric ratio.
    """
    a = normalize(needle)
    b = normalize(haystack)
    if not a or not b:
        return 0.0
    if len(a) >= len(b):
        return SequenceMatcher(None, a, b).ratio()

    # Use the longest matching blocks to pick candidate alignment offsets, then
    # score the quote against a window of its own length at each offset.
    matcher = SequenceMatcher(None, a, b)
    best = 0.0
    for i, j, size in matcher.get_matching_blocks():
        if size == 0:
            continue
        start = max(0, j - i)
        window = b[start : start + len(a)]
        ratio = SequenceMatcher(None, a, window).ratio()
        if ratio > best:
            best = ratio
            if best == 1.0:
                break
    return best

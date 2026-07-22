"""Companion verifier for claim 1 — evidence must be substantive, not trivial.

:mod:`ai.evals.checks.grounding` answers "is this quote actually in the
transcript?". It uses partial matching on purpose, so a faithful *fragment* of a
long utterance is not penalised — but the flip side is that a very short quote
can match almost anywhere. Per the layered-eval principle (each grader catches
what the others miss; see Anthropic's "Demystifying Evals for AI Agents"),
substance is a **separate** check: even a grounded quote is weak evidence if it
is too short to mean anything. Combined: grounded AND substantive = trustworthy.

Whether an evidence string is a genuine quote versus the model's own verdict is
a *semantic* judgement that needs an LLM grader — out of scope for this
deterministic check, which only covers the length/triviality proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ai.evals.checks.matching import normalize
from ai.evals.checks.payload import is_timecode_only, iter_quote_items

# Evidence shorter than this many words is treated as non-substantive.
DEFAULT_MIN_WORDS = 3


@dataclass(frozen=True)
class SubstanceItem:
    quote: str
    path: str
    word_count: int


@dataclass
class SubstanceResult:
    min_words: int
    items: list[SubstanceItem] = field(default_factory=list)

    def _substantive(self, item: SubstanceItem) -> bool:
        return item.word_count >= self.min_words

    @property
    def total(self) -> int:
        return len(self.items)

    @property
    def substantive_count(self) -> int:
        return sum(1 for i in self.items if self._substantive(i))

    @property
    def rate(self) -> float:
        """Fraction of evidence quotes that clear the length bar. Higher is better."""
        return self.substantive_count / self.total if self.total else 1.0

    @property
    def low_substance_items(self) -> list[SubstanceItem]:
        return [i for i in self.items if not self._substantive(i)]


def check_substance(payload: dict, min_words: int = DEFAULT_MIN_WORDS) -> SubstanceResult:
    """Flag evidence/phrase quotes that are too short to be real evidence.

    Needs no transcript — it judges the quote text itself.
    """
    result = SubstanceResult(min_words=min_words)
    for item in iter_quote_items(payload):
        if is_timecode_only(item.quote):
            # A bare timecode pointer is not a quote at all; grounding counts it
            # as its own failure, so don't double-flag it here as "too short".
            continue
        result.items.append(
            SubstanceItem(
                quote=item.quote,
                path=item.path,
                word_count=len(normalize(item.quote).split()),
            )
        )
    return result

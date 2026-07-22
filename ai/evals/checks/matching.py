"""Fuzzy matching for the verifiers — re-exported from the pipeline's own primitive.

The analyser verifies quotes with the same function (``call_v3.qc``), and a checker that
measured "is this quoted" by a different definition than the pipeline uses would compare two
different things. One implementation, one threshold vocabulary, both sides.
"""

from __future__ import annotations

from ai.reports.shared.matching import normalize, partial_ratio

__all__ = ["normalize", "partial_ratio"]

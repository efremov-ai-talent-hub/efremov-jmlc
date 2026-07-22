"""Deterministic verifiers for call-analysis output quality.

Each verifier is a pure function over ``(report_payload, transcript)`` and needs
no LLM and no Inspect AI runner. They target concrete, machine-checkable
failure modes of the current call processing:

- :mod:`ai.evals.checks.grounding` — evidence/phrase must be a real quote from
  the transcript at the claimed timecode (not an invented phrase or a verdict).
- :mod:`ai.evals.checks.speaker` — "bad/good manager phrases" must actually be
  spoken by the manager, not the client.

Both return rich per-item detail so failures can be read, not just counted.
"""

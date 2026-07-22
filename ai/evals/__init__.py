"""Evaluation harness for LLM call/meeting processing.

This package is intentionally pipelines-free (see .importlinter contract
``ai-independent``): it is a reusable library, not part of the data platform.
The deterministic verifiers under ``ai.evals.checks`` need neither an LLM nor
the Inspect AI runner — they are pure ``(transcript, payload) -> result``
functions and can be run over already-stored reports to characterise current
quality before any refactor.
"""

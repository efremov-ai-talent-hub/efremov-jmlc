"""Reads that decide what a flow should work on next.

Each query answers one question: which calls still need a transcript, which
still need a report from a given analyser. Both exclude calls that have already
failed too often — that counter comes from ``analysis.llm_calls`` rows with
``status='failed'``, which is the journal of what was attempted. There is no
separate attempts table: a failed LLM call is already recorded there against its
``call_id``, and a second table tracking the same thing could only disagree.

The cut-off keys on ``artifact_type``, written by this package, rather than on
``stage``, which belongs to the domain layer and is free-form by its own
documentation — the real values there are pass0 / qc / main / analysis /
transcribe / speaker_detect / polish / segment, and pinning a reader to any
subset of them silently stops counting when that set changes.

It counts rows, not runs. One report run emits several LLM calls, so the budget
in ``ANALYSIS_MAX_FAILED_ATTEMPTS`` is spent faster than "N failed runs" — set
it accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection

from pipelines import config
from pipelines.analysis.tables._sql import qualified
from pipelines.analysis.tables.writer import (
    ARTIFACT_TRANSCRIPTION,
    STATUS_FAILED,
    call_report_artifact_type,
)


@dataclass(frozen=True)
class CallCandidate:
    call_id: str
    audio_key: str
    duration_seconds: float | None


@dataclass(frozen=True)
class ReportCandidate:
    call_id: str
    transcription_id: str
    transcription_version: int
    transcript: str


def _limit(value: int | None, default: int) -> int:
    """Clamp to at least 1: `LIMIT 0` returns nothing and `LIMIT -1` is a parse error."""
    chosen = default if value is None else value
    return max(1, int(chosen))


# Calls whose failures have piled up are left alone. Without this a call that
# fails deterministically — unreadable audio, a prompt the model always refuses —
# would be retried on every run, forever, at full LLM cost.
_FAILED_CTE = f"""
    failed AS (
        SELECT call_id, count(*) AS n
        FROM {qualified("llm_calls")}
        WHERE status = :failed AND artifact_type = :artifact_type
        GROUP BY call_id
    )
"""


def fetch_calls_needing_transcription(
    conn: Connection, *, limit: int | None = None
) -> list[CallCandidate]:
    """Calls that have audio and no current transcript.

    A current transcript with NULL text would block re-transcription while never
    becoming a report candidate, so such rows do not count as current here.
    """
    query = text(
        f"""
        WITH {_FAILED_CTE}
        SELECT c.call_id, c.audio_key, c.duration_seconds
        FROM {qualified("calls")} c
        LEFT JOIN {qualified("transcriptions")} t
               ON t.call_id = c.call_id AND t.is_current = true AND t.text IS NOT NULL
        LEFT JOIN failed f ON f.call_id = c.call_id
        WHERE c.audio_key IS NOT NULL
          AND t.call_id IS NULL
          AND coalesce(f.n, 0) < :max_failed
        ORDER BY c.started_at
        LIMIT {_limit(limit, config.ANALYSIS_LIMIT_TRANSCRIPTION)}
        """
    )
    rows = conn.execute(
        query,
        {
            "artifact_type": ARTIFACT_TRANSCRIPTION,
            "failed": STATUS_FAILED,
            "max_failed": config.ANALYSIS_MAX_FAILED_ATTEMPTS,
        },
    )
    return [
        CallCandidate(
            call_id=r.call_id, audio_key=r.audio_key, duration_seconds=r.duration_seconds
        )
        for r in rows
    ]


def fetch_calls_needing_report(
    conn: Connection, *, analyser_version: str, limit: int | None = None
) -> list[ReportCandidate]:
    """Calls with a current transcript but no current report from this analyser.

    Scoped by ``analyser_version`` throughout — including the failure count, so
    that repeated v2 failures do not also block v3 from ever being tried.
    """
    query = text(
        f"""
        WITH failed AS (
            SELECT call_id, count(*) AS n
            FROM {qualified("llm_calls")}
            WHERE status = :failed AND artifact_type = :artifact_type
            GROUP BY call_id
        )
        SELECT t.call_id,
               t.transcription_id,
               t.version AS transcription_version,
               t.text AS transcript
        FROM {qualified("transcriptions")} t
        LEFT JOIN {qualified("call_reports")} r
               ON r.call_id = t.call_id
              AND r.is_current = true
              AND r.analyser_version = :analyser_version
        LEFT JOIN failed f ON f.call_id = t.call_id
        WHERE t.is_current = true
          AND t.text IS NOT NULL
          AND r.call_id IS NULL
          AND coalesce(f.n, 0) < :max_failed
        ORDER BY t.created_at
        LIMIT {_limit(limit, config.ANALYSIS_LIMIT_CALL_REPORT)}
        """
    )
    rows = conn.execute(
        query,
        {
            "artifact_type": call_report_artifact_type(analyser_version),
            "analyser_version": analyser_version,
            "failed": STATUS_FAILED,
            "max_failed": config.ANALYSIS_MAX_FAILED_ATTEMPTS,
        },
    )
    return [
        ReportCandidate(
            call_id=r.call_id,
            transcription_id=r.transcription_id,
            transcription_version=r.transcription_version,
            transcript=r.transcript,
        )
        for r in rows
    ]

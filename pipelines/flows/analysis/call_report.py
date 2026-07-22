"""Analyse transcripts into call reports.

Two analysers are carried, v2 and v3. They are not successive versions of one
thing: each has its own prompts, schema and scoring, so a call can hold a
current report from each and the two are never compared with one another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prefect import flow, get_run_logger, task

from ai.reports.call_v2.analyser import analyse_call_v2
from ai.reports.call_v3.analyser import analyse_call_v3
from ai.shared import llm_tracing
from pipelines import config
from pipelines.analysis.tables import queries as q
from pipelines.analysis.tables import writer as w
from pipelines.clients.trino import make_trino_engine, trino_conn_from_config
from pipelines.flows.analysis._orchestration import run_with_batched_persist

ANALYSERS = {"v2": analyse_call_v2, "v3": analyse_call_v3}


@dataclass
class ReportResult:
    candidate: q.ReportCandidate
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    stamps: list[Any] = field(default_factory=list)


@task(retries=0)
def analyse_candidate(
    candidate: q.ReportCandidate, *, analyser_version: str, model: str
) -> ReportResult:
    """Run one analyser over one transcript. Failures come back as data."""
    logger = get_run_logger()
    analyse = ANALYSERS[analyser_version]
    try:
        with llm_tracing.collect_stamps() as stamps, llm_tracing.bind(
            entity_type="call", entity_id=candidate.call_id
        ):
            payload = analyse(candidate.transcript, model, config)
        return ReportResult(
            candidate=candidate, ok=True, payload=payload, stamps=list(stamps)
        )
    except Exception as exc:  # noqa: BLE001 — carried as data
        logger.warning("%s analysis failed for %s: %s", analyser_version, candidate.call_id, exc)
        return ReportResult(candidate=candidate, ok=False, error=str(exc)[:2000])


def _persist(results: list[ReportResult], *, analyser_version: str, model: str) -> int:
    conn_cfg = trino_conn_from_config()
    engine = make_trino_engine(conn_cfg, schema=config.ANALYSIS_SCHEMA)
    ok = [r for r in results if r.ok and r.payload]
    written = 0
    artifact_type = w.call_report_artifact_type(analyser_version)
    with engine.connect() as conn:
        rows: list[w.CallReportRow] = []
        if ok:
            versions = w.next_versions_for_calls(
                conn,
                table="call_reports",
                call_ids=[r.candidate.call_id for r in ok],
                analyser_version=analyser_version,
            )
            rows = [
                w.CallReportRow(
                    call_id=r.candidate.call_id,
                    version=versions[r.candidate.call_id],
                    analyser_version=analyser_version,
                    report=r.payload,
                    model=model,
                    report_id=w.new_id(),
                )
                for r in ok
            ]
            w.insert_call_reports_batch(conn, rows)
            written = len(rows)

            # Lineage: which transcript version this report consumed. Without it
            # a re-transcription cannot tell which reports it invalidates.
            by_call = {r.candidate.call_id: r.candidate for r in ok}
            w.insert_report_inputs_batch(
                conn,
                [
                    w.ReportInputRow(
                        report_id=row.report_id,
                        report_version=row.version,
                        source_type=w.SOURCE_TRANSCRIPTION,
                        source_id=by_call[row.call_id].transcription_id,
                        source_version=by_call[row.call_id].transcription_version,
                    )
                    for row in rows
                ],
            )

        journal = [
            w.LLMCallRow(
                call_id=r.candidate.call_id,
                status=w.STATUS_OK if r.ok else w.STATUS_FAILED,
                artifact_type=artifact_type,
                artifact_id=next(
                    (row.report_id for row in rows if row.call_id == r.candidate.call_id),
                    None,
                ),
                error=r.error,
                stamps=stamp,
            )
            for r in results
            for stamp in (r.stamps or [None])
        ]
        w.record_llm_calls_batch(conn, journal)

        if rows:
            w.demote_current_for_calls(
                conn,
                table="call_reports",
                id_column="report_id",
                call_ids=[row.call_id for row in rows],
                exclude_ids=[row.report_id for row in rows],
                analyser_version=analyser_version,
            )
    return written


@flow(name="analysis-call-report")
def call_report_flow(
    analyser_version: str = "v3",
    limit: int | None = None,
    model: str = config.ANALYSIS_CHAT_MODEL,
) -> dict[str, int]:
    logger = get_run_logger()
    if analyser_version not in ANALYSERS:
        raise ValueError(f"unknown analyser_version {analyser_version!r}, expected one of {sorted(ANALYSERS)}")

    conn_cfg = trino_conn_from_config()
    engine = make_trino_engine(conn_cfg, schema=config.ANALYSIS_SCHEMA)
    with engine.connect() as conn:
        candidates = q.fetch_calls_needing_report(
            conn, analyser_version=analyser_version, limit=limit
        )
    logger.info("%s report candidates: %d", analyser_version, len(candidates))
    if not candidates:
        return {"candidates": 0, "written": 0}

    written, total = run_with_batched_persist(
        candidates,
        parallelism=1,
        batch_size=config.WRITE_CHUNK_SIZE,
        submit_analyse=lambda c: analyse_candidate.submit(
            c, analyser_version=analyser_version, model=model
        ),
        persist_batch=lambda chunk: _persist(
            chunk, analyser_version=analyser_version, model=model
        ),
    )
    logger.info("%s reports written: %d of %d", analyser_version, written, total)
    return {"candidates": total, "written": written}

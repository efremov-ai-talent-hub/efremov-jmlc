"""Transcribe calls that have audio but no current transcript."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prefect import flow, get_run_logger, task

from ai.shared import llm_tracing
from ai.transcription.transcriber import transcribe_audio
from pipelines import config
from pipelines.analysis.tables import queries as q
from pipelines.analysis.tables import writer as w
from pipelines.clients.trino import make_trino_engine, trino_conn_from_config
from pipelines.flows.analysis._orchestration import run_with_batched_persist
from pipelines.storage import minio


@dataclass
class TranscriptionResult:
    """Outcome of one candidate. Never raises out of the task.

    A raised exception would abort the whole wave and lose the sibling results
    that already cost money; carrying the failure as data lets it be recorded in
    the journal alongside the successes.
    """

    call_id: str
    ok: bool
    text: str = ""
    error: str | None = None
    stamps: list[Any] = field(default_factory=list)


@task(retries=0)
def transcribe_candidate(
    candidate: q.CallCandidate, *, whisper_model: str, chat_model: str
) -> TranscriptionResult:
    logger = get_run_logger()
    try:
        client = minio.make_minio_client()
        audio = minio.download_bytes(
            client, config.MINIO_RAW_FILES_BUCKET, candidate.audio_key
        )
        # bind() scopes the call to this entity so every LLM record inside
        # carries it; collect_stamps() buffers those records for the writer.
        with llm_tracing.collect_stamps() as stamps, llm_tracing.bind(
            entity_type="call", entity_id=candidate.call_id
        ):
            text = transcribe_audio(
                audio, whisper_model=whisper_model, chat_model=chat_model
            )
        return TranscriptionResult(
            call_id=candidate.call_id, ok=True, text=text, stamps=list(stamps)
        )
    except Exception as exc:  # noqa: BLE001 — carried as data, see docstring
        logger.warning("transcription failed for %s: %s", candidate.call_id, exc)
        return TranscriptionResult(
            call_id=candidate.call_id, ok=False, error=str(exc)[:2000]
        )


def _persist(results: list[TranscriptionResult], *, model: str) -> int:
    """Write one chunk: artifacts, then the journal, then demote.

    Order matters — see pipelines.analysis.tables.writer. The demote runs last
    and excludes the rows just written, so the call is never left without a
    current transcript.
    """
    conn_cfg = trino_conn_from_config()
    engine = make_trino_engine(conn_cfg, schema=config.ANALYSIS_SCHEMA)
    ok = [r for r in results if r.ok and r.text]
    written = 0
    with engine.connect() as conn:
        rows: list[w.TranscriptionRow] = []
        if ok:
            versions = w.next_versions_for_calls(
                conn, table="transcriptions", call_ids=[r.call_id for r in ok]
            )
            rows = [
                w.TranscriptionRow(
                    call_id=r.call_id,
                    version=versions[r.call_id],
                    text=r.text,
                    model=model,
                    transcription_id=w.new_id(),
                )
                for r in ok
            ]
            w.insert_transcriptions_batch(conn, rows)
            written = len(rows)

        journal = [
            w.LLMCallRow(
                call_id=r.call_id,
                status=w.STATUS_OK if r.ok else w.STATUS_FAILED,
                artifact_type=w.ARTIFACT_TRANSCRIPTION,
                artifact_id=next(
                    (row.transcription_id for row in rows if row.call_id == r.call_id),
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
                table="transcriptions",
                id_column="transcription_id",
                call_ids=[row.call_id for row in rows],
                exclude_ids=[row.transcription_id for row in rows],
            )
    return written


@flow(name="analysis-transcription")
def transcription_flow(
    limit: int | None = None,
    whisper_model: str = "transcription-gigaam",
    chat_model: str = "gigachat-lite",
) -> dict[str, int]:
    """Pick up untranscribed calls and transcribe them.

    Model names are the proxy's model groups, not provider model ids — the proxy
    is what maps them onto an actual backend.
    """
    logger = get_run_logger()
    conn_cfg = trino_conn_from_config()
    engine = make_trino_engine(conn_cfg, schema=config.ANALYSIS_SCHEMA)
    with engine.connect() as conn:
        candidates = q.fetch_calls_needing_transcription(conn, limit=limit)
    logger.info("transcription candidates: %d", len(candidates))
    if not candidates:
        return {"candidates": 0, "written": 0}

    written, total = run_with_batched_persist(
        candidates,
        parallelism=1,  # ASR on CPU: concurrency here buys nothing and starves the box
        batch_size=config.WRITE_CHUNK_SIZE,
        submit_analyse=lambda c: transcribe_candidate.submit(
            c, whisper_model=whisper_model, chat_model=chat_model
        ),
        persist_batch=lambda chunk: _persist(chunk, model=whisper_model),
    )
    logger.info("transcribed %d of %d", written, total)
    return {"candidates": total, "written": written}

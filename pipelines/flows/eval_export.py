"""Publish an evaluation dataset from the analysis tables to object storage.

The dataset pairs each current transcript with the current report produced for
it, which is what an eval run scores. It is written to the eval-datasets bucket
as JSONL — one self-contained record per line, so a run can stream it without
loading the whole file, and a published dataset stays readable even if the
tables move on underneath it.

Datasets are immutable once written: the key carries a UTC timestamp, and an
existing key is never overwritten. An eval result is only meaningful next to the
exact input it scored, so silently replacing a dataset would invalidate every
result that referenced it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from prefect import flow, get_run_logger
from sqlalchemy import text

from ai.shared import json_serde
from pipelines import config
from pipelines.analysis.tables._sql import qualified
from pipelines.clients.trino import make_trino_engine, trino_conn_from_config
from pipelines.storage import minio


def dataset_key(analyser_version: str, stamp: str) -> str:
    return f"call-analysis/{analyser_version}/{stamp}.jsonl"


@flow(name="eval-export-dataset")
def eval_export_flow(
    analyser_version: str = "v3",
    limit: int | None = None,
) -> dict[str, object]:
    """Export transcript/report pairs for one analyser."""
    logger = get_run_logger()
    conn_cfg = trino_conn_from_config()
    engine = make_trino_engine(conn_cfg, schema=config.ANALYSIS_SCHEMA)

    limit_sql = f"LIMIT {max(1, int(limit))}" if limit else ""
    query = text(
        f"""
        SELECT c.call_id,
               c.duration_seconds,
               t.transcription_id,
               t.version AS transcription_version,
               t.text AS transcript,
               t.model AS transcription_model,
               r.report_id,
               r.version AS report_version,
               r.report_json,
               r.model AS report_model,
               r.created_at AS report_created_at
        FROM {qualified("call_reports")} r
        JOIN {qualified("transcriptions")} t
          ON t.call_id = r.call_id AND t.is_current = true
        JOIN {qualified("calls")} c ON c.call_id = r.call_id
        WHERE r.is_current = true AND r.analyser_version = :analyser_version
        ORDER BY r.created_at
        {limit_sql}
        """
    )

    with engine.connect() as conn:
        rows = list(conn.execute(query, {"analyser_version": analyser_version}))

    if not rows:
        logger.info("nothing to export for analyser %s", analyser_version)
        return {"records": 0, "key": None}

    lines = []
    for r in rows:
        lines.append(
            json_serde.dumps(
                {
                    "call_id": r.call_id,
                    "duration_seconds": r.duration_seconds,
                    "transcript": r.transcript,
                    "transcription": {
                        "id": r.transcription_id,
                        "version": r.transcription_version,
                        "model": r.transcription_model,
                    },
                    "report": {
                        "id": r.report_id,
                        "version": r.report_version,
                        "analyser_version": analyser_version,
                        "model": r.report_model,
                        "created_at": r.report_created_at,
                        # Kept as a string: this is the analyser's payload
                        # verbatim, and re-encoding it here would let a change in
                        # this flow alter what an eval scores.
                        "payload": r.report_json,
                    },
                }
            )
        )
    body = ("\n".join(lines) + "\n").encode("utf-8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    key = dataset_key(analyser_version, stamp)
    client = minio.make_minio_client()
    if minio.object_exists(client, config.MINIO_EVAL_DATASETS_BUCKET, key):
        raise RuntimeError(f"dataset {key} already exists; refusing to overwrite")
    minio.upload_bytes(
        client,
        config.MINIO_EVAL_DATASETS_BUCKET,
        key,
        body,
        content_type="application/x-ndjson",
    )
    logger.info("exported %d records to %s", len(lines), key)
    return {"records": len(lines), "key": key, "bucket": config.MINIO_EVAL_DATASETS_BUCKET}

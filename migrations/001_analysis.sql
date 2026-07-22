-- Data model for call processing. Applied through Trino by the iceberg-migrate
-- service; every statement is idempotent, so it re-runs on each deploy.
--
-- Column sets are derived from what the ported domain layer actually emits:
-- ai/transcription/transcriber.py, ai/reports/call_v2, ai/reports/call_v3 and
-- ai/shared/llm_tracing.py (LLMStamps drives analysis.llm_calls one-to-one).

CREATE SCHEMA IF NOT EXISTS iceberg.analysis;

-- Source records. Seeded, not ingested: audio lives in the raw-files bucket and
-- the object key is held here, so no separate link table is needed.
CREATE TABLE IF NOT EXISTS iceberg.analysis.calls (
    call_id           varchar,
    audio_key         varchar,
    duration_seconds  double,
    started_at        timestamp(6) with time zone,
    manager           varchar,
    source            varchar,
    created_at        timestamp(6) with time zone
);

-- Transcripts. Append-only and versioned: a re-run adds a new version and the
-- previous one is demoted, because Iceberg does not take kindly to row updates.
-- `text` is stored inline rather than by reference — the volumes here are small
-- and being able to read a transcript straight from Trino is worth more than the
-- scan width. `segments_json` keeps the timed segments as serialized JSON;
-- Trino's JSON functions cover the querying we need without a sixth table.
CREATE TABLE IF NOT EXISTS iceberg.analysis.transcriptions (
    transcription_id     varchar,
    call_id              varchar,
    version              integer,
    is_current           boolean,
    text                 varchar,
    segments_json        varchar,
    language             varchar,
    duration_seconds     double,
    model                varchar,
    created_at           timestamp(6) with time zone,
    invalidated_at       timestamp(6) with time zone,
    invalidated_reason   varchar
);

-- Call reports, same versioning contract as transcripts. `analyser_version`
-- records which analyser produced the row (v2 / v3) — both are carried, and a
-- report is only comparable with others from the same analyser.
-- `report_json` holds the analyser payload as serialized JSON: its shape is
-- defined by the prompt and schema of each analyser version, not by this table.
CREATE TABLE IF NOT EXISTS iceberg.analysis.call_reports (
    report_id            varchar,
    call_id              varchar,
    version              integer,
    is_current           boolean,
    analyser_version     varchar,
    report_json          varchar,
    model                varchar,
    created_at           timestamp(6) with time zone,
    invalidated_at       timestamp(6) with time zone,
    invalidated_reason   varchar
);

-- One row per LLM call routed through the proxy, including failed ones
-- (`status='failed'`, `artifact_id` null) — that is what makes this the
-- processing journal as well as the trace anchor.
--
-- The trace works both ways: from a record, read `request_id` and search the
-- proxy's own logs; from a spend log, the same identifiers arrive in its
-- metadata. Cost and token columns allow aggregation without going back to the
-- proxy at all.
--
-- Fields mirror ai/shared/llm_tracing.py::LLMStamps one-to-one. `entity_type`
-- from the tracing layer is always "call" here, so the writer maps its
-- `entity_id` onto `call_id` and the discriminator is not stored.
CREATE TABLE IF NOT EXISTS iceberg.analysis.llm_calls (
    llm_call_id        varchar,
    call_id            varchar,
    artifact_type      varchar,
    artifact_id        varchar,
    status             varchar,
    error              varchar,
    kind               varchar,
    stage              varchar,
    worker_label       varchar,
    flow_run_id        varchar,
    task_run_id        varchar,
    deployment_name    varchar,
    session_id         varchar,
    request_id         varchar,
    model              varchar,
    model_id           varchar,
    model_group        varchar,
    cost_usd           double,
    tokens_in          integer,
    tokens_out         integer,
    latency_ms         integer,
    overhead_ms        integer,
    provider_proc_ms   integer,
    retries            integer,
    fallbacks          integer,
    cache_hit          boolean,
    created_at         timestamp(6) with time zone
);

-- Lineage: which source version fed which report version. This is what makes
-- versioning useful — without it a new transcript can be produced but the
-- reports it invalidates cannot be found.
CREATE TABLE IF NOT EXISTS iceberg.analysis.report_inputs (
    report_id        varchar,
    report_version   integer,
    source_type      varchar,
    source_id        varchar,
    source_version   integer,
    created_at       timestamp(6) with time zone
);

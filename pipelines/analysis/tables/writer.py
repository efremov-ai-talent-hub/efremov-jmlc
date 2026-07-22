"""Writes into the analysis tables.

Artifacts are append-only and versioned: a re-run inserts a new row with
``version = max(version) + 1`` and flips ``is_current`` on the previous one.
Content is never rewritten — the only update these tables see is that boolean
flip, which keeps history intact.

**These calls do not share a transaction.** The Trino DBAPI connection runs in
autocommit and the SQLAlchemy dialect's ``do_begin`` is a no-op, so every
statement commits on its own — insert and demote are two commits with a window
in between.

Insert first, then demote passing ``exclude_ids`` with the ids just written.
That ordering never leaves a call without a current artifact: the old row stays
current until the new one exists, and the window holds *two* current rows rather
than none. Two is recoverable — the next demote collapses it — while zero makes
the call look untranscribed and it is re-processed, at cost, on every run.
``exclude_ids`` is what makes this work: without it the demote would also clear
the row just inserted, which is the same zero-current state by another route.

Inserts are batched because each statement is one Iceberg commit, and a commit
writes a metadata file. Batch size is bounded by statement length rather than
row count: the Trino client inlines every parameter as a literal into an
``EXECUTE IMMEDIATE`` string, so transcripts count against Trino's
``query.max-length``.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ai.shared import json_serde
from pipelines.analysis.tables._sql import qualified

# Artifact kinds. These are written and read only by this package, which is why
# the retry cut-off keys on them: `stage` belongs to the domain layer, is
# free-form by its own documentation, and reading it here would couple us to
# labels that change without notice.
ARTIFACT_TRANSCRIPTION = "transcription"
_ARTIFACT_CALL_REPORT = "call_report"


def call_report_artifact_type(analyser_version: str) -> str:
    """Artifact kind for a report, carrying the analyser that produced it.

    The analyser has to live in this column rather than be joined in later: a
    failed attempt writes ``artifact_id = NULL`` by contract — no report row
    exists to join to — so any lookup through ``call_reports`` matches nothing
    for exactly the rows the retry cut-off needs to count.
    """
    return f"{_ARTIFACT_CALL_REPORT}:{analyser_version}"

STATUS_OK = "ok"
STATUS_FAILED = "failed"

# Lineage source kinds for report_inputs.source_type.
SOURCE_TRANSCRIPTION = "transcription"


def new_id() -> str:
    """Identifier for a new row.

    UUIDs rather than a sequence: Iceberg has none, and ``max(id) + 1`` would
    race whenever two workers insert at once.
    """
    return str(uuid.uuid4())


def as_entity_id(value: Any) -> str:
    """Normalise an identifier to the varchar form these tables store."""
    return str(value)


def _now() -> datetime:
    # Always UTC: the Trino client renders a tz-aware datetime as
    # TIMESTAMP '... UTC', and a fixed non-UTC offset would render as
    # 'UTC+03:00', which Trino rejects.
    return datetime.now(timezone.utc)


# --------------------------------------------------------------- versioning


def next_versions_for_calls(
    conn: Connection,
    *,
    table: str,
    call_ids: list[str],
    analyser_version: str | None = None,
) -> dict[str, int]:
    """Next version per call, in one round trip.

    ``analyser_version`` scopes the sequence for ``call_reports``: v2 and v3 are
    different analysers rather than successive versions of one, so each keeps its
    own numbering. Without it a call analysed by both would produce v2→1, v3→2,
    v2→3, and ``report_inputs.report_version`` would stop meaning "the Nth report
    from this analyser".

    Calls with no rows yet are absent from the grouped result and are filled in
    with 1 rather than silently dropped.
    """
    if not call_ids:
        return {}
    params: dict[str, Any] = {f"id{i}": v for i, v in enumerate(call_ids)}
    placeholders = ", ".join(f":{k}" for k in params)
    scope = ""
    if analyser_version is not None:
        scope = " AND analyser_version = :analyser_version"
        params["analyser_version"] = analyser_version
    query = text(
        f"""
        SELECT call_id, max(version) AS v
        FROM {qualified(table)}
        WHERE call_id IN ({placeholders}){scope}
        GROUP BY call_id
        """
    )
    found = {row.call_id: int(row.v) + 1 for row in conn.execute(query, params)}
    return {cid: found.get(cid, 1) for cid in call_ids}


def demote_current_for_calls(
    conn: Connection,
    *,
    table: str,
    id_column: str,
    call_ids: list[str],
    exclude_ids: list[str],
    analyser_version: str | None = None,
) -> None:
    """Clear ``is_current`` for the listed calls in a single statement.

    ``exclude_ids`` must carry the ids just inserted. This runs *after* the
    insert, so the new row also satisfies ``is_current = true``; without the
    exclusion it would be demoted along with the old one and the call would end
    up with no current artifact at all.

    ``analyser_version`` is required in practice for ``call_reports``: without it
    a new v3 report un-currents the call's v2 report, which then re-qualifies as
    a v2 candidate and is analysed again on the next run, forever.

    One UPDATE for the whole batch — each statement is an Iceberg commit, and a
    commit per call is what turns a routine re-run into hundreds of metadata
    files.
    """
    if not call_ids:
        return
    params: dict[str, Any] = {f"id{i}": v for i, v in enumerate(call_ids)}
    placeholders = ", ".join(f":{k}" for k in params)
    scope = ""
    if analyser_version is not None:
        scope = " AND analyser_version = :analyser_version"
        params["analyser_version"] = analyser_version
    keep = ""
    if exclude_ids:
        keep_params = {f"keep{i}": v for i, v in enumerate(exclude_ids)}
        params.update(keep_params)
        keep = f" AND {id_column} NOT IN ({', '.join(f':{k}' for k in keep_params)})"
    conn.execute(
        text(
            f"""
            UPDATE {qualified(table)}
            SET is_current = false
            WHERE call_id IN ({placeholders}) AND is_current = true{scope}{keep}
            """
        ),
        params,
    )


def _insert_batch(conn: Connection, *, table: str, columns: list[str], rows: list[dict]) -> None:
    """Multi-row INSERT so the whole batch lands in one Iceberg commit."""
    if not rows:
        return
    params: dict[str, Any] = {}
    tuples = []
    for i, row in enumerate(rows):
        keys = []
        for col in columns:
            # `p{i}_{col}` rather than `{col}_{i}`: a column whose name ends in
            # _<digits> would otherwise collide with another row's binding.
            key = f"p{i}_{col}"
            # row[col], not row.get(col) — a missing key means the column list and
            # the row builder have drifted apart, and writing NULL would hide it.
            params[key] = row[col]
            keys.append(f":{key}")
        tuples.append("(" + ", ".join(keys) + ")")
    conn.execute(
        text(
            f"INSERT INTO {qualified(table)} ({', '.join(columns)}) VALUES {', '.join(tuples)}"
        ),
        params,
    )


# ------------------------------------------------------------------ rows


@dataclass
class TranscriptionRow:
    call_id: str
    version: int
    text: str
    segments: list[dict] | None = None
    language: str | None = None
    duration_seconds: float | None = None
    model: str | None = None
    transcription_id: str = ""

    def as_params(self) -> dict[str, Any]:
        return {
            "transcription_id": self.transcription_id or new_id(),
            "call_id": self.call_id,
            "version": self.version,
            "is_current": True,
            "text": self.text,
            "segments_json": json_serde.dumps(self.segments)
            if self.segments is not None
            else None,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "created_at": _now(),
            "invalidated_at": None,
            "invalidated_reason": None,
        }


@dataclass
class CallReportRow:
    call_id: str
    version: int
    analyser_version: str
    report: dict
    model: str | None = None
    report_id: str = ""

    def as_params(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id or new_id(),
            "call_id": self.call_id,
            "version": self.version,
            "is_current": True,
            "analyser_version": self.analyser_version,
            "report_json": json_serde.dumps(self.report),
            "model": self.model,
            "created_at": _now(),
            "invalidated_at": None,
            "invalidated_reason": None,
        }


@dataclass
class CallRow:
    """A source call.

    Not an artifact: no version, no ``is_current``. Calls arrive from the seeder
    rather than being produced here, so the only rule is that ``call_id`` stays
    stable — everything downstream keys on it, and a re-seed that changed it
    would orphan every transcript and report already made for that call.
    """

    call_id: str
    audio_key: str
    duration_seconds: float | None
    started_at: datetime | None
    manager: str | None
    source: str

    def as_params(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "audio_key": self.audio_key,
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at,
            "manager": self.manager,
            "source": self.source,
            "created_at": _now(),
        }


@dataclass
class ReportInputRow:
    report_id: str
    report_version: int
    source_type: str
    source_id: str
    source_version: int

    def as_params(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_version": self.report_version,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "created_at": _now(),
        }


_STAMP_FIELDS = (
    "kind",
    "stage",
    "worker_label",
    "flow_run_id",
    "task_run_id",
    "deployment_name",
    "session_id",
    "request_id",
    "model",
    "model_id",
    "model_group",
    "cost_usd",
    "tokens_in",
    "tokens_out",
    "latency_ms",
    "overhead_ms",
    "provider_proc_ms",
    "retries",
    "fallbacks",
    "cache_hit",
)


@dataclass
class LLMCallRow:
    """One LLM call, successful or not.

    ``artifact_id`` is None on failure — no artifact row was written — while
    ``call_id`` stays populated, so a failed attempt is still attributable to the
    call it was made for. That is what makes this table the processing journal as
    well as the anchor for tracing a request into the proxy's logs, and it is why
    the retry cut-off can be computed from it.

    ``stamps`` takes the ``LLMCallRecord`` the tracing layer collects, or a plain
    mapping. Unknown fields are ignored rather than rejected: the tracing record
    carries more than this table stores, and gaining a field there should not
    break writes here.
    """

    call_id: str
    status: str
    artifact_type: str
    artifact_id: str | None = None
    error: str | None = None
    stamps: Any = None

    def as_params(self) -> dict[str, Any]:
        raw = self.stamps
        if raw is None:
            stamps: dict[str, Any] = {}
        elif is_dataclass(raw) and not isinstance(raw, type):
            stamps = asdict(raw)
        else:
            stamps = dict(raw)
        params: dict[str, Any] = {
            "llm_call_id": new_id(),
            "call_id": self.call_id,
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "status": self.status,
            "error": self.error,
            "created_at": _now(),
        }
        for field in _STAMP_FIELDS:
            params[field] = stamps.get(field)
        return params


# --------------------------------------------------------------- inserts

_CALL_COLUMNS = [
    "call_id",
    "audio_key",
    "duration_seconds",
    "started_at",
    "manager",
    "source",
    "created_at",
]

_TRANSCRIPTION_COLUMNS = [
    "transcription_id",
    "call_id",
    "version",
    "is_current",
    "text",
    "segments_json",
    "language",
    "duration_seconds",
    "model",
    "created_at",
    "invalidated_at",
    "invalidated_reason",
]

_CALL_REPORT_COLUMNS = [
    "report_id",
    "call_id",
    "version",
    "is_current",
    "analyser_version",
    "report_json",
    "model",
    "created_at",
    "invalidated_at",
    "invalidated_reason",
]

_REPORT_INPUT_COLUMNS = [
    "report_id",
    "report_version",
    "source_type",
    "source_id",
    "source_version",
    "created_at",
]

_LLM_CALL_COLUMNS = [
    "llm_call_id",
    "call_id",
    "artifact_type",
    "artifact_id",
    "status",
    "error",
    *_STAMP_FIELDS,
    "created_at",
]


def insert_calls_batch(conn: Connection, rows: list[CallRow]) -> None:
    _insert_batch(
        conn,
        table="calls",
        columns=_CALL_COLUMNS,
        rows=[r.as_params() for r in rows],
    )


def existing_call_ids(conn: Connection, call_ids: list[str]) -> set[str]:
    """Which of these calls are already recorded.

    The seeder is expected to be re-run, so it asks first rather than inserting
    and hoping: Iceberg has no unique constraint to lean on, and a second insert
    of the same call_id would simply create a duplicate row that every
    downstream join then multiplies.
    """
    if not call_ids:
        return set()
    params = {f"id{i}": v for i, v in enumerate(call_ids)}
    placeholders = ", ".join(f":{k}" for k in params)
    query = text(
        f"SELECT call_id FROM {qualified('calls')} WHERE call_id IN ({placeholders})"
    )
    return {row.call_id for row in conn.execute(query, params)}


def insert_transcriptions_batch(conn: Connection, rows: list[TranscriptionRow]) -> None:
    _insert_batch(
        conn,
        table="transcriptions",
        columns=_TRANSCRIPTION_COLUMNS,
        rows=[r.as_params() for r in rows],
    )


def insert_call_reports_batch(conn: Connection, rows: list[CallReportRow]) -> None:
    _insert_batch(
        conn,
        table="call_reports",
        columns=_CALL_REPORT_COLUMNS,
        rows=[r.as_params() for r in rows],
    )


def insert_report_inputs_batch(conn: Connection, rows: list[ReportInputRow]) -> None:
    _insert_batch(
        conn,
        table="report_inputs",
        columns=_REPORT_INPUT_COLUMNS,
        rows=[r.as_params() for r in rows],
    )


def record_llm_calls_batch(conn: Connection, rows: list[LLMCallRow]) -> None:
    _insert_batch(
        conn,
        table="llm_calls",
        columns=_LLM_CALL_COLUMNS,
        rows=[r.as_params() for r in rows],
    )

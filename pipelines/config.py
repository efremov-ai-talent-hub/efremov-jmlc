"""Environment-driven settings for the pipelines package.

Everything the flows need to reach the stack lives here rather than being read
from ``os.environ`` at the point of use, so the wire to the environment stays in
one place and the defaults match docker-compose.
"""

from __future__ import annotations

import os


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


# ---------------------------------------------------------------- Trino
TRINO_HOST = os.getenv("TRINO_HOST", "trino")
TRINO_PORT = _int("TRINO_PORT", 8080)
TRINO_USER = os.getenv("TRINO_USER", "pipelines")
TRINO_CATALOG = os.getenv("TRINO_CATALOG", "iceberg")
TRINO_HTTP_SCHEME = os.getenv("TRINO_HTTP_SCHEME", "http")

# The single schema this project writes to. See migrations/001_analysis.sql.
ANALYSIS_SCHEMA = os.getenv("ANALYSIS_SCHEMA", "analysis")

# ---------------------------------------------------------------- MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
# No defaults: the stand's real values live in .env, and baking them in here
# would let a misnamed variable authenticate successfully instead of failing.
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "")
MINIO_RAW_FILES_BUCKET = os.getenv("MINIO_RAW_FILES_BUCKET", "raw-files")
MINIO_EVAL_DATASETS_BUCKET = os.getenv("MINIO_EVAL_DATASETS_BUCKET", "eval-datasets")

# ---------------------------------------------------------------- Models
# Proxy model groups, not provider model ids — see infra/litellm/config.yaml for
# what is available. These are the flows' defaults; a deployment can still pass
# its own, but leaving the parameter unset here means the model is changed by
# editing .env and restarting the worker, with no re-registration.
ANALYSIS_CHAT_MODEL = os.getenv("ANALYSIS_CHAT_MODEL", "gigachat-lite")
ANALYSIS_WHISPER_MODEL = os.getenv("ANALYSIS_WHISPER_MODEL", "transcription-gigaam")

# ---------------------------------------------------------------- Analysis
# How many candidates a single flow run picks up. Kept small by default: this
# is a demo stand, and every candidate costs an LLM call.
ANALYSIS_LIMIT_TRANSCRIPTION = _int("ANALYSIS_LIMIT_TRANSCRIPTION", 10)
ANALYSIS_LIMIT_CALL_REPORT = _int("ANALYSIS_LIMIT_CALL_REPORT", 10)

# Rows per Iceberg commit. Iceberg writes a metadata file per commit, so
# committing per row would litter the warehouse with tiny files. Kept small
# because the Trino client inlines every parameter as a literal into a single
# EXECUTE IMMEDIATE string: a batch of full transcripts counts against Trino's
# query.max-length (1M characters by default), and that only shows up on real
# data, never in a smoke test.
WRITE_CHUNK_SIZE = _int("WRITE_CHUNK_SIZE", 10)

# A call is dropped from the candidate set after this many failed LLM calls.
# Counted from analysis.llm_calls (status='failed'), the journal of what was
# attempted — there is no separate attempts table. Note it counts calls, not
# runs: one report run emits several, so this budget is spent faster than
# "N failed runs" would suggest.
ANALYSIS_MAX_FAILED_ATTEMPTS = _int("ANALYSIS_MAX_FAILED_ATTEMPTS", 3)

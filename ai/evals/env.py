"""Load the handful of env vars an eval run needs from the repo-root ``.env``.

Values are taken literally apart from surrounding whitespace: the rendered ``.env``
never quotes them and carries no ``export`` prefix, so no unquoting is attempted.

Ansible renders that file from the ``env/<env>.yml`` blocks, so it holds the whole
platform config — DB passwords, the LiteLLM master key, SSO secrets. An eval run needs
a narrow slice of it (S3 creds for datasets, the gateway endpoint for labelers), so
loading is **allowlisted by prefix**: a runner names what it needs and nothing else
reaches the process environment. Anything already exported wins (``setdefault``).
"""

from __future__ import annotations

import os
from pathlib import Path

# parents[2] = repo root (this file is ai/evals/…).
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

# Datasets live in MinIO; ai.evals.dataset builds its client from EVAL_S3_* alone
# (region included), so the ambient AWS_* vars are deliberately not in the allowlist.
S3_PREFIXES = ("EVAL_S3_",)

# What ai.transcription's client resolves its endpoint + key from — the labelers reuse
# it, so the diarization case needs these on top of the dataset creds. The bare
# ``OPENAI_`` names are listed one by one instead of as a single ``OPENAI_`` prefix, so
# the allowlist does not silently widen as the rendered .env grows (they are still
# matched as prefixes — no ``OPENAI_*`` var extends them today).
#
# LITELLM_PROXY_* is here because ai.shared.llm_tracing swings the client onto the
# proxy when it is on: without them a labeler documented as "the exact prod polish
# step" would quietly take the legacy direct path and measure something else.
GATEWAY_PREFIXES = (
    "ANALYSIS_OPENAI_",
    "ENRICHMENT_OPENAI_",
    "OPENAI_API_BASE_URL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "LITELLM_PROXY_ENABLED",
    "LITELLM_PROXY_BASE_URL",
    "LITELLM_PROXY_API_KEY",
)

# The diarization case loads a dataset AND calls the gateway.
DIARIZATION_PREFIXES = S3_PREFIXES + GATEWAY_PREFIXES


def load_env_defaults(prefixes: tuple[str, ...]) -> None:
    """Set env vars whose name starts with one of ``prefixes`` from the repo-root .env."""
    try:
        text = ENV_FILE.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith(prefixes):
            os.environ.setdefault(key, value.strip())

"""OpenAI-compatible client resolution shared by report analysers.

Public counterpart of the private helpers in ``call_v2.core``. Works against any
OpenAI-compatible endpoint — the LiteLLM gateway in production, or a local
llama-server / vLLM when running a self-hosted model.
"""

from __future__ import annotations

import os

from openai import OpenAI

from ai.shared import llm_tracing

_BASE_URL_VARS = (
    "ANALYSIS_OPENAI_BASE_URL",
    "ENRICHMENT_OPENAI_BASE_URL",
    "OPENAI_API_BASE_URL",
    "OPENAI_BASE_URL",
)
_API_KEY_VARS = (
    "ANALYSIS_OPENAI_API_KEY",
    "ENRICHMENT_OPENAI_API_KEY",
    "OPENAI_API_KEY",
)
_MODEL_VARS = ("ANALYSIS_OPENAI_MODEL", "OPENAI_MODEL")

DEFAULT_BASE_URL = "https://api.openai.com/v1"


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def resolve_model_name(default_model: str) -> str:
    return _first_env(_MODEL_VARS) or default_model or "gpt-4o-mini"


def get_analysis_client() -> OpenAI:
    """Build the analysis client. ``LITELLM_PROXY_ENABLED=1`` swaps in the proxy."""
    base_url = _first_env(_BASE_URL_VARS) or DEFAULT_BASE_URL
    api_key = _first_env(_API_KEY_VARS)
    if not api_key:
        raise RuntimeError(
            "ANALYSIS_OPENAI_API_KEY (or ENRICHMENT_OPENAI_API_KEY / OPENAI_API_KEY) is required"
        )
    return OpenAI(
        **llm_tracing.resolve_openai_kwargs(default_api_key=api_key, default_base_url=base_url)
    )

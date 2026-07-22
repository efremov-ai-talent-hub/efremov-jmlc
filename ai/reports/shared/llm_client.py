"""OpenAI-compatible client resolution shared by report analysers.

Public counterpart of the private helpers in ``call_v2.core``. Works against any
OpenAI-compatible endpoint — the LiteLLM gateway in production, or a local
llama-server / vLLM when running a self-hosted model.
"""

from __future__ import annotations

import os
from typing import Any

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


def resolve_model_from_env(default_model: str) -> str:
    """Environment first, argument as the fallback.

    The inverse of :func:`resolve_model`, and deliberately so: the evals runner
    documents ANALYSIS_OPENAI_MODEL as an override of whatever a run requests.
    """
    return _first_env(_MODEL_VARS) or default_model or "gpt-4o-mini"


def resolve_model(model: Any, cfg: Any) -> str:
    """The model name the caller asked for.

    Accepts a plain string — in this project a model is the name of a proxy model
    group — or an object carrying ``.model_name``, or a cfg with
    ``call_analysis_model``. The caller always wins over the environment; use
    ``resolve_model_from_env`` where the environment should win instead.

    Raises rather than substituting a default: a model nobody chose is a bug in
    the caller, and a silent substitution sends every request under a name the
    proxy does not serve while the run still looks configured.
    """
    if isinstance(model, str) and model.strip():
        return model.strip()
    name = getattr(model, "model_name", None) or getattr(cfg, "call_analysis_model", None)
    if name and str(name).strip():
        return str(name).strip()
    raise ValueError(
        "no model to call: pass a proxy model group name (see infra/litellm/config.yaml)"
    )


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

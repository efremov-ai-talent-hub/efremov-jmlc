"""LLM-call tracing helper + LiteLLM proxy resolver.

Single source of truth for two concerns:

1. **Outbound tracing metadata.** Every chat-completion / audio-transcription
   call site reads a ContextVar-scoped ``LLMContext`` and pushes its fields
   onto the request via ``extra_body={"metadata": ...}`` (LiteLLM stores it on
   the spend log row) and ``extra_headers={"x-litellm-tags": ...}`` (LiteLLM
   indexes it for filtering and exposes it as a Prometheus label).

   The ContextVar is set once per Prefect task body — ``with bind(kind=...,
   stage=..., call_id=...): ...`` — and read deep inside ``ChatGPTModel`` /
   factory callables. ContextVar is the only clean way to thread tags through
   ``pydantic_ai.Agent``, which constructs its own model objects out of our
   reach.

2. **LiteLLM proxy switch.** Workers historically point at OpenAI directly via
   ``ENRICHMENT_OPENAI_*`` env vars. When ``LITELLM_PROXY_ENABLED=1``, every
   call site routes through the LiteLLM proxy: ``base_url`` becomes
   ``LITELLM_PROXY_BASE_URL`` and ``api_key`` becomes the per-kind virtual key
   (``LITELLM_VK_<KIND_UPPER>``). Off — bytes-for-bytes the same as today.

   The dark-launch flag is per-process: flipping it on a single environment
   does not require a code change.

Everything in this module is best-effort and side-effect-free at import time;
imports do not require Prefect to be installed (we attempt to read
``prefect.runtime`` only if available).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator, Literal, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMContext:
    """Snapshot of "what record / what flow / what stage" for an LLM call.

    Any field may be ``None`` — the helper only emits non-None values into
    request metadata. Callers fill in the IDs they know about; the rest stay
    empty.

    ``kind`` is the canonical model-group name (one of the eight defined in
    ``infra/litellm/config.yaml`` — ``analysis-call``, ``analysis-meeting``,
    ``analysis-manager``, ``analysis-lead``, ``enrichment-dealcards``,
    ``enrichment-leads``, ``transcription-whisper``, ``transcription-chat``).
    The helper uses ``kind`` to resolve the per-kind virtual key.

    ``stage`` is a free-form label scoping the call inside a kind
    (``pass0``, ``qc``, ``main``, ``report``, ``polish``, ``speaker_detect``,
    ``audit``, ``note_selector``, ``instalment``, ``discount``, ...).

    ``entity_type`` + ``entity_id`` identify the source record a call was made
    for: one varchar slot per side, so a single journal table covers every kind
    of subject without a column per kind. This project only produces
    ``entity_type="call"`` today; the pair is kept because the tracing layer is
    deliberately agnostic about what it is tracing.
    """

    entity_type: str | None = None
    entity_id: str | None = None
    kind: str | None = None
    stage: str | None = None
    flow_run_id: str | None = None
    task_run_id: str | None = None
    deployment_name: str | None = None


@dataclass(frozen=True)
class RequestExtras:
    """Kwargs shaped for ``openai.OpenAI().chat.completions.create(...)``.

    Always safe to splat — both dicts are well-formed even when the ContextVar
    is empty. ``extra_body`` is omitted for the audio API (Whisper rejects
    unknown body params); ``extra_headers`` carries the tags either way.
    """

    extra_body: dict[str, Any]
    extra_headers: dict[str, str]


# ---------------------------------------------------------------------------
# ContextVar plumbing
# ---------------------------------------------------------------------------


_CURRENT: ContextVar[LLMContext] = ContextVar("llm_tracing_current", default=LLMContext())


def current() -> LLMContext:
    """Read the current LLMContext. Returns an empty context if unset."""
    return _CURRENT.get()


@contextmanager
def bind(**overrides: Any) -> Iterator[LLMContext]:
    """Push a merged ``LLMContext`` onto the ContextVar for the duration of the block.

    Fields not given as overrides inherit from the parent context, so nested
    binds can refine the picture without re-stating everything (e.g. a task
    binds ``kind`` + ``call_id``; a sub-helper later binds ``stage="qc"``
    without re-stating ``call_id``).

    ``flow_run_id`` / ``task_run_id`` / ``deployment_name`` are auto-populated
    from ``prefect.runtime`` if available and not explicitly overridden. The
    fields are still ``None``-tolerant when called outside a flow (ad-hoc
    scripts, unit tests).
    """
    parent = _CURRENT.get()
    auto = _prefect_runtime_snapshot()
    merged_kwargs = {
        **{
            k: v
            for k, v in auto.items()
            # Only fill auto-fields that the parent didn't already carry and
            # the caller didn't explicitly override.
            if getattr(parent, k) is None and k not in overrides
        },
        **{k: v for k, v in overrides.items() if v is not None},
    }
    # ``replace`` raises on unknown fields, which is what we want — bind()
    # callers should not invent fields ad hoc; extend ``LLMContext`` instead.
    new_ctx = replace(parent, **merged_kwargs)
    token = _CURRENT.set(new_ctx)
    try:
        yield new_ctx
    finally:
        _CURRENT.reset(token)


def _prefect_runtime_snapshot() -> dict[str, str | None]:
    """Best-effort read of Prefect runtime IDs.

    Outside a flow/task (e.g. CLI scripts, unit tests) every field is ``None``.
    Inside a flow/task, all three are populated.
    """
    try:
        from prefect.runtime import deployment, flow_run, task_run  # type: ignore
    except Exception:  # pragma: no cover — Prefect optional at import time
        return {"flow_run_id": None, "task_run_id": None, "deployment_name": None}

    def _maybe(get_id_attr: Any, get_name_attr: Any = None) -> str | None:
        try:
            value = get_id_attr() if callable(get_id_attr) else get_id_attr
            return str(value) if value is not None else None
        except Exception:
            return None

    return {
        "flow_run_id": _maybe(getattr(flow_run, "get_id", None)),
        "task_run_id": _maybe(getattr(task_run, "get_id", None)),
        "deployment_name": _maybe(getattr(deployment, "get_name", None)),
    }


# ---------------------------------------------------------------------------
# Request-extras builder
# ---------------------------------------------------------------------------


def build_request_extras(
    *,
    user_label: str | None = None,
    mode: Literal["chat", "audio"] = "chat",
) -> RequestExtras:
    """Build the OpenAI-SDK ``extra_body`` / ``extra_headers`` for the current context.

    The function is pure-ish (only reads the ContextVar) and is safe to call
    when the context is empty — it returns minimally-shaped dicts in that case
    so call sites can always splat the result unconditionally.

    ``user_label`` is the existing hard-coded ``worker_*`` string that each
    call site already passes via OpenAI's ``user=`` parameter. We mirror it
    into ``metadata.worker_label`` for cross-correlation; the original
    ``user=`` argument is left untouched by callers — LiteLLM stores it in its
    native ``end_user`` column.

    ``mode='audio'`` omits ``extra_body`` (Whisper rejects unknown body params)
    but still emits ``extra_headers`` so tags reach the proxy.
    """
    ctx = current()
    md: dict[str, Any] = {k: v for k, v in asdict(ctx).items() if v is not None}
    if user_label:
        md["worker_label"] = user_label

    # Tags — flat ``key=value`` list, comma-separated. We do NOT dump every
    # metadata field as a tag (would inflate Prometheus label cardinality) —
    # only the low-cardinality discriminators are tag-worthy.
    tag_keys = ("kind", "stage", "deployment_name")
    tags = [f"{k}={md[k]}" for k in tag_keys if k in md]
    headers: dict[str, str] = {}
    if tags:
        headers["x-litellm-tags"] = ",".join(tags)
    # The dedicated ``LiteLLM_SpendLogs.metadata`` column is populated from
    # this header — NOT from ``extra_body.metadata``. The latter lands inside
    # the ``request`` JSON, which the UI shows under "Request" but not under
    # the "Metadata" block, and which the ``/spend/logs?metadata_key=…``
    # filter does not search. Sending both is intentional: ``extra_body``
    # keeps the metadata visible inside the request payload for debugging,
    # the header makes it indexable as a first-class field.
    if md:
        headers["x-litellm-spend-logs-metadata"] = json.dumps(md, ensure_ascii=False)

    if mode == "audio":
        return RequestExtras(extra_body={}, extra_headers=headers)
    body: dict[str, Any] = {"metadata": md} if md else {}
    # GigaChat-side knob: disable the provider profanity filter. Legitimate
    # real-estate sales language (e.g. discussions of price, layouts, even
    # neutral words combined with intonation hints in the prompt) triggers
    # GigaChat's classifier with finish_reason='blacklist' and empty content —
    # which surfaces downstream as ``ValueError("No JSON block found in model
    # output")`` from extract_json. We always send it; LiteLLM's GigaChat
    # transformation explicitly forwards profanity_check from optional_params
    # into the request body, and for non-GigaChat deployments LiteLLM drops it
    # via the proxy's drop_params=true. On the direct-OpenAI path (proxy off
    # — AI-dev local, hard fallback) OpenAI would 400 on an unknown param, so
    # we only emit it when the proxy is in the path.
    if proxy_enabled():
        body["profanity_check"] = False
    return RequestExtras(extra_body=body, extra_headers=headers)


# ---------------------------------------------------------------------------
# Proxy switch
# ---------------------------------------------------------------------------


def _truthy(raw: str | None) -> bool:
    return bool(raw) and str(raw).strip().lower() in {"1", "true", "yes", "on"}


def proxy_enabled() -> bool:
    """``LITELLM_PROXY_ENABLED`` master flag — defaults to off (no behavior change)."""
    return _truthy(os.getenv("LITELLM_PROXY_ENABLED"))


def resolve_openai_kwargs(
    *,
    default_api_key: str | None,
    default_base_url: str | None,
) -> dict[str, Any]:
    """Resolve ``api_key`` / ``base_url`` for an OpenAI client constructor.

    - ``LITELLM_PROXY_ENABLED`` off — returns the caller's defaults verbatim
      (preserves byte-for-byte today's behavior).
    - ``LITELLM_PROXY_ENABLED`` on — substitutes ``base_url`` with
      ``LITELLM_PROXY_BASE_URL`` and ``api_key`` with ``LITELLM_PROXY_API_KEY``
      (a single shared key, rendered into ``.env`` by Ansible from the master
      key). Missing values fall back to defaults with a loud warning — better
      to keep the flow running on the legacy path than to break in a way
      that's hard to diagnose at 3 AM.

    Per-kind isolation (rate-limits, throttle) lives on the LiteLLM side via
    ``tpm`` / ``rpm`` on each ``model_list`` entry in ``config.yaml`` — that's
    declarative and Ansible-managed. Per-worker spend attribution is captured
    on every request via OpenAI's ``user=worker_*`` field (LiteLLM stores it
    in the ``end_user`` spend-log column). So one shared API key is enough.

    Callers do ``OpenAI(**resolve_openai_kwargs(default_api_key=..., default_base_url=...))``.
    The function does not touch other constructor kwargs (timeout, http_client,
    etc.) — callers thread those independently.
    """
    out: dict[str, Any] = {}
    if default_api_key is not None:
        out["api_key"] = default_api_key
    if default_base_url:
        out["base_url"] = default_base_url

    if not proxy_enabled():
        return out

    proxy_base = (os.getenv("LITELLM_PROXY_BASE_URL") or "").strip()
    proxy_key = (os.getenv("LITELLM_PROXY_API_KEY") or "").strip()
    if not proxy_base or not proxy_key:
        logger.warning(
            "LITELLM_PROXY_ENABLED=1 but LITELLM_PROXY_BASE_URL or "
            "LITELLM_PROXY_API_KEY is empty; falling back to direct OpenAI path",
        )
        return out

    if not current().kind:
        # Not a fallback — proxy call will still succeed — but `kind` is what
        # drives Prometheus tags / spend-log grouping. Missing kind almost
        # always means a Prefect task forgot `with llm_tracing.bind(kind=...)`.
        logger.warning(
            "LITELLM_PROXY_ENABLED=1 but LLMContext.kind is unset; "
            "tags/metadata will be missing the kind dimension. "
            "Likely cause: missing llm_tracing.bind(kind=...) in the calling flow.",
        )

    out["api_key"] = proxy_key
    out["base_url"] = proxy_base
    return out


# ---------------------------------------------------------------------------
# Response-side capture: LLMCallRecord + stamps collection
# ---------------------------------------------------------------------------
#
# The capture layer is opt-in: callers wrap a code region in ``with
# collect_stamps() as stamps: ...`` and every LLM call routed through
# ``chat_create`` / ``async_chat_create`` / ``transcribe_create`` inside that
# region appends a fully-populated ``LLMCallRecord`` to the list.
#
# Outside a ``collect_stamps()`` block the helpers are silent no-ops: tests,
# CLI scripts and AI-dev workflows that don't care about Iceberg-side audit
# can run without changes.


@dataclass(frozen=True)
class LLMCallRecord:
    """One LiteLLM (or direct-OpenAI) call, ready to be persisted.

    Fields split into three groups:

    * **From LLMContext at call time** — captured by snapshotting the
      ContextVar inside ``chat_create``. This is how a stamp learns which
      ``entity_type`` / ``entity_id`` / ``flow_run_id`` / ``kind`` / ``stage``
      it belongs to without the call site having to pass them.

    * **From the LiteLLM response headers** (``x-litellm-*``,
      ``llm_provider-*``) — present only when the proxy is in the path.
      Direct-OpenAI calls leave these as ``None``.

    * **From the parsed response body** — token usage and the requested model
      name. Same for direct OpenAI calls.

    Every field is ``Optional``: a record from a non-proxy run still carries
    LLMContext fields, just with empty header/cost columns.
    """

    # --- LLMContext snapshot ---
    kind: str | None = None
    stage: str | None = None
    worker_label: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    flow_run_id: str | None = None
    task_run_id: str | None = None
    deployment_name: str | None = None

    # --- LiteLLM identifiers ---
    # `session_id`  = LiteLLM's internal call UUID from `x-litellm-call-id`
    #                 header. Shown as "Session ID" in the Admin UI.
    # `request_id`  = upstream provider's response id from `parsed.id`
    #                 (chatcmpl-* for OpenAI, similar for GigaChat; transcr-*
    #                 minted by our GigaAM wrapper, which the audio API has no
    #                 native id for). Shown as "Request ID" in the Admin UI;
    #                 this is the field the /ui/logs filter searches by — keep
    #                 it queryable on our side too. NULL only for providers
    #                 that emit no response id.
    session_id: str | None = None
    request_id: str | None = None
    model_id: str | None = None
    model_group: str | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    overhead_ms: int | None = None
    retries: int | None = None
    fallbacks: int | None = None
    cache_hit: bool | None = None
    provider_proc_ms: int | None = None

    # --- Response body (usage object) ---
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    # --- Arbitrary opaque payload ---
    # Free-form dict for ad-hoc producer-defined fields (experiment id,
    # custom slicing, A/B markers, ...). Serialized to ``metadata varchar``
    # via ``json_serde.dumps`` at write time. ``None`` today — there is no
    # producer wired in yet; the column is created so future callers can
    # populate it without a schema migration.
    metadata: dict[str, Any] | None = None


def _coerce_float(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _coerce_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _coerce_bool(raw: Any) -> bool | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return None


def _provider_processing_ms(headers: Mapping[str, str]) -> int | None:
    """Pick the upstream provider's processing-time header (e.g. ``llm_provider-openai-processing-ms``).

    LiteLLM forwards the original provider headers prefixed with
    ``llm_provider-``. The exact suffix varies per provider; we don't pin
    one, we just take the first matching ``*processing-ms`` header.
    """
    for k, v in headers.items():
        key = k.lower() if hasattr(k, "lower") else str(k).lower()
        if key.startswith("llm_provider-") and key.endswith("processing-ms"):
            return _coerce_int(v)
    return None


def stamp_from_response(
    parsed: Any,
    headers: Mapping[str, str] | None,
    *,
    requested_model: str | None = None,
    worker_label: str | None = None,
) -> LLMCallRecord:
    """Build an ``LLMCallRecord`` from a parsed response + headers + current LLMContext.

    Pure function: it does not mutate anything, does not push to the ContextVar
    list. Suitable for unit tests with fake responses.

    ``parsed`` is the OpenAI SDK object (``ChatCompletion`` /
    ``Transcription`` / etc.). Token counts come from ``parsed.usage`` when
    present; Whisper responses don't carry usage, so those stay ``None``.
    """
    h: Mapping[str, str] = headers or {}
    ctx = current()
    md = {k: v for k, v in asdict(ctx).items() if v is not None}

    # Try to read token counts off ``parsed.usage`` — works for chat
    # completions and any other endpoint that follows OpenAI's usage shape.
    tokens_in: int | None = None
    tokens_out: int | None = None
    usage = getattr(parsed, "usage", None)
    if usage is not None:
        tokens_in = _coerce_int(getattr(usage, "prompt_tokens", None))
        tokens_out = _coerce_int(getattr(usage, "completion_tokens", None))

    model_from_parsed = getattr(parsed, "model", None)
    # parsed.id — chatcmpl-* on chat completions, transcr-* on GigaAM audio
    # (it mints its own; native audio API has none).
    request_id_from_body = getattr(parsed, "id", None)
    return LLMCallRecord(
        kind=md.get("kind"),
        stage=md.get("stage"),
        # worker_label is NOT part of LLMContext (it's a per-call kwarg the SDK
        # passes via `user=`), so the wrapper extracts it from the create()
        # kwargs and passes it in explicitly.
        worker_label=worker_label,
        entity_type=md.get("entity_type"),
        entity_id=md.get("entity_id"),
        flow_run_id=md.get("flow_run_id"),
        task_run_id=md.get("task_run_id"),
        deployment_name=md.get("deployment_name"),
        session_id=h.get("x-litellm-call-id"),
        request_id=str(request_id_from_body) if request_id_from_body is not None else None,
        model_id=h.get("x-litellm-model-id"),
        model_group=h.get("x-litellm-model-group"),
        cost_usd=_coerce_float(h.get("x-litellm-response-cost")),
        latency_ms=_coerce_int(h.get("x-litellm-response-duration-ms")),
        overhead_ms=_coerce_int(h.get("x-litellm-overhead-duration-ms")),
        retries=_coerce_int(h.get("x-litellm-attempted-retries")),
        fallbacks=_coerce_int(h.get("x-litellm-attempted-fallbacks")),
        cache_hit=_coerce_bool(h.get("x-litellm-cache-hit")),
        provider_proc_ms=_provider_processing_ms(h),
        model=str(model_from_parsed) if model_from_parsed is not None else requested_model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


_STAMPS: ContextVar[list[LLMCallRecord] | None] = ContextVar("llm_tracing_stamps", default=None)


@contextmanager
def collect_stamps() -> Iterator[list[LLMCallRecord]]:
    """Collect every LLM call that happens inside the block into a list.

    Outside a ``collect_stamps()`` block, ``chat_create`` / ``transcribe_create``
    still work but their records are discarded — so the AI-dev workflow does
    not need Iceberg / LiteLLM at all to exercise the same code paths.

    Nested blocks each get their own buffer (no leakage upward).
    """
    buffer: list[LLMCallRecord] = []
    token = _STAMPS.set(buffer)
    try:
        yield buffer
    finally:
        _STAMPS.reset(token)


def push_stamp(record: LLMCallRecord) -> None:
    """Append a record to the active ``collect_stamps()`` buffer (no-op if none)."""
    buf = _STAMPS.get()
    if buf is not None:
        buf.append(record)


# ---------------------------------------------------------------------------
# OpenAI SDK wrappers — use these instead of direct .create() to get capture
# ---------------------------------------------------------------------------
#
# These thin wrappers replace
#
#     resp = client.chat.completions.create(model=..., messages=..., ...)
#
# with
#
#     resp = chat_create(client, model=..., messages=..., ...)
#
# returning the same ``ChatCompletion`` object so the surrounding code is
# unchanged. Behind the scenes we go through ``with_raw_response.create`` to
# inspect ``x-litellm-*`` headers and push a stamp.


def _worker_label_from_kwargs(kwargs: dict[str, Any]) -> str | None:
    """Pick the OpenAI ``user=`` field out of create() kwargs.

    Chat completions accept ``user`` as a top-level kwarg; the audio
    transcriptions endpoint rejects unknown body params and we route the
    worker label via ``extra_body={"user": "..."}`` instead. Check both.
    """
    direct = kwargs.get("user")
    if direct:
        return str(direct)
    body = kwargs.get("extra_body") or {}
    if isinstance(body, dict):
        nested = body.get("user")
        if nested:
            return str(nested)
    return None


def chat_create(client: Any, /, **kwargs: Any) -> Any:
    """Sync wrapper around ``client.chat.completions.create(**kwargs)`` that records the call.

    Returns the parsed ``ChatCompletion`` exactly like ``.create()`` would. If
    response inspection fails for any reason (older SDK, missing headers,
    direct-OpenAI path) the stamp is silently dropped — the call result is
    still returned to the caller.
    """
    raw = client.chat.completions.with_raw_response.create(**kwargs)
    parsed = raw.parse()
    try:
        push_stamp(
            stamp_from_response(
                parsed,
                raw.headers,
                requested_model=kwargs.get("model"),
                worker_label=_worker_label_from_kwargs(kwargs),
            )
        )
    except Exception:  # pragma: no cover — capture must never break the call
        logger.exception("llm_tracing: failed to capture stamp; continuing")
    return parsed


async def async_chat_create(client: Any, /, **kwargs: Any) -> Any:
    """Async counterpart of :func:`chat_create` for ``AsyncOpenAI`` clients."""
    raw = await client.chat.completions.with_raw_response.create(**kwargs)
    parsed = raw.parse()
    try:
        push_stamp(
            stamp_from_response(
                parsed,
                raw.headers,
                requested_model=kwargs.get("model"),
                worker_label=_worker_label_from_kwargs(kwargs),
            )
        )
    except Exception:  # pragma: no cover
        logger.exception("llm_tracing: failed to capture async stamp; continuing")
    return parsed


def transcribe_create(client: Any, /, **kwargs: Any) -> Any:
    """Wrapper around ``client.audio.transcriptions.create(**kwargs)``.

    Whisper responses don't carry a ``usage`` object, so token counts stay
    ``None``; the LiteLLM headers (cost, latency, retries, model_id) are
    still recorded.
    """
    raw = client.audio.transcriptions.with_raw_response.create(**kwargs)
    parsed = raw.parse()
    try:
        push_stamp(
            stamp_from_response(
                parsed,
                raw.headers,
                requested_model=kwargs.get("model"),
                worker_label=_worker_label_from_kwargs(kwargs),
            )
        )
    except Exception:  # pragma: no cover
        logger.exception("llm_tracing: failed to capture transcription stamp; continuing")
    return parsed


__all__ = [
    "LLMContext",
    "LLMCallRecord",
    "RequestExtras",
    "bind",
    "current",
    "build_request_extras",
    "resolve_openai_kwargs",
    "proxy_enabled",
    "collect_stamps",
    "push_stamp",
    "stamp_from_response",
    "chat_create",
    "async_chat_create",
    "transcribe_create",
]

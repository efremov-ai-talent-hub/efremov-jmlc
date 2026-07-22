from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ai.shared import llm_tracing

logger = logging.getLogger(__name__)


def _role_to_str(role: Any) -> str:
    if hasattr(role, "value"):
        role = role.value
    role = str(role)
    if role in {"system", "user", "assistant"}:
        return role
    if "assistant" in role or "model" in role:
        return "assistant"
    if "system" in role:
        return "system"
    return "user"


def _extract_text(msg: ModelMessage) -> str:
    """Concatenate every part's text — kept for backward compatibility with
    earlier callers / tests. The production request path now iterates parts
    individually via :func:`_part_text` so it can preserve roles."""
    parts = getattr(msg, "parts", []) or []
    out: list[str] = []
    for part in parts:
        text = _part_text(part)
        if text:
            out.append(text)
    return "".join(out).strip()


def _part_text(part: Any) -> str:
    """Best-effort string extraction from a single pydantic_ai message part.

    Falls through ``content`` → ``text`` → ``str(part)`` because pydantic_ai
    part shapes vary by part type (``SystemPromptPart.content``,
    ``TextPart.content``, ``UserPromptPart.content`` — all strings; tool
    parts have richer shapes but we don't speak tools here).
    """
    if hasattr(part, "content") and part.content:
        return str(part.content).strip()
    if hasattr(part, "text") and part.text:
        return str(part.text).strip()
    value = str(part).strip()
    return value if value and value != "None" else ""


def _part_role(part: Any, *, fallback: str = "user") -> str:
    """Map a pydantic_ai message part to a chat-completion role.

    Detection is by class name (``SystemPromptPart`` / ``UserPromptPart`` /
    ``TextPart`` / ``RetryPromptPart`` / ``ToolReturnPart`` / ``ToolCallPart``)
    so we don't take a hard dependency on the exact symbol path inside
    pydantic_ai — that import has moved between minor versions.

    ``fallback`` is what we return when nothing in the class name hints at a
    role (e.g. a mock part in tests).
    """
    name = type(part).__name__.lower()
    if "system" in name:
        return "system"
    if "userprompt" in name or "retryprompt" in name:
        return "user"
    if "toolcall" in name or "toolreturn" in name:
        # Tool exchanges are intentionally projected into assistant — we don't
        # have a tool/function loop here, so the safest is to keep them as
        # context attached to the conversation.
        return "assistant"
    if "text" in name:
        return "assistant"
    return fallback


def _clean_to_json(text: str) -> str:
    value = (text or "").strip()
    if "```" in value:
        first = value.find("```")
        last = value.rfind("```")
        if last > first:
            inner = value[first + 3 : last].strip()
            lines = inner.splitlines()
            if lines and lines[0].lower().strip() in {"json", "javascript"}:
                inner = "\n".join(lines[1:]).strip()
            value = inner
    left = value.find("{")
    right = value.rfind("}")
    if left != -1 and right != -1 and right > left:
        value = value[left : right + 1].strip()
    return value


class ChatGPTModel(Model):
    system = "openai"

    def __init__(
        self,
        api_key_env: str = "ANALYSIS_OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = 60.0,
        model_name: str | None = None,
        user: str | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._model_name = (
            model_name
            or os.getenv("ANALYSIS_OPENAI_MODEL")
            or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        )
        self._user = user

    @property
    def model_name(self) -> str:
        return self._model_name

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: dict,
    ) -> ModelResponse:
        # Match ai.reports.call.core._get_client() so pass0 and pydantic_ai
        # agents behave the same when only ENRICHMENT_* is set.
        api_key = (
            (os.getenv("ANALYSIS_OPENAI_API_KEY") or "").strip()
            or (os.getenv("ENRICHMENT_OPENAI_API_KEY") or "").strip()
            or (os.getenv("OPENAI_API_KEY") or "").strip()
        )
        if not api_key:
            raise RuntimeError(
                "ANALYSIS_OPENAI_API_KEY (or ENRICHMENT_OPENAI_API_KEY / OPENAI_API_KEY) is required"
            )

        # Iterate parts of each pydantic_ai message and emit one chat-completion
        # message per part. Before this, every part — including SystemPromptPart
        # — was concatenated under role="user", which: (a) confused providers
        # that lean on the system role for instruction-following (GigaChat in
        # particular degenerated into repetition loops on long combined-user
        # prompts), and (b) made the spend-log payload misleading for tracing.
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            parts = getattr(msg, "parts", []) or []
            msg_role_hint = getattr(msg, "role", None)
            fallback_role = _role_to_str(msg_role_hint) if msg_role_hint is not None else "user"
            for part in parts:
                text = _part_text(part)
                if not text:
                    continue
                role = _part_role(part, fallback=fallback_role)
                chat_messages.append({"role": role, "content": text})
        if not chat_messages:
            raise RuntimeError("ChatGPTModel: messages are empty")

        base_url = (
            (os.getenv("ANALYSIS_OPENAI_BASE_URL") or "").strip()
            or (os.getenv("ENRICHMENT_OPENAI_BASE_URL") or "").strip()
            or (os.getenv("OPENAI_API_BASE_URL") or "").strip()
            or (os.getenv("OPENAI_BASE_URL") or "").strip()
            or self._base_url
        )
        # When LITELLM_PROXY_ENABLED=1, swap api_key/base_url to the proxy
        # (per-kind virtual key resolved from current LLMContext.kind).
        # Off — kwargs come through verbatim.
        client_kwargs = llm_tracing.resolve_openai_kwargs(
            default_api_key=api_key, default_base_url=base_url
        )
        masked_key = (
            client_kwargs["api_key"][:7] + "..." + client_kwargs["api_key"][-4:]
            if len(client_kwargs.get("api_key") or "") > 11
            else "***"
        )
        logger.info(
            "ChatGPTModel: base_url=%s, api_key=%s, model=%s",
            client_kwargs.get("base_url"),
            masked_key,
            self._model_name,
        )

        extras = llm_tracing.build_request_extras(user_label=self._user)
        client = OpenAI(**client_kwargs)
        # llm_tracing.chat_create wraps the SDK with `with_raw_response.create`
        # so the response headers (x-litellm-call-id, cost, latency, ...) are
        # captured into the active collect_stamps() buffer. Direct-OpenAI path
        # (no proxy in front) still works — stamps for those calls have all
        # litellm_* fields as None.
        resp = llm_tracing.chat_create(
            client,
            model=self._model_name,
            messages=chat_messages,
            temperature=0.0,
            extra_body=extras.extra_body,
            extra_headers=extras.extra_headers,
            **({"user": self._user} if self._user else {}),
        )
        content = resp.choices[0].message.content
        cleaned = _clean_to_json(content)
        return ModelResponse(parts=[TextPart(cleaned)])

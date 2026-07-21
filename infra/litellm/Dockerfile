# Custom LiteLLM image with the GigaChat finish_reason patch baked in.
#
# Why a custom image at all: upstream LiteLLM's Pydantic Literal[...] for
# Choices.finish_reason doesn't include GigaChat's "blacklist" / "error"
# values. Without the patch, every blocked-by-filter response throws
# ValidationError and the client sees HTTP 500 with no usable info.
# Issue tracker: https://github.com/BerriAI/litellm/issues/21348
#
# The patch is anchor-based — if upstream changes the line
# `finish_reason = choice.get("finish_reason", "stop")` in
# litellm/llms/gigachat/chat/transformation.py, the build FAILS, which
# forces a manual review instead of silently degrading.
#
# Pinned to a specific stable release for deterministic builds.
# Floating tags (main-latest, main-stable) caused unpredictable behaviour —
# a rebuild could silently pick up a different commit and regress streaming
# / A2A behaviour. See infra/litellm/README.md "Upgrading LiteLLM" for the
# bump procedure.
#
# Floor: v1.89.0 (SSE buffering fix — required for A2A streaming transparency).
# Also requires: A2A Agent Gateway (v1.80.8+) — needed for chat-agent platform.
FROM ghcr.io/berriai/litellm:v1.89.3

# Patch GigaChat finish_reason mapping (blacklist/error -> content_filter)
# until upstream fixes it. See: https://github.com/BerriAI/litellm/issues/21348
COPY patches/gigachat_finish_reason.py /tmp/gigachat_finish_reason.py
RUN python /tmp/gigachat_finish_reason.py && rm /tmp/gigachat_finish_reason.py

# Patch A2A PascalCase→wire method map: native a2a-python sends `SendMessage`
# and `SendStreamingMessage` (PascalCase JSON-RPC method names), LiteLLM's
# _PASCAL_TO_WIRE map omits them, so every native A2A client gets a 400
# "Method '...' not found". Anchor-based; build FAILS if upstream refactors.
# See patches/a2a_method_map.py header.
COPY patches/a2a_method_map.py /tmp/a2a_method_map.py
RUN python /tmp/a2a_method_map.py && rm /tmp/a2a_method_map.py

# Patch OpenTelemetry to tolerate a non-dict response_obj. LiteLLM's OTEL callback
# calls response_obj.get(...) blindly; MCP `/mcp` request "responses" are JSON
# lists, so it raises "'list' object has no attribute 'get'" and logs an error on
# every MCP call (caught — request still completes — but noisy, and it drops the
# span attributes for those calls). Mirrors upstream fix BerriAI/litellm#26713
# (not yet released as of the pinned v1.89.3). Anchor-based; build FAILS if the
# set_attributes signature changes. See patches/otel_nondict_response.py header.
COPY patches/otel_nondict_response.py /tmp/otel_nondict_response.py
RUN python /tmp/otel_nondict_response.py && rm /tmp/otel_nondict_response.py

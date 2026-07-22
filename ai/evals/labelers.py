"""Text speaker-labelers for the diarization case — the system under test.

Each takes a label-STRIPPED transcript (timecodes + text, no ``[РОЛЬ]``) and returns
the transcript with ``[МЕНЕДЖЕР]``/``[КЛИЕНТ]`` re-applied, via ONE LLM call to the
LiteLLM gateway. This is the only LLM in the diarization case — the grader
(:mod:`ai.evals.checks.diarization`) is deterministic. Needs the gateway env (same as
:mod:`ai.transcription`), whose production client + polish step are reused so the
``polish`` labeler measures exactly what prod does.

- ``label_via_polish`` — the exact prod polish step. On a stripped (mono-view)
  transcript it measures the labels prod already invents blind on real mono calls.
- ``label_semantic`` — a dedicated diarize-from-text prompt (the v3 candidate).
"""

from __future__ import annotations

import re

from ai.shared import llm_tracing
from ai.transcription.transcriber import _get_openai_client, _llm_polish_transcript

# Matches the "[МЕНЕДЖЕР] " / "[КЛИЕНТ] " role tag only (timecodes are digits, untouched).
_ROLE_TAG = re.compile(r"\[(?:МЕНЕДЖЕР|КЛИЕНТ)\]\s*")


def strip_labels(transcript: str) -> str:
    """Drop role tags, keep timecodes + text — the channel-blind 'mono-view' input."""
    return _ROLE_TAG.sub("", transcript)


_SEMANTIC_SYSTEM = (
    "Ты размечаешь спикеров в транскрипте телефонного звонка в отдел продаж "
    "недвижимости. В транскрипте НЕТ меток спикеров.\n"
    "Для КАЖДОЙ строки определи ПО СМЫСЛУ, кто говорит — менеджер по продажам или "
    "клиент — и поставь метку [МЕНЕДЖЕР] или [КЛИЕНТ] сразу ПОСЛЕ таймкода.\n"
    "Ориентиры: менеджер представляет компанию, консультирует, ведёт к сделке; "
    "клиент спрашивает, возражает, интересуется условиями.\n"
    "НЕ меняй таймкоды и текст, НЕ добавляй и НЕ удаляй строки. Верни ВЕСЬ транскрипт "
    "строка за строкой.\n"
    "Формат каждой строки: [00.00–00.00] [МЕНЕДЖЕР] текст"
)


def label_semantic(stripped: str, *, chat_model: str) -> str:
    """v3 candidate: assign a speaker role to every line from TEXT alone."""
    with llm_tracing.bind(kind="transcription-chat", stage="eval_diarize"):
        client = _get_openai_client()
        extras = llm_tracing.build_request_extras(user_label="eval_diarize")
        response = llm_tracing.chat_create(
            client,
            model=chat_model,
            messages=[
                {"role": "system", "content": _SEMANTIC_SYSTEM},
                {"role": "user", "content": stripped},
            ],
            temperature=0,
            user="eval_diarize",
            extra_body=extras.extra_body,
            extra_headers=extras.extra_headers,
            timeout=300,
        )
    return str(response.choices[0].message.content or "")


def label_via_polish(stripped: str, *, chat_model: str) -> str:
    """Prod baseline: the exact polish step — the blind labelling prod does today."""
    with llm_tracing.bind(kind="transcription-chat"):
        client = _get_openai_client()
    return _llm_polish_transcript(stripped, client=client, chat_model=chat_model)


LABELERS = {"semantic": label_semantic, "polish": label_via_polish}

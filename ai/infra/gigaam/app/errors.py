"""Ошибки враппера с HTTP-классификацией.

Ключ к «не залипающим» звонкам: транзиентное (5xx) vs терминальное (4xx). Флоу
по коду решает, писать transient (ретраебельно) или failed (после N). Поэтому
важно НЕ отдавать голый 500 на всё подряд — каждый этап бросает осмысленный тип.
"""

from __future__ import annotations


class WrapperError(Exception):
    http_status: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, http_status: int | None = None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status


class AudioError(WrapperError):
    """Терминальное: битый/неподдерживаемый/пустой аудиофайл. Ретраить бессмысленно."""

    http_status = 422
    code = "audio_error"


class TranscriptionTransientError(WrapperError):
    """Транзиентное: OOM, таймаут, временный сбой модели/VAD. Ретраебельно."""

    http_status = 503
    code = "transient_error"


def to_error_body(exc: WrapperError) -> dict:
    """OpenAI-подобное тело ошибки (LiteLLM/SDK донесут код/сообщение во флоу)."""
    return {"error": {"message": exc.message, "type": exc.code, "code": exc.code}}

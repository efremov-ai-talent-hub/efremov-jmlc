"""Провайдер-агностичная модель транскрипта — lingua franca между движком и
OpenAI-маппингом. Сюда же добавляются будущие поля (insights/emotion/speaker)
без правки адаптеров, которые их не заполняют.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class CanonicalTranscript:
    text: str
    language: str  # полное имя, как в whisper-ответе ("russian")
    duration: float
    words: list[Word] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    # ШОВ под будущее: optional insights/emotion/speaker добавляются здесь.

"""Интерфейс ASR-движка — шов под сменные адаптеры (GigaAM сейчас; завтра
SaluteSpeech/Whisper и т.п. без правки api/mapping/segmentation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..canonical import CanonicalTranscript


class TranscriptionEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Тёплая загрузка модели в RAM один раз при старте сервиса."""

    @abstractmethod
    def transcribe(self, audio_path: str) -> CanonicalTranscript:
        """Распознать файл. Должен бросать errors.AudioError (терминал) /
        TranscriptionTransientError (транзиент), а не голый Exception."""

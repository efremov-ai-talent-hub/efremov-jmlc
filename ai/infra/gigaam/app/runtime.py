"""Рантайм: тёплый движок + гейт конкурентности (CPU-предохранитель).

Гейт = семафор на 1: на машине не запускается два инференса разом, кто бы ни
постучал. Это backstop на стороне сервиса — очерёдность вызовов задаёт
вызывающий (оркестратор). Гейт НЕ дропает (никаких 503-by-busy,
переупорядочивающих очередь) — он блокирует и пропускает по одному.
"""

from __future__ import annotations

import anyio

from . import metrics
from .canonical import CanonicalTranscript
from .config import Config
from .engines.gigaam import GigaAMEngine


class Runtime:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.engine = GigaAMEngine(cfg)
        self._gate = anyio.Semaphore(1)
        self.loaded = False

    def load(self) -> None:
        self.engine.load()
        self.loaded = True
        metrics.model_loaded.set(1)

    async def transcribe(self, audio_path: str) -> CanonicalTranscript:
        # Блокирующий CPU-инференс уводим в поток, чтобы event loop (а значит
        # /health и /metrics) оставался живым во время длинной транскрипции.
        async with self._gate:
            metrics.inflight.inc()
            try:
                return await anyio.to_thread.run_sync(self.engine.transcribe, audio_path)
            finally:
                metrics.inflight.dec()

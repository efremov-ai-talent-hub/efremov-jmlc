"""Конфиг враппера из окружения.

Все значения — из env (их задаёт docker-compose из .env). Дефолты безопасны
для локального запуска.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


@dataclass(frozen=True)
class Config:
    # ASR-движок и модель
    model_name: str
    threads: int

    # VAD-нарезка (точность ASR; дефолт = заводской GigaAM, крупные точные куски)
    vad_max_duration: float | None
    vad_min_duration: float | None
    vad_strict_limit: float | None

    # склейка слов -> выходные сегменты
    glue_max_seconds: float  # верхний кап длины сегмента при склейке

    # кэш весов (том)
    download_root: str

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model_name=os.getenv("GIGAAM_MODEL", "v3_e2e_rnnt"),
            threads=_int("GIGAAM_THREADS", 3),
            # None → не передаём в transcribe_longform, движок берёт свои дефолты
            vad_max_duration=_float("GIGAAM_VAD_MAX_DUR", None)
            if os.getenv("GIGAAM_VAD_MAX_DUR")
            else None,
            vad_min_duration=_float("GIGAAM_VAD_MIN_DUR", None)
            if os.getenv("GIGAAM_VAD_MIN_DUR")
            else None,
            vad_strict_limit=_float("GIGAAM_VAD_STRICT_DUR", None)
            if os.getenv("GIGAAM_VAD_STRICT_DUR")
            else None,
            glue_max_seconds=_float("GIGAAM_GLUE_MAX_SEC", 8.0),
            download_root=os.getenv("GIGAAM_DOWNLOAD_ROOT", "/cache/gigaam"),
        )

    def vad_kwargs(self) -> dict:
        out: dict[str, float] = {}
        if self.vad_max_duration is not None:
            out["max_duration"] = self.vad_max_duration
        if self.vad_min_duration is not None:
            out["min_duration"] = self.vad_min_duration
        if self.vad_strict_limit is not None:
            out["strict_limit_duration"] = self.vad_strict_limit
        return out

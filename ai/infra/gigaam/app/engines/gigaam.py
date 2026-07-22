"""GigaAM-движок: грузит v3 (git-версия gigaam) один раз, транскрибирует
longform с пословными таймкодами. Все квирки GigaAM локализованы здесь.
"""

from __future__ import annotations

import logging
import os
import time

from ..canonical import CanonicalTranscript, Word
from ..config import Config
from ..errors import AudioError, TranscriptionTransientError
from .base import TranscriptionEngine

logger = logging.getLogger("gigaam.engine")


def _patch_torch_load() -> None:
    """torch 2.6+ дефолт weights_only=True ломает чекпойнт pyannote/segmentation-3.0
    (longform-VAD). Веса с HF — доверенные, форсим weights_only=False."""
    import torch

    if getattr(torch.load, "_gigaam_patched", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    _load._gigaam_patched = True  # type: ignore[attr-defined]
    torch.load = _load  # type: ignore[assignment]


def _wf(w) -> tuple[float, float, str]:
    """(start, end, text) слова — объект или dict (защитно к версии gigaam)."""
    if isinstance(w, dict):
        return (
            float(w.get("start", 0)),
            float(w.get("end", 0)),
            w.get("text") or w.get("word") or "",
        )
    return float(getattr(w, "start", 0)), float(getattr(w, "end", 0)), getattr(w, "text", "")


def _words_of(seg) -> list:
    w = getattr(seg, "words", None)
    if w is None and isinstance(seg, dict):
        w = seg.get("words")
    return w or []


def _seg_start(seg) -> float:
    if isinstance(seg, dict):
        b = seg.get("boundaries")
        return float(b[0]) if b else float(seg.get("start", 0))
    return float(getattr(seg, "start", 0))


class GigaAMEngine(TranscriptionEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    def load(self) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cfg.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.cfg.threads))
        import torch

        torch.set_num_threads(self.cfg.threads)
        _patch_torch_load()
        import gigaam

        t0 = time.monotonic()
        try:
            self._model = gigaam.load_model(
                self.cfg.model_name, download_root=self.cfg.download_root
            )
        except TypeError:
            self._model = gigaam.load_model(self.cfg.model_name)
        logger.info("GigaAM '%s' loaded in %.1fs", self.cfg.model_name, time.monotonic() - t0)

    def transcribe(self, audio_path: str) -> CanonicalTranscript:
        if self._model is None:
            raise TranscriptionTransientError("engine not loaded yet")

        from pydub import AudioSegment

        # 1) длительность (required-поле ответа). Ошибка чтения = терминальный аудио-сбой.
        try:
            duration = len(AudioSegment.from_file(audio_path)) / 1000.0
        except Exception as e:  # noqa: BLE001
            raise AudioError(f"cannot read audio: {e}") from e

        # 2) распознавание. Падение здесь трактуем как транзиентное (OOM/таймаут/runtime);
        # терминальные форматные проблемы обычно отлавливаются на чтении выше.
        try:
            segs = self._model.transcribe_longform(
                audio_path, word_timestamps=True, **self.cfg.vad_kwargs()
            )
        except Exception as e:  # noqa: BLE001
            raise TranscriptionTransientError(f"transcription failed: {e}") from e

        words = self._flatten_words(segs)
        text = " ".join(w.text for w in words).strip()
        return CanonicalTranscript(
            text=text,
            language="russian",
            duration=duration,
            words=words,
            segments=[],  # склейку делает segmentation на уровне сервиса
        )

    @staticmethod
    def _flatten_words(segs) -> list[Word]:
        """Слова call-глобально. Определяем, относительны ли таймкоды слов сегменту
        (тогда +seg.start) — по первому сегменту с start>2с."""
        seg_list = list(segs) if isinstance(segs, (list, tuple)) else getattr(segs, "segments", [])
        relative: bool | None = None
        for seg in seg_list:
            ws = _words_of(seg)
            ss = _seg_start(seg)
            if ws and ss > 2.0 and relative is None:
                relative = _wf(ws[0])[0] < ss - 1.0
        out: list[Word] = []
        for seg in seg_list:
            off = _seg_start(seg) if relative else 0.0
            for w in _words_of(seg):
                s, e, t = _wf(w)
                out.append(Word(start=off + s, end=off + e, text=t))
        return out

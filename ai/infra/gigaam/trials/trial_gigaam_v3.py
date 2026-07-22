#!/usr/bin/env python3
"""Замер GigaAM-v3 на CPU (путь HuggingFace AutoModel) + контроль гранулярности.

v3 нет в pip-пакете gigaam — грузим через transformers + trust_remote_code.
word-level таймкодов у v3 НЕТ (подтверждено по исходнику), но крупность кусков
регулируется VAD-параметрами: внутренний model.model.transcribe_longform(**kwargs)
пробрасывает их в segment_audio_file(max_duration=22, min_duration=15,
strict_limit_duration=30, new_chunk_threshold=0.2). Дефолт = крупные 15-22с куски.

Запуск из docker-compose. Свип гранулярности БЕЗ пересборки — через -e:

    docker compose run --rm trial-v3 /audio/call.mp3 e2e_rnnt          # дефолт
    docker compose run --rm -e GIGAAM_MAX_DUR=6 -e GIGAAM_MIN_DUR=2 \\
        -e GIGAAM_STRICT_DUR=8 trial-v3 /audio/call.mp3 e2e_rnnt       # мельче

Ревизии: ctc, rnnt, e2e_ctc, e2e_rnnt (e2e_* — с пунктуацией и нормализацией).
"""

import os
import sys
import time

THREADS = int(os.environ.get("GIGAAM_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch  # noqa: E402

torch.set_num_threads(THREADS)

# torch 2.6+ сменил дефолт torch.load на weights_only=True. Чекпойнт
# pyannote/segmentation-3.0 (его грузит longform v3) содержит глобал
# TorchVersion, который строгий анпиклер отвергает → UnpicklingError. Веса
# pyannote с HF — доверенный источник, поэтому возвращаем weights_only=False.
_orig_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    # Принудительно, а не setdefault: lightning/pyannote передают
    # weights_only=True явным аргументом, и setdefault его не перебил бы.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

from pydub import AudioSegment  # noqa: E402
from transformers import AutoModel  # noqa: E402


def audio_seconds(path: str) -> float:
    return len(AudioSegment.from_file(path)) / 1000.0


def _vad_kwargs() -> dict:
    """VAD-параметры из окружения — только заданные (остальные = дефолты модели)."""
    env_map = {
        "GIGAAM_MAX_DUR": "max_duration",
        "GIGAAM_MIN_DUR": "min_duration",
        "GIGAAM_STRICT_DUR": "strict_limit_duration",
        "GIGAAM_NEW_CHUNK_THR": "new_chunk_threshold",
    }
    out = {}
    for env, kw in env_map.items():
        val = os.environ.get(env)
        if val is not None:
            out[kw] = float(val)
    return out


def _seg_fields(seg):
    """v3 longform → list[dict] {'transcription', 'boundaries':(s,e)}; защитно."""
    if isinstance(seg, dict):
        b = seg.get("boundaries") or (seg.get("start"), seg.get("end"))
        return float(b[0]), float(b[1]), seg.get("transcription") or seg.get("text") or ""
    return (
        float(getattr(seg, "start", 0.0)),
        float(getattr(seg, "end", 0.0)),
        getattr(seg, "text", ""),
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: trial_gigaam_v3.py <audio_file> [revision]")
    path = sys.argv[1]
    revision = sys.argv[2] if len(sys.argv) > 2 else "e2e_rnnt"
    vad = _vad_kwargs()

    dur = audio_seconds(path)
    print(f"audio={dur:.0f}s  model=GigaAM-v3:{revision}  threads={THREADS}", flush=True)
    print(f"VAD-параметры: {vad or 'дефолт (max=22/min=15/strict=30)'}", flush=True)

    t0 = time.monotonic()
    model = AutoModel.from_pretrained(
        "ai-sage/GigaAM-v3", revision=revision, trust_remote_code=True
    )
    if hasattr(model, "to"):
        model = model.to("cpu")
    if hasattr(model, "eval"):
        model.eval()
    print(f"load={time.monotonic() - t0:.1f}s", flush=True)

    # С кастомными VAD-параметрами зовём ВНУТРЕННИЙ model.model.transcribe_longform
    # (его **kwargs идут в segment_audio_file); внешний HF-обёртка kwargs не пробрасывает.
    inner = getattr(model, "model", model)
    t0 = time.monotonic()
    if vad:
        out = inner.transcribe_longform(path, **vad)
    else:
        out = model.transcribe_longform(path)
    compute = time.monotonic() - t0

    rtf = compute / dur if dur else float("nan")
    print(f"compute={compute:.0f}s  RTF={rtf:.2f}x", flush=True)

    segs = [_seg_fields(s) for s in out] if isinstance(out, (list, tuple)) else []
    if segs:
        lens = sorted(e - s for s, e, _ in segs)
        n = len(lens)
        avg = sum(lens) / n
        print(
            f"гранулярность: {n} сегм., длительность мин={lens[0]:.1f}s "
            f"средн={avg:.1f}s медиана={lens[n // 2]:.1f}s макс={lens[-1]:.1f}s",
            flush=True,
        )
    print("--- транскрипт (первые сегменты) ---", flush=True)
    for s, e, text in segs[:25]:
        print(f"[{s:6.1f}-{e:6.1f}] ({e - s:4.1f}s) {text}", flush=True)
    if not segs:
        print(f"(неожиданная форма ответа, type={type(out).__name__}): {out!r:.500}", flush=True)


if __name__ == "__main__":
    main()

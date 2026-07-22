#!/usr/bin/env python3
"""Замер GigaAM на CPU перед тем, как строить serving-враппер.

Цель — получить честный RTF (real-time factor = время счёта / длительность
аудио) на реальном звонке, с ограничением числа потоков, чтобы цифра отражала
реальность "делим машину с другими сервисами", а не "забрал все ядра".

Запускается ТОЛЬКО через docker-compose из этой папки (см. README.md):

    cp .env.example .env        # вписать HF_TOKEN
    # положить запись звонка в ./audio/call.mp3
    docker compose run --rm trial /audio/call.mp3 v3_rnnt

Сравнивать вывод транскрипта по-русски с whisper-транскриптом того же
звонка — это и есть оценка качества.
"""

import os
import sys
import time

# Лимит потоков ДО импорта torch — иначе он уже захватит все ядра.
# Значение приходит из compose (GIGAAM_THREADS); дефолт 3 из 6 — оставляем
# половину машины под остальное (на проде это критично).
THREADS = int(os.environ.get("GIGAAM_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch  # noqa: E402

torch.set_num_threads(THREADS)

import gigaam  # noqa: E402
from pydub import AudioSegment  # noqa: E402


def audio_seconds(path: str) -> float:
    return len(AudioSegment.from_file(path)) / 1000.0


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: trial_gigaam.py <audio_file> [model_name]")
    path = sys.argv[1]
    # v3_rnnt — лучшая точность; v3_ctc — быстрее; v3_e2e_rnnt — с пунктуацией/
    # нормализацией (читабельнее для звонков). Прогоните несколько и сравните.
    model_name = sys.argv[2] if len(sys.argv) > 2 else "v3_rnnt"

    dur = audio_seconds(path)
    print(f"audio={dur:.0f}s  model={model_name}  threads={THREADS}", flush=True)

    t0 = time.monotonic()
    # download_root на том ./cache — иначе gigaam 0.1.0 качает веса заново
    # каждый прогон (HF_HOME/TORCH_HOME он не слушает). try/except — на случай
    # версии без этого параметра.
    try:
        model = gigaam.load_model(model_name, download_root="/cache/gigaam")
    except TypeError:
        model = gigaam.load_model(model_name)
    print(f"load={time.monotonic() - t0:.1f}s  (разовая загрузка модели в RAM)", flush=True)

    t0 = time.monotonic()
    segments = model.transcribe_longform(path)
    compute = time.monotonic() - t0

    rtf = compute / dur if dur else float("nan")
    print(f"compute={compute:.0f}s  RTF={rtf:.2f}x", flush=True)
    print("  (RTF<1 — быстрее реалтайма; RTF=2 — 2 сек счёта на 1 сек аудио)", flush=True)
    print("--- транскрипт (первые сегменты) ---", flush=True)
    print(f"(raw первого сегмента: {segments[0]!r})", flush=True)
    for seg in segments[:15]:
        # gigaam 0.1.0 отдаёт dict; новее — объект. Поддерживаем оба.
        if isinstance(seg, dict):
            bounds = seg.get("boundaries") or [None, None]
            start = seg.get("start", bounds[0])
            end = seg.get("end", bounds[1])
            text = seg.get("transcription") or seg.get("text") or ""
        else:
            start, end = getattr(seg, "start", None), getattr(seg, "end", None)
            text = getattr(seg, "text", "")
        ts = f"{float(start):6.1f}-{float(end):6.1f}" if start is not None else "   ?   "
        print(f"[{ts}] {text}", flush=True)


if __name__ == "__main__":
    main()

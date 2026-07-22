#!/usr/bin/env python3
"""Тур по возможностям GigaAM на реальном звонке (pip-пакет, v1/v2-семейство).

Эмоции (emo) и SSL-эмбеддинги есть только в pip-пакете — в HF-карточке v3 их
нет, поэтому демо на v2. Архитектура та же, смысл возможностей идентичен.

Показывает:
  1) ASR с ДОСЛОВНЫМИ таймкодами (word_timestamps) — суперсет под диаризацию
  2) распознавание эмоций по окнам звонка (модель emo)
  3) SSL-эмбеддинг — вектор аудио, основа кластеризации спикеров (моно-диаризация)

Запуск:
    docker compose run --rm features /audio/call.mp3
"""

import os
import sys

THREADS = int(os.environ.get("GIGAAM_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch  # noqa: E402

torch.set_num_threads(THREADS)

import gigaam  # noqa: E402
from pydub import AudioSegment  # noqa: E402


def _load(name: str):
    try:
        return gigaam.load_model(name, download_root="/cache/gigaam")
    except TypeError:
        return gigaam.load_model(name)


def slice_to_tmp(path: str, start_s: float, end_s: float) -> str:
    audio = AudioSegment.from_file(path)[int(start_s * 1000) : int(end_s * 1000)]
    out = f"/tmp/slice_{int(start_s)}_{int(end_s)}.wav"
    audio.export(out, format="wav")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: trial_features.py <audio_file>")
    path = sys.argv[1]

    # 1) короткий ASR (первые 20с). word_timestamps в pip 0.1.0 нет (TypeError) —
    # проверяем наличие, чтобы зафиксировать факт; word-level смотрим на v3.
    print("=== 1) короткий transcribe (первые 20с) ===", flush=True)
    asr = _load("v2_rnnt")
    sl = slice_to_tmp(path, 0, 20)
    try:
        res = asr.transcribe(sl, word_timestamps=True)
        words = getattr(res, "words", None) or (res.get("words") if isinstance(res, dict) else None)
        for w in list(words or [])[:40]:
            wt = getattr(w, "word", None) or getattr(w, "text", None)
            if wt is None and isinstance(w, dict):
                wt = w.get("word") or w.get("text")
            print(f"  [{getattr(w, 'start', '?')}-{getattr(w, 'end', '?')}] {wt}", flush=True)
        if not words:
            print(f"  (нет .words — raw: {repr(res)[:300]})", flush=True)
    except TypeError:
        res = asr.transcribe(sl)
        print("  word_timestamps НЕ поддержан в pip 0.1.0 — проверить на v3", flush=True)
        print(f"  текст: {repr(res)[:300]}", flush=True)

    # 2) ЭМОЦИИ по окнам — показываем, что меняется по ходу звонка
    print("\n=== 2) эмоции (модель emo) по окнам ===", flush=True)
    emo = _load("emo")
    dur = len(AudioSegment.from_file(path)) / 1000.0
    windows = [(0, 15), (dur / 2, dur / 2 + 15), (max(0, dur - 15), dur)]
    for a, b in windows:
        try:
            probs = emo.get_probs(slice_to_tmp(path, a, b))
            top = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            pretty = ", ".join(f"{k}={v:.2f}" for k, v in top)
            print(f"  [{a:.0f}-{b:.0f}s] {pretty}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{a:.0f}-{b:.0f}s] ошибка: {e}", flush=True)

    # 3) SSL-ЭМБЕДДИНГ — вектор аудио (основа кластеризации спикеров)
    print("\n=== 3) SSL-эмбеддинг ===", flush=True)
    try:
        ssl = _load("v2_ssl")
        emb, _ = ssl.embed_audio(slice_to_tmp(path, 0, 20))
        print(f"  shape={tuple(emb.shape)} dtype={emb.dtype}", flush=True)
        print("  (такой вектор на сегмент → кластеризация = кто говорит, даже на моно)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ошибка: {e}", flush=True)


if __name__ == "__main__":
    main()

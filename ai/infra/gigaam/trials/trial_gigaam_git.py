#!/usr/bin/env python3
"""GigaAM из git main: проверка v3 + word_timestamps на CPU.

PyPI застрял на 0.1.0 (v1/v2, без word_timestamps). git main даёт v3 через
load_model и word-level таймкоды. Проверяем три вещи:
  1) load_model("v3_e2e_rnnt") вообще грузится из git-версии;
  2) transcribe(slice, word_timestamps=True) -> result.words (короткое аудио);
  3) transcribe_longform(path, word_timestamps=True, **vad) -> есть ли .words
     у сегментов на длинном аудио (README говорит нет — проверяем опытом).

Запуск:
    docker compose build trial-git
    docker compose run --rm trial-git /audio/call.mp3 v3_e2e_rnnt
    # VAD как в trial-v3: -e GIGAAM_MAX_DUR=6 -e GIGAAM_MIN_DUR=2 -e GIGAAM_STRICT_DUR=12
"""

import os
import sys
import time

THREADS = int(os.environ.get("GIGAAM_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch  # noqa: E402

torch.set_num_threads(THREADS)

# torch 2.6+ дефолт weights_only=True ломает чекпойнт pyannote — форсим False.
_orig_torch_load = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_trusted

import gigaam  # noqa: E402
from pydub import AudioSegment  # noqa: E402


def audio_seconds(path: str) -> float:
    return len(AudioSegment.from_file(path)) / 1000.0


def slice_to_tmp(path: str, a: float, b: float) -> str:
    out = "/tmp/slice.wav"
    AudioSegment.from_file(path)[int(a * 1000) : int(b * 1000)].export(out, format="wav")
    return out


def _vad_kwargs() -> dict:
    env_map = {
        "GIGAAM_MAX_DUR": "max_duration",
        "GIGAAM_MIN_DUR": "min_duration",
        "GIGAAM_STRICT_DUR": "strict_limit_duration",
        "GIGAAM_NEW_CHUNK_THR": "new_chunk_threshold",
    }
    return {kw: float(os.environ[e]) for e, kw in env_map.items() if e in os.environ}


def _words_of(seg):
    w = getattr(seg, "words", None)
    if w is None and isinstance(seg, dict):
        w = seg.get("words")
    return w or []


def _seg_text(seg):
    if isinstance(seg, dict):
        return seg.get("transcription") or seg.get("text") or ""
    return getattr(seg, "text", "")


def _wf(w):
    """(start, end, text) слова — объект или dict."""
    if isinstance(w, dict):
        return (
            float(w.get("start", 0)),
            float(w.get("end", 0)),
            w.get("text") or w.get("word") or "",
        )
    return float(getattr(w, "start", 0)), float(getattr(w, "end", 0)), getattr(w, "text", "")


def _flatten_abs_words(segs):
    """Все слова call-глобально. Определяем, относительны ли таймкоды слов
    сегменту (тогда прибавляем seg.start) — по первому сегменту с start>2с."""
    relative = None
    for seg in segs:
        ws = _words_of(seg)
        seg_start = float(
            getattr(seg, "start", 0) or (seg.get("start", 0) if isinstance(seg, dict) else 0)
        )
        if ws and seg_start > 2.0 and relative is None:
            relative = _wf(ws[0])[0] < seg_start - 1.0
    out = []
    for seg in segs:
        seg_start = float(
            getattr(seg, "start", 0) or (seg.get("start", 0) if isinstance(seg, dict) else 0)
        )
        off = seg_start if relative else 0.0
        for w in _words_of(seg):
            s, e, t = _wf(w)
            out.append((off + s, off + e, t))
    return out, bool(relative)


def _glue_sentences(words, max_dur=8.0):
    """Склейка слов в сегменты: граница = конец предложения (.?!…) или max_dur."""
    segs, cur, cur_start = [], [], None
    for s, e, t in words:
        if cur_start is None:
            cur_start = s
        cur.append(t)
        if t.rstrip().endswith((".", "?", "!", "…")) or (e - cur_start) >= max_dur:
            segs.append((cur_start, e, " ".join(cur)))
            cur, cur_start = [], None
    if cur:
        segs.append((cur_start, words[-1][1], " ".join(cur)))
    return segs


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: trial_gigaam_git.py <audio_file> [model_name]")
    path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "v3_e2e_rnnt"
    vad = _vad_kwargs()

    print(
        f"gigaam={getattr(gigaam, '__version__', '?')}  model={model_name}  threads={THREADS}",
        flush=True,
    )

    t0 = time.monotonic()
    try:
        model = gigaam.load_model(model_name, download_root="/cache/gigaam")
    except TypeError:
        model = gigaam.load_model(model_name)
    print(f"load={time.monotonic() - t0:.1f}s", flush=True)

    # 1) короткий transcribe с word_timestamps
    print("\n=== 1) transcribe(word_timestamps=True), первые 20с ===", flush=True)
    try:
        res = model.transcribe(slice_to_tmp(path, 0, 20), word_timestamps=True)
        words = getattr(res, "words", None) or (res.get("words") if isinstance(res, dict) else None)
        for w in list(words or [])[:25]:
            print(
                f"  [{getattr(w, 'start', '?'):.2f}-{getattr(w, 'end', '?'):.2f}] {getattr(w, 'text', w)}",
                flush=True,
            )
        if not words:
            print(f"  нет .words; raw: {repr(res)[:200]}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ОШИБКА: {type(e).__name__}: {e}", flush=True)

    # 2) longform с word_timestamps — главный вопрос
    print("\n=== 2) transcribe_longform(word_timestamps=True) ===", flush=True)
    dur = audio_seconds(path)
    t0 = time.monotonic()
    try:
        out = model.transcribe_longform(path, word_timestamps=True, **vad)
    except TypeError as e:
        print(f"  word_timestamps не принят longform ({e}); пробую без него", flush=True)
        out = model.transcribe_longform(path, **vad)
    compute = time.monotonic() - t0

    segs = list(out) if isinstance(out, (list, tuple)) else getattr(out, "segments", [])
    rtf = compute / dur if dur else float("nan")
    n_words = sum(len(_words_of(s)) for s in segs)
    print(f"  RTF={rtf:.2f}x  сегментов={len(segs)}  слов с таймкодами={n_words}", flush=True)
    print(f"  → longform отдаёт слова: {'ДА' if n_words else 'НЕТ'}", flush=True)

    # A) штатные крупные сегменты longform — ПОЛНЫЙ текст (оценка качества)
    print("\n--- A) крупные сегменты longform (как есть) ---", flush=True)
    for seg in segs:
        s = float(getattr(seg, "start", 0) or (seg.get("start", 0) if isinstance(seg, dict) else 0))
        e = float(getattr(seg, "end", 0) or (seg.get("end", 0) if isinstance(seg, dict) else 0))
        print(f"[{s:6.1f}-{e:6.1f}] {_seg_text(seg)}", flush=True)

    # B) ПЕРЕСБОРКА из слов в мелкие сегменты (предложения / ≤8с) — то, что
    # реально отдавал бы враппер. Видно и качество, и зернистость.
    if n_words:
        words, rel = _flatten_abs_words(segs)

        # B0) СЫРЫЕ слова из longform — доказательство, что пословные таймкоды
        # реально есть (склейка в B строится из НИХ, не выдумана).
        print(
            f"\n--- B0) сырые слова из longform ({'отн.сегменту' if rel else 'абсолютные'} таймкоды), первые 40 из {len(words)} ---",
            flush=True,
        )
        for s, e, t in words[:40]:
            print(f"  [{s:6.2f}-{e:6.2f}] {t}", flush=True)

        # B) пересборка ИЗ этих слов в мелкие сегменты (то, что отдаёт враппер)
        print("\n--- B) пересборка по словам в сегменты (по предложениям / ≤8с) ---", flush=True)
        fine = _glue_sentences(words, max_dur=8.0)
        print(f"  {len(fine)} мелких сегм. из {len(words)} слов:", flush=True)
        for s, e, t in fine:
            print(f"[{s:6.1f}-{e:6.1f}] ({e - s:4.1f}s) {t}", flush=True)

        # 3) ЭМОЦИЯ по сегментам — траектория тона, выровненная с текстом
        print("\n=== 3) эмоция по сегментам (модель emo) ===", flush=True)
        ch = AudioSegment.from_file(path).channels
        print(
            f"  каналов: {ch} ({'стерео → можно по-спикерно по каналам' if ch >= 2 else 'моно → только смешанная эмоция'})",
            flush=True,
        )
        try:
            emo = gigaam.load_model("emo", download_root="/cache/gigaam")
        except TypeError:
            emo = gigaam.load_model("emo")
        MIN_EMO = 1.5  # короче — эмоция ненадёжна
        for s, e, t in fine:
            if e - s < MIN_EMO:
                continue
            try:
                probs = emo.get_probs(slice_to_tmp(path, s, e))
                ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                top = ordered[0][0]
                dist = " ".join(f"{k}={v:.2f}" for k, v in ordered)
                print(f"[{s:6.1f}-{e:6.1f}] {top:8s} | {dist} | {t[:55]}", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"[{s:6.1f}-{e:6.1f}] emo error: {ex}", flush=True)


if __name__ == "__main__":
    main()

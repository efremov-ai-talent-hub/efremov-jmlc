#!/usr/bin/env python3
"""ONNX vs PyTorch для ЯДРА GigaAM на CPU — стоит ли ради экономии CPU.

ВАЖНО: ONNX-путь gigaam (load_onnx/infer_onnx) — сырое распознавание короткого
аудио (≤25-30с). Он НЕ делает longform-нарезку, word_timestamps и, возможно,
e2e-нормализацию — это Python-обвязка вокруг PyTorch-модели. Поэтому меряем
ускорение ЯДРА на коротком клипе, не всего пайплайна.

Оговорка по потокам: число потоков onnxruntime задаётся при создании сессии
внутри load_onnx — мы им отсюда не управляем. Контейнер ограничен cpus:3
(compose), torch — GIGAAM_THREADS. Сравнение ориентировочное; смотрим порядок.

Запуск:
    docker compose build trial-onnx
    docker compose run --rm trial-onnx /audio/call.mp3 v3_e2e_rnnt
"""

import os
import sys
import time

THREADS = int(os.environ.get("GIGAAM_THREADS", "3"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import torch  # noqa: E402

torch.set_num_threads(THREADS)

_orig_load = torch.load


def _l(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _l

import gigaam  # noqa: E402
from pydub import AudioSegment  # noqa: E402

CLIP_SEC = 20.0


def clip20(path: str) -> str:
    out = "/tmp/clip20.wav"
    AudioSegment.from_file(path)[: int(CLIP_SEC * 1000)].export(out, format="wav")
    return out


def _text(res):
    return getattr(res, "text", None) or (res if isinstance(res, str) else repr(res))


def main() -> None:
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "v3_e2e_rnnt"
    clip = clip20(path)
    print(f"model={name}  threads(torch)={THREADS}  clip={CLIP_SEC:.0f}s", flush=True)

    try:
        model = gigaam.load_model(name, download_root="/cache/gigaam")
    except TypeError:
        model = gigaam.load_model(name)

    # 1) PyTorch
    model.transcribe(clip)  # прогрев
    t0 = time.monotonic()
    res = model.transcribe(clip)
    pt = time.monotonic() - t0
    print(f"\n[PyTorch] {pt:.2f}s  RTF={pt / CLIP_SEC:.3f}x", flush=True)
    print(f"  text: {_text(res)}", flush=True)

    # 2) ONNX export
    onnx_dir = f"/cache/onnx_{name}"
    try:
        os.makedirs(onnx_dir, exist_ok=True)
        t0 = time.monotonic()
        model.to_onnx(dir_path=onnx_dir)
        print(f"\n[ONNX] export {time.monotonic() - t0:.1f}s (разово)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"\n[ONNX] export FAILED: {type(e).__name__}: {e}", flush=True)
        return

    # 3) ONNX infer
    try:
        from gigaam.onnx_utils import infer_onnx, load_onnx

        sessions, cfg = load_onnx(onnx_dir, name)
        infer_onnx([clip], cfg, sessions)  # прогрев
        t0 = time.monotonic()
        out = infer_onnx([clip], cfg, sessions)
        on = time.monotonic() - t0
        print(f"[ONNX] {on:.2f}s  RTF={on / CLIP_SEC:.3f}x", flush=True)
        print(f"  out: {out[0] if isinstance(out, (list, tuple)) else out}", flush=True)
        print(f"\n→ ядро ONNX/PyTorch: {pt / on:.2f}x (>1 = ONNX быстрее)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[ONNX] infer FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

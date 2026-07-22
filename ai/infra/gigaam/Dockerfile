# Продакшен-образ враппера GigaAM (OpenAI-совместимый /v1/audio/transcriptions).
# gigaam из git (v3 + word_timestamps); отдельных пинов на hf_hub/numpy НЕ
# ставим — их резолвит сам git-стек (см. историю trials/Dockerfile.git).
# Всё остальное закреплено по версиям ниже.
FROM python:3.12-slim@sha256:6c4dd321d176d61ea848dc8c73a4f7dbae8f70e0ee48bb411ea2f045b599fa8e

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# CPU-only torch, pinned to the exact version gigaam[longform] requires
# (torch==2.10.*). Unpinned, this line installs the newest CPU build (2.13 today)
# and the gigaam install below then has to satisfy ==2.10.* from PyPI — where the
# linux wheel is the CUDA variant, dragging in the nvidia-* stack and silently
# defeating "CPU-only". torchaudio is pinned here for the same reason.
RUN pip install --no-cache-dir torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cpu

# gigaam из git + longform-экстра (pyannote VAD).
# Pinned to a commit: GigaAM publishes no PyPI release carrying v3, so it has to
# come from git — but a floating main makes the build irreproducible and lets any
# upstream commit break the deploy. Bump deliberately: change the sha, rebuild,
# re-run trials/.
RUN pip install --no-cache-dir \
    "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git@559d88d6b72541412743929f633a6ae7c9950b85" \
    pydub==0.25.1

# Веб-слой.
RUN pip install --no-cache-dir \
    fastapi==0.139.2 \
    "uvicorn[standard]==0.51.0" \
    python-multipart==0.0.32 \
    prometheus-client==0.25.0 \
    anyio==4.14.2

# The three installs above resolve independently, so a later layer can silently
# downgrade something an earlier one needed — pip only warns. Fail the build here
# instead of shipping a broken dependency graph.
RUN pip check

COPY app/ /srv/app/

ENV PYTHONPATH=/srv
EXPOSE 8000

# Один воркер: модель грузится один раз, держится тёплой; конкурентность
# инференса ограничена гейтом внутри приложения (не uvicorn-воркерами, иначе
# N копий модели в RAM).
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

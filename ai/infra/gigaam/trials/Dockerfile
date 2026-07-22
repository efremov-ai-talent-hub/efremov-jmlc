# Экспериментальный образ для замера GigaAM на CPU.
# Самодостаточный — не часть основного стека.
FROM python:3.12-slim

# ffmpeg нужен pydub (длительность/декод) и longform-нарезке GigaAM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch отдельной строкой: иначе pip тянет CUDA-колёса на сотни МБ,
# которые на этой машине бесполезны (GPU нет).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# [longform] подтягивает pyannote для VAD-нарезки длинного аудио.
RUN pip install --no-cache-dir "gigaam[longform]" pydub

# huggingface_hub>=1.0 (окт 2025) убрал use_auth_token, который зовёт старый
# pyannote из gigaam[longform] → TypeError. Держим <1.0.
# numpy<2: onnxruntime (тянется pyannote) собран под numpy 1.x, на 2.x ругается
# _ARRAY_API not found.
RUN pip install --no-cache-dir "huggingface_hub<1.0" "numpy<2"

COPY trial_gigaam.py /app/trial_gigaam.py
COPY trial_features.py /app/trial_features.py

ENTRYPOINT ["python3", "trial_gigaam.py"]

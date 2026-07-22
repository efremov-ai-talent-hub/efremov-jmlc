"""Prometheus-метрики враппера (операционные; tracing/стоимость — на LiteLLM).

Главное под CPU-headroom: rtf и stage_seconds. inflight — занят ли слот.
Бэклог/порядок наблюдаются в Prefect, не здесь.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

requests_total = Counter("gigaam_requests_total", "Transcription requests", ["status"])
request_seconds = Histogram("gigaam_request_seconds", "End-to-end request time")
rtf = Histogram(
    "gigaam_rtf",
    "compute / audio_duration",
    buckets=(0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
)
audio_seconds_total = Counter("gigaam_audio_seconds_total", "Audio seconds processed")
stage_seconds = Histogram("gigaam_stage_seconds", "Per-stage time", ["stage"])
inflight = Gauge("gigaam_inflight", "Inferences currently running (gate)")
model_loaded = Gauge("gigaam_model_loaded", "1 if model loaded")

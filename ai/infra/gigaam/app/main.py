"""FastAPI-обёртка: OpenAI-совместимый /v1/audio/transcriptions + /health + /metrics.

Тонкий слой: разбирает OpenAI-поля (толерантно к лишним), зовёт рантайм, склеивает
слова в сегменты, мапит в whisper-схему. Движок/склейка/маппинг — за ним, сменные.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import mapping, metrics, segmentation
from .config import Config
from .errors import AudioError, TranscriptionTransientError, WrapperError, to_error_body
from .runtime import Runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gigaam.api")


class _AccessNoiseFilter(logging.Filter):
    """Глушим access-лог поллинга /health (healthcheck) и /metrics (prometheus),
    чтобы json-file-лог не пух впустую при долгой работе. Полезные
    POST /v1/audio/transcriptions остаются."""

    def filter(self, record: logging.LogRecord) -> bool:
        m = record.getMessage()
        return "/health" not in m and "/metrics" not in m


logging.getLogger("uvicorn.access").addFilter(_AccessNoiseFilter())

CFG = Config.from_env()
RT = Runtime(CFG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    RT.load()  # тёплая загрузка модели один раз при старте
    yield


app = FastAPI(title="gigaam-wrapper", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if RT.loaded else "loading",
        "model": CFG.model_name,
        "loaded": RT.loaded,
    }


@app.get("/metrics")
async def metrics_endpoint():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    t0 = time.monotonic()
    # Уникальный id ответа: whisper его не даёт, но GigaAM — наш сервис, поэтому
    # чеканим сами. Едет в теле как `id` (вызывающий снимает в request_id) и в
    # заголовке x-request-id; по нему ищется запрос в LiteLLM /ui/logs.
    req_id = "transcr-" + uuid.uuid4().hex
    form = await request.form()  # толерантно к любым лишним полям (prompt/user/temperature/...)

    upload = form.get("file")
    if upload is None or not hasattr(upload, "filename"):
        return _error(AudioError("no audio file in 'file' field"))

    response_format = str(form.get("response_format") or "json")
    # OpenAI шлёт как timestamp_granularities[] (повтор) или timestamp_granularities
    granularities = (
        form.getlist("timestamp_granularities[]")
        or form.getlist("timestamp_granularities")
        or ["segment"]
    )

    suffix = os.path.splitext(getattr(upload, "filename", "") or "")[1] or ".audio"
    tmp_path = None
    try:
        data = await upload.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        ct = await RT.transcribe(tmp_path)

        ts = time.monotonic()
        ct.segments = segmentation.glue_words(ct.words, max_seconds=CFG.glue_max_seconds)
        metrics.stage_seconds.labels(stage="segment").observe(time.monotonic() - ts)

        compute = time.monotonic() - t0
        metrics.stage_seconds.labels(stage="transcribe").observe(compute)
        if ct.duration > 0:
            metrics.rtf.observe(compute / ct.duration)
            metrics.audio_seconds_total.inc(ct.duration)
        metrics.requests_total.labels(status="ok").inc()
        metrics.request_seconds.observe(compute)
        logger.info("transcription ok id=%s dur=%.1fs compute=%.1fs", req_id, ct.duration, compute)

        hdr = {"x-request-id": req_id}
        if response_format == "text":
            return PlainTextResponse(mapping.to_text(ct), headers=hdr)
        if response_format == "json":
            return JSONResponse(mapping.to_json(ct, request_id=req_id), headers=hdr)
        return JSONResponse(
            mapping.to_verbose_json(ct, granularities=granularities, request_id=req_id), headers=hdr
        )

    except WrapperError as exc:
        return _error(exc)
    except Exception as exc:  # noqa: BLE001 — неизвестное трактуем как ТРАНЗИЕНТНОЕ (ретраебельно)
        logger.exception("unexpected transcription error")
        return _error(TranscriptionTransientError(f"unexpected: {exc}"))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _error(exc: WrapperError) -> JSONResponse:
    metrics.requests_total.labels(status="error").inc()
    logger.warning("request failed [%s %d]: %s", exc.code, exc.http_status, exc.message)
    return JSONResponse(status_code=exc.http_status, content=to_error_body(exc))

from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import Any

from ai.shared import llm_tracing
from ai.shared.branding import SELLER_BRAND

logger = logging.getLogger(__name__)


# NB: no process-level client cache. Under LITELLM_PROXY_ENABLED the api_key
# depends on the current LLMContext.kind (transcription-whisper vs
# transcription-chat resolve to different per-kind virtual keys), so caching a
# single client across kinds would route traffic under the wrong key. OpenAI
# client construction is cheap; the cost is dwarfed by Whisper latency.

# OpenAI Whisper API hard-caps uploads at 25 MB (26_214_400 bytes). We keep
# a 1 MB safety margin so a 24.5 MB file doesn't 413 on a single multipart
# boundary mismatch.
_WHISPER_MAX_BYTES = 24 * 1024 * 1024
# Default split when the file exceeds the cap. 10 min × ~128 kbps mp3 ≈ 9.6 MB,
# comfortably inside the limit. Overlap covers words straddling the boundary;
# duplicates are dropped at stitch time.
_WHISPER_CHUNK_SEC = 600
_WHISPER_CHUNK_OVERLAP_SEC = 5


# Vocabulary hint for ASR: domain terms only. Project names and addresses are
# deliberately absent — listing them would tie this stand to one company's
# portfolio. Supply them through SELLER_PROJECTS to run against a real catalogue.
_SELLER_PROJECTS = os.getenv("SELLER_PROJECTS", "").strip()

WHISPER_PROMPT = (
    f"Компания «{SELLER_BRAND}». "
    "Термины: эскроу, ДДУ, рассрочка, траншевая ипотека, семейная ипотека, субсидированная ипотека, "
    "аккредитив, переуступка, цессия, первоначальный взнос, материнский капитал, бронирование, РВЭ, "
    "апартаменты, пентхаус, евродвушка, евротрёшка, машиноместо."
) + (f" Проекты: {_SELLER_PROJECTS}." if _SELLER_PROJECTS else "")


def _get_openai_client() -> Any:
    from openai import OpenAI

    base_url = (
        (os.getenv("ANALYSIS_OPENAI_BASE_URL") or "").strip()
        or (os.getenv("ENRICHMENT_OPENAI_BASE_URL") or "").strip()
        or (os.getenv("OPENAI_API_BASE_URL") or "").strip()
        or (os.getenv("OPENAI_BASE_URL") or "").strip()
        or "https://api.openai.com/v1"
    )
    api_key = (
        (os.getenv("ANALYSIS_OPENAI_API_KEY") or "").strip()
        or (os.getenv("ENRICHMENT_OPENAI_API_KEY") or "").strip()
        or (os.getenv("OPENAI_API_KEY") or "").strip()
    )
    if not api_key:
        raise ValueError(
            "ANALYSIS_OPENAI_API_KEY (or ENRICHMENT_OPENAI_API_KEY / OPENAI_API_KEY) is required"
        )

    # LITELLM_PROXY_ENABLED=1 swaps base_url/api_key to the proxy. The current
    # LLMContext.kind (set by the caller — see ai/transcription callers) picks
    # the right virtual key.
    return OpenAI(
        **llm_tracing.resolve_openai_kwargs(default_api_key=api_key, default_base_url=base_url)
    )


def _get_audio_channels(audio_data: bytes) -> int:
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        return int(audio.channels or 1)
    except Exception:
        return 1


def _assign_speakers_by_energy(
    audio_data: bytes, segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    c0, c1 = audio.split_to_mono()

    result: list[dict[str, Any]] = []
    for seg in segments:
        start_ms = int(float(seg["start"]) * 1000)
        end_ms = int(float(seg["end"]) * 1000)

        c0_slice = c0[start_ms:end_ms]
        c1_slice = c1[start_ms:end_ms]

        c0_energy = c0_slice.dBFS if len(c0_slice) > 0 else -100
        c1_energy = c1_slice.dBFS if len(c1_slice) > 0 else -100

        speaker = 1 if c0_energy >= c1_energy else 2
        result.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": f"[{speaker}] {str(seg['text'])}",
            }
        )
    return result


def _whisper_call(
    audio_file: io.BytesIO, *, model: str, client: Any, prompt: str
) -> list[dict[str, Any]]:
    """Single Whisper API call. ``audio_file.name`` must be set so the SDK
    uploads with the right multipart filename (Whisper sniffs format from it).
    """
    # Whisper rejects unknown body params besides `user`; use mode="audio" so
    # llm_tracing emits only the x-litellm-tags header. bind() must wrap the
    # actual call so transcribe_create can read kind/stage from the live
    # LLMContext when it snapshots the stamp.
    with llm_tracing.bind(kind="transcription-whisper", stage="transcribe"):
        extras = llm_tracing.build_request_extras(user_label="worker_transcription", mode="audio")
        response = llm_tracing.transcribe_create(
            client,
            model=model,
            file=audio_file,
            language="ru",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            prompt=prompt,
            extra_body={"user": "worker_transcription"},
            extra_headers=extras.extra_headers,
            # Hard cap so a stuck Whisper proxy can't block the worker indefinitely.
            # 30 min covers ~1h audio at 1-2x realtime; longer recordings fail fast.
            timeout=1800,
        )
    return [
        {"start": float(s.start), "end": float(s.end), "text": str(s.text).strip()}
        for s in (getattr(response, "segments", []) or [])
    ]


def _split_audio_for_whisper(
    audio_bytes: bytes,
    *,
    chunk_sec: int = _WHISPER_CHUNK_SEC,
    overlap_sec: int = _WHISPER_CHUNK_OVERLAP_SEC,
) -> list[tuple[bytes, float]]:
    """Slice an audio blob into time-bounded mp3 chunks for Whisper.

    Each chunk overlaps the previous one by ``overlap_sec`` so a word
    straddling the boundary is captured on both sides; the duplicate is
    removed by :func:`_stitch_chunked_segments` after Whisper returns
    timecodes. Returns a list of ``(mp3_bytes, offset_seconds)`` pairs,
    where ``offset_seconds`` is the chunk's global start in the source.

    Pydub uses ffmpeg under the hood; the worker image (Dockerfile.prefect-worker)
    installs ffmpeg with libmp3lame, so the in-memory mp3 export works without
    extra deps. Default mp3 bitrate (~128 kbps) keeps quality close to the
    source — we're not optimising for size here, just slicing.
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    total_ms = len(audio)
    chunk_ms = chunk_sec * 1000
    overlap_ms = overlap_sec * 1000
    step_ms = chunk_ms - overlap_ms
    if step_ms <= 0:
        raise ValueError(f"chunk_sec ({chunk_sec}) must exceed overlap_sec ({overlap_sec})")

    chunks: list[tuple[bytes, float]] = []
    cursor_ms = 0
    while cursor_ms < total_ms:
        end_ms = min(cursor_ms + chunk_ms, total_ms)
        segment = audio[cursor_ms:end_ms]
        buf = io.BytesIO()
        segment.export(buf, format="mp3")
        chunks.append((buf.getvalue(), cursor_ms / 1000.0))
        if end_ms >= total_ms:
            break
        cursor_ms += step_ms
    return chunks


def _stitch_chunked_segments(
    chunk_results: list[tuple[list[dict[str, Any]], float]],
    *,
    overlap_sec: float = _WHISPER_CHUNK_OVERLAP_SEC,
) -> list[dict[str, Any]]:
    """Concatenate per-chunk segments into a single timeline.

    Each segment's ``start`` / ``end`` is local to its chunk; we shift them by
    ``offset_sec`` to recover global timecodes. A segment from chunk N+1 whose
    shifted ``start`` falls inside the overlap zone with chunk N is dropped —
    Whisper transcribes both sides of the overlap and would otherwise emit a
    near-duplicate. Half-overlap is the cut threshold: forgiving enough that
    speech legitimately starting at the boundary isn't lost.
    """
    out: list[dict[str, Any]] = []
    last_chunk_end_sec: float | None = None
    for segs, offset_sec in chunk_results:
        threshold = (
            (last_chunk_end_sec - overlap_sec / 2.0) if last_chunk_end_sec is not None else None
        )
        for s in segs:
            shifted_start = float(s["start"]) + offset_sec
            shifted_end = float(s["end"]) + offset_sec
            if threshold is not None and shifted_start < threshold:
                continue
            out.append({"start": shifted_start, "end": shifted_end, "text": s["text"]})
        if segs:
            last_chunk_end_sec = float(segs[-1]["end"]) + offset_sec
    return out


def _whisper_transcribe(
    audio_bytes: bytes, *, model: str, client: Any, prompt: str
) -> list[dict[str, Any]]:
    """Transcribe an audio blob via Whisper, chunking past the 25 MB upload limit.

    Most meetings fit in a single 25 MB upload — we take the fast path then.
    For longer recordings (typically 1h+ of stereo mp3) the audio is sliced
    into 10-min chunks with a 5-sec overlap, fed to Whisper sequentially,
    then stitched into a single global timeline.

    Chunking is transparent to the surrounding pipeline:
      * ``_assign_speakers_by_energy`` runs on the original ``audio_bytes``
        (passed separately by ``transcribe_audio``), not on the chunked
        re-encoded mp3s, so stereo speaker labelling keeps working.
      * Returned segments already carry global timecodes.
    """
    if len(audio_bytes) <= _WHISPER_MAX_BYTES:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.mp3"
        return _whisper_call(audio_file, model=model, client=client, prompt=prompt)

    logger.info(
        "Whisper input %d bytes (> %d cap); chunking into %ds slices",
        len(audio_bytes),
        _WHISPER_MAX_BYTES,
        _WHISPER_CHUNK_SEC,
    )
    chunks = _split_audio_for_whisper(audio_bytes)
    logger.info("Whisper chunking: %d chunks ready", len(chunks))
    chunk_results: list[tuple[list[dict[str, Any]], float]] = []
    for idx, (chunk_bytes, offset_sec) in enumerate(chunks):
        if len(chunk_bytes) > _WHISPER_MAX_BYTES:
            # Shouldn't happen at default chunk_sec unless source is at an
            # extreme bitrate; log and try anyway — Whisper will 413 and the
            # outer try/except surfaces the cause.
            logger.warning(
                "Chunk %d still %d bytes (> %d cap) — Whisper will likely 413",
                idx,
                len(chunk_bytes),
                _WHISPER_MAX_BYTES,
            )
        audio_file = io.BytesIO(chunk_bytes)
        audio_file.name = f"audio_chunk_{idx:03d}.mp3"
        t0 = time.monotonic()
        segs = _whisper_call(audio_file, model=model, client=client, prompt=prompt)
        logger.info(
            "Whisper chunk %d/%d: %d segments in %.2fs (offset=%.1fs, bytes=%d)",
            idx + 1,
            len(chunks),
            len(segs),
            time.monotonic() - t0,
            offset_sec,
            len(chunk_bytes),
        )
        chunk_results.append((segs, offset_sec))
    return _stitch_chunked_segments(chunk_results)


def _parse_speaker_mapping(answer: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in answer.splitlines():
        line = line.strip()
        if line.startswith("1="):
            mapping[1] = line[2:].strip()
        elif line.startswith("2="):
            mapping[2] = line[2:].strip()
    if 1 not in mapping or 2 not in mapping:
        return {1: "КЛИЕНТ", 2: "МЕНЕДЖЕР"}
    return mapping


def _llm_detect_speakers(
    numbered_transcript: str, client: Any, *, chat_model: str
) -> dict[int, str]:
    prompt = (
        "Ниже — транскрипт телефонного звонка в отдел продаж недвижимости.\n"
        "Участники помечены как [1] и [2].\n"
        "Определи, кто из них менеджер по продажам, а кто клиент.\n"
        "Ответь СТРОГО в формате:\n"
        "1=МЕНЕДЖЕР\n2=КЛИЕНТ\n"
        "или\n"
        "1=КЛИЕНТ\n2=МЕНЕДЖЕР\n"
        "Никаких пояснений, только две строки.\n\n"
        f"{numbered_transcript[:3000]}"
    )
    # Soft-fail: speaker mapping is cosmetic — provider errors (content filter,
    # network, or LiteLLM Pydantic validation on unknown finish_reason) must
    # not lose the Whisper transcript that already exists. bind() spans the
    # call so stage="speaker_detect" reaches the captured stamp.
    with llm_tracing.bind(kind="transcription-chat", stage="speaker_detect"):
        extras = llm_tracing.build_request_extras(user_label="worker_speaker_detect")
        try:
            response = llm_tracing.chat_create(
                client,
                model=chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
                user="worker_speaker_detect",
                extra_body=extras.extra_body,
                extra_headers=extras.extra_headers,
                # Small payload (3K in, 20 out); 60s is generous for any sane backend.
                timeout=60,
            )
        except Exception as exc:
            logger.warning(
                "LLM speaker detection raised %s, falling back to default mapping: %s",
                type(exc).__name__,
                str(exc)[:200],
            )
            return {1: "КЛИЕНТ", 2: "МЕНЕДЖЕР"}
    answer = (response.choices[0].message.content or "").strip()
    mapping = _parse_speaker_mapping(answer)
    if mapping == {1: "КЛИЕНТ", 2: "МЕНЕДЖЕР"}:
        logger.warning("LLM speaker detection parse failed: %r, using default mapping", answer)
    return mapping


def _llm_polish_transcript(transcript: str, client: Any, *, chat_model: str) -> str:
    system_prompt = (
        "Ты — редактор транскриптов телефонных звонков в отдел продаж недвижимости.\n\n"
        f"Это разговор между менеджером по продажам и клиентом по поводу покупки жилой недвижимости компании «{SELLER_BRAND}».\n\n"
        "В разговоре могут обсуждаться проекты компании, условия покупки, планировки, цены, "
        "рассрочка, ипотека, этапы сделки, локации и адреса объектов.\n\n"
        # Каталог объектов намеренно не зашит: он привязал бы стенд к портфелю
        # конкретной компании. Передать реальный список можно через SELLER_PROJECTS.
        + (f"Проекты компании: {_SELLER_PROJECTS}.\n\n" if _SELLER_PROJECTS else "")
        + "Термины: эскроу счет, рассрочка, ипотека, первоначальный взнос, субсидированная ипотека, траншевая ипотека, "
        "апартаменты, квартира, пентхаус, чистовая отделка, предчистовая отделка, ДДУ, бронирование, машиноместо.\n\n"
        "Мессенджеры и соцсети которые могут упоминаться: WhatsApp, Telegram, VK (ВКонтакте), Max (ранее TamTam), Авито.\n\n"
        "Твоя задача:\n"
        "1) Если транскрипт содержит метки [МЕНЕДЖЕР] и [КЛИЕНТ] — проверь их правильность. "
        "Считай что 80% меток верные, но бывают ошибки. Исправляй метку ТОЛЬКО если уверен на 100% "
        "(например фраза «Компания Девелопер, добрый день» точно принадлежит менеджеру).\n"
        "2) Исправь явные ошибки распознавания речи: неправильные слова, искажённые названия проектов, "
        "адреса, термины. Исправляй ТОЛЬКО если уверен в правильном варианте.\n"
        "3) НЕ меняй таймкоды, НЕ удаляй и НЕ добавляй строки, НЕ меняй структуру.\n"
        "4) Верни ПОЛНЫЙ исправленный транскрипт, строка за строкой, без пояснений."
    )
    # Soft-fail: polish is cosmetic — provider errors (content filter, network,
    # or LiteLLM Pydantic validation on unknown finish_reason) must not lose
    # the Whisper-produced transcript that already exists. bind() spans the
    # call so stage="polish" reaches the captured stamp.
    with llm_tracing.bind(kind="transcription-chat", stage="polish"):
        extras = llm_tracing.build_request_extras(user_label="worker_polish")
        try:
            response = llm_tracing.chat_create(
                client,
                model=chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": transcript},
                ],
                temperature=0,
                user="worker_polish",
                extra_body=extras.extra_body,
                extra_headers=extras.extra_headers,
                # Polish round-trips the full transcript both ways. 5 min handles
                # ~30K-char meeting transcripts on a healthy backend; longer hangs
                # fail fast and we keep the unpolished Whisper output.
                timeout=300,
            )
        except Exception as exc:
            logger.warning(
                "LLM polish raised %s, keeping original transcript: %s",
                type(exc).__name__,
                str(exc)[:200],
            )
            return transcript
    polished = str(response.choices[0].message.content or "").strip()
    if len(polished) < len(transcript) * 0.5:
        logger.warning(
            "LLM polish returned suspiciously short result (%d vs %d chars), keeping original",
            len(polished),
            len(transcript),
        )
        return transcript
    return polished


def _format_segments(segments: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"[{float(seg['start']):06.2f}–{float(seg['end']):06.2f}] {str(seg['text'])}"
        for seg in segments
    )


def transcribe_audio(
    audio_data: bytes,
    *,
    whisper_model: str,
    chat_model: str,
) -> str:
    """Sales calls/meetings transcription: stereo 2-speaker labelling + sales-tuned polish.

    ``whisper_model``: speech-to-text (e.g. ``whisper-1-proxy`` on LiteLLM).
    ``chat_model``: speaker detection + polish steps (must match LiteLLM aliases, not bare ``gpt-4o-mini``).
    """
    if not audio_data:
        raise ValueError("Empty audio data provided")

    try:
        # Two clients because Whisper and chat helpers resolve to different
        # LiteLLM virtual keys under LITELLM_PROXY_ENABLED. When the proxy is
        # off both clients are byte-identical.
        with llm_tracing.bind(kind="transcription-whisper"):
            whisper_client = _get_openai_client()
        with llm_tracing.bind(kind="transcription-chat"):
            chat_client = _get_openai_client()
        num_channels = _get_audio_channels(audio_data)
        logger.info("Audio channels detected: %d", num_channels)

        if num_channels >= 2:
            t0 = time.monotonic()
            segs = _whisper_transcribe(
                audio_data,
                model=whisper_model,
                client=whisper_client,
                prompt=WHISPER_PROMPT,
            )
            logger.info(
                "Whisper done (stereo): %d segments in %.2fs", len(segs), time.monotonic() - t0
            )
            labeled_segs = _assign_speakers_by_energy(audio_data, segs)

            speakers_used: set[int] = set()
            for seg in labeled_segs:
                text = str(seg["text"])
                if text.startswith("[1]"):
                    speakers_used.add(1)
                elif text.startswith("[2]"):
                    speakers_used.add(2)

            if len(speakers_used) < 2:
                logger.warning(
                    "All segments on one channel (speakers_used=%s), returning without speaker labels",
                    sorted(speakers_used),
                )
                result = _format_segments(segs)
            else:
                numbered_transcript = _format_segments(labeled_segs)
                t0 = time.monotonic()
                mapping = _llm_detect_speakers(
                    numbered_transcript, client=chat_client, chat_model=chat_model
                )
                logger.info("Speaker detection done in %.2fs: %s", time.monotonic() - t0, mapping)
                result = re.sub(r"\[1\]", f"[{mapping[1]}]", numbered_transcript)
                result = re.sub(r"\[2\]", f"[{mapping[2]}]", result)
        else:
            t0 = time.monotonic()
            segs = _whisper_transcribe(
                audio_data,
                model=whisper_model,
                client=whisper_client,
                prompt=WHISPER_PROMPT,
            )
            logger.info(
                "Whisper done (mono): %d segments in %.2fs", len(segs), time.monotonic() - t0
            )
            result = _format_segments(segs)

        if not result:
            raise RuntimeError("Transcription returned empty result")

        logger.info("Polishing transcript with LLM (%d chars)...", len(result))
        t0 = time.monotonic()
        result = _llm_polish_transcript(result, client=chat_client, chat_model=chat_model)
        logger.info("Polish done in %.2fs (%d chars)", time.monotonic() - t0, len(result))
        logger.info("Successfully transcribed audio (%d bytes)", len(audio_data))
        return result
    except Exception as exc:
        logger.error("Error transcribing audio: %s: %s", type(exc).__name__, exc, exc_info=True)
        raise RuntimeError(f"Transcription failed: {exc}") from exc


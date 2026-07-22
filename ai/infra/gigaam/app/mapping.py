"""CanonicalTranscript → OpenAI-схема ответа транскрипции.

Контракт сверен с реальным логом + исходником openai SDK: TranscriptionSegment
имеет ВСЕ 10 полей required → эмитим полный объект (реальные id/start/end/text,
заглушки для tokens/logprob'ов, которых GigaAM не даёт), иначе response.segments
падает на pydantic-валидации у вызывающего.
"""

from __future__ import annotations

from .canonical import CanonicalTranscript, Segment


def _segment_obj(idx: int, seg: Segment) -> dict:
    return {
        "id": idx,
        "seek": 0,
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text,
        # GigaAM не даёт whisper-токенов/логпробов — нейтральные заглушки ради
        # required-полей TranscriptionSegment у SDK вызывающего.
        "tokens": [],
        "temperature": 0.0,
        "avg_logprob": 0.0,
        "compression_ratio": 0.0,
        "no_speech_prob": 0.0,
    }


def to_verbose_json(ct: CanonicalTranscript, *, granularities: list[str], request_id: str) -> dict:
    body: dict = {
        # whisper-ответы id не несут; добавляем свой, т.к. GigaAM — наш сервис.
        # Вызывающий снимает его как response.id — по нему запрос находится в
        # LiteLLM /ui/logs (фильтра по session_id там нет).
        "id": request_id,
        "task": "transcribe",
        "text": ct.text,
        "language": ct.language,
        "duration": round(ct.duration, 3),
    }
    # verbose_json у whisper всегда несёт segments; words — только если запрошены.
    body["segments"] = [_segment_obj(i, s) for i, s in enumerate(ct.segments)]
    if "word" in granularities:
        body["words"] = [
            {"word": w.text, "start": round(w.start, 3), "end": round(w.end, 3)} for w in ct.words
        ]
    return body


def to_json(ct: CanonicalTranscript, *, request_id: str) -> dict:
    return {"id": request_id, "text": ct.text}


def to_text(ct: CanonicalTranscript) -> str:
    return ct.text

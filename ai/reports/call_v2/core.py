from __future__ import annotations

import json
import logging
import os
import re
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from ai.reports.call_v2.prompts import (
    PASS0_FEW_SHOT_EXAMPLES,
    PASS0_SYSTEM_PROMPT,
    QC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)
from ai.reports.call_v2.scoring import calc_manager_score_1to10
from ai.reports.shared.chatgpt_model import ChatGPTModel
from ai.reports.shared.utils import extract_json, load_agent_schema
from ai.shared import llm_tracing

logger = logging.getLogger(__name__)


_CONNECTION_LOST_NA = {
    "closing_meeting_datetime_fixed",
    "closing_next_actions_agreed",
    "closing_contacts_collected",
    "closing_questions_clarified",
    "closing_farewell_standard",
}


def _get_client() -> OpenAI:
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
        raise RuntimeError(
            "ANALYSIS_OPENAI_API_KEY (or ENRICHMENT_OPENAI_API_KEY / OPENAI_API_KEY) is required"
        )
    # LITELLM_PROXY_ENABLED=1 swaps to the proxy + per-kind virtual key.
    return OpenAI(
        **llm_tracing.resolve_openai_kwargs(default_api_key=api_key, default_base_url=base_url)
    )


def _model_name(default_model: str) -> str:
    return (
        os.getenv("ANALYSIS_OPENAI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or default_model
        or "gpt-4o-mini"
    )


def _apply_repeat_na(qc_scores: dict[str, Any], *, is_primary: bool) -> dict[str, Any]:
    if is_primary:
        return qc_scores
    # For repeat sales calls (not primary), mark onboarding criteria as N/A —
    # they were already covered in the first call of the chain.
    from ai.reports.call_v2.scoring import _REPEAT_SKIP

    result = dict(qc_scores)
    for key in _REPEAT_SKIP:
        if key in result:
            result[key] = {"score": None, "evidence": "н/п — повторный звонок в цепочке"}
    return result


def _apply_service_na(qc_scores: dict[str, Any]) -> dict[str, Any]:
    # Сервисный звонок не оценивается: обнуляем ВСЕ QC-критерии в N/A
    # (v1-поведение — service не получает manager_score, но summary/портрет остаются).
    result: dict[str, Any] = {}
    for key in qc_scores.keys():
        result[key] = {"score": None, "evidence": "н/п — сервисный звонок"}
    return result


def _apply_connection_lost_na(qc_scores: dict[str, Any]) -> dict[str, Any]:
    result = dict(qc_scores)
    for key in _CONNECTION_LOST_NA:
        if key in result:
            result[key] = {"score": None, "evidence": "н/п — связь прервана"}
    return result


def _parse_transcript_segments(transcript_text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    pattern = re.compile(r"\[(\d+\.\d+)[–-](\d+\.\d+)\]\s*(.*)")
    for line in transcript_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            segments.append(
                {
                    "start": float(match.group(1)),
                    "end": float(match.group(2)),
                    "text": match.group(3).strip(),
                }
            )
    return segments


def _find_real_timecode(
    citation: str,
    llm_timecode: str,
    segments: list[dict[str, Any]],
    window: float = 30.0,
    threshold: float = 0.6,
) -> str | None:
    if not citation or not segments:
        return None

    llm_start = None
    if llm_timecode and llm_timecode not in ("not_specified", ""):
        match = re.match(r"(\d+\.?\d*)", llm_timecode)
        if match:
            llm_start = float(match.group(1))

    def best_match(candidates: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
        best_score, best_seg = 0.0, None
        for segment in candidates:
            score = SequenceMatcher(None, citation.lower(), segment["text"].lower()).ratio()
            if score > best_score:
                best_score, best_seg = score, segment
        return best_score, best_seg

    if llm_start is not None:
        window_segments = [
            segment for segment in segments if abs(segment["start"] - llm_start) <= window
        ]
        score, segment = best_match(window_segments)
        if score >= threshold and segment:
            return f"{segment['start']:06.2f}–{segment['end']:06.2f}"

    score, segment = best_match(segments)
    if score >= threshold and segment:
        return f"{segment['start']:06.2f}–{segment['end']:06.2f}"
    return None


def _fix_timecodes(data: Any, segments: list[dict[str, Any]]) -> Any:
    if isinstance(data, dict):
        if "timecode" in data:
            citation = data.get("evidence") or data.get("phrase") or data.get("reason") or ""
            llm_timecode = data.get("timecode", "")
            real_timecode = _find_real_timecode(str(citation), str(llm_timecode), segments)
            if real_timecode:
                data["timecode"] = real_timecode
        for value in data.values():
            _fix_timecodes(value, segments)
    elif isinstance(data, list):
        for value in data:
            _fix_timecodes(value, segments)
    return data


def _run_pass0(client: OpenAI, transcript: str, model: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": PASS0_SYSTEM_PROMPT},
        *PASS0_FEW_SHOT_EXAMPLES,
        {"role": "user", "content": transcript},
    ]
    # bind() scope must wrap the actual call too — chat_create reads the
    # current LLMContext from inside the SDK call to snapshot stage=pass0
    # into the captured stamp.
    with llm_tracing.bind(stage="pass0"):
        extras = llm_tracing.build_request_extras(user_label="worker_pass0_filter_v2")
        try:
            response = llm_tracing.chat_create(
                client,
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0,
                user="worker_pass0_filter_v2",
                extra_body=extras.extra_body,
                extra_headers=extras.extra_headers,
            )
            result = extract_json((response.choices[0].message.content or "").strip())
        except Exception:
            result = {"call_type": "sales", "reason": "parse_error", "connection_lost": False}

    call_type = result.get("call_type", "sales")
    if call_type not in {"junk", "service", "sales"}:
        call_type = "sales"
    return {
        "call_type": call_type,
        "reason": result.get("reason", ""),
        "connection_lost": bool(result.get("connection_lost", False)),
    }


def _agent_class():
    from pydantic_ai import Agent

    return Agent


def run_qc_pass(transcript: str, qc_agent: Any, qc_schema: str) -> dict[str, Any]:
    prompt = f"""Заполни строго по схеме:
{qc_schema}

Транскрипция:
{transcript}
"""
    # Override the outer flow's stage="main" so QC sub-passes are correctly
    # tagged in spend logs and in the LLM call journal.
    with llm_tracing.bind(stage="qc"):
        raw = qc_agent.run_sync(prompt, output_type=str).output
    return extract_json(raw)


def analyze_call_transcript(
    transcript_text: str,
    *,
    model_name: str,
    is_primary: bool = True,
    chain_context: str | None = None,
) -> dict[str, Any]:
    schema = load_agent_schema("call_analysis.schema.json")
    qc_schema = load_agent_schema("qc_schema.json")
    client = _get_client()
    resolved_model = _model_name(model_name)

    pass0 = _run_pass0(client, transcript_text, resolved_model)
    call_type = pass0["call_type"]
    connection_lost = pass0["connection_lost"]
    logger.info(
        "Pass0 v2 call_type=%s reason=%s connection_lost=%s is_primary=%s",
        call_type,
        pass0.get("reason"),
        connection_lost,
        is_primary,
    )

    if call_type == "junk":
        return {
            "call_type": "junk",
            "is_scoreable": False,
            "is_primary": is_primary,
            "reason": pass0.get("reason", "junk"),
            "connection_lost": connection_lost,
            "phases": {},
            "qc_scores": {},
            "manager_score_1to10": None,
        }

    Agent = _agent_class()
    qc_model = ChatGPTModel(model_name=resolved_model, user="worker_analysis_qc_v2")
    main_model = ChatGPTModel(model_name=resolved_model, user="worker_analysis_main_v2")
    qc_agent = Agent(qc_model, retries=2, system_prompt=QC_SYSTEM_PROMPT)
    main_system_prompt = SYSTEM_PROMPT
    if chain_context:
        main_system_prompt = f"{chain_context}\n{SYSTEM_PROMPT}"
    agent = Agent(main_model, retries=2, system_prompt=main_system_prompt)

    qc_data = run_qc_pass(transcript_text, qc_agent, qc_schema)

    qc_summary = json.dumps(qc_data.get("qc_scores", {}), ensure_ascii=False, indent=2)
    main_prompt = (
        f"Заполни строго по схеме:\n{schema}\n\n"
        f"Транскрипция:\n{transcript_text}\n\n"
        f"QC-анализ (дополнительная информация — используй как подсказку):\n{qc_summary}\n"
    )
    # Explicit stage="main" — same site-level visibility as run_qc_pass above.
    with llm_tracing.bind(stage="main"):
        raw = agent.run_sync(main_prompt, output_type=str).output
    data = extract_json(raw)

    segments = _parse_transcript_segments(transcript_text)
    data = _fix_timecodes(data, segments)
    data["phases"] = qc_data.get("phases", {})
    data["call_type"] = call_type
    data["is_primary"] = is_primary
    data["reason"] = pass0.get("reason", "")
    data["connection_lost"] = connection_lost

    # Скорится только sales без обрыва связи.
    # service → анализ делается (summary, портрет), но manager_score_1to10 = None.
    data["is_scoreable"] = call_type == "sales" and not connection_lost

    qc_scores = qc_data.get("qc_scores", {})
    if call_type == "service":
        qc_scores = _apply_service_na(qc_scores)
    elif call_type == "sales" and not is_primary:
        qc_scores = _apply_repeat_na(qc_scores, is_primary=is_primary)
    if connection_lost:
        qc_scores = _apply_connection_lost_na(qc_scores)
    data["qc_scores"] = qc_scores

    if data["is_scoreable"]:
        data["manager_score_1to10"] = calc_manager_score_1to10(
            qc_scores,
            call_type=call_type,
            is_primary=is_primary,
        )
    else:
        data["manager_score_1to10"] = None
    return data

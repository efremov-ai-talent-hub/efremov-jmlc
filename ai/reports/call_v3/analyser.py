"""v3 analyser — тот же контракт и chain-aware поведение, что у v2.

Флоу не замечает разницы, кроме ``analyser_version``: обёртка результата и сигнатуры
совпадают, поэтому переключение версии — это вызов другого метода.

Обёртка и chain-хелперы объявлены здесь СВОИ (а не импортируются из приватных v2): версия
владеет своей политикой, и рефактор v2 не должен молча ломать v3. Публичные вещи, которые
переиспользуются намеренно, импортируются как публичные — chain-контекст и scoring.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ai.reports.call_v2.prompts import build_chain_context_prompt
from ai.reports.call_v2.scoring import aggregate_chain_metrics, calc_manager_score_chain
from ai.reports.call_v3.core import analyze_call_transcript

_CONFIDENCE_MAP = {"low": 0.33, "medium": 0.66, "high": 0.9}


def _confidence_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return _CONFIDENCE_MAP.get(str(value).strip().lower())


def _deal_probability_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number > 1:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _wrap_payload_result(payload: dict[str, Any]) -> dict[str, Any]:
    score_value = payload.get("manager_score_1to10")
    score = float(score_value) if isinstance(score_value, (int, float)) else None
    return {
        "score": score,
        "call_type": payload.get("call_type"),
        "is_scoreable": bool(payload.get("is_scoreable", False)),
        "deal_probability": _deal_probability_to_float(payload.get("deal_probability_pct")),
        "confidence": _confidence_to_float(payload.get("confidence")),
        "connection_lost": bool(payload.get("connection_lost", False)),
        "payload": payload,
    }


def _chain_input_fingerprint(previous_reports: Iterable[dict[str, Any]]) -> str:
    pairs: list[tuple[str, int]] = []
    for report in previous_reports:
        if not isinstance(report, dict):
            continue
        call_id = report.get("call_id")
        version = report.get("version")
        if call_id is None or version is None:
            continue
        pairs.append((str(call_id), int(version)))
    pairs.sort()
    raw = "|".join(f"{cid}:{ver}" for cid, ver in pairs)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _filter_context_reports(previous_reports: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for report in previous_reports:
        if not isinstance(report, dict):
            continue
        if report.get("call_type") == "junk":
            continue
        if report.get("connection_lost"):
            continue
        filtered.append(report)
    return filtered


def _collect_sales_qc(reports: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sales_qc: list[dict[str, Any]] = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        if report.get("call_type") != "sales" or report.get("connection_lost"):
            continue
        payload = report.get("payload")
        if not isinstance(payload, dict):
            continue
        qc = payload.get("qc_scores")
        if isinstance(qc, dict):
            sales_qc.append(qc)
    return sales_qc


def _resolve_model(model: Any, cfg: Any) -> str:
    return str(
        getattr(model, "model_name", None) or getattr(cfg, "call_analysis_model", "gpt-4o-mini")
    )


def analyse_call_v3(
    transcript: str,
    model: Any,
    cfg: Any,
    *,
    is_primary: bool = True,
) -> dict[str, Any]:
    payload = analyze_call_transcript(
        transcript,
        model_name=_resolve_model(model, cfg),
        is_primary=is_primary,
    )
    payload["analyser_version"] = "v3"
    return _wrap_payload_result(payload)


def analyse_call_in_chain(
    transcript: str,
    previous_reports: list[dict[str, Any]],
    chain_meta: dict[str, Any],
    model: Any,
    cfg: Any,
) -> dict[str, Any]:
    """Chain-aware анализ повторного звонка в цепочке (зеркало контракта v2)."""
    context_reports = _filter_context_reports(previous_reports)
    chain_context = build_chain_context_prompt(
        context_reports,
        current_chain_position=chain_meta.get("chain_position"),
    )

    payload = analyze_call_transcript(
        transcript,
        model_name=_resolve_model(model, cfg),
        is_primary=False,
        chain_context=chain_context,
    )

    payload["analyser_version"] = "v3-chain"
    payload["chain_meta"] = {
        "chain_id": chain_meta.get("chain_id"),
        "chain_position": chain_meta.get("chain_position"),
        "prev_call_event_id": chain_meta.get("prev_call_event_id"),
        "is_new_chain": chain_meta.get("is_new_chain"),
        "gap_from_prev_minutes": chain_meta.get("gap_from_prev_minutes"),
    }
    payload["chain_input_fingerprint"] = _chain_input_fingerprint(previous_reports)
    payload["score_own"] = payload.get("manager_score_1to10")

    if payload.get("is_scoreable") and payload.get("call_type") == "sales":
        chain_metrics = aggregate_chain_metrics(
            previous_sales_qc=_collect_sales_qc(context_reports),
            current_qc=payload.get("qc_scores", {}),
        )
        payload["qc_scores_chain_cumulative"] = chain_metrics
        payload["manager_score_1to10"] = calc_manager_score_chain(
            current_qc=payload.get("qc_scores", {}),
            chain_metrics=chain_metrics,
            call_type=payload["call_type"],
            is_primary=False,
        )
    else:
        payload["qc_scores_chain_cumulative"] = None

    return _wrap_payload_result(payload)


def analyse_call(transcript: str, model: Any, cfg: Any) -> dict[str, Any]:
    return analyse_call_v3(transcript, model, cfg, is_primary=True)

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ai.reports.call_v2.core import analyze_call_transcript
from ai.reports.call_v2.prompts import build_chain_context_prompt
from ai.reports.call_v2.scoring import (
    aggregate_chain_metrics,
    calc_manager_score_chain,
)


def _confidence_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    mapping = {"low": 0.33, "medium": 0.66, "high": 0.9}
    return mapping.get(str(value).strip().lower())


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


def analyse_call_v2(
    transcript: str,
    model: Any,
    cfg: Any,
    *,
    is_primary: bool = True,
) -> dict[str, Any]:
    resolved_model = getattr(model, "model_name", None) or getattr(
        cfg, "call_analysis_model", "gpt-4o-mini"
    )
    payload = analyze_call_transcript(
        transcript,
        model_name=str(resolved_model),
        is_primary=is_primary,
    )
    payload["analyser_version"] = "v2"
    return _wrap_payload_result(payload)


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
        if report.get("call_type") != "sales":
            continue
        if report.get("connection_lost"):
            continue
        payload = report.get("payload")
        if not isinstance(payload, dict):
            continue
        qc = payload.get("qc_scores")
        if isinstance(qc, dict):
            sales_qc.append(qc)
    return sales_qc


def analyse_call_in_chain(
    transcript: str,
    previous_reports: list[dict[str, Any]],
    chain_meta: dict[str, Any],
    model: Any,
    cfg: Any,
) -> dict[str, Any]:
    """Chain-aware analysis of a call that is NOT the first in its chain.

    The caller is responsible for loading ``previous_reports`` — the list of
    prior reports for the same ``chain_id`` (is_current=true), ordered by
    chain_position ascending.
    """

    resolved_model = getattr(model, "model_name", None) or getattr(
        cfg, "call_analysis_model", "gpt-4o-mini"
    )

    context_reports = _filter_context_reports(previous_reports)
    chain_context = build_chain_context_prompt(
        context_reports,
        current_chain_position=chain_meta.get("chain_position"),
    )

    # chain-aware run = repeat call inside a chain; is_primary is forced to False
    payload = analyze_call_transcript(
        transcript,
        model_name=str(resolved_model),
        is_primary=False,
        chain_context=chain_context,
    )

    payload["analyser_version"] = "v2-chain"
    payload["chain_meta"] = {
        "chain_id": chain_meta.get("chain_id"),
        "chain_position": chain_meta.get("chain_position"),
        "prev_call_event_id": chain_meta.get("prev_call_event_id"),
        "is_new_chain": chain_meta.get("is_new_chain"),
        "gap_from_prev_minutes": chain_meta.get("gap_from_prev_minutes"),
    }
    payload["chain_input_fingerprint"] = _chain_input_fingerprint(previous_reports)

    # Own-call score stays for drill-down/debug, even if we aggregate for the
    # headline score below. junk/connection_lost cases were handled inside
    # analyze_call_transcript (manager_score_1to10 already None).
    payload["score_own"] = payload.get("manager_score_1to10")

    if payload.get("is_scoreable") and payload.get("call_type") == "sales":
        sales_qc = _collect_sales_qc(context_reports)
        chain_metrics = aggregate_chain_metrics(
            previous_sales_qc=sales_qc,
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


# Backward-compatible alias matching v1's name so flow code can route via
# `analyse_call` import but pick up v2 via version flag without touching call_v2's
# primary API. Passing is_primary=True by default preserves legacy behaviour
# when no chain context is available.
def analyse_call(transcript: str, model: Any, cfg: Any) -> dict[str, Any]:
    return analyse_call_v2(transcript, model, cfg, is_primary=True)

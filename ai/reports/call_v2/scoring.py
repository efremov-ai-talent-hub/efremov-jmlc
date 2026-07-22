from __future__ import annotations

from typing import Any

QC_WEIGHTS = {
    "contact_greeting_standard": 1,
    "contact_name_and_object": 1,
    "contact_name_used_3plus": 1,
    "needs_purchase_goal": 2,
    "needs_key_aspects": 2,
    "needs_budget_and_payment": 2,
    "presentation_project_advantages_5plus": 2,
    "presentation_developer_advantages": 2,
    "presentation_link_benefits_to_needs": 2,
    "presentation_promotions": 2,
    "presentation_related_objects": 2,
    "presentation_meeting_offer": 2,
    "objections_all_handled": 2,
    "objections_techniques_used": 2,
    "closing_meeting_datetime_fixed": 2,
    "closing_next_actions_agreed": 2,
    "closing_contacts_collected": 2,
    "closing_questions_clarified": 1,
    "closing_farewell_standard": 1,
    "general_active_listening": 1,
    "general_polite_tone": 1,
    "general_initiative_control": 1,
    "general_professional_speech": 1,
}

# TODO: согласовать конкретный список с командой качества.
# Поля, которые неприменимы к повторному звонку (is_primary=False) —
# менеджер уже прошёл эти шаги в первом контакте.
_REPEAT_SKIP = {
    "contact_greeting_standard",
    "contact_name_and_object",
    "needs_purchase_goal",
    "needs_key_aspects",
    "needs_budget_and_payment",
    "presentation_project_advantages_5plus",
}

# Сервисный звонок не скорится в v2 (решение от 2026-04-21):
# _apply_service_na в core.py зануляет ВСЕ qc_scores в N/A, manager_score_1to10 = None.
# Поэтому отдельного _SERVICE_SKIP не требуется — calc_manager_score_1to10 для service
# просто не вызывается.

_HALF_POINT_ALLOWED = {
    "presentation_meeting_offer",
    "needs_key_aspects",
    "general_active_listening",
    "contact_greeting_standard",
    "presentation_link_benefits_to_needs",
    "contact_name_and_object",
}


def _skip_keys(*, is_primary: bool) -> set[str]:
    return set(_REPEAT_SKIP) if not is_primary else set()


def calc_manager_score_1to10(
    qc_scores: dict,
    *,
    call_type: str = "sales",  # noqa: ARG001 — kept for API compatibility; service is not scored
    is_primary: bool = True,
) -> int:
    total = 0.0
    max_total = 0.0
    skip = _skip_keys(is_primary=is_primary)

    # presentation_project_advantages_5plus uses {count}, objections_all_handled uses
    # {total, handled} — they have no `score` key by design and must not be early-skipped
    # by the score-is-None check below.
    _no_score_keys = {"presentation_project_advantages_5plus", "objections_all_handled"}

    for key, weight in QC_WEIGHTS.items():
        if key in skip:
            continue
        item = qc_scores.get(key, 0) or 0
        if isinstance(item, dict) and item.get("score") is None and key not in _no_score_keys:
            continue

        if key == "presentation_project_advantages_5plus":
            count = item.get("count", 0) if isinstance(item, dict) else 0
            value = min(count / 5.0, 1.0)
        elif key == "objections_techniques_used":
            objections = qc_scores.get("objections_all_handled", {})
            if isinstance(objections, dict) and (objections.get("total") or 0) == 0:
                continue
            value = item.get("score", 0) if isinstance(item, dict) else item
        elif key == "objections_all_handled":
            if isinstance(item, dict):
                total_count = item.get("total", 0) or 0
                handled_count = item.get("handled", 0) or 0
                value = 1.0 if total_count == 0 else handled_count / total_count
            else:
                value = float(item) if item in (0, 1) else 0
        elif isinstance(item, dict):
            value = item.get("score", 0) or 0
        else:
            value = item

        if key not in {"presentation_project_advantages_5plus", "objections_all_handled"}:
            if key in _HALF_POINT_ALLOWED:
                if value not in (0, 0.5, 1):
                    value = 0
            else:
                if value not in (0, 1):
                    value = 0

        total += float(value) * float(weight)
        max_total += float(weight)

    if max_total == 0:
        return 0

    quality_pct = (total / max_total) * 100.0
    score = round(quality_pct / 10.0)
    return max(1, min(10, int(score)))


def aggregate_chain_metrics(
    previous_sales_qc: list[dict[str, Any]],
    current_qc: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate QC metrics across a chain of sales calls (previous + current)."""

    def _total_handled(qc: dict) -> tuple[int, int]:
        obj = qc.get("objections_all_handled", {}) if isinstance(qc, dict) else {}
        if not isinstance(obj, dict):
            return 0, 0
        return int(obj.get("total", 0) or 0), int(obj.get("handled", 0) or 0)

    def _field_has_positive(qc: dict, key: str) -> bool:
        item = qc.get(key, 0) if isinstance(qc, dict) else 0
        if isinstance(item, dict):
            score = item.get("score")
            return isinstance(score, (int, float)) and float(score) > 0
        return bool(item)

    objections_total = 0
    objections_handled = 0
    greeting_any = False
    needs_any = False
    next_step_any = False

    for qc in [*previous_sales_qc, current_qc]:
        if not isinstance(qc, dict):
            continue
        total_i, handled_i = _total_handled(qc)
        objections_total += total_i
        objections_handled += handled_i
        greeting_any = greeting_any or _field_has_positive(qc, "contact_greeting_standard")
        needs_any = needs_any or _field_has_positive(qc, "needs_purchase_goal")
        next_step_any = next_step_any or _field_has_positive(qc, "closing_next_actions_agreed")

    return {
        "objections_total_chain": objections_total,
        "objections_handled_chain": objections_handled,
        "greeting_any": greeting_any,
        "needs_any": needs_any,
        "next_step_any": next_step_any,
    }


def calc_manager_score_chain(
    current_qc: dict[str, Any],
    chain_metrics: dict[str, Any],
    *,
    call_type: str,
    is_primary: bool,  # noqa: ARG001 — kept for API symmetry with calc_manager_score_1to10
) -> int:
    """Compute manager score using cumulative chain metrics that override
    current-call values where a positive result anywhere in the chain should
    credit the manager (greeting, needs, next step, objections handled).

    Note: chain scoring intentionally bypasses _REPEAT_SKIP — repeat-call
    fairness is already handled by the cumulative override below (e.g. if the
    manager greeted in call #1, contact_greeting_standard gets 1 here).
    Skipping those keys would discard the credit instead of granting it.
    """

    merged: dict[str, Any] = dict(current_qc) if isinstance(current_qc, dict) else {}

    merged["objections_all_handled"] = {
        "total": int(chain_metrics.get("objections_total_chain", 0) or 0),
        "handled": int(chain_metrics.get("objections_handled_chain", 0) or 0),
        "evidence": "chain_cumulative",
    }

    def _override_positive(key: str, flag: bool) -> None:
        if not flag:
            return
        existing = merged.get(key)
        if isinstance(existing, dict):
            merged[key] = {**existing, "score": 1}
        else:
            merged[key] = {"score": 1, "evidence": "chain_cumulative"}

    _override_positive("contact_greeting_standard", bool(chain_metrics.get("greeting_any")))
    _override_positive("needs_purchase_goal", bool(chain_metrics.get("needs_any")))
    _override_positive("closing_next_actions_agreed", bool(chain_metrics.get("next_step_any")))

    return calc_manager_score_1to10(merged, call_type=call_type, is_primary=True)

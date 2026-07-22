"""v3 core — двухстадийный отчёт по звонку (analysis → format), клиент + менеджер.

Отличия от call_v2.core:

* Клиентский отчёт строится ДВУМЯ вызовами: (A) свободный NL-анализ с дословными цитатами и
  таймкодами, без схемы; (B) раскладка этого анализа по ПЛОСКОЙ схеме с quote-gating
  (значение только при цитате-опоре) и unknown-default. Плоско ⇒ меньше вложенности ⇒ меньше
  шансов уронить скобку.
* Менеджерская часть: qc-стадия играет роль «рассуждения» (критерии + evidence), поверх неё
  один форматный вызов на психопрофиль и фразы. Сильные/слабые стороны ОТБИРАЕТ КОД из
  qc_scores, слова-паразиты СЧИТАЕТ КОД по словарю — модели остаётся только суждение.
* Роли решаются ДО вызовов модели: детерминированный детектор смотрит на продавцовые маркеры
  и, если метки диаризации перепутаны, ИСПРАВЛЯЕТ транскрипт — все стадии дальше видят
  корректные роли. Просить модель «читай метки наоборот» бесполезно: замерено, что она
  игнорирует подсказку и берёт фразы по буквальной метке. Вердикт стадии A сохраняется
  рядом как ЗАМЕР, кто чаще прав, но на данные не влияет.
* Толерантный парс (снимается висячая запятая — законный парсер-фикс) + Instructor-ретрай с
  текстом ошибки. Скобки НЕ достраиваются: брошенная скобка — невалидный JSON модели.
* ``on_step`` — необязательный приёмник шагов: каждый LLM-вызов отдаётся наружу целиком
  (промпты, сырой ответ, тайминг, токены). В проде None — поведение не меняется.

Общая механика берётся из ``ai.reports.shared`` (публичные функции), а не из приватных v2.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Callable

from ai.reports.call_v2.prompts import (  # публичные: pass0 переиспользуется как есть
    PASS0_FEW_SHOT_EXAMPLES,
    PASS0_SYSTEM_PROMPT,
)
from ai.reports.call_v2.scoring import calc_manager_score_1to10
from ai.reports.call_v3 import manager as mgr
from ai.reports.call_v3 import qc
from ai.reports.call_v3.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_TEMPLATE,
    FORMAT_SYSTEM_PROMPT,
    FORMAT_USER_TEMPLATE,
    MANAGER_SYSTEM_PROMPT,
    MANAGER_USER_TEMPLATE,
    QC_ROLE_PREAMBLE,
    RETRY_SUFFIX,
    qc_system_prompt,
)
from ai.reports.call_v3.schema_gen import response_format
from ai.reports.call_v3.timecodes import repair_timecodes
from ai.reports.shared.llm_client import get_analysis_client, resolve_model_name
from ai.reports.shared.transcript import has_speaker_labels, parse_segments, relabel_swapped
from ai.reports.shared.utils import extract_json, load_agent_schema
from ai.shared import llm_tracing

logger = logging.getLogger(__name__)

StepSink = Callable[[str, dict[str, Any]], None]

CLIENT_SCHEMA = "call_analysis_v3.schema.json"
MANAGER_SCHEMA = "call_analysis_v3_manager.schema.json"
QC_SCHEMA = qc.SCHEMA

_LABELS_RE = re.compile(r"LABELS:\s*(swapped|ok|none)\b", re.IGNORECASE)


def guided_enabled() -> bool:
    """Grammar-constrained decoding. Требует эндпоинт, который умеет ``response_format``
    (llama-server / vLLM у self-hosted модели). Прод-шлюз этого не обещает, поэтому по
    умолчанию выключено и включается явно через ANALYSIS_GUIDED_JSON."""
    return (os.getenv("ANALYSIS_GUIDED_JSON") or "").strip().lower() in {"1", "true", "yes", "on"}


def _as_object(parsed: Any) -> dict[str, Any]:
    """Верхний уровень обязан быть объектом: список тоже валидный JSON, но не наш контракт."""
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def tolerant_extract_json(text: str) -> dict[str, Any]:
    """extract_json + законный парсер-фикс: снять висячую запятую (её допускает JSON5).

    Скобки НЕ достраиваем — угадывать, где модель потеряла `}`, не задача парсера.
    Не-объект на верхнем уровне поднимается как ValueError, чтобы его подхватил ретрай,
    а не TypeError где-то ниже по коду.
    """
    try:
        return _as_object(extract_json(text))
    except Exception:
        match = re.search(r"\{.*\}", (text or "").strip(), flags=re.S)
        candidate = match.group(0) if match else (text or "")
        candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
        return _as_object(json.loads(candidate))


def _usage_of(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _chat(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None,
    user: str,
    stage: str,
    on_step: StepSink | None,
    step_name: str,
    extra: dict[str, Any] | None = None,
    guided: dict[str, Any] | None = None,
) -> str:
    started = time.monotonic()
    error: str | None = None
    raw = ""
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    try:
        with llm_tracing.bind(stage=stage):
            extras = llm_tracing.build_request_extras(user_label=user)
            # Лимит выставляется только там, где он осмыслен: обрезанный по max_tokens ответ
            # приходит как невалидный JSON, и его легко ошибочно записать модели в формат-провалы.
            call_kwargs = {"max_tokens": max_tokens} if max_tokens is not None else {}
            # Грамматика включается ТОЛЬКО на форматных стадиях: свободный анализ под
            # ограничением декодера теряет в качестве рассуждения.
            if guided is not None:
                call_kwargs["response_format"] = guided
            response = llm_tracing.chat_create(
                client,
                model=model,
                messages=messages,
                temperature=0,
                user=user,
                extra_body=extras.extra_body,
                extra_headers=extras.extra_headers,
                **call_kwargs,
            )
        choice = response.choices[0]
        raw = (choice.message.content or "").strip()
        finish_reason = getattr(choice, "finish_reason", None)
        usage = _usage_of(response)
    except Exception as exc:  # noqa: BLE001 — шаг обязан быть записан даже при падении
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if on_step is not None:
            record: dict[str, Any] = {
                "step": step_name,
                "stage": stage,
                "model": model,
                "messages": messages,
                "raw": raw,
                "error": error,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "usage": usage,
                "finish_reason": finish_reason,
                "guided": guided is not None,
            }
            if extra:
                record.update(extra)
            # Приёмник шагов — диагностика; его падение не должно подменять собой
            # настоящую ошибку LLM-вызова, которая летит из блока try.
            try:
                on_step(step_name, record)
            except Exception:
                logger.warning("on_step sink failed for step=%s", step_name, exc_info=True)
    return raw


def _format_with_retry(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None,
    user: str,
    step_name: str,
    on_step: StepSink | None,
    stage: str = "main",
    guided: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Форматный вызов: толерантный парс, при провале — ретрай с текстом ошибки.

    ``stage`` обязан приходить снаружи: qc-подпроходы должны попадать в spend-логи и
    в журнал вызовов LLM со своей стадией, иначе пер-стадийная аналитика врёт.
    """
    raw = _chat(
        client,
        model,
        messages,
        max_tokens=max_tokens,
        user=user,
        stage=stage,
        on_step=on_step,
        step_name=step_name,
        guided=guided,
    )
    try:
        return tolerant_extract_json(raw), 0
    except ValueError as exc:
        # Только ошибка РАЗБОРА заслуживает ретрая. Транспортные ошибки (таймаут, 5xx,
        # rate-limit) поднимаются из _chat выше и не маскируются под «модель не смогла».
        retry_messages = [
            *messages,
            {"role": "assistant", "content": raw},
            {"role": "user", "content": RETRY_SUFFIX.format(error=str(exc)[:200])},
        ]
        raw2 = _chat(
            client,
            model,
            retry_messages,
            max_tokens=max_tokens,
            user=f"{user}_retry",
            stage=stage,
            on_step=on_step,
            step_name=f"{step_name}_retry",
            extra={"retry_after_error": str(exc)[:300]},
            guided=guided,
        )
        return tolerant_extract_json(raw2), 1


def _run_pass0(
    client: Any, transcript: str, model: str, on_step: StepSink | None
) -> dict[str, Any]:
    # Классификатор берётся у v2 ДОСЛОВНО (вместе с few-shot): иначе сравнение v2↔v3
    # смешало бы изменения клиентского отчёта с переклассификацией junk/sales.
    messages = [
        {"role": "system", "content": PASS0_SYSTEM_PROMPT},
        *PASS0_FEW_SHOT_EXAMPLES,
        {"role": "user", "content": transcript},
    ]
    try:
        raw = _chat(
            client,
            model,
            messages,
            max_tokens=200,
            user="worker_pass0_filter_v3",
            stage="pass0",
            on_step=on_step,
            step_name="pass0",
        )
        result = tolerant_extract_json(raw)
    except ValueError:
        # Нечитаемый ответ классификатора не повод ронять звонок — идём как sales.
        # Транспортные ошибки не глушим: они летят выше.
        result = {"call_type": "sales", "reason": "parse_error", "connection_lost": False}

    call_type = result.get("call_type", "sales")
    if call_type not in {"junk", "service", "sales"}:
        call_type = "sales"
    return {
        "call_type": call_type,
        "reason": result.get("reason", ""),
        "connection_lost": bool(result.get("connection_lost", False)),
    }


def _run_qc(
    client: Any,
    transcript: str,
    qc_schema: str,
    model: str,
    on_step: StepSink | None,
    guided: dict[str, Any] | None = None,
    roles: str = "",
) -> tuple[dict[str, Any], bool]:
    """QC: критерии v2 + преамбула v3 (роли и требование дословных цитат)."""
    messages = [
        {"role": "system", "content": QC_ROLE_PREAMBLE.format(roles=roles) + qc_system_prompt()},
        {
            "role": "user",
            "content": f"Заполни строго по схеме:\n{qc_schema}\n\nТранскрипция:\n{transcript}\n",
        },
    ]
    try:
        data, _ = _format_with_retry(
            client,
            model,
            messages,
            # 23 критерия × (цитата + вердикт). Схема v3 развела эти два поля, и ответ стал
            # вдвое длиннее v2: замерено 1578 токенов на коротком звонке и обрыв ровно на
            # прежнем потолке 6000 — оборванный JSON шёл в парсер как «модель не смогла в
            # формат» и ронял звонок в непроскориваемые.
            max_tokens=12000,
            user="worker_analysis_qc_v3",
            step_name="qc",
            on_step=on_step,
            stage="qc",
            guided=guided,
        )
        return data, True
    except ValueError:
        # Модель не смогла в формат — это законный исход, звонок не роняем. Но пустой qc
        # НЕЛЬЗЯ протащить как обычный результат: calc_manager_score_1to10({}) вернёт 1 —
        # худшую оценку, неотличимую от честно плохого звонка. Поэтому наверх идёт флаг,
        # а вызывающий снимает is_scoreable.
        # Транспортные ошибки сюда НЕ попадают: они летят выше, чтобы флоу переретраил звонок.
        logger.warning(
            "v3 qc pass returned unparseable JSON; call marked not scoreable", exc_info=True
        )
        return {}, False


def _run_analysis(
    client: Any, transcript: str, model: str, chain_context: str | None, on_step: StepSink | None
) -> tuple[str, str]:
    """Стадия A: свободный анализ клиента. Возвращает (текст, метка ролей)."""
    system = ANALYSIS_SYSTEM_PROMPT
    if chain_context:
        system = f"{chain_context}\n{system}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": ANALYSIS_USER_TEMPLATE.format(transcript=transcript)},
    ]
    text = _chat(
        client,
        model,
        messages,
        max_tokens=4000,
        user="worker_analysis_reason_v3",
        stage="analysis",
        on_step=on_step,
        step_name="analysis",
    )
    # Берём ПОСЛЕДНЕЕ совпадение: модель нередко цитирует список вариантов из промпта,
    # а ответом является финальная строка.
    found = _LABELS_RE.findall(text or "")
    return text, (found[-1].lower() if found else "none")


def analyze_call_transcript(
    transcript_text: str,
    *,
    model_name: str,
    is_primary: bool = True,
    chain_context: str | None = None,
    on_step: StepSink | None = None,
) -> dict[str, Any]:
    """Чистая точка входа v3: транскрипт → payload. Без Prefect и без БД."""
    client = get_analysis_client()
    model = resolve_model_name(model_name)

    # Роли определяем ДО любых вызовов модели: детектор — чистый код. Если метки
    # диаризации перепутаны, транскрипт исправляется ОДИН раз, и дальше все стадии видят
    # корректные роли. Просить модель «читай метки наоборот» бесполезно — замерено, что
    # она это игнорирует и берёт фразы по буквальной метке.
    detected_swap, swap_evidence = mgr.detect_labels_swapped(parse_segments(transcript_text))
    labels_verified = detected_swap is not None
    labels_corrected = detected_swap is True
    if labels_corrected:
        transcript_text = relabel_swapped(transcript_text)
    segments = parse_segments(transcript_text)

    # Обрезка по лимиту и «модель не смогла в формат» — разные диагнозы, и их нельзя
    # смешивать в замере. Потолки щедрые (держат вырожденную генерацию), а факт обрезки
    # фиксируется явно по finish_reason.
    truncated_stages: list[str] = []

    def sink(name: str, record: dict[str, Any]) -> None:
        if record.get("finish_reason") == "length":
            truncated_stages.append(name)
        if on_step is not None:
            on_step(name, record)

    # Шаблон схемы нужен дважды: текстом — в промпт, разобранным — для грамматики декодера.
    qc_schema_text = load_agent_schema(QC_SCHEMA)
    client_schema_text = load_agent_schema(CLIENT_SCHEMA)
    manager_schema_text = load_agent_schema(MANAGER_SCHEMA)
    guided_on = guided_enabled()
    qc_guided = response_format("qc", json.loads(qc_schema_text)) if guided_on else None
    client_guided = (
        response_format("call_client_v3", json.loads(client_schema_text)) if guided_on else None
    )
    manager_guided = (
        response_format("call_manager_v3", json.loads(manager_schema_text)) if guided_on else None
    )

    pass0 = _run_pass0(client, transcript_text, model, sink)
    call_type = pass0["call_type"]
    connection_lost = pass0["connection_lost"]
    logger.info(
        "Pass0 v3 call_type=%s reason=%s connection_lost=%s is_primary=%s",
        call_type,
        pass0.get("reason"),
        connection_lost,
        is_primary,
    )

    if call_type == "junk":
        return {
            "analyser_version": "v3",
            "call_type": "junk",
            "is_scoreable": False,
            "is_primary": is_primary,
            "reason": pass0.get("reason", "junk"),
            "connection_lost": connection_lost,
            "phases": {},
            "qc_scores": {},
            "manager_score_1to10": None,
        }

    qc_data, qc_ok = _run_qc(
        client,
        transcript_text,
        qc_schema_text,
        model,
        sink,
        guided=qc_guided,
        roles=mgr.roles_hint(segments, verified=labels_verified),
    )
    # qc приходит из сырого ответа модели: тип полей не гарантирован, а ниже по коду они
    # используются уже вне try. Приводим к ожидаемой форме здесь.
    qc_scores_raw = qc_data.get("qc_scores")
    if not isinstance(qc_scores_raw, dict):
        qc_scores_raw = {}
    # Цитату из qc проверяем кодом: модель заявляет её в отдельном поле, но заявление — не факт.
    qc_scores_raw = qc.project_qc_scores(qc_scores_raw, segments)
    qc_phases = qc_data.get("phases")
    if not isinstance(qc_phases, dict):
        qc_phases = {}

    analysis_text, labels_flag = _run_analysis(client, transcript_text, model, chain_context, sink)

    # ── клиентская часть ──
    client_messages = [
        {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FORMAT_USER_TEMPLATE.format(
                schema=client_schema_text,
                transcript=transcript_text,
                analysis=analysis_text,
            ),
        },
    ]
    data, client_retries = _format_with_retry(
        client,
        model,
        client_messages,
        max_tokens=6000,
        user="worker_analysis_format_v3",
        step_name="format_client",
        on_step=sink,
        guided=client_guided,
    )

    # N/A-политику применяем ДО менеджерской стадии: иначе неприменимые критерии уедут
    # в промпт как «проваленные», и weaknesses в отчёте будут противоречить qc_scores
    # в том же payload («не зафиксировал время встречи» на звонке с обрывом связи).
    qc_scores = qc_scores_raw
    if call_type == "service":
        qc_scores = mgr.apply_service_na(qc_scores)
    elif call_type == "sales" and not is_primary:
        qc_scores = mgr.apply_repeat_na(qc_scores)
    if connection_lost:
        qc_scores = mgr.apply_connection_lost_na(qc_scores)

    # ── менеджерская часть: критерии отбирает код, модель только формулирует ──
    passed, failed = mgr.split_qc_criteria(qc_scores)
    manager_messages = [
        {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": MANAGER_USER_TEMPLATE.format(
                schema=manager_schema_text,
                roles_hint=mgr.roles_hint(segments, verified=labels_verified),
                transcript=transcript_text,
                passed=json.dumps(passed, ensure_ascii=False, indent=1),
                failed=json.dumps(failed, ensure_ascii=False, indent=1),
            ),
        },
    ]
    try:
        manager_data, manager_retries = _format_with_retry(
            client,
            model,
            manager_messages,
            max_tokens=4000,
            user="worker_analysis_manager_v3",
            step_name="format_manager",
            on_step=sink,
            guided=manager_guided,
        )
    except ValueError:
        # Формат не сдался даже после ретрая — оставляем код-деривацию (она не зависит от
        # модели). Транспортные ошибки не глушим.
        logger.warning(
            "v3 manager pass returned unparseable JSON; keeping code-derived part", exc_info=True
        )
        manager_data, manager_retries = {}, 1

    manager_data["filler_words"] = mgr.count_filler_words(
        segments, roles_verified=detected_swap is not None
    )
    manager_data["qc_passed_count"] = len(passed)
    manager_data["qc_failed_count"] = len(failed)
    data["manager"] = manager_data

    data, timecodes_fixed = repair_timecodes(data, segments)

    data["analyser_version"] = "v3"
    data["analysis_nl"] = analysis_text
    data["speaker_labels"] = {
        "present": has_speaker_labels(segments),
        # Вердикт детерминированного детектора по ИСХОДНОМУ транскрипту. Проверяющая
        # сторона доверяет ему только при source == "detector" — самоотчёту модели нельзя.
        "swapped": detected_swap if detected_swap is not None else (labels_flag == "swapped"),
        "source": "detector" if detected_swap is not None else "model",
        "corrected_in_prompt": labels_corrected,
        "model_flag": labels_flag,
        "detected_swap": detected_swap,
        "detector_evidence": swap_evidence,
    }
    data["phases"] = qc_phases
    data["call_type"] = call_type
    data["is_primary"] = is_primary
    data["reason"] = pass0.get("reason", "")
    data["connection_lost"] = connection_lost
    data["qc_failed"] = not qc_ok
    data["is_scoreable"] = call_type == "sales" and not connection_lost and qc_ok
    data["format_retries"] = {"client": client_retries, "manager": manager_retries}
    data["timecodes_repaired"] = timecodes_fixed
    data["guided_json"] = guided_on
    data["truncated_stages"] = truncated_stages

    data["qc_scores"] = qc_scores
    data["qc_quote_stats"] = qc.summarise(qc_scores)

    data["manager_score_1to10"] = (
        calc_manager_score_1to10(qc_scores, call_type=call_type, is_primary=is_primary)
        if data["is_scoreable"]
        else None
    )
    return data

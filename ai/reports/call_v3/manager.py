"""Менеджерская часть v3 — то, что считается КОДОМ, а не моделью.

Принцип: LLM оставляем только настоящее суждение (психопрофиль, удачные/неудачные фразы).
Всё, что выводимо детерминированно, считаем кодом:

* **strengths / weaknesses** отбираются из ``qc_scores`` (пройденные / проваленные критерии).
  Модель их только ФОРМУЛИРУЕТ по переданному списку — выдумать новый пункт она не может, а
  strengths и weaknesses структурно не могут противоречить друг другу: множества критериев
  не пересекаются. (В v2 промпт был вынужден просить «не противоречьте сами себе».)
* **слова-паразиты** считаются по явному словарю над репликами менеджера. В v2 у модели
  запрашивалась «приблизительная оценка» ЧИСЛА — оценка количества это ровно то, что нельзя
  доверять модели.

Атрибуция реплик менеджеру опирается на метки транскрипта, но только ПОСЛЕ того, как
``detect_labels_swapped`` проверил их по смыслу и вызывающий код исправил перепутанные
(см. ``core``): дальше по конвейеру метка уже достоверна.
"""

from __future__ import annotations

import re
from typing import Any

from ai.shared.branding import seller_greeting_markers
from ai.reports.shared.transcript import (
    CLIENT_TAG,
    MANAGER_TAG,
    Segment,
    has_speaker_labels,
    manager_segments,
)

# Критерии — свойства ВСЕГО разговора, а не отдельной фразы. Подтверждающей цитаты у них не
# бывает: «тон был вежливым на протяжении разговора» доказывается отсутствием грубости, а не
# фрагментом. Замерено: на них приходилось 23 из 47 «противоречий» — то есть метрика требовала
# невозможного и записывала честный ответ модели в нарушения.
NOT_CITABLE = frozenset(
    {
        "general_polite_tone",
        "general_professional_speech",
        "general_initiative_control",
        "general_active_listening",
    }
)

# Критерии, неприменимые к повторному звонку в цепочке — менеджер прошёл их в первом контакте.
# Зеркалит политику call_v2.scoring на момент создания v3; это policy самой версии, поэтому
# объявлена здесь явно, а не импортируется из приватных v2.
REPEAT_SKIP = frozenset(
    {
        "contact_greeting_standard",
        "contact_name_and_object",
        "needs_purchase_goal",
        "needs_key_aspects",
        "needs_budget_and_payment",
        "presentation_project_advantages_5plus",
    }
)

# Критерии закрытия, неприменимые при обрыве связи.
CONNECTION_LOST_NA = frozenset(
    {
        "closing_meeting_datetime_fixed",
        "closing_next_actions_agreed",
        "closing_contacts_collected",
        "closing_questions_clarified",
        "closing_farewell_standard",
    }
)

# Явный, аудируемый словарь. Многословные идут первыми — иначе «как бы» распадётся на «как»+«бы».
FILLER_WORDS: tuple[str, ...] = (
    "это самое",
    "так сказать",
    "скажем так",
    "как бы",
    "в общем",
    "короче",
    "собственно",
    "получается",
    "значит",
    "типа",
    "вот",
    "ну",
    "эээ",
    "ммм",
)


def _na(reason: str) -> dict[str, Any]:
    return {"score": None, "evidence": reason}


def apply_service_na(qc_scores: dict[str, Any]) -> dict[str, Any]:
    """Сервисный звонок не скорится — все критерии в N/A (портрет и summary остаются)."""
    return {key: _na("н/п — сервисный звонок") for key in qc_scores}


def apply_repeat_na(qc_scores: dict[str, Any]) -> dict[str, Any]:
    result = dict(qc_scores)
    for key in REPEAT_SKIP:
        if key in result:
            result[key] = _na("н/п — повторный звонок в цепочке")
    return result


def apply_connection_lost_na(qc_scores: dict[str, Any]) -> dict[str, Any]:
    result = dict(qc_scores)
    for key in CONNECTION_LOST_NA:
        if key in result:
            result[key] = _na("н/п — связь прервана")
    return result


def _criterion_text(value: Any) -> str:
    """Текст критерия для менеджерского промпта: цитата И вердикт.

    Цитаты без вердикта мало — модель не поймёт, за что критерий засчитан. Вердикта без цитаты
    достаточно, когда цитировать нечего. А неподтверждённые цитаты приходят из ``comment`` уже
    помеченными (см. ``qc.REJECTED_PREFIX``): выдавать их за доказательство нельзя.
    """
    if not isinstance(value, dict):
        return ""
    evidence = str(value.get("evidence") or "")
    comment = str(value.get("comment") or "")
    if evidence and not evidence.strip().lower().startswith("н/п"):
        return f"«{evidence}» — {comment}" if comment else f"«{evidence}»"
    if comment:
        return comment
    # Осталось только «н/п — не прозвучало»: для проваленного критерия это и есть содержание
    # («действия не было»), а «цитата не найдена» промпту ничего не сообщает.
    return evidence if value.get("quote_state") == "nothing_to_quote" else ""


# Два критерия по контракту не имеют поля score: у одного ``count``, у другого ``total``/
# ``handled``. ``calc_manager_score_1to10`` их считает, а наивная проверка «score is None»
# выбрасывает — и тогда отчёт молча теряет из strengths/weaknesses работу с возражениями и
# презентацию преимуществ, а счётчики пройденных/проваленных врут на два.
_ADVANTAGES_BAR = 5


def criterion_passed(key: str, body: Any, qc_scores: dict[str, Any] | None = None) -> bool | None:
    """Пройден ли критерий. ``None`` — неприменим (N/A или возражений не было).

    Отдельная функция, потому что ответ нужен и отбору strengths/weaknesses, и замеру
    противоречий в :mod:`ai.reports.call_v3.qc`, и разъезжаться этим двум нельзя.
    """
    # N/A проверяем ПЕРВЫМ. У «неприменимого» тела есть score=None и нет ни count, ни
    # total/handled — без этой проверки критерий, снятый политикой N/A, читался бы как
    # «count отсутствует, значит 0, значит провален», и попадал бы в weaknesses ровно там,
    # где мы его сознательно сняли (каждый повторный звонок в цепочке, каждый сервисный).
    if isinstance(body, dict) and "score" in body and body["score"] is None:
        return None

    if key == "presentation_project_advantages_5plus":
        count = body.get("count") if isinstance(body, dict) else body
        try:
            return float(count or 0) >= _ADVANTAGES_BAR
        except (TypeError, ValueError):
            return None
    if key == "objections_all_handled":
        if not isinstance(body, dict):
            return None
        try:
            total, handled = float(body.get("total") or 0), float(body.get("handled") or 0)
        except (TypeError, ValueError):
            return None
        if total == 0:
            return None  # возражений не было — критерий неприменим, а не выигран
        return handled >= total
    if key == "objections_techniques_used" and _no_objections(qc_scores):
        # Зеркалит calc_manager_score_1to10: без возражений технику работы с ними применять
        # не к чему. Иначе отчёт пишет «техники не применялись» там, где оценка не штрафует.
        return None

    score = body.get("score") if isinstance(body, dict) else body
    if score is None:
        return None
    try:
        return float(score) > 0
    except (TypeError, ValueError):
        # qc парсится из сырого JSON: "н/п" или "1/1" не должны ронять весь звонок
        return None


def _no_objections(qc_scores: dict[str, Any] | None) -> bool:
    """Возражений в звонке не было. Отсутствие поля ``total`` — это НЕ ноль.

    Модель иногда отвечает по этому критерию просто ``{"score": 1}``; читать это как «возражений
    не было» значит снимать соседний критерий с оценки на пустом месте.
    """
    objections = (qc_scores or {}).get("objections_all_handled")
    if not isinstance(objections, dict) or objections.get("total") is None:
        return False
    try:
        return float(objections["total"]) == 0
    except (TypeError, ValueError):
        return False


def split_qc_criteria(
    qc_scores: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Разделить критерии на пройденные и проваленные — основа для strengths/weaknesses.

    Неприменимое (N/A, отсутствие возражений) не попадает никуда: критерий не выигран и не
    провален.
    """
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for key, value in (qc_scores or {}).items():
        verdict = criterion_passed(key, value, qc_scores)
        if verdict is None:
            continue
        item = {"criterion": key, "evidence": _criterion_text(value)}
        (passed if verdict else failed).append(item)
    return passed, failed


def count_filler_words(
    segments: list[Segment], *, roles_verified: bool = True
) -> dict[str, Any] | None:
    """Посчитать слова-паразиты в репликах менеджера.

    ``None`` — когда реплики нельзя атрибутировать (моно без меток). Честное «неизвестно»
    вместо выдуманного числа.
    """
    if not has_speaker_labels(segments):
        return None
    mgr = manager_segments(segments, labels_swapped=False)
    if not mgr:
        return None

    text = " ".join(segment.text for segment in mgr).lower()
    counts: dict[str, int] = {}
    for filler in FILLER_WORDS:
        pattern = r"(?<![\w-])" + re.escape(filler) + r"(?![\w-])"
        found = len(re.findall(pattern, text))
        if found:
            counts[filler] = found
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "manager_segments": len(mgr),
        "source": "deterministic_dictionary",
        # Детектор ролей иногда воздерживается (нет маркеров, свидетельство из одной реплики,
        # неубедительный перевес). Тогда метки не проверены, и при их перестановке это число —
        # паразиты КЛИЕНТА. Считать всё равно честно, выдавать за проверенное — нет.
        "roles_verified": roles_verified,
    }


# Только ИСКЛЮЧИТЕЛЬНО продавцовые фразы: корпоративное приветствие, перевод звонка,
# самопредставление отдела продаж. Нейтральные вроде «меня зовут» / «чем могу помочь»
# сюда НЕ входят — их произносят и клиенты, а детектор ПЕРЕОПРЕДЕЛЯЕТ вердикт модели.
_SELLER_MARKERS: tuple[str, ...] = seller_greeting_markers() + (
    "отдел продаж",
    "отдела продаж",
    "передаю ваш звонок",
    "личному менеджеру",
    "личный менеджер",
    "ваш менеджер",
)


# Минимальный перевес маркеров, при котором вердикт считается надёжным.
_SWAP_MARGIN = 2


def detect_labels_swapped(segments: list[Segment]) -> tuple[bool | None, dict[str, Any]]:
    """Детерминированно проверить, не перепутаны ли метки спикеров.

    Считает продавцовые маркеры в каждой группе меток. Если группа [КЛИЕНТ] говорит их
    заметно чаще — метки перепутаны. Возвращает ``(вердикт, доказательства)``; вердикт
    ``None``, когда меток нет или сигнал неубедителен (тогда решает модель).

    Нужен потому, что стадия A ошибается: на смоук-звонке она сказала «ok», хотя строки
    «Приветствую вас от команды «Девелопер»» и «Передаю ваш звонок личному менеджеру» были
    помечены [КЛИЕНТ]. От этой метки зависят и qc, и менеджерские метрики.
    """
    if not has_speaker_labels(segments):
        return None, {"reason": "no_labels"}

    # Первая реплика — автоприветствие робота/IVR; оба промпта велят её игнорировать, и здесь
    # она тоже не в счёт. Иначе одна строка «Передаю ваш звонок личному менеджеру» даёт сразу
    # два маркера и в одиночку выбирает весь порог — тогда детектор измеряет, в какой канал
    # попал IVR, а не перепутаны ли метки.
    tally = {MANAGER_TAG: 0, CLIENT_TAG: 0}
    hit_segments = {MANAGER_TAG: 0, CLIENT_TAG: 0}
    for segment in segments[1:]:
        if segment.speaker not in tally:
            continue
        text = segment.text.lower()
        hits = sum(1 for marker in _SELLER_MARKERS if marker in text)
        if hits:
            tally[segment.speaker] += hits
            hit_segments[segment.speaker] += 1

    manager_hits, client_hits = tally[MANAGER_TAG], tally[CLIENT_TAG]
    evidence = {
        "seller_markers_in_manager_label": manager_hits,
        "seller_markers_in_client_label": client_hits,
        "segments_with_markers": dict(hit_segments),
    }
    # Маркеры должны прийти минимум из ДВУХ разных реплик: одна фраза, набравшая порог сама
    # по себе, — это не свидетельство систематической перестановки меток.
    if manager_hits == 0 and client_hits == 0:
        return None, {**evidence, "reason": "no_markers"}
    leader = MANAGER_TAG if manager_hits > client_hits else CLIENT_TAG
    if hit_segments[leader] < 2:
        return None, {**evidence, "reason": "single_segment_evidence"}
    # Требуется запас: перевес в один маркер слишком легко получить случайно, а цена
    # ошибки высока — от вердикта зависят qc, подсчёт паразитов и менеджерский промпт.
    if abs(manager_hits - client_hits) < _SWAP_MARGIN:
        return None, {**evidence, "reason": "inconclusive", "margin_required": _SWAP_MARGIN}
    return client_hits > manager_hits, evidence


def roles_hint(segments: list[Segment], *, verified: bool) -> str:
    """Короткая подсказка о ролях для промптов.

    Варианта «метки перепутаны» здесь нет: перепутанные метки чинятся в транскрипте до вызова
    модели, просьба «читай метки наоборот» на практике игнорируется. Но ``detect_labels_swapped``
    молчит чаще, чем отвечает (нет маркеров, одна реплика, недостаточный перевес), и выдавать
    непроверенные метки за проверенные — это ровно тот дефект, ради которого детектор и заведён.
    ``verified`` = детектор вынес вердикт; иначе метки остаются гипотезой.
    """
    if not has_speaker_labels(segments):
        return (
            "меток спикеров нет (моно) — определи продавца по смыслу реплик "
            "(представляется от «Девелопера», презентует объект, задаёт квалифицирующие вопросы)"
        )
    if verified:
        return (
            "метки в транскрипте проверены и при необходимости уже исправлены: "
            "[МЕНЕДЖЕР] — продавец, [КЛИЕНТ] — покупатель"
        )
    return (
        "метки в транскрипте есть, но НЕ подтверждены и БЫВАЮТ ПЕРЕПУТАНЫ — сверь их со смыслом "
        "реплик и оценивай того, кто ведёт себя как продавец («Девелопер», презентация объекта, "
        "квалифицирующие вопросы)"
    )

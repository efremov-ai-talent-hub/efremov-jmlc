"""Ремонт таймкодов под форму v3.

В v2 таймкод всегда лежал ВНУТРИ объекта рядом с ``evidence``, поэтому починка обходила
вложенные словари и искала ключ ``timecode``. В v3 форма другая и ремонт должен покрыть оба вида:

* плоские пары ``<field>_quote`` / ``<field>_timecode`` у скалярных квал-полей;
* элементы списков ``requirements`` / ``objections`` / фразы менеджера — ``quote`` + ``timecode``.

Модель называет таймкод по памяти и часто промахивается; здесь он приводится к реальному
сегменту транскрипта по тексту цитаты. Если цитата не находится — таймкод НЕ трогаем: пусть
grounding-чек увидит промах, а не заглаженный след.
"""

from __future__ import annotations

from typing import Any

from ai.reports.shared.transcript import Segment, find_real_timecode

_QUOTE_SUFFIX = "_quote"
_TIMECODE_SUFFIX = "_timecode"
# Куда кладётся исходное утверждение модели, когда код его исправил. Без этого проверка
# размещения меряет ответ кода, а не модели, и получается верной по построению — ровно тот
# дефект, что уже убран из qc-стадии.
_CLAIMED_SUFFIX = "_timecode_claimed"
CLAIMED_KEY = "timecode_claimed"


def _repair_pairs(node: dict[str, Any], segments: list[Segment]) -> int:
    """Плоские пары <field>_quote / <field>_timecode."""
    fixed = 0
    for key in list(node.keys()):
        if not key.endswith(_QUOTE_SUFFIX):
            continue
        field = key[: -len(_QUOTE_SUFFIX)]
        tc_key = f"{field}{_TIMECODE_SUFFIX}"
        if tc_key not in node:
            continue
        quote = str(node.get(key) or "").strip()
        if not quote:
            continue
        real = find_real_timecode(quote, str(node.get(tc_key) or ""), segments)
        if real and real != node.get(tc_key):
            node[f"{field}{_CLAIMED_SUFFIX}"] = node.get(tc_key)
            node[tc_key] = real
            fixed += 1
    return fixed


def _repair_items(node: dict[str, Any], segments: list[Segment]) -> int:
    """Элементы списков: объект с обоими ключами quote + timecode."""
    if "timecode" not in node:
        return 0
    quote = str(node.get("quote") or node.get("evidence") or node.get("phrase") or "").strip()
    if not quote:
        return 0
    real = find_real_timecode(quote, str(node.get("timecode") or ""), segments)
    if real and real != node.get("timecode"):
        node[CLAIMED_KEY] = node.get("timecode")
        node["timecode"] = real
        return 1
    return 0


def repair_timecodes(data: Any, segments: list[Segment]) -> tuple[Any, int]:
    """Пройти payload и привязать таймкоды к реальным сегментам. Возвращает (payload, сколько починено)."""
    fixed = 0
    if isinstance(data, dict):
        fixed += _repair_pairs(data, segments)
        fixed += _repair_items(data, segments)
        for value in data.values():
            _, sub = repair_timecodes(value, segments)
            fixed += sub
    elif isinstance(data, list):
        for value in data:
            _, sub = repair_timecodes(value, segments)
            fixed += sub
    return data, fixed

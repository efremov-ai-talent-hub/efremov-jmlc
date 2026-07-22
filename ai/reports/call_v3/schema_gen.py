"""Формальная JSON Schema, выведенная из описательного шаблона.

Схемы в ``static/agent/*.json`` — это ШАБЛОНЫ для промпта: значения описывают тип словами
(«string — цитата…»), а перечисления записаны как ``a|b|c``. Для grammar-constrained decoding
нужна формальная JSON Schema. Второй файл заводить нельзя — два источника правды разъедутся,
поэтому формальная схема ВЫВОДИТСЯ из шаблона по той же конвенции, на которую уже опирается
проверка enum-conformance в evals.

Правила вывода:
* строка вида ``a|b|c`` (только строчные идентификаторы) → ``enum``;
* прочая строка → ``string``; ``bool`` → ``boolean``; ЛЮБОЕ число → ``number``.
  Именно ``number``, а не ``integer``: в шаблоне qc стоит целое ``0``, но половинные оценки
  (0.5) легитимны для части критериев и учитываются скорингом. Грамматика с ``integer``
  физически не дала бы их выдать — оценка менеджера поехала бы молча;
* список → ``array`` с типом по первому элементу;
* словарь → ``object`` со всеми ключами в ``required`` и ``additionalProperties: false``.

Строгость намеренная: декодер обязан выдать ровно наши поля с ровно допустимыми значениями —
это и убирает out-of-enum, недостающие поля и мусор вокруг JSON.
"""

from __future__ import annotations

import re
from typing import Any

# Идентификатор enum-варианта: строчные буквы/цифры/подчёркивание. Прозу с «|» так не спутать.
_ENUM_TOKEN = re.compile(r"^[a-z0-9_]+$")


def _is_enum_literal(value: str) -> bool:
    if "|" not in value:
        return False
    tokens = [token.strip() for token in value.split("|")]
    return len(tokens) >= 2 and all(_ENUM_TOKEN.match(token) for token in tokens)


def _convert(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        properties = {key: _convert(item) for key, item in value.items()}
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
    if isinstance(value, list):
        item = _convert(value[0]) if value else {"type": "string"}
        return {"type": "array", "items": item}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, (int, float)):
        # Намеренно number, а не integer: половинные оценки qc обязаны остаться выразимыми.
        return {"type": "number"}
    if isinstance(value, str) and _is_enum_literal(value):
        return {"type": "string", "enum": [token.strip() for token in value.split("|")]}
    return {"type": "string"}


def to_json_schema(template: dict[str, Any]) -> dict[str, Any]:
    """Собрать формальную JSON Schema из описательного шаблона."""
    return _convert(template)


def response_format(name: str, template: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-совместимый ``response_format`` для guided-декодирования."""
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": to_json_schema(template), "strict": True},
    }

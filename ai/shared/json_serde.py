"""JSON encoding with datetime / date / Decimal support for prompts and DB payloads."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any


def json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, date):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(f"Object of type {type(o).__name__!r} is not JSON serializable")


def dumps(value: Any, **kwargs: Any) -> str:
    """Like ``json.dumps`` but encodes :class:`~datetime.datetime`, :class:`~datetime.date`, and :class:`~decimal.Decimal`."""
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(value, default=json_default, **kwargs)

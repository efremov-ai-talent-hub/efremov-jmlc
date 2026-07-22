from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPORTS_ROOT = Path(__file__).resolve().parent.parent


def load_agent_schema(name: str) -> str:
    schema_path = REPORTS_ROOT / "static" / "agent" / name
    if not schema_path.exists():
        raise FileNotFoundError(schema_path.resolve())
    return schema_path.read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    value = (text or "").strip()
    try:
        return json.loads(value)
    except Exception:
        pass

    match = re.search(r"\{.*\}", value, flags=re.S)
    if not match:
        raise ValueError("No JSON block found in model output")
    return json.loads(match.group(0))

"""Build-time patch: extend LiteLLM's A2A PascalCase→wire method map.

Anchored on this exact entry in litellm/proxy/agent_endpoints/a2a_endpoints.py:
    "GetTask": "tasks/get",

Why: native `a2a-python` v1.x clients (the canonical A2A SDK) send JSON-RPC
methods in PascalCase — including `SendMessage` and `SendStreamingMessage`.
LiteLLM's invoke handler normalises PascalCase to A2A wire spec via
`_PASCAL_TO_WIRE.get(method, method)`, but the map omits the two basic
message methods. They fall through `method == "message/send"` /
`method == "message/stream"` checks and hit the default branch:
    return _jsonrpc_error(request_id, -32601, f"Method '{method}' not found")
which returns HTTP 400, breaking every native a2a-sdk client. The
`tasks/*` and `agent/getAuthenticatedExtendedCard` mappings are present;
the two main ones were forgotten.

This patch inserts the missing entries right after `"GetTask"`. Tracked
upstream (TODO: open issue + PR once we have a clean repro outside our
stack).

Image ships two copies of the litellm source; we patch both, same as
the GigaChat patch. Build fails if neither copy contains the anchor —
that signals upstream changed the file and the patch needs review.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ANCHOR = re.compile(
    r'^(\s*)"GetTask":\s*"tasks/get",\s*$',
    re.MULTILINE,
)
SENTINEL = "# a2a-patch: SendMessage/SendStreamingMessage wire mapping"


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    try:
        from litellm.proxy.agent_endpoints import a2a_endpoints

        paths.append(Path(a2a_endpoints.__file__))
    except Exception as exc:
        print(f"WARN: could not import installed a2a_endpoints: {exc}", file=sys.stderr)
    app_path = Path("/app/litellm/proxy/agent_endpoints/a2a_endpoints.py")
    if app_path.exists() and app_path not in paths:
        paths.append(app_path)
    return paths


def _patch_file(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    if SENTINEL in src:
        return "already-patched"

    def _inject(match: re.Match) -> str:
        indent = match.group(1)
        return (
            f'{indent}{SENTINEL}\n'
            f'{indent}"SendMessage": "message/send",\n'
            f'{indent}"SendStreamingMessage": "message/stream",\n'
            f"{match.group(0)}"
        )

    new, n = ANCHOR.subn(_inject, src, count=1)
    if n != 1:
        return "anchor-not-found"

    path.write_text(new, encoding="utf-8")
    return "patched"


def main() -> int:
    candidates = _candidate_paths()
    if not candidates:
        print("ERROR: no a2a_endpoints.py copies found", file=sys.stderr)
        return 2

    any_patched = False
    for path in candidates:
        result = _patch_file(path)
        print(f"{result}: {path}")
        if result == "patched":
            any_patched = True
        elif result == "anchor-not-found":
            print(
                '  expected: "GetTask": "tasks/get",',
                file=sys.stderr,
            )

    if not any_patched and not any(
        SENTINEL in Path(p).read_text(encoding="utf-8") for p in candidates
    ):
        print(
            "FATAL: a2a method-map anchor not found in any copy of "
            "a2a_endpoints.py — upstream likely refactored. Patch needs review.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

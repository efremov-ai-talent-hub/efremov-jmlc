"""Build-time patch: map GigaChat blacklist/error finish_reason to content_filter.

Anchored on this exact line in litellm/llms/gigachat/chat/transformation.py:
    finish_reason = choice.get("finish_reason", "stop")

Why: GigaChat returns finish_reason="blacklist" when content filter blocks the
response. LiteLLM's Pydantic Literal[...] for Choices.finish_reason doesn't
include "blacklist" or "error", so the whole response raises ValidationError
inside transform_response — clients see HTTP 500 with no usable content.

This patch normalises those values to "content_filter" (semantically the
closest OpenAI-compatible value: no usable output, model refused).

The image ships TWO copies of the litellm source tree:
  - /usr/lib/python3.*/site-packages/litellm/...   (used by the CLI entrypoint)
  - /app/litellm/...                                (source checkout, used when
                                                    run as `python -m litellm`)
We patch both to avoid drift if the entrypoint changes or someone runs the
server from /app.

Fails the build if neither copy contains the anchor — that signals upstream
changed the file and the patch needs review.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ANCHOR = re.compile(
    r'^(\s*)finish_reason = choice\.get\("finish_reason", "stop"\)\s*$',
    re.MULTILINE,
)
SENTINEL = "# gigachat-patch: blacklist/error -> content_filter"


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    # Primary: the installed package the entrypoint actually loads.
    try:
        from litellm.llms.gigachat.chat import transformation
        paths.append(Path(transformation.__file__))
    except Exception as exc:
        print(f"WARN: could not import installed transformation: {exc}", file=sys.stderr)
    # Secondary: the source checkout under /app, if present.
    app_path = Path("/app/litellm/llms/gigachat/chat/transformation.py")
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
            f"{match.group(0)}\n"
            f"{indent}{SENTINEL}\n"
            f"{indent}if finish_reason in ('blacklist', 'error'):\n"
            f"{indent}    finish_reason = 'content_filter'"
        )

    new, n = ANCHOR.subn(_inject, src, count=1)
    if n != 1:
        return "anchor-not-found"

    path.write_text(new, encoding="utf-8")
    return "patched"


def main() -> int:
    candidates = _candidate_paths()
    if not candidates:
        print("ERROR: no transformation.py copies found", file=sys.stderr)
        return 2

    any_patched = False
    for path in candidates:
        result = _patch_file(path)
        print(f"{result}: {path}")
        if result == "patched":
            any_patched = True
        elif result == "anchor-not-found":
            print(
                f"  expected: finish_reason = choice.get(\"finish_reason\", \"stop\")",
                file=sys.stderr,
            )

    # Anchor missing in EVERY copy is a hard fail — upstream likely changed.
    # Anchor missing in ONE copy while another patched is fine (e.g. /app/ may
    # have older/different source). At least one copy must be effective.
    if not any_patched and not any(
        SENTINEL in Path(p).read_text(encoding="utf-8") for p in candidates
    ):
        print("ERROR: anchor not found in any transformation.py", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())

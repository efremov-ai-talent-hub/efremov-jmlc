"""Parse call transcripts into structured segments.

Transcript lines have the shape produced by
``ai.transcription.transcriber._format_segments``::

    [018.50–022.30] [МЕНЕДЖЕР] текст реплики

- Timecodes are **seconds** (float), not MM:SS.
- The separator is an en-dash (U+2013) or a plain hyphen.
- The role label is **optional**: mono recordings carry no
  ``[МЕНЕДЖЕР]``/``[КЛИЕНТ]`` marker.

The stock parser in ``ai.reports.call_v2.core._parse_transcript_segments``
discards the role; the speaker-attribution verifier needs it, so this parser
keeps it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ROLE_MANAGER = "МЕНЕДЖЕР"
ROLE_CLIENT = "КЛИЕНТ"

# [start–end] optional-[ROLE] text
_LINE = re.compile(
    r"\[(\d+(?:\.\d+)?)[–-](\d+(?:\.\d+)?)\]\s*"
    r"(?:\[(МЕНЕДЖЕР|КЛИЕНТ)\]\s*)?"
    r"(.*)"
)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    role: str | None  # ROLE_MANAGER | ROLE_CLIENT | None (mono recording)
    text: str


def parse_transcript(text: str) -> list[Segment]:
    """Parse a transcript string into ordered segments.

    Lines that do not match the timecode pattern (blank lines, headers) are
    skipped rather than raising — transcripts occasionally carry stray lines.
    """
    segments: list[Segment] = []
    for line in text.splitlines():
        match = _LINE.match(line.strip())
        if not match:
            continue
        start, end, role, body = match.groups()
        segments.append(
            Segment(
                start=float(start),
                end=float(end),
                role=role,
                text=body.strip(),
            )
        )
    return segments


def has_roles(segments: list[Segment]) -> bool:
    """True if any segment carries a speaker role (stereo recording).

    Mono recordings produce no role labels, so speaker-attribution checks
    cannot run and must report ``unknown`` instead of a violation.
    """
    return any(seg.role is not None for seg in segments)

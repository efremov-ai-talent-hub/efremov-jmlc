"""Seed the stand with calls: audio into object storage, metadata into Iceberg.

Run from the worker container, which already carries the dependencies and can
reach both MinIO and Trino:

    make seed

Idempotent by design — it is meant to be re-run. ``call_id`` is the file stem,
so seeding the same directory twice touches the same rows rather than creating
new ones. That stability matters beyond convenience: transcripts and reports key
on ``call_id``, and a re-seed that minted fresh ids would orphan everything
already produced for those calls while leaving the old rows to be picked up and
processed again.

Objects are re-uploaded unconditionally. `put_object` is idempotent, and doing
it every time means a row whose object went missing repairs itself instead of
becoming a call the transcription flow retries and fails on forever.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipelines import config
from pipelines.analysis.tables.writer import CallRow, existing_call_ids, insert_calls_batch
from pipelines.clients.trino import make_trino_engine, trino_conn_from_config
from pipelines.storage.minio import audio_key, make_minio_client, upload_bytes

logger = logging.getLogger(__name__)

# Formats pydub can open through ffmpeg, which the worker image installs.
_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

_CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}


def _duration_seconds(path: Path) -> float | None:
    """Real duration, read from the file rather than guessed.

    Best-effort: a file pydub cannot open still seeds, with a NULL duration. The
    column is informational — nothing downstream refuses to run without it — and
    failing the whole seed over one unreadable file would be the wrong trade.
    """
    try:
        from pydub import AudioSegment

        return round(len(AudioSegment.from_file(path)) / 1000.0, 3)
    except Exception as exc:
        logger.warning("could not read duration of %s: %s", path.name, exc)
        return None


def _started_at(path: Path) -> datetime:
    """The file's modification time.

    Not `now()`: these are recordings of calls that happened at some point, and
    stamping them with the moment they were seeded would put every call at the
    same instant and make `ORDER BY started_at` meaningless. mtime is at least a
    real property of the file.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def seed(directory: Path) -> int:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in _AUDIO_SUFFIXES)
    if not files:
        logger.warning("no audio files in %s — nothing to seed", directory)
        return 0

    client = make_minio_client()
    bucket = config.MINIO_RAW_FILES_BUCKET
    engine = make_trino_engine(trino_conn_from_config(), schema=config.ANALYSIS_SCHEMA)

    with engine.connect() as conn:
        known = existing_call_ids(conn, [p.stem for p in files])

        rows: list[CallRow] = []
        for path in files:
            call_id = path.stem
            suffix = path.suffix.lower()
            key = audio_key(call_id, suffix.lstrip("."))

            upload_bytes(
                client,
                bucket,
                key,
                path.read_bytes(),
                _CONTENT_TYPES.get(suffix, "application/octet-stream"),
            )
            logger.info("uploaded %s -> s3://%s/%s", path.name, bucket, key)

            if call_id in known:
                logger.info("call %s already recorded — object refreshed, row left alone", call_id)
                continue

            rows.append(
                CallRow(
                    call_id=call_id,
                    audio_key=key,
                    duration_seconds=_duration_seconds(path),
                    started_at=_started_at(path),
                    manager=None,
                    source="seed",
                )
            )

        if rows:
            insert_calls_batch(conn, rows)
        logger.info("seeded %d new call(s); %d already present", len(rows), len(known))
        return len(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/seeds/audio"),
        help="directory of audio files (default: /seeds/audio, mounted into the worker)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        logger.error("%s is not a directory", args.directory)
        return 1
    seed(args.directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())

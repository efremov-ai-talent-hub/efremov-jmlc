"""Eval dataset — the unit fed to the verifiers, plus loaders for it.

One ``Sample`` shape serves both run modes the eval has to support:

- **smoke** — a few hand-written samples kept in ``ai/evals/data/*.jsonl``,
  checked into the repo, run offline.
- **published** — a larger ``(source, older result)`` set built by a
  ``pipelines`` flow from MinIO + Iceberg and published as JSONL to MinIO/S3.
  :func:`load_jsonl` reads it the same way — only the location changes (a local
  path vs an ``s3://bucket/key`` URI). Reading ``s3://`` uses the read-only
  eval-reader credentials from the environment (see :func:`_s3_client`).

A sample carries the call transcript (``input``) and, optionally, the report
``reference`` produced for it. ``reference`` is what enables scoring an
already-produced result as-is and, later, pairwise old-vs-new comparison.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Env vars carrying the read-only eval-reader account (consumer side). Ansible
# renders these into .env from the env/<env>.yml `evals` block (creds default to
# the eval-reader account minio-init provisions; endpoint defaults to the local
# MinIO host port). The runner loads .env automatically, so they are normally
# present without manual export; setting them in the shell overrides .env.
_ENV_ENDPOINT = "EVAL_S3_ENDPOINT_URL"
_ENV_ACCESS_KEY = "EVAL_S3_ACCESS_KEY"
_ENV_SECRET_KEY = "EVAL_S3_SECRET_KEY"
_ENV_REGION = "EVAL_S3_REGION"
_DEFAULT_REGION = "us-east-1"


@dataclass(frozen=True)
class Sample:
    id: str
    input: str  # source — the call transcript
    reference: dict[str, Any] | None = None  # older result — the report payload, if any
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_remote(location: str | Path) -> bool:
    """True for an ``s3://`` URI, False for a local filesystem path."""
    return isinstance(location, str) and urlparse(location).scheme == "s3"


def _s3_client():
    """Build a boto3 S3 client for MinIO from the eval-reader env vars.

    ai.evals is a consumer and never imports ``pipelines`` (one-way layering),
    so it carries its own minimal client rather than reusing pipelines.storage.
    """
    endpoint = os.environ.get(_ENV_ENDPOINT)
    access_key = os.environ.get(_ENV_ACCESS_KEY)
    secret_key = os.environ.get(_ENV_SECRET_KEY)
    missing = [
        name
        for name, value in (
            (_ENV_ENDPOINT, endpoint),
            (_ENV_ACCESS_KEY, access_key),
            (_ENV_SECRET_KEY, secret_key),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Reading s3:// datasets needs the eval-reader credentials in the "
            f"environment; missing: {', '.join(missing)}."
        )
    import boto3  # lazy: offline/local runs never pay for the s3 dependency

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.environ.get(_ENV_REGION, _DEFAULT_REGION),
    )


def _read_text(location: str | Path) -> str:
    """Read a whole text object as UTF-8, from local disk or ``s3://``."""
    if _is_remote(location):
        parsed = urlparse(str(location))
        body = (
            _s3_client()
            .get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
            .read()
        )
        return body.decode("utf-8")
    return Path(location).read_text(encoding="utf-8")


def load_jsonl(location: str | Path) -> list[Sample]:
    """Load samples from a JSONL source (one JSON object per line).

    ``location`` is a local path or an ``s3://bucket/key`` URI; both are read
    the same way.
    """
    samples: list[Sample] = []
    for raw in _read_text(location).splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        samples.append(
            Sample(
                id=obj["id"],
                input=obj["input"],
                reference=obj.get("reference"),
                metadata=obj.get("metadata", {}),
            )
        )
    return samples


def list_datasets(location: str | Path) -> list[str]:
    """List ``*.jsonl`` datasets at a location, sorted.

    ``location`` is a local directory (or a single ``.jsonl`` file) or an
    ``s3://bucket/prefix``. Returns local paths or ``s3://`` URIs, each ready to
    hand to :func:`load_jsonl`.
    """
    if _is_remote(location):
        parsed = urlparse(str(location))
        bucket = parsed.netloc
        paginator = _s3_client().get_paginator("list_objects_v2")
        found: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=parsed.path.lstrip("/")):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    found.append(f"s3://{bucket}/{obj['Key']}")
        return sorted(found)
    path = Path(location)
    if path.is_dir():
        return sorted(str(p) for p in path.glob("*.jsonl"))
    if path.suffix == ".jsonl" and path.exists():
        return [str(path)]
    return []

"""Object storage helpers.

Audio lives in the raw-files bucket and is referenced from ``analysis.calls`` by
key; nothing here writes to Iceberg.
"""

from __future__ import annotations

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from pipelines import config


def audio_key(call_id: str, fmt: str = "mp3") -> str:
    return f"calls/{call_id}.{fmt}"


def make_minio_client() -> BaseClient:
    if not config.MINIO_ACCESS_KEY or not config.MINIO_SECRET_KEY:
        raise ValueError("MINIO_ROOT_USER / MINIO_ROOT_PASSWORD are required")
    return boto3.client(
        "s3",
        endpoint_url=config.MINIO_ENDPOINT,
        aws_access_key_id=config.MINIO_ACCESS_KEY,
        aws_secret_access_key=config.MINIO_SECRET_KEY,
        # Path-style: MinIO is reached by hostname, and virtual-host style would
        # try to resolve <bucket>.minio.
        config=Config(s3={"addressing_style": "path"}),
    )


def download_bytes(client: BaseClient, bucket: str, key: str) -> bytes:
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def upload_bytes(
    client: BaseClient, bucket: str, key: str, data: bytes, content_type: str
) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def object_exists(client: BaseClient, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except client.exceptions.ClientError:
        return False

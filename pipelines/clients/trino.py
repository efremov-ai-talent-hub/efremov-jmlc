"""Trino engine factory.

Single canonical path: ``trino_conn_from_config()`` then
``make_trino_engine(conn, schema=...)``. Callers do not read ``config.TRINO_*``
directly, so the wire to the environment stays in one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from pipelines import config


@dataclass(frozen=True)
class TrinoConn:
    """Connection parameters. Schema is supplied per-engine, not stored here."""

    host: str
    port: int
    user: str
    catalog: str
    http_scheme: str


def trino_conn_from_config() -> TrinoConn:
    return TrinoConn(
        host=config.TRINO_HOST,
        port=config.TRINO_PORT,
        user=config.TRINO_USER,
        catalog=config.TRINO_CATALOG,
        http_scheme=config.TRINO_HTTP_SCHEME,
    )


def make_trino_engine(
    conn: TrinoConn,
    *,
    schema: str,
    pool_pre_ping: bool = False,
) -> Engine:
    """SQLAlchemy engine bound to ``schema``.

    ``pool_pre_ping`` checks each connection before use — worth setting when one
    engine is reused across many statements over minutes.
    """
    from trino.sqlalchemy import URL

    url = URL(
        host=conn.host,
        port=conn.port,
        catalog=conn.catalog,
        schema=schema,
        user=conn.user,
        password=None,
    )
    return create_engine(
        url,
        connect_args={"http_scheme": conn.http_scheme},
        pool_pre_ping=pool_pre_ping,
    )

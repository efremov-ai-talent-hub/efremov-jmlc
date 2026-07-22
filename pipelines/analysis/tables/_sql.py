"""Shared SQL fragments for the analysis tables."""

from __future__ import annotations

from pipelines import config


def qualified(table: str) -> str:
    """Fully-qualified table name.

    Interpolated, not bound — Trino takes no parameters in identifier position.
    Every argument reaching here is a module-level literal; do not pass caller
    data through it.

    Catalog and schema are left unquoted so a value with uppercase keeps
    resolving the way Trino's own case-folding expects.
    """
    return f'{config.TRINO_CATALOG}.{config.ANALYSIS_SCHEMA}."{table}"'

"""Prefect orchestration shared by the analysis flows.

Private to ``flows.analysis`` — nothing outside imports it.
"""

from __future__ import annotations

from typing import Any, Callable

from prefect.futures import wait


def run_with_batched_persist(
    items: list[Any],
    *,
    parallelism: int,
    batch_size: int,
    submit_analyse: Callable[[Any], Any],
    persist_batch: Callable[[list[Any]], int],
) -> tuple[int, int]:
    """Analyse in waves, persist in batches, with the two sized independently.

    ``submit_analyse`` returns a Prefect future and must not raise — it returns a
    result object that ``persist_batch`` knows how to interpret, so one bad call
    does not abort the wave. Results accumulate until ``batch_size``, then are
    written; the tail is flushed at the end. Returns ``(succeeded, total)``.

    The two knobs are decoupled on purpose: parallelism is bounded by what the
    LLM endpoint will take, while batch size is bounded by Iceberg commit
    economics and by statement length. Tying them together would force one of
    the two to be wrong.
    """
    succeeded = 0
    buffer: list[Any] = []
    for i in range(0, len(items), parallelism):
        wave = items[i : i + parallelism]
        futures = [submit_analyse(item) for item in wave]
        wait(futures)
        for f in futures:
            buffer.append(f.result())
        while len(buffer) >= batch_size:
            chunk, buffer = buffer[:batch_size], buffer[batch_size:]
            succeeded += persist_batch(chunk)
    if buffer:
        succeeded += persist_batch(buffer)
    return succeeded, len(items)

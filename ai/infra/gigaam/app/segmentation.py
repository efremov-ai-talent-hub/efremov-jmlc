"""Склейка слов в сегменты — провайдер-агностично.

Зернистость транскрипта решается ЗДЕСЬ (пост-обработка), а не параметрами VAD
движка: транскрибируем крупными точными кусками (полный контекст, нет артефактов
на стыках), а на выходную зернитость режем по словам. Граница сегмента — конец
предложения (пунктуация e2e) или верхний кап длины.
"""

from __future__ import annotations

from .canonical import Segment, Word

_SENTENCE_END = (".", "?", "!", "…")


def glue_words(words: list[Word], *, max_seconds: float) -> list[Segment]:
    segments: list[Segment] = []
    cur: list[Word] = []
    start: float | None = None
    for w in words:
        if start is None:
            start = w.start
        cur.append(w)
        end_of_sentence = w.text.rstrip().endswith(_SENTENCE_END)
        too_long = (w.end - start) >= max_seconds
        if end_of_sentence or too_long:
            segments.append(Segment(start=start, end=w.end, text=" ".join(x.text for x in cur)))
            cur, start = [], None
    if cur:
        segments.append(
            Segment(start=start or 0.0, end=cur[-1].end, text=" ".join(x.text for x in cur))
        )
    return segments

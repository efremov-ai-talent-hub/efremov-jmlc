"""Pure transcript primitives shared by report analysers.

Public counterparts of the private helpers that grew inside ``call_v2.core``.
v2 keeps its own copies (it is frozen); v3 and anything newer import from here
instead of reaching into another version's privates.

A transcript line looks like::

    [001.29–002.45] [МЕНЕДЖЕР] Добрый день, Александр.

The speaker tag is optional — it is present only for two-channel (stereo)
recordings. **The tag is not authoritative**: diarization mislabels channels
often enough that callers must decide roles by meaning and pass the outcome in
(see ``manager_segments``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

from ai.reports.shared.matching import partial_ratio

_LINE_RE = re.compile(r"\[(\d+\.\d+)[–-](\d+\.\d+)\]\s*(.*)")
_SPEAKER_RE = re.compile(r"^\[(МЕНЕДЖЕР|КЛИЕНТ)\]\s*(.*)", re.IGNORECASE)
_SWAP_TAG_RE = re.compile(r"\[(МЕНЕДЖЕР|КЛИЕНТ)\]", re.IGNORECASE)

MANAGER_TAG = "МЕНЕДЖЕР"
CLIENT_TAG = "КЛИЕНТ"


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None  # MANAGER_TAG / CLIENT_TAG as written in the transcript

    @property
    def timecode(self) -> str:
        return f"{self.start:06.2f}–{self.end:06.2f}"


def parse_segments(transcript_text: str) -> list[Segment]:
    """Parse timecoded lines, keeping the speaker tag when the transcript has one."""
    segments: list[Segment] = []
    for line in (transcript_text or "").splitlines():
        match = _LINE_RE.match(line.strip())
        if not match:
            continue
        body = match.group(3).strip()
        speaker: str | None = None
        tagged = _SPEAKER_RE.match(body)
        if tagged:
            speaker = tagged.group(1).upper()
            body = tagged.group(2).strip()
        segments.append(
            Segment(
                start=float(match.group(1)), end=float(match.group(2)), text=body, speaker=speaker
            )
        )
    return segments


def has_speaker_labels(segments: list[Segment]) -> bool:
    return any(segment.speaker for segment in segments)


def relabel_swapped(transcript_text: str) -> str:
    """Swap the two speaker tags in a transcript whose channels were labelled the wrong way round.

    Only the tags move — text and timecodes are untouched. Handing a corrected transcript to the
    model is strictly more reliable than telling it "the labels lie, invert them as you read":
    that instruction was measured to be ignored.
    """
    return _SWAP_TAG_RE.sub(
        lambda m: f"[{CLIENT_TAG if m.group(1).upper() == MANAGER_TAG else MANAGER_TAG}]",
        transcript_text or "",
    )


def manager_segments(segments: list[Segment], *, labels_swapped: bool) -> list[Segment] | None:
    """Segments spoken by the manager, or ``None`` when roles cannot be attributed.

    ``labels_swapped`` comes from the analysis stage, which decides who is the
    buyer and who is the seller **by meaning**. Without speaker tags (mono) there
    is nothing to attribute per line, and the honest answer is ``None`` — callers
    must then report the derived metric as unknown rather than guess it.
    """
    if not has_speaker_labels(segments):
        return None
    wanted = CLIENT_TAG if labels_swapped else MANAGER_TAG
    return [segment for segment in segments if segment.speaker == wanted]


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    """Склеить соседние реплики ОДНОГО спикера.

    Дословная цитата нередко перетекает через границу сегментов — диаризация режет речь по
    паузам, а не по фразам. Без склейки честная цитата не находится и попадает в «не найдено».

    Реплики без метки (моно) НЕ склеиваются: там ``speaker is None`` у всех, и склейка сшила бы
    весь звонок в один сегмент. Тогда цитата получила бы таймкод всего звонка, а фраза,
    собранная из вопроса менеджера и ответа клиента, «подтвердилась» бы как одна реплика.
    """
    merged: list[Segment] = []
    for segment in segments:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.speaker is not None
            and previous.speaker == segment.speaker
        ):
            merged[-1] = Segment(
                start=previous.start,
                end=segment.end,
                text=f"{previous.text} {segment.text}".strip(),
                speaker=segment.speaker,
            )
        else:
            merged.append(segment)
    return merged


# Короткая цитата подтверждается почти в любом звонке («да» — подстрока половины реплик), и это
# ровно та лазейка, которой метрика честности играется в приятную сторону. Но «Добрый день» —
# законная дословная цитата для критерия приветствия. Поэтому: минимум два слова и восемь букв,
# а для коротких цитат планка почти абсолютная — но по тексту БЕЗ знаков препинания, иначе
# скопированная фраза с точкой на конце объявляется выдумкой.
MIN_QUOTE_WORDS = 2
MIN_QUOTE_CHARS = 8
EXACT_MATCH_BELOW_CHARS = 20
SHORT_QUOTE_THRESHOLD = 0.95

# Длина, начиная с которой слово годится в отпечаток для дешёвого предфильтра.
_FINGERPRINT_LEN = 5

# Многоточие и типографские кавычки — ровно то, что модель ставит чаще обычных: без них
# скопированное «Добрый день…» снова объявлялось бы выдумкой.
_PUNCTUATION = str.maketrans("", "", ".,!?;:()[]«»„“”‘’\"'…—–-")


def _depunctuate(text: str) -> str:
    return " ".join(text.translate(_PUNCTUATION).split())


def prepare_candidates(segments: list[Segment]) -> list[Segment]:
    """Реплики плюс их склейки — то, с чем сравнивается цитата.

    Отдельная функция, чтобы вызывающий считал список ОДИН раз на звонок, а не на каждый из
    23 критериев.
    """
    return [*segments, *_merge_adjacent(segments)]


def _fingerprints(quote: str) -> set[str]:
    """Начала длинных слов цитаты — дешёвый признак «тут может быть совпадение».

    Именно НАЧАЛА, а не слова целиком: ``partial_ratio`` терпит четверть расхождения, а в
    русском эта четверть — обычно окончание («рассматриваете» ↔ «рассматриваю»). Фильтр по
    целому слову отбрасывал такие пары, хотя они совпадают на 0.84.
    """
    return {w[:_FINGERPRINT_LEN] for w in quote.lower().split() if len(w) >= _FINGERPRINT_LEN}


@lru_cache(maxsize=4096)
def _depunctuated_text(segment: Segment) -> str:
    """Очищенный текст реплики. Кэш — потому что на звонок приходится 23 критерия и один и тот
    же набор реплик; сегмент неизменяем, так что кэшировать его безопасно."""
    return _depunctuate(segment.text).lower()


def find_quote_segment(
    quote: str,
    segments: list[Segment],
    *,
    threshold: float = 0.75,
    candidates: list[Segment] | None = None,
) -> Segment | None:
    """Реплика, из которой скопирована цитата, либо ``None``, если ни из какой.

    Сравнение — ``partial_ratio``: промпт просит ФРАГМЕНТ, и сопоставление с репликой целиком
    отвергало бы ровно то, что просили. Знаки препинания снимаются с ОБЕИХ сторон: они не
    относятся к тому, произносил ли человек эти слова, а длину цитаты меняют.
    """
    bare = _depunctuate(quote or "")
    if not bare or not segments:
        return None
    if len(bare.split()) < MIN_QUOTE_WORDS or len(bare.replace(" ", "")) < MIN_QUOTE_CHARS:
        return None

    # Порог считаем по очищенной длине — иначе два восклицательных знака переводят цитату через
    # границу «короткая/длинная» и меняют вердикт на тех же самых словах.
    bar = SHORT_QUOTE_THRESHOLD if len(bare) < EXACT_MATCH_BELOW_CHARS else threshold
    marks = _fingerprints(bare)
    pool = candidates if candidates is not None else prepare_candidates(segments)
    texts = [_depunctuated_text(candidate) for candidate in pool]

    def scan(use_prefilter: bool) -> tuple[float, Segment | None]:
        best_score, best_segment = 0.0, None
        for candidate, text in zip(pool, texts, strict=True):
            if use_prefilter and marks and not any(mark in text for mark in marks):
                continue
            score = partial_ratio(bare, text)
            if score > best_score:
                best_score, best_segment = score, candidate
                if best_score == 1.0:
                    break
        return best_score, best_segment

    best_score, best_segment = scan(use_prefilter=True)
    if best_score < bar and marks:
        # Ничего не дотянуло до порога — возможно, нужную реплику снял предфильтр (так бывает,
        # когда все длинные слова разошлись уже в начале: «посмотрели» ↔ «смотрели»). Условие
        # именно по ПОРОГУ, а не по «ничего не нашлось»: почти любой посторонний сегмент даёт
        # ненулевой счёт, и проверка на None молча оставляла вердикт за дешёвым фильтром.
        best_score, best_segment = scan(use_prefilter=False)
    return best_segment if best_score >= bar else None


def parse_timecode_start(timecode: str) -> float | None:
    if not timecode or timecode in ("not_specified", "unknown", ""):
        return None
    match = re.match(r"(\d+\.?\d*)", str(timecode).strip())
    return float(match.group(1)) if match else None


def find_real_timecode(
    citation: str,
    claimed_timecode: str,
    segments: list[Segment],
    *,
    window: float = 30.0,
    threshold: float = 0.6,
) -> str | None:
    """Snap a cited quote to the transcript segment it actually came from.

    Searches near the model's own timecode first (that is the claim being
    checked), then the whole transcript. Returns ``None`` when nothing matches
    well enough — the caller keeps the model's value so the grounding check can
    still see that it was wrong.
    """
    if not citation or not segments:
        return None

    def best_match(candidates: list[Segment]) -> tuple[float, Segment | None]:
        best_score, best_segment = 0.0, None
        for segment in candidates:
            score = SequenceMatcher(None, citation.lower(), segment.text.lower()).ratio()
            if score > best_score:
                best_score, best_segment = score, segment
        return best_score, best_segment

    claimed_start = parse_timecode_start(claimed_timecode)
    if claimed_start is not None:
        near = [s for s in segments if abs(s.start - claimed_start) <= window]
        score, segment = best_match(near)
        if score >= threshold and segment:
            return segment.timecode

    score, segment = best_match(segments)
    if score >= threshold and segment:
        return segment.timecode
    return None

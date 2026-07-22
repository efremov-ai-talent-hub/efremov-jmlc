"""QC-стадия v3: цитата и вердикт — РАЗНЫЕ поля, и цитата проверяется кодом.

Замерено дважды на одних и тех же звонках: в схеме v2 у критерия одно текстовое поле
``evidence``, и оно почти никогда не цитата — дословных 5–7%, выдуманных 41–44%. Правка
инструкций не помогла ни разу: ни преамбула «пиши дословно» (изменилось 5 ячеек из 111), ни
переписанное правило 2 самого промпта (4 из 115). Вывод один: пока цитата и вердикт делят одно
поле, модель пишет туда вердикт — ей есть что сказать, и сказать это больше негде.

Поэтому здесь два рычага вместо уговоров:

* **схема** разводит ``quote`` (дословно или «н/п — не прозвучало») и ``comment`` (вердикт), а
  guided-декодер обязывает заполнить оба — конкуренции за одно поле больше нет;
* **код** сверяет ``quote`` с транскриптом. Что не нашлось — не объявляется цитатой:
  ``quote_verified: false``, и текст уезжает в ``comment``. Отчёт при этом ничего не теряет,
  но и не выдаёт пересказ за цитату.

Наружу отдаётся привычная форма ``{"score": …, "evidence": …}``: её читают и
``calc_manager_score_1to10``, и витрины, и проверки. Меняется не контракт, а то, чему в нём
можно верить.

Подтверждений у критерия МНОЖЕСТВО (``quotes`` — массив). Один слот на критерий был ошибкой:
«обратился по имени 3+ раза» доказывается тремя цитатами, «5+ преимуществ» — пятью, и модель
честно пыталась сшить их в одну строку («Спасибо. Да. До свидания. Всего доброго.»), после чего
такая сшивка не совпадала ни с одной репликой. Заодно исчезает вторая болячка: «цитировать
нечего» теперь пишется ровно одним способом — пустым массивом, а не то маркером, то пустой
строкой (замерено: 80 против 111 — одно поведение, разложенное по двум корзинам метрики).

Часть критериев (:data:`ai.reports.call_v3.manager.NOT_CITABLE`) цитировать нечем ПО ПРИРОДЕ:
вежливость тона доказывается отсутствием грубости. У них пустой массив — правильный ответ, и в
замере они идут своим состоянием, а не как нарушение.

ВАЖНО ПРО ЗАМЕР. Раз код убирает неподтверждённые цитаты, проверка grounding по qc становится
зелёной по построению и больше ничего не ловит. Настоящий замер стадии — :func:`summarise`, и в
нём отдельно считаются ПРОТИВОРЕЧИЯ: критерий засчитан, а подтвердить его нечем. Без этого
счётчика метрика играбельна в приятную сторону — модель, которая везде отмалчивается, выглядит
безупречно честной.
"""

from __future__ import annotations

from typing import Any

from ai.reports.call_v3 import manager as mgr
from ai.reports.call_v3.manager import criterion_passed
from ai.reports.shared.transcript import Segment, find_quote_segment, prepare_candidates

SCHEMA = "qc_v3.schema.json"

# Маркеры для поля ``evidence``, которое читают скоринг и витрины. Они начинаются с «н/п» —
# проверки трактуют такой префикс как «нечего оценивать», а не как цитату.
NOTHING_TO_QUOTE = "н/п — подтверждений не приведено"
QUOTE_NOT_FOUND = "н/п — приведённые цитаты не найдены в транскрипте"
NOT_CITABLE = "н/п — критерий о разговоре в целом, отдельной цитаты не бывает"

# Пометка для текста, который модель выдавала за цитату, а код её в звонке не нашёл.
REJECTED_PREFIX = "не найдено в транскрипте: "

# Порог для ``find_quote_segment``: он сравнивает цитату с наиболее подходящей ЧАСТЬЮ реплики
# (промпт просит фрагмент, а не реплику целиком), поэтому планка выше, чем у сравнения со всей
# репликой — там дословный фрагмент длинного ответа не набирал и 0.5.
MATCH_THRESHOLD = 0.75

_SCORE_KEYS = ("score", "count", "total", "handled")

# Состояние критерия по его подтверждениям. Ключи ``summarise`` — они же, плюс
# ``contradictions``, ``unmeasured`` и счётчики самих цитат. Публичное имя: печать замера в
# ``ai.evals.analysis_runner`` читает его отсюда, чтобы состояния не разъезжались.
STATES = ("verified", "partly_verified", "nothing_to_quote", "unverified", "not_citable")


def _is_marker(text: str) -> bool:
    """«н/п …» — в этом поле не доказательство, а его отсутствие."""
    return text.strip().lower().startswith("н/п")


def _text(body: dict[str, Any], key: str) -> str:
    value = body.get(key)
    return value.strip() if isinstance(value, str) else ""


def _claimed_quotes(body: dict[str, Any]) -> list[tuple[str, str]]:
    """Заявленные подтверждения как ``(цитата, таймкод)``.

    Понимает и прежнюю форму с одним полем ``quote``: модель без grammar-декодера иногда
    отвечает по старой схеме, и терять её текст только из-за формы нельзя.
    """
    claims: list[tuple[str, str]] = []
    raw = body.get("quotes")
    if isinstance(raw, dict):  # один объект вместо массива — частая форма без grammar-декодера
        raw = [raw]
    elif isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                text, timecode = _text(entry, "quote"), _text(entry, "timecode")
            elif isinstance(entry, str):
                text, timecode = entry.strip(), ""
            else:
                continue
            if text and not _is_marker(text):
                claims.append((text, timecode))
    single = _text(body, "quote")
    if not claims and single and not _is_marker(single):
        claims.append((single, _text(body, "timecode")))
    return claims


def project_qc_scores(raw: dict[str, Any], segments: list[Segment]) -> dict[str, Any]:
    """Свести ответ qc к ``{score, evidence, …}``, проверив КАЖДОЕ подтверждение кодом."""
    result: dict[str, Any] = {}
    candidates = prepare_candidates(segments)

    for name, body in (raw or {}).items():
        if not isinstance(body, dict):
            # Без guided-декодера модель иногда отдаёт голое число вместо объекта. Это законный
            # для скоринга вид (``calc_manager_score_1to10`` его понимает), и выбросить ключ
            # значило бы молча превратить пройденный критерий в проваленный.
            result[name] = body
            continue

        item: dict[str, Any] = {key: body[key] for key in _SCORE_KEYS if key in body}
        comment = _text(body, "comment")
        claims = _claimed_quotes(body)

        verified: list[dict[str, str]] = []
        rejected: list[str] = []
        for text, claimed_timecode in claims:
            segment = find_quote_segment(
                text, segments, threshold=MATCH_THRESHOLD, candidates=candidates
            )
            if segment is None:
                rejected.append(text)
            else:
                verified.append(
                    {
                        "quote": text,
                        # Только то, что заявила МОДЕЛЬ: проверки должны иметь возможность
                        # этот таймкод не подтвердить. Подставить сюда найденный кодом значит
                        # сделать точность размещения верной по построению. Ответ кода лежит
                        # рядом, в ``segment_timecode``.
                        "timecode": claimed_timecode,
                        "segment_timecode": segment.timecode,
                    }
                )

        if name in mgr.NOT_CITABLE:
            state, evidence = "not_citable", NOT_CITABLE
        elif verified and not rejected:
            state, evidence = "verified", verified[0]["quote"]
        elif verified:
            state, evidence = "partly_verified", verified[0]["quote"]
        elif claims:
            state, evidence = "unverified", QUOTE_NOT_FOUND
        else:
            state, evidence = "nothing_to_quote", NOTHING_TO_QUOTE

        # Отвергнутое не пропадает, но и не выдаётся за доказательство: содержание уезжает в
        # вердикт С ПОМЕТКОЙ. Дальше этот текст уходит в менеджерский промпт под именем
        # ``evidence``, и подавать там непроверенное как проверенное нельзя.
        marked = [f"{REJECTED_PREFIX}{text}" for text in rejected]
        merged = " ".join(part for part in (comment, *marked) if part).strip()
        item.update(
            {
                "evidence": evidence,
                "quotes": verified,
                "quotes_claimed": len(claims),
                "quote_verified": bool(verified),
                "quote_state": state,
                "comment": merged,
            }
        )
        if verified:
            item["timecode"] = verified[0]["segment_timecode"]
        result[name] = item
    return result


# Критерии, утверждающие КОЛИЧЕСТВО. Для двух оно приходит числом от модели, для третьего зашито
# в сам критерий («обратился по имени 3 и более раз»).
_QUANTITY_FIELD = {
    "presentation_project_advantages_5plus": "count",
    # Промпт просит по элементу на КАЖДОЕ возражение, то есть на total, а не на отработанные.
    "objections_all_handled": "total",
}
_QUANTITY_BAR = {"contact_name_used_3plus": 3}


def _understated(key: str, body: dict[str, Any]) -> bool:
    """Критерий утверждает количество, а подтверждений приведено меньше."""
    if key in _QUANTITY_BAR:
        required: Any = _QUANTITY_BAR[key]
    elif key in _QUANTITY_FIELD:
        required = body.get(_QUANTITY_FIELD[key])
    else:
        return False
    if required is None:
        return False
    try:
        return len(body.get("quotes") or []) < float(required)
    except (TypeError, ValueError):
        return False


def summarise(qc_scores: dict[str, Any]) -> dict[str, int]:
    """Замер qc-стадии по ИТОГОВЫМ критериям — после того, как применены N/A.

    Считать до N/A нельзя: у сервисного звонка все критерии заменяются на «неприменимо», их
    цитаты никуда не идут, и замер смешивал бы измеренное с выброшенным.

    ``contradictions`` — критерий засчитан, а подтвердить его нечем. Критерии из
    :data:`ai.reports.call_v3.manager.NOT_CITABLE` сюда не попадают: у них цитаты не бывает,
    и требовать её значит мерить не то.
    """
    stats = dict.fromkeys(
        (*STATES, "contradictions", "unmeasured", "quotes_claimed", "quotes_verified"), 0
    )
    for key, body in (qc_scores or {}).items():
        state = body.get("quote_state") if isinstance(body, dict) else None
        if state not in STATES:
            # Нечего измерять: критерий снят политикой N/A либо пришёл скаляром без полей.
            # Считаем отдельно, чтобы сумма состояний + unmeasured сходилась с числом критериев.
            stats["unmeasured"] += 1
            continue
        stats[state] += 1
        if state != "not_citable":
            # Критерий, которому цитата не положена, в счётчик цитат не идёт: иначе доля
            # подтверждённых считается по множеству, часть которого мы вообще не оцениваем.
            stats["quotes_claimed"] += int(body.get("quotes_claimed") or 0)
            stats["quotes_verified"] += len(body.get("quotes") or [])
        if criterion_passed(key, body, qc_scores) and (
            state in ("nothing_to_quote", "unverified") or _understated(key, body)
        ):
            # Засчитан, а подтвердить нечем — либо цитат нет вовсе, либо их меньше, чем
            # утверждаемое количество. Второе важно отдельно: критерии «3+ обращений по имени» и
            # «5+ преимуществ» — ровно те, ради которых заводился массив, и без этой проверки
            # доля подтверждённых выглядит лучше, чем есть.
            stats["contradictions"] += 1
    return stats

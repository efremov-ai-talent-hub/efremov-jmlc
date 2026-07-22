"""Run the call analyser over a dataset and score the FRESH result — score-fresh mode.

``ai.evals.runner`` scores a report that already exists (no LLM, no network). This runner
produces the report first, by calling the analyser itself, and then applies the very same
verifiers. Both modes print the same blocks, so a fresh run and the historical baseline are
directly comparable — that is the whole point of reusing ``runner.print_block``.

Usage::

    python -m ai.evals.analysis_runner DATASET --out DIR [--version v3] [--guided]
    python -m ai.evals.analysis_runner ds.jsonl --out runs/v3 --limit 5
    python -m ai.evals.analysis_runner s3://eval-datasets/calls.jsonl --out runs/v3

``DATASET`` and ``--out`` are always given by the caller: the runner owns no location of its
own. It talks to whatever OpenAI-compatible endpoint ``ANALYSIS_OPENAI_BASE_URL`` points at —
the LiteLLM gateway in production, a local llama-server when a model is being evaluated
off-platform — so no Prefect and no platform services are involved.

Every LLM step is written out in full (prompts, raw response, latency, usage, finish reason)
next to the report and its scores, because an eval whose intermediate steps are not inspectable
cannot be argued with. ``--version v2`` also works; v2 has no step sink (it is frozen), so only
its report and scores are recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.evals.checks.enum_conformance import EnumResult, check_enum_conformance
from ai.evals.checks.grounding import GroundingResult, check_grounding
from ai.evals.checks.speaker import SpeakerResult, check_speaker_attribution
from ai.evals.checks.substance import DEFAULT_MIN_WORDS, SubstanceResult, check_substance
from ai.evals.dataset import Sample, load_jsonl
from ai.evals.env import GATEWAY_PREFIXES, S3_PREFIXES, load_env_defaults

# Записываем то, что РЕАЛЬНО применилось: ANALYSIS_GUIDED_JSON может стоять в окружении
# и без --guided, и тогда флаг из аргументов соврал бы про сам прогон.
from ai.evals.runner import print_block
from ai.reports.call_v3.core import guided_enabled
from ai.reports.call_v3.qc import STATES as _QC_STATES

DEFAULT_VERSION = "v3"
DEFAULT_MODEL = "GigaChat-2-Lite"


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    """Filesystem-safe directory name for a call id."""
    return _UNSAFE.sub("_", value).strip("_") or "sample"


@dataclass
class CallRun:
    """One call put through the analyser, with everything needed to audit the outcome."""

    sample: Sample
    payload: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float = 0.0
    steps: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.payload is not None


def resolved_model(requested: str) -> str:
    """The model the analyser will actually use.

    ``ANALYSIS_OPENAI_MODEL`` outranks the caller's ``--model``, so recording the requested name
    would put a model into the artefacts that never ran — a provenance defect in an eval.
    """
    from ai.reports.shared.llm_client import resolve_model_name

    return resolve_model_name(requested)


def _analyse(version: str, transcript: str, model: str, steps: list[dict[str, Any]]):
    """Call the requested analyser version, collecting steps when the version emits them."""
    if version == "v3":
        from ai.reports.call_v3.core import analyze_call_transcript

        def sink(name: str, record: dict[str, Any]) -> None:
            steps.append({"step": name, **record})

        return analyze_call_transcript(transcript, model_name=model, on_step=sink)

    from ai.reports.call_v2.core import analyze_call_transcript as analyze_v2

    return analyze_v2(transcript, model_name=model)


def run_sample(sample: Sample, *, version: str, model: str) -> CallRun:
    """Analyse one call. A failure is recorded, not raised: one bad call must not
    abort the run and silently shrink the denominator."""
    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        payload = _analyse(version, sample.input, model, steps)
        error = None
    except Exception:
        payload, error = None, traceback.format_exc()
    return CallRun(
        sample=sample,
        payload=payload,
        error=error,
        duration_s=time.monotonic() - started,
        steps=steps,
    )


def score_payload(
    payload: dict[str, Any], transcript: str
) -> tuple[GroundingResult, SpeakerResult, SubstanceResult, EnumResult]:
    return (
        check_grounding(payload, transcript),
        check_speaker_attribution(payload, transcript),
        check_substance(payload),
        check_enum_conformance(payload),
    )


def _scores_dict(
    g: GroundingResult, sp: SpeakerResult, su: SubstanceResult, en: EnumResult
) -> dict[str, Any]:
    """The printed metrics as data — same definitions, so runs stay comparable."""
    return {
        "grounding": {
            "quotes": g.total,
            "grounded_placed_rate": g.grounded_placed_rate,
            "grounded_unplaced_rate": g.grounded_unplaced_rate,
            "weak_rate": g.weak_rate,
            "misplacement_rate": g.misplacement_rate,
            "fabrication_rate": g.fabrication_rate,
            "timecode_only_rate": g.timecode_only_rate,
            "timecode_accuracy": g.timecode_accuracy,
        },
        "substance": {
            "quotes": su.total,
            "substantive_rate": su.rate,
            "min_words": su.min_words,
        },
        "speaker": {
            "manager": sp.manager_count,
            "client": sp.client_count,
            "accuracy": sp.accuracy,
            "violations": len(sp.violations),
        },
        "enum": {
            "fields": en.total,
            "conformance_rate": en.conformance_rate,
            "novel": en.novel_count,
            "sentinel": en.sentinel_count,
            "multiselect": en.multiselect_count,
            "echo": en.echo_count,
        },
    }


def guided_for(version: str) -> bool:
    """Действовало ли grammar-декодирование. Для v2 — всегда нет.

    ``guided_enabled()`` читает переменную окружения, а ``call_v2`` её не смотрит вовсе. Если
    в шелле осталась экспортированная ``ANALYSIS_GUIDED_JSON=1`` от прогона v3, наивное чтение
    проштампует v2-артефакты как guided — и провенанс соврёт ровно в ту сторону, ради которой
    вся эта запись и ведётся.
    """
    return version == "v3" and guided_enabled()


def _call_metadata(run: CallRun, *, version: str, model: str) -> dict[str, Any]:
    """Attribution for one call: which call, which run parameters, what the analyser reported.

    ``sample.metadata`` carries the call's own characteristics (provider, channels, duration)
    and is passed through untouched — a number without its call's properties cannot be read.
    """
    payload = run.payload or {}
    return {
        "call_id": run.sample.id,
        "sample_metadata": run.sample.metadata,
        "run": {
            "analyser_version": version,
            "model": model,
            "guided_json": guided_for(version),
            "duration_s": round(run.duration_s, 2),
            # v2 не отдаёт шаги наружу, и ноль здесь означал бы «ретраев не было», хотя на деле
            # это «нечем измерить». Два плеча нельзя сравнивать по полю, которое заполняет одно.
            "llm_steps": len(run.steps) if run.steps else None,
            "retries": (
                sum(1 for s in run.steps if s.get("step", "").endswith("_retry"))
                if run.steps
                else None
            ),
        },
        "analyser": {
            "analyser_version": payload.get("analyser_version"),
            "call_type": payload.get("call_type"),
            "connection_lost": payload.get("connection_lost"),
            "qc_failed": payload.get("qc_failed"),
            # Настоящий замер qc-стадии: grounding по ней зелёный по построению, см. call_v3/qc.py.
            "qc_quote_stats": payload.get("qc_quote_stats"),
            "is_scoreable": payload.get("is_scoreable"),
            "manager_score_1to10": payload.get("manager_score_1to10"),
            "truncated_stages": payload.get("truncated_stages"),
            "speaker_labels": payload.get("speaker_labels"),
            "timecodes_repaired": payload.get("timecodes_repaired"),
        },
        "error": run.error,
    }


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def persist(out_dir: Path, run: CallRun, *, version: str, model: str) -> Path:
    """Write one call's steps, report and metadata under ``out_dir/<call>/``."""
    call_dir = out_dir / _slug(run.sample.id)
    _write_json(call_dir / "meta.json", _call_metadata(run, version=version, model=model))
    (call_dir / "transcript.txt").write_text(run.sample.input, encoding="utf-8")
    if run.payload is not None:
        _write_json(call_dir / "report.json", run.payload)
    for index, step in enumerate(run.steps, start=1):
        _write_json(
            call_dir / "steps" / f"{index:02d}_{_slug(step.get('step', 'step'))}.json", step
        )
    return call_dir


def print_qc_stats(stats: Counter) -> None:
    """Print the qc stage's own measurement.

    Grounding over qc evidence is green by construction — the analyser drops what it cannot
    verify — so this is the number that actually says how often qc cites the call. Printing it
    next to the check blocks keeps the two from being confused.
    """
    measurable = sum(stats[k] for k in _QC_STATES)
    if not measurable:
        return
    parts = " | ".join(f"{k} {stats[k]} ({stats[k] / measurable:.0%})" for k in _QC_STATES)
    print(f"  qc quotes  | criteria {measurable:3d} | {parts}")
    print(f"      quotes claimed {stats['quotes_claimed']}, verified {stats['quotes_verified']}")
    print(
        f"      contradictions {stats['contradictions']} (criterion passed with nothing to cite)"
        + (f" | unmeasured {stats['unmeasured']}" if stats["unmeasured"] else "")
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai.evals.analysis_runner",
        description="Run the call analyser over a dataset and score the fresh result.",
    )
    parser.add_argument("dataset", help="local path or s3://bucket/key URI")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="directory for per-call steps, reports, scores and the run summary",
    )
    parser.add_argument(
        "--version", default=DEFAULT_VERSION, choices=("v2", "v3"), help="analyser version"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model name passed to the analyser (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--limit", type=int, default=None, help="analyse at most N samples")
    parser.add_argument(
        "--guided",
        action="store_true",
        help="constrain the format stages with a JSON grammar (v3 only; needs an endpoint "
        "supporting response_format)",
    )
    return parser


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv[1:])
    load_env_defaults(S3_PREFIXES + GATEWAY_PREFIXES)
    if args.guided:
        if args.version != "v3":
            print("--guided applies to v3 only")
            return 2
        os.environ["ANALYSIS_GUIDED_JSON"] = "1"

    samples = load_jsonl(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        # Отдельный код: «нечего анализировать» и «часть звонков упала» — разные исходы, и
        # обёртка вокруг раннера должна их различать.
        print(f"No samples in {args.dataset}")
        return 2

    model = resolved_model(args.model)
    if model != args.model:
        print(f"note: ANALYSIS_OPENAI_MODEL overrides --model {args.model!r} → {model!r}")

    out_dir: Path = args.out
    print(
        f"Dataset: {args.dataset}  ({len(samples)} sample(s))\n"
        f"Analyser: {args.version}  model: {model}  guided: {guided_for(args.version)}\n"
        f"Output: {out_dir}\n"
    )

    all_g, all_sp, all_su = [], [], []
    en_total, en_viol = 0, []
    summary: list[dict[str, Any]] = []
    qc_stats: Counter = Counter()
    failed = 0

    for sample in samples:
        run = run_sample(sample, version=args.version, model=model)
        call_dir = persist(out_dir, run, version=args.version, model=model)
        if not run.ok:
            failed += 1
            first_line = (run.error or "").strip().splitlines()[-1:] or [""]
            print(f"[{sample.id}] FAILED after {run.duration_s:.1f}s — {first_line[0]}")
            print(f"  traceback: {call_dir / 'meta.json'}\n")
            summary.append({"call_id": sample.id, "ok": False, "duration_s": run.duration_s})
            continue

        qc_stats.update(run.payload.get("qc_quote_stats") or {})
        g, sp, su, en = score_payload(run.payload, sample.input)
        scores = _scores_dict(g, sp, su, en)
        _write_json(call_dir / "scores.json", scores)
        all_g += g.items
        all_sp += sp.items
        all_su += su.items
        en_total += en.total
        en_viol += en.violations
        print_block(
            f"[{sample.id}]  {run.duration_s:.1f}s, {len(run.steps)} LLM step(s)", g, sp, su, en
        )
        print()
        summary.append(
            {"call_id": sample.id, "ok": True, "duration_s": run.duration_s, "scores": scores}
        )

    print("=" * 72)
    aggregate = (
        GroundingResult(items=all_g),
        SpeakerResult(items=all_sp),
        SubstanceResult(min_words=DEFAULT_MIN_WORDS, items=all_su),
        EnumResult(total=en_total, violations=en_viol),
    )
    print_block(f"AGGREGATE ({len(samples) - failed}/{len(samples)} analysed)", *aggregate)
    print_qc_stats(qc_stats)
    _write_json(
        out_dir / "summary.json",
        {
            "dataset": str(args.dataset),
            "analyser_version": args.version,
            "model": model,
            "guided_json": guided_for(args.version),
            "samples": len(samples),
            "analysed": len(samples) - failed,
            "failed": failed,
            "aggregate": _scores_dict(*aggregate),
            "qc_quote_stats": dict(qc_stats),
            "calls": summary,
        },
    )
    if failed:
        print(f"\n{failed}/{len(samples)} call(s) failed to analyse — see meta.json in each.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

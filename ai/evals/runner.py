"""Score a JSONL dataset of calls with the deterministic verifiers and print a report.

Usage::

    python -m ai.evals.runner                         # the checked-in smoke set
    python -m ai.evals.runner path/to/dataset.jsonl   # a local dataset
    python -m ai.evals.runner s3://eval-datasets/x.jsonl  # a published dataset
    python -m ai.evals.runner --list                  # list bundled datasets
    python -m ai.evals.runner --list s3://eval-datasets/  # list published ones

This is **score-reference** mode: each sample's ``reference`` payload (an
already-produced report) is checked against its ``input`` transcript. No LLM and
no network — it measures a result that already exists, exactly as written. It is
the same loop that will later score the older results in a published S3 dataset.

Scoring a *fresh* run instead — calling the analyser and checking what it just
produced — is ``ai.evals.analysis_runner``; it reuses :func:`print_block` from here
so both modes print the same blocks and stay comparable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai.evals.checks.enum_conformance import (
    EnumResult,
    EnumViolationKind,
    check_enum_conformance,
)
from ai.evals.checks.grounding import GroundingResult, check_grounding
from ai.evals.checks.speaker import SpeakerResult, check_speaker_attribution
from ai.evals.checks.substance import DEFAULT_MIN_WORDS, SubstanceResult, check_substance
from ai.evals.dataset import Sample, list_datasets, load_jsonl
from ai.evals.env import S3_PREFIXES, load_env_defaults

DEFAULT_DATASET = Path(__file__).parent / "data" / "smoke.jsonl"


def _pct(value: float | None) -> str:
    """Format a rate as a percentage, or ``н/п`` when it could not be computed."""
    return "н/п" if value is None else f"{value:5.0%}"


def score_sample(
    sample: Sample,
) -> tuple[GroundingResult, SpeakerResult, SubstanceResult, EnumResult]:
    payload = sample.reference or {}
    return (
        check_grounding(payload, sample.input),
        check_speaker_attribution(payload, sample.input),
        check_substance(payload),
        check_enum_conformance(payload),
    )


def print_block(
    label: str,
    g: GroundingResult,
    sp: SpeakerResult,
    su: SubstanceResult,
    en: EnumResult,
) -> None:
    print(f"{label}")
    print(
        f"  grounding  | quotes {g.total:2d} | grounded(@tc) {_pct(g.grounded_placed_rate)} | "
        f"grounded(no-tc) {_pct(g.grounded_unplaced_rate)} | weak {_pct(g.weak_rate)} | "
        f"misplaced {_pct(g.misplacement_rate)} | fabricated {_pct(g.fabrication_rate)} | "
        f"timecode-only {_pct(g.timecode_only_rate)} | timecode {_pct(g.timecode_accuracy)}"
    )
    for it in g.fabricated_items:
        anchored = "н/п" if it.anchored_ratio is None else f"{it.anchored_ratio:.2f}"
        print(
            f"      fabricated  {it.path}: «{it.quote}» "
            f"(at timecode {anchored}, elsewhere {it.full_ratio:.2f})"
        )
    for it in g.timecode_only_items:
        print(f"      timecode-only  {it.path}: «{it.quote}» (a segment pointer, not a quote)")
    for it in g.misplaced_items:
        print(
            f"      misplaced   {it.path}: «{it.quote}» "
            f"(timecode {it.timecode} → really at {it.best_segment_timecode})"
        )
    print(
        f"  substance  | quotes {su.total:2d} | substantive {_pct(su.rate)} (>= {su.min_words} words)"
    )
    for it in su.low_substance_items:
        print(f"      too short   {it.path}: «{it.quote}» ({it.word_count} word(s))")
    print(
        f"  speaker    | manager {sp.manager_count:2d} | client {sp.client_count:2d} | "
        f"correct {_pct(sp.accuracy)} | violations {len(sp.violations):2d}"
    )
    for it in sp.violations:
        print(
            f"      misattributed  {it.path}: «{it.phrase}» "
            f"is the client's line (segment {it.matched_segment_timecode})"
        )
    print(
        f"  enum       | fields {en.total:2d} | conformant {_pct(en.conformance_rate)} | "
        f"novel {en.novel_count} | sentinel {en.sentinel_count} | "
        f"multiselect {en.multiselect_count} | echo {en.echo_count}"
    )
    for v in en.violations:
        if v.kind is EnumViolationKind.NOVEL:
            print(f"      novel        {v.path}: «{v.value}»")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai.evals.runner",
        description=(
            "Score a JSONL dataset (local path or s3:// URI) with the deterministic verifiers."
        ),
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="local path or s3://bucket/key URI; default: bundled smoke set",
    )
    parser.add_argument(
        "--list",
        dest="list_location",
        nargs="?",
        const=str(DEFAULT_DATASET.parent),
        metavar="LOCATION",
        help=(
            "list available .jsonl datasets at LOCATION (local dir or s3:// "
            "prefix; default: bundled data dir) and exit"
        ),
    )
    return parser


def main(argv: list[str]) -> int:
    args = _build_parser().parse_args(argv[1:])
    load_env_defaults(S3_PREFIXES)

    if args.list_location is not None:
        found = list_datasets(args.list_location)
        if not found:
            print(f"No .jsonl datasets at {args.list_location}")
            return 1
        for loc in found:
            print(loc)
        return 0

    location: str | Path = args.dataset if args.dataset is not None else DEFAULT_DATASET
    samples = load_jsonl(location)
    if not samples:
        print(f"No samples in {location}")
        return 1

    print(f"Dataset: {location}  ({len(samples)} sample(s))\n")

    all_g, all_sp, all_su = [], [], []
    en_total, en_viol = 0, []
    for sample in samples:
        g, sp, su, en = score_sample(sample)
        all_g += g.items
        all_sp += sp.items
        all_su += su.items
        en_total += en.total
        en_viol += en.violations
        print_block(f"[{sample.id}]", g, sp, su, en)
        print()

    print("=" * 72)
    print_block(
        "AGGREGATE (pooled over all samples)",
        GroundingResult(items=all_g),
        SpeakerResult(items=all_sp),
        SubstanceResult(min_words=DEFAULT_MIN_WORDS, items=all_su),
        EnumResult(total=en_total, violations=en_viol),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

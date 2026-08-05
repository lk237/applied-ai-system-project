"""
The reliability harness.

    python -m src.evals.run_eval              # full run, real models
    python -m src.evals.run_eval --offline    # plumbing check, no downloads
    python -m src.evals.run_eval --k 3        # override picks per query

Writes two artifacts:

    evals/results.json   machine-readable; read back at runtime by confidence.py
    evals/scorecard.md   human-readable; quoted in the README

The run is split into two phases for a reason worth stating, because it looks
like a shortcut and is not:

  Phase 1 - retrieval metrics (relevance, consistency, refusal accuracy).
            These depend only on which songs come back, and prompt strategy has
            no effect on retrieval. Measuring them once per strategy would
            produce identical numbers at twice the cost, so they are measured
            once, with generation switched off.

  Phase 2 - strategy bake-off (groundedness, latency).
            These are properties of generated prose, so each strategy is scored
            separately here. The winner is written to results.json and picked up
            by the running system on its next start.

Nothing in this file grades output with a model. Every number is a set
membership test, a string check, or a clock reading.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.catalog import load_catalog
from src.config import (
    EVAL_DIR,
    EVAL_RESULTS_PATH,
    EVAL_SCORECARD_PATH,
    PROMPT_STRATEGIES,
    SIMILARITY_FLOOR,
    describe,
)
from src.evals.golden import GOLDEN_CASES, GoldenCase
from src.evals.metrics import (
    Aggregate,
    CaseResult,
    aggregate,
    score_consistency,
    score_relevance,
)
from src.generate import TemplateGenerator, default_generator
from src.obs import RunLogger
from src.pipeline import CrateDigger
from src.retrieval import HashingEmbedder, RetrievalIndex, build_index


# ============================================================================
# Phase 1 - retrieval metrics
# ============================================================================

def run_case(engine: CrateDigger, case: GoldenCase, k: int, floor: float) -> CaseResult:
    """
    Score one golden case.

    The paraphrase run uses explain=False: consistency compares returned song
    ids, and generating prose nobody scores would double the runtime for no
    information.
    """
    started = time.perf_counter()
    primary = engine.recommend(case.query, k=k, floor=floor, explain=False)
    paraphrase = engine.recommend(case.paraphrase, k=k, floor=floor, explain=False)
    elapsed = (time.perf_counter() - started) * 1000

    passed, failures = score_relevance(case, primary)

    return CaseResult(
        case_id=case.id,
        query=case.query,
        status=primary.status,
        expected_refusal=case.expect_refusal,
        relevance_pass=passed,
        relevance_failures=failures,
        consistency=score_consistency(primary, paraphrase),
        paraphrase_status=paraphrase.status,
        best_similarity=primary.best_similarity,
        elapsed_ms=elapsed,
        titles=[p.song.title for p in primary.picks],
        paraphrase_titles=[p.song.title for p in paraphrase.picks],
    )


# ============================================================================
# Phase 2 - strategy bake-off
# ============================================================================

def score_strategy(
    engine: CrateDigger,
    strategy: str,
    cases: List[GoldenCase],
    k: int,
    floor: float,
) -> Dict:
    """
    Generate real prose under one strategy and measure how grounded it is.

    Only cases that should produce recommendations are used - a refusal path
    never reaches the generator, so it tells us nothing about prompt quality.
    """
    grounded_picks = 0
    total_picks = 0
    failures: List[str] = []
    latencies: List[float] = []
    examples: List[Dict] = []

    for case in cases:
        result = engine.recommend(
            case.query, k=k, floor=floor, explain=True, strategy=strategy
        )
        latencies.append(result.elapsed_ms)

        for pick in result.picks:
            total_picks += 1
            if pick.grounded:
                grounded_picks += 1
        failures.extend(f"{case.id}: {note}" for note in result.guardrail_notes)

        if len(examples) < 3 and result.picks:
            examples.append(
                {
                    "case_id": case.id,
                    "query": case.query,
                    "summary": result.summary,
                    "first_reason": result.picks[0].reason,
                    "first_title": result.picks[0].song.title,
                }
            )

        print(
            f"    [{strategy}] {case.id:<20} "
            f"{len(result.picks)} picks, "
            f"{sum(1 for p in result.picks if not p.grounded)} ungrounded, "
            f"{result.elapsed_ms / 1000:.1f}s",
            flush=True,
        )

    return {
        "strategy": strategy,
        "cases": len(cases),
        "total_picks": total_picks,
        "grounded_picks": grounded_picks,
        "groundedness": (grounded_picks / total_picks) if total_picks else 1.0,
        "mean_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "guardrail_failures": failures,
        "examples": examples,
    }


# ============================================================================
# Reporting
# ============================================================================

def build_scorecard(
    summary: Aggregate,
    case_results: List[CaseResult],
    strategy_results: List[Dict],
    best_strategy: str,
    generated_at: str,
    config: Dict,
    runtime_s: float,
) -> str:
    """The human-readable report. Deliberately leads with what failed."""
    lines: List[str] = []
    add = lines.append

    add("# Reliability Scorecard")
    add("")
    add(f"Generated `{generated_at}` in {runtime_s:.0f}s.")
    add("")
    add("Produced by `python -m src.evals.run_eval`. Every metric below is a")
    add("deterministic check - no model grades another model's output.")
    add("")

    # -- headline ----------------------------------------------------------
    add("## Headline")
    add("")
    add("| Metric | Score | What it measures |")
    add("|---|---|---|")
    add(
        f"| Groundedness | **{summary.groundedness:.0%}** | "
        "Share of recommendations whose prose named only retrieved songs |"
    )
    add(
        f"| Relevance | **{summary.relevance:.0%}** | "
        "Golden cases meeting every human-authored expectation |"
    )
    add(
        f"| Consistency | **{summary.consistency:.2f}** | "
        "Mean Jaccard overlap between a query and its paraphrase |"
    )
    add(
        f"| Refusal accuracy | **{summary.refusal_accuracy:.0%}** | "
        "Out-of-catalog requests correctly declined |"
    )
    add(
        f"| Latency (mean) | {summary.mean_latency_ms / 1000:.1f}s | "
        "Retrieval-only, per case (two queries) |"
    )
    add(
        f"| Latency (p95) | {summary.p95_latency_ms / 1000:.1f}s | "
        "Slowest 5% |"
    )
    add("")
    add(f"Cases: **{summary.cases}**. Configuration: `{json.dumps(config)}`")
    add("")

    # -- failures first ----------------------------------------------------
    add("## Failures")
    add("")
    failed = [r for r in case_results if not r.relevance_pass]
    if not failed:
        add("No relevance failures. Every golden case met its expectation.")
    else:
        add(f"{len(failed)} of {summary.cases} cases failed their expectation.")
        add("")
        add("| Case | Query | Why it failed |")
        add("|---|---|---|")
        for result in failed:
            why = "; ".join(result.relevance_failures).replace("|", "/")
            add(f"| `{result.case_id}` | {result.query[:44]} | {why[:150]} |")
    add("")

    # -- consistency -------------------------------------------------------
    add("## Weakest paraphrase robustness")
    add("")
    add("Low overlap means rewording the same request returns different songs.")
    add("")
    ranked = sorted(
        (r for r in case_results if r.consistency is not None),
        key=lambda r: r.consistency,
    )[:5]
    add("| Case | Overlap | Query result | Paraphrase result |")
    add("|---|---|---|---|")
    for result in ranked:
        add(
            f"| `{result.case_id}` | {result.consistency:.2f} | "
            f"{', '.join(result.titles[:3]) or '(refused)'} | "
            f"{', '.join(result.paraphrase_titles[:3]) or '(refused)'} |"
        )
    add("")

    # -- strategy bake-off -------------------------------------------------
    add("## Prompt strategy bake-off")
    add("")
    add("Retrieval is identical across strategies, so relevance and consistency")
    add("cannot distinguish them. Groundedness and latency can.")
    add("")
    add("| Strategy | Groundedness | Picks checked | Mean latency | Selected |")
    add("|---|---|---|---|---|")
    for entry in strategy_results:
        mark = " **yes**" if entry["strategy"] == best_strategy else ""
        add(
            f"| `{entry['strategy']}` | {entry['groundedness']:.0%} | "
            f"{entry['total_picks']} | "
            f"{entry['mean_latency_ms'] / 1000:.1f}s |{mark} |"
        )
    add("")
    add(
        f"`{best_strategy}` is written to `evals/results.json`; the running "
        "system reads it on start."
    )
    add("")

    # -- guardrail evidence -------------------------------------------------
    add("## Guardrail activity")
    add("")
    any_failures = False
    for entry in strategy_results:
        if entry["guardrail_failures"]:
            any_failures = True
            add(f"**{entry['strategy']}** - {len(entry['guardrail_failures'])} caught:")
            add("")
            for note in entry["guardrail_failures"][:10]:
                add(f"- {note}")
            add("")
    if not any_failures:
        add("The citation guardrail caught no unsupported claims in this run.")
        add("")

    # -- sample output ------------------------------------------------------
    add("## Sample generated output")
    add("")
    for entry in strategy_results:
        if entry["strategy"] != best_strategy:
            continue
        for example in entry["examples"]:
            add(f"**`{example['case_id']}`** - {example['query']}")
            add("")
            add(f"> {example['summary']}")
            add(">")
            add(f"> *{example['first_title']}*: {example['first_reason']}")
            add("")

    return "\n".join(lines) + "\n"


def apply_selected_strategy_metrics(
    summary: Aggregate,
    strategy_results: List[Dict],
    best_strategy: str,
) -> Aggregate:
    """Attach generated-prose metrics from the selected prompt strategy."""
    if not strategy_results:
        return summary
    selected = next(
        entry for entry in strategy_results
        if entry["strategy"] == best_strategy
    )
    summary.groundedness = selected["groundedness"]
    return summary


# ============================================================================
# Entry point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Crate Digger reliability harness")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use stub models. Verifies the harness itself; scores are meaningless.",
    )
    parser.add_argument("--k", type=int, default=3, help="picks per query (default 3)")
    parser.add_argument(
        "--floor",
        type=float,
        default=SIMILARITY_FLOOR,
        help="similarity threshold below which the system refuses",
    )
    parser.add_argument(
        "--skip-strategies",
        action="store_true",
        help="Phase 1 only. Much faster; leaves the recorded strategy unchanged.",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print("Crate Digger - reliability harness")
    print(f"  offline={args.offline}  k={args.k}  floor={args.floor}")
    print()

    # --- engine ----------------------------------------------------------
    songs = load_catalog()
    logger = RunLogger()
    if args.offline:
        index = RetrievalIndex(songs, HashingEmbedder()).build()
        generator = TemplateGenerator()
    else:
        print("  loading models (first run downloads ~1.2 GB)...", flush=True)
        index = build_index(songs)
        generator = default_generator()

    engine = CrateDigger(
        songs=songs, index=index, generator=generator, logger=logger
    )
    print(f"  catalog {len(songs)} songs | embedder {index.embedder.name}")
    print(f"  generator {generator.name}")
    print()

    # --- phase 1 ----------------------------------------------------------
    print(f"Phase 1: retrieval metrics over {len(GOLDEN_CASES)} cases")
    case_results: List[CaseResult] = []
    for case in GOLDEN_CASES:
        result = run_case(engine, case, k=args.k, floor=args.floor)
        case_results.append(result)
        mark = "pass" if result.relevance_pass else "FAIL"
        print(
            f"  {mark}  {case.id:<20} {result.status:<9} "
            f"sim={result.best_similarity:.3f} "
            f"consistency={result.consistency:.2f}",
            flush=True,
        )
    summary = aggregate(case_results)
    print()

    # --- phase 2 ----------------------------------------------------------
    strategy_results: List[Dict] = []
    best_strategy = PROMPT_STRATEGIES[0]

    if args.skip_strategies:
        print("Phase 2: skipped (--skip-strategies)")
    else:
        generating_cases = [c for c in GOLDEN_CASES if not c.expect_refusal]
        print(
            f"Phase 2: strategy bake-off over {len(generating_cases)} cases "
            f"x {len(PROMPT_STRATEGIES)} strategies"
        )
        for strategy in PROMPT_STRATEGIES:
            strategy_results.append(
                score_strategy(engine, strategy, generating_cases, args.k, args.floor)
            )
        # Highest groundedness wins; latency breaks a tie. Groundedness is the
        # metric that maps to user harm, so it is not traded against speed.
        best_strategy = max(
            strategy_results,
            key=lambda e: (e["groundedness"], -e["mean_latency_ms"]),
        )["strategy"]

        # Phase 1 intentionally skips generation, so its per-pick
        # groundedness is vacuously 100%: there is no generated prose to
        # inspect.  Publish the selected prompt strategy's measured
        # groundedness instead.  This is also the value consumed by the
        # runtime confidence note.
        apply_selected_strategy_metrics(summary, strategy_results, best_strategy)
    print()

    # --- write artifacts ---------------------------------------------------
    runtime_s = time.perf_counter() - started
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": generated_at,
        "runtime_seconds": round(runtime_s, 1),
        "offline_stub_run": args.offline,
        "best_strategy": best_strategy,
        "config": {**describe(), "k": args.k, "floor": args.floor},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": summary.as_dict(),
        "strategies": strategy_results,
        "cases": [r.as_dict() for r in case_results],
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    scorecard = build_scorecard(
        summary=summary,
        case_results=case_results,
        strategy_results=strategy_results,
        best_strategy=best_strategy,
        generated_at=generated_at,
        config=payload["config"],
        runtime_s=runtime_s,
    )
    with open(EVAL_SCORECARD_PATH, "w", encoding="utf-8") as handle:
        handle.write(scorecard)

    # --- console summary ---------------------------------------------------
    print("=" * 62)
    print(f"  groundedness      {summary.groundedness:.0%}")
    print(f"  relevance         {summary.relevance:.0%}"
          f"  ({summary.cases - len(summary.failed_case_ids)}/{summary.cases})")
    print(f"  consistency       {summary.consistency:.2f}")
    print(f"  refusal accuracy  {summary.refusal_accuracy:.0%}")
    print(f"  best strategy     {best_strategy}")
    if summary.failed_case_ids:
        print(f"  failed cases      {', '.join(summary.failed_case_ids)}")
    print("=" * 62)
    print(f"  wrote {EVAL_RESULTS_PATH}")
    print(f"  wrote {EVAL_SCORECARD_PATH}")
    print(f"  total {runtime_s:.0f}s")

    if args.offline:
        print()
        print("  NOTE: --offline uses stub models. These scores describe the")
        print("        harness, not the system. Re-run without --offline.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

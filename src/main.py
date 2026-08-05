"""
Command line interface for Crate Digger.

    python -m src.main "chill lofi for studying"
    python -m src.main "aggressive workout music" --k 5
    python -m src.main "korean pop idol group"            # near-miss handling
    python -m src.main "upbeat pop" --energy 2.0          # watch it clamp
    python -m src.main --baseline                         # Module 3 scorer
    python -m src.main --demo                             # scripted walkthrough

Exit codes are meaningful so this can be driven from a script:
    0  recommendations produced
    1  usage error
    2  request refused (nothing in the catalog matched)
    3  request rejected (input failed validation)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from src.catalog import load_catalog
from src.config import SIMILARITY_FLOOR, TOP_K, describe
from src.pipeline import CrateDigger, Recommendation
from src.recommender import Recommender, UserProfile

RULE = "=" * 74
THIN = "-" * 74


# ============================================================================
# Rendering
# ============================================================================

def render(result: Recommendation, show_similarity: bool = True) -> None:
    print()
    print(RULE)
    print(f'  "{result.query}"')
    print(RULE)

    for warning in result.warnings:
        print(f"  [guardrail] {warning}")
    if result.warnings:
        print()

    if result.status == "rejected":
        print(f"  REJECTED: {result.message}")
        print(f"\n  ({result.elapsed_ms:.0f} ms - no model was loaded)")
        return

    if result.status == "refused":
        print(f"  REFUSED: {result.message}")
        if result.confidence:
            print(f"\n  confidence: {result.confidence.band}")
        print(f"  ({result.elapsed_ms:.0f} ms)")
        return

    if result.summary:
        print(f"  {result.summary}")
        print()

    for rank, pick in enumerate(result.picks, start=1):
        flag = "   [ungrounded - text replaced]" if not pick.grounded else ""
        similarity = f"   sim={pick.similarity:.3f}" if show_similarity else ""
        print(f"  {rank}. {pick.song.title} - {pick.song.artist}{similarity}{flag}")
        print(
            f"     {pick.song.genre} / {pick.song.mood} / "
            f"energy {pick.song.energy:.2f} / {pick.song.tempo_bpm} BPM"
        )
        print(f"     {pick.reason}")
        print()

    if result.guardrail_notes:
        print(THIN)
        print("  GUARDRAIL - unsupported claims removed:")
        for note in result.guardrail_notes:
            print(f"    - {note}")
        print()

    if result.confidence:
        print(THIN)
        print(f"  confidence: {result.confidence.band.upper()}")
        print(f"  {result.confidence.note}")

    print(THIN)
    print(
        f"  strategy={result.strategy}   "
        f"top_similarity={result.best_similarity:.3f}   "
        f"{result.elapsed_ms:.0f} ms"
    )


def render_baseline(k: int) -> None:
    """
    The original Module 3 rule-based scorer, for side-by-side comparison.

    Kept so the README can show what the AI layer actually changed: this path
    accepts structured fields only and has no way to answer a natural-language
    request at all.
    """
    songs = load_catalog()
    engine = Recommender(songs)
    profiles = {
        "High-Energy Pop": UserProfile("pop", "happy", 0.9, False),
        "Chill Lofi": UserProfile("lofi", "chill", 0.25, True),
        "Deep Intense Rock": UserProfile("rock", "intense", 0.85, False),
    }
    print(f"\nBASELINE - rule-based scorer over {len(songs)} songs")
    print("(genre +2.0, mood +1.0, energy closeness up to +1.5)")

    for name, profile in profiles.items():
        print()
        print(RULE)
        print(
            f"  [{name}] genre={profile.favorite_genre} "
            f"mood={profile.favorite_mood} energy={profile.target_energy}"
        )
        print(RULE)
        for rank, song in enumerate(engine.recommend(profile, k=k), start=1):
            print(f"  {rank}. {engine.explain_recommendation(profile, song)}")


# ============================================================================
# Demo
# ============================================================================

DEMO_QUERIES = [
    (
        "chill lofi for studying",
        {},
        "1. Direct hit - the catalog covers this well",
    ),
    (
        "catchy pop music with guitars and big harmonies",
        {},
        "2. Near-genre test - exact-match scoring would ignore 'indie pop'",
    ),
    (
        "qqqq zzzz vvvv xxxx wwww",
        {},
        "3. Nonsense - the system should refuse rather than guess",
    ),
    (
        "",
        {},
        "4. Empty input - rejected before any model is loaded",
    ),
    (
        "upbeat pop",
        {"target_energy": 2.0},
        "5. Out-of-range energy - clamped, and the user is told",
    ),
]


def run_demo(engine: CrateDigger, k: int) -> None:
    print("\nCRATE DIGGER - scripted demo")
    print(f"config: {json.dumps(describe())}")
    for query, kwargs, note in DEMO_QUERIES:
        print()
        print(f"### {note}")
        render(engine.recommend(query, k=k, **kwargs))


# ============================================================================
# Entry point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Crate Digger - grounded music recommender",
    )
    parser.add_argument("query", nargs="?", help="what you want to listen to")
    parser.add_argument(
        "--k", type=int, default=TOP_K, help=f"how many picks (default {TOP_K})"
    )
    parser.add_argument(
        "--energy", type=float, default=None,
        help="target energy 0.0-1.0; out-of-range values are clamped, not rejected",
    )
    parser.add_argument(
        "--floor", type=float, default=SIMILARITY_FLOOR,
        help="similarity below which the system refuses to answer",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--no-explain", action="store_true",
        help="skip generation; retrieval plus deterministic reasons only (fast)",
    )
    parser.add_argument("--demo", action="store_true", help="scripted walkthrough")
    parser.add_argument(
        "--baseline", action="store_true",
        help="run the original Module 3 rule-based scorer",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="stub models, no downloads (plumbing check only)",
    )
    args = parser.parse_args(argv)

    if args.baseline:
        render_baseline(args.k)
        return 0

    if not args.query and not args.demo:
        parser.print_help()
        print('\nExample:\n  python -m src.main "chill lofi for studying"')
        return 1

    started = time.perf_counter()
    print("loading models (first run downloads ~1.2 GB)...", file=sys.stderr)
    engine = CrateDigger(offline=args.offline)
    print(f"ready in {time.perf_counter() - started:.1f}s", file=sys.stderr)

    if args.demo:
        run_demo(engine, args.k)
        return 0

    result = engine.recommend(
        args.query,
        k=args.k,
        target_energy=args.energy,
        floor=args.floor,
        explain=not args.no_explain,
    )

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        render(result)

    return {"ok": 0, "refused": 2, "rejected": 3}.get(result.status, 1)


if __name__ == "__main__":
    sys.exit(main())

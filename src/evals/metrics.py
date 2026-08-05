"""
The four reliability metrics.

Every metric here is deterministic. No model grades another model's output,
which is a deliberate choice: the local generator is far too small to be a
trustworthy judge, and a weak judge produces numbers that look rigorous while
meaning nothing. A string check that cannot be argued with is worth more than a
plausible score from an unreliable grader.

  groundedness  Did the model only discuss songs that were actually retrieved?
                Pure set membership. This is the metric that catches invention.

  consistency   Does a paraphrase of the same request return the same songs?
                Jaccard overlap of returned song ids. Measures robustness to
                wording rather than to re-running identical input.

  relevance     Do the returned songs satisfy the human-authored expectation
                for that case? All-or-nothing per case.

  latency       Wall-clock milliseconds per request, and the retrieval/generation
                split. Grounds any efficiency claim in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.evals.golden import GoldenCase
from src.pipeline import Pick, Recommendation


# ============================================================================
# Per-case result
# ============================================================================

@dataclass
class CaseResult:
    case_id: str
    query: str
    status: str                       # ok | refused | rejected
    expected_refusal: bool

    relevance_pass: bool = False
    relevance_failures: List[str] = field(default_factory=list)

    grounded_picks: int = 0
    total_picks: int = 0

    consistency: Optional[float] = None   # None for refusal cases
    paraphrase_status: str = ""

    best_similarity: float = 0.0
    elapsed_ms: float = 0.0
    titles: List[str] = field(default_factory=list)
    paraphrase_titles: List[str] = field(default_factory=list)
    guardrail_notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "status": self.status,
            "expected_refusal": self.expected_refusal,
            "relevance_pass": self.relevance_pass,
            "relevance_failures": self.relevance_failures,
            "grounded_picks": self.grounded_picks,
            "total_picks": self.total_picks,
            "consistency": (
                round(self.consistency, 4) if self.consistency is not None else None
            ),
            "paraphrase_status": self.paraphrase_status,
            "best_similarity": round(self.best_similarity, 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "titles": self.titles,
            "paraphrase_titles": self.paraphrase_titles,
            "guardrail_notes": self.guardrail_notes,
        }


# ============================================================================
# Relevance
# ============================================================================

def score_relevance(case: GoldenCase, result: Recommendation) -> tuple[bool, List[str]]:
    """
    Check a result against the case's human-authored expectations.

    Returns (passed, failure_descriptions). Only the assertions the case
    actually specifies are evaluated.
    """
    failures: List[str] = []

    # --- refusal cases: refusing IS the correct answer --------------------
    if case.expect_refusal:
        if result.status == "refused":
            return True, []
        titles = ", ".join(p.song.title for p in result.picks[:3])
        return False, [
            f"expected a refusal but got {result.status} "
            f"(best similarity {result.best_similarity:.3f}; returned {titles})"
        ]

    # --- everything else must produce picks -------------------------------
    if result.status != "ok" or not result.picks:
        return False, [f"expected recommendations but got {result.status}"]

    top = result.picks[0]
    picks: Sequence[Pick] = result.picks

    if case.genres_any:
        # Substring match so a request satisfied by "indie pop" counts against
        # an expectation of "pop" - the whole point of semantic retrieval.
        genre = top.song.genre.lower()
        if not any(term.lower() in genre for term in case.genres_any):
            failures.append(
                f"top pick genre '{top.song.genre}' is not one of "
                f"{list(case.genres_any)}"
            )

    if case.moods_any:
        mood = top.song.mood.lower()
        if not any(term.lower() == mood for term in case.moods_any):
            failures.append(
                f"top pick mood '{top.song.mood}' is not one of "
                f"{list(case.moods_any)}"
            )

    if case.energy_at_least is not None and top.song.energy < case.energy_at_least:
        failures.append(
            f"top pick energy {top.song.energy:.2f} < "
            f"required {case.energy_at_least:.2f}"
        )

    if case.energy_at_most is not None and top.song.energy > case.energy_at_most:
        failures.append(
            f"top pick energy {top.song.energy:.2f} > "
            f"allowed {case.energy_at_most:.2f}"
        )

    if case.all_energy_at_least is not None:
        low = [p for p in picks if p.song.energy < case.all_energy_at_least]
        if low:
            names = ", ".join(f"{p.song.title} ({p.song.energy:.2f})" for p in low)
            failures.append(
                f"energy below {case.all_energy_at_least:.2f} for: {names}"
            )

    if case.all_energy_at_most is not None:
        high = [p for p in picks if p.song.energy > case.all_energy_at_most]
        if high:
            names = ", ".join(f"{p.song.title} ({p.song.energy:.2f})" for p in high)
            failures.append(
                f"energy above {case.all_energy_at_most:.2f} for: {names}"
            )

    return not failures, failures


# ============================================================================
# Consistency
# ============================================================================

def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    """
    Overlap between two sets of song ids.

    Defined as 1.0 when both sides are empty, because two refusals of the same
    request agree perfectly - that is consistent behaviour, not a missing value.
    """
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def score_consistency(primary: Recommendation, paraphrase: Recommendation) -> float:
    return jaccard(
        [p.song.id for p in primary.picks],
        [p.song.id for p in paraphrase.picks],
    )


# ============================================================================
# Aggregation
# ============================================================================

@dataclass
class Aggregate:
    cases: int
    groundedness: float
    consistency: float
    relevance: float
    refusal_accuracy: float
    mean_latency_ms: float
    p95_latency_ms: float
    failed_case_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "cases": self.cases,
            "groundedness": round(self.groundedness, 4),
            "consistency": round(self.consistency, 4),
            "relevance": round(self.relevance, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "failed_case_ids": self.failed_case_ids,
        }


def aggregate(results: Sequence[CaseResult]) -> Aggregate:
    """Roll per-case results into the summary the scorecard publishes."""
    if not results:
        return Aggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_picks = sum(r.total_picks for r in results)
    grounded_picks = sum(r.grounded_picks for r in results)
    # A run that produced no picks at all is vacuously grounded: nothing was
    # claimed, so nothing was unsupported.
    groundedness = grounded_picks / total_picks if total_picks else 1.0

    consistencies = [r.consistency for r in results if r.consistency is not None]
    consistency = sum(consistencies) / len(consistencies) if consistencies else 1.0

    relevance = sum(1 for r in results if r.relevance_pass) / len(results)

    refusal_results = [r for r in results if r.expected_refusal]
    refusal_accuracy = (
        sum(1 for r in refusal_results if r.status == "refused") / len(refusal_results)
        if refusal_results else 1.0
    )

    latencies = sorted(r.elapsed_ms for r in results)
    mean_latency = sum(latencies) / len(latencies)
    p95_index = max(0, int(round(0.95 * (len(latencies) - 1))))

    return Aggregate(
        cases=len(results),
        groundedness=groundedness,
        consistency=consistency,
        relevance=relevance,
        refusal_accuracy=refusal_accuracy,
        mean_latency_ms=mean_latency,
        p95_latency_ms=latencies[p95_index],
        failed_case_ids=[r.case_id for r in results if not r.relevance_pass],
    )

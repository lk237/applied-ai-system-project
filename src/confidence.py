"""
The feedback loop: stored eval results changing runtime behaviour.

This is the module that makes the reliability harness load-bearing instead of
decorative. Two things are read out of evals/results.json at startup:

  1. Which prompt strategy to use. The harness scores every strategy in
     config.PROMPT_STRATEGIES and records the winner. generate.py asks this
     module which one to use rather than hard-coding a choice.

  2. How to describe confidence. A retrieval similarity is turned into a band,
     but the band is only stated as measured accuracy if the harness has
     actually been run. Before any eval exists the system says so, rather than
     inventing a number it has not earned.

Cold start is a first-class case: a fresh clone has no results.json, and the
system must still work. It falls back to the configured default strategy and
reports confidence as "unverified".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    DEFAULT_STRATEGY,
    EVAL_RESULTS_PATH,
    PROMPT_STRATEGIES,
)


@dataclass(frozen=True)
class EvalSummary:
    """What the runtime needs from a completed eval run."""
    exists: bool
    best_strategy: str
    groundedness: Optional[float] = None
    consistency: Optional[float] = None
    relevance: Optional[float] = None
    cases: Optional[int] = None
    generated_at: Optional[str] = None

    @property
    def provenance(self) -> str:
        if not self.exists:
            return (
                "No evaluation on record - run `python -m src.evals.run_eval` "
                "to measure this system."
            )
        return (
            f"Calibrated on {self.cases} golden cases "
            f"(groundedness {self.groundedness:.0%}, "
            f"relevance {self.relevance:.0%}), run {self.generated_at}."
        )


def load_eval_summary(path: Optional[Path] = None) -> EvalSummary:
    """
    Read evals/results.json if present.

    Every failure mode - missing file, corrupt JSON, unknown strategy name -
    degrades to the cold-start default instead of raising. A broken eval artifact
    must not be able to take down the recommender.
    """
    source = Path(path) if path else EVAL_RESULTS_PATH

    if not source.exists():
        return EvalSummary(exists=False, best_strategy=DEFAULT_STRATEGY)

    try:
        with open(source, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return EvalSummary(exists=False, best_strategy=DEFAULT_STRATEGY)

    summary = data.get("summary", {})
    winner = data.get("best_strategy", DEFAULT_STRATEGY)

    # Guard against a results file written by an older config that named a
    # strategy this build no longer implements.
    if winner not in PROMPT_STRATEGIES:
        winner = DEFAULT_STRATEGY

    def _num(key: str) -> Optional[float]:
        value = summary.get(key)
        return float(value) if isinstance(value, (int, float)) else None

    return EvalSummary(
        exists=True,
        best_strategy=winner,
        groundedness=_num("groundedness"),
        consistency=_num("consistency"),
        relevance=_num("relevance"),
        cases=summary.get("cases"),
        generated_at=data.get("generated_at"),
    )


@dataclass(frozen=True)
class Confidence:
    """A confidence verdict attached to one answer."""
    band: str          # high | medium | low
    similarity: float
    verified: bool     # was this band calibrated against a real eval run?
    note: str

    def as_dict(self) -> dict:
        return asdict(self)


def assess(similarity: float, summary: Optional[EvalSummary] = None) -> Confidence:
    """
    Turn a top retrieval similarity into a stated confidence.

    The band comes from thresholds; the *claim* attached to it comes from the
    eval record. This separation is deliberate - the system should never imply
    measured reliability it does not have.
    """
    record = summary if summary is not None else load_eval_summary()

    if similarity >= CONFIDENCE_HIGH:
        band = "high"
        shape = "The catalog contains a strong match for this request."
    elif similarity >= CONFIDENCE_MEDIUM:
        band = "medium"
        shape = "The closest tracks are related but not an exact fit."
    else:
        band = "low"
        shape = "This is a weak match; treat the suggestions as approximate."

    if record.exists and record.relevance is not None:
        note = (
            f"{shape} On {record.cases} evaluated cases this system placed a "
            f"correct-property track in the top {record.relevance:.0%} of the "
            f"time, with {record.groundedness:.0%} of claims traceable to the "
            "catalog."
        )
    else:
        note = f"{shape} Confidence is unverified - no evaluation has been run yet."

    return Confidence(
        band=band,
        similarity=round(float(similarity), 4),
        verified=record.exists,
        note=note,
    )


def active_strategy(summary: Optional[EvalSummary] = None) -> str:
    """The prompt strategy the harness found best, or the configured default."""
    record = summary if summary is not None else load_eval_summary()
    return record.best_strategy

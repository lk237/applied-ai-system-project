"""
Confidence and the eval feedback loop.

The loop is the part of this system most likely to fail silently: if
results.json cannot be read, the app still works and simply reports unverified
confidence, so a broken feedback loop looks exactly like a working one. These
tests are what stop that going unnoticed.
"""

from __future__ import annotations

import json

import pytest

from src.confidence import (
    EvalSummary,
    active_strategy,
    assess,
    load_eval_summary,
)
from src.config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, DEFAULT_STRATEGY


def _write_results(tmp_path, payload) -> "object":
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


REAL_RUN = {
    "generated_at": "2026-08-04T12:00:00+00:00",
    "best_strategy": "explanatory",
    "offline_stub_run": False,
    "summary": {
        "cases": 26,
        "groundedness": 0.94,
        "consistency": 0.71,
        "relevance": 0.81,
    },
}


# ============================================================================
# Cold start
# ============================================================================

def test_missing_results_file_falls_back_to_default(tmp_path):
    summary = load_eval_summary(tmp_path / "absent.json")
    assert summary.exists is False
    assert summary.best_strategy == DEFAULT_STRATEGY


def test_corrupt_results_file_does_not_raise(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("{ this is not json", encoding="utf-8")
    summary = load_eval_summary(path)
    assert summary.exists is False


def test_unknown_strategy_name_falls_back(tmp_path):
    """A results file from an older build must not select a dead strategy."""
    path = _write_results(tmp_path, {**REAL_RUN, "best_strategy": "no_such_strategy"})
    assert load_eval_summary(path).best_strategy == DEFAULT_STRATEGY


# ============================================================================
# Real runs
# ============================================================================

def test_real_run_is_loaded(tmp_path):
    summary = load_eval_summary(_write_results(tmp_path, REAL_RUN))
    assert summary.exists
    assert summary.best_strategy == "explanatory"
    assert summary.groundedness == pytest.approx(0.94)
    assert summary.cases == 26


def test_active_strategy_reads_the_winner(tmp_path):
    summary = load_eval_summary(_write_results(tmp_path, REAL_RUN))
    assert active_strategy(summary) == "explanatory"


def test_provenance_cites_the_run(tmp_path):
    summary = load_eval_summary(_write_results(tmp_path, REAL_RUN))
    assert "26" in summary.provenance
    assert "2026-08-04" in summary.provenance


# ============================================================================
# The stub-run guard
# ============================================================================

def test_offline_stub_run_is_ignored(tmp_path):
    """
    An --offline harness run scores stub models. Treating those numbers as
    calibration would let the app claim measured accuracy it never had.
    """
    path = _write_results(tmp_path, {**REAL_RUN, "offline_stub_run": True})
    summary = load_eval_summary(path)
    assert summary.exists is False
    assert summary.groundedness is None


def test_stub_run_produces_unverified_confidence(tmp_path):
    summary = load_eval_summary(
        _write_results(tmp_path, {**REAL_RUN, "offline_stub_run": True})
    )
    verdict = assess(0.9, summary)
    assert verdict.verified is False
    assert "unverified" in verdict.note.lower()


# ============================================================================
# Banding
# ============================================================================

@pytest.mark.parametrize(
    "similarity,expected",
    [
        (0.95, "high"),
        (CONFIDENCE_HIGH, "high"),
        (CONFIDENCE_HIGH - 0.01, "medium"),
        (CONFIDENCE_MEDIUM, "medium"),
        (CONFIDENCE_MEDIUM - 0.01, "low"),
        (0.0, "low"),
    ],
)
def test_bands_are_assigned_at_the_documented_thresholds(similarity, expected):
    assert assess(similarity, EvalSummary(exists=False, best_strategy="terse")).band == expected


def test_unverified_confidence_says_so_plainly():
    verdict = assess(0.9, EvalSummary(exists=False, best_strategy="terse"))
    assert verdict.verified is False
    assert "unverified" in verdict.note.lower()


def test_verified_confidence_quotes_measured_numbers(tmp_path):
    summary = load_eval_summary(_write_results(tmp_path, REAL_RUN))
    verdict = assess(0.9, summary)
    assert verdict.verified is True
    assert "unverified" not in verdict.note.lower()
    assert "26" in verdict.note        # case count is cited


def test_confidence_is_serialisable():
    payload = assess(0.5, EvalSummary(exists=False, best_strategy="terse")).as_dict()
    assert set(payload) == {"band", "similarity", "verified", "note"}
    json.dumps(payload)  # must not raise

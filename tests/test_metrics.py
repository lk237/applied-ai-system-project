"""
Metric arithmetic and golden-set integrity.

The metrics decide what the scorecard claims, so a bug here would quietly
misreport reliability - the worst possible failure for a project whose whole
argument is "measure it, don't assert it". These tests pin the arithmetic to
worked examples rather than to whatever the code currently returns.
"""

from __future__ import annotations

import pytest

from src.evals.golden import (
    GOLDEN_CASES,
    GoldenCase,
    recommendation_cases,
    refusal_cases,
)
from src.evals.metrics import Aggregate, CaseResult, aggregate, jaccard, score_relevance
from src.evals.run_eval import apply_selected_strategy_metrics
from src.pipeline import Pick, Recommendation
from src.recommender import Song


def _song(song_id=1, title="T", genre="pop", mood="happy", energy=0.5) -> Song:
    return Song(
        id=song_id, title=title, artist="A", genre=genre, mood=mood,
        energy=energy, tempo_bpm=100, valence=0.5, danceability=0.5,
        acousticness=0.5, description="d",
    )


def _result(status="ok", songs=None) -> Recommendation:
    picks = [
        Pick(song=s, similarity=0.5, reason="r", grounded=True)
        for s in (songs or [])
    ]
    return Recommendation(status=status, query="q", picks=picks)


# ============================================================================
# Jaccard
# ============================================================================

@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([1, 2, 3], [1, 2, 3], 1.0),        # identical
        ([1, 2, 3], [4, 5, 6], 0.0),        # disjoint
        ([1, 2], [2, 3], 1 / 3),            # one shared of three total
        ([], [], 1.0),                      # two refusals agree
        ([1], [], 0.0),                     # one refused, one did not
        ([1, 1, 2], [1, 2], 1.0),           # duplicates are set-collapsed
    ],
)
def test_jaccard_worked_examples(left, right, expected):
    assert jaccard(left, right) == pytest.approx(expected)


def test_two_refusals_count_as_perfectly_consistent():
    """
    Refusing the same request twice is consistent behaviour, not missing data.
    Scoring it 0.0 would penalise the system for correctly declining.
    """
    assert jaccard([], []) == 1.0


# ============================================================================
# Relevance scoring
# ============================================================================

def test_refusal_case_passes_when_refused():
    case = GoldenCase("c", "q", "p", "r", expect_refusal=True)
    passed, failures = score_relevance(case, _result(status="refused"))
    assert passed and not failures


def test_refusal_case_fails_when_answered():
    case = GoldenCase("c", "q", "p", "r", expect_refusal=True)
    passed, failures = score_relevance(case, _result(songs=[_song()]))
    assert not passed
    assert "expected a refusal" in failures[0]


def test_recommendation_case_fails_when_refused():
    case = GoldenCase("c", "q", "p", "r", genres_any=("pop",))
    passed, failures = score_relevance(case, _result(status="refused"))
    assert not passed


def test_genre_assertion_uses_substring_matching():
    """
    'indie pop' must satisfy an expectation of 'pop'. This is the assertion that
    encodes the whole point of semantic retrieval over exact matching.
    """
    case = GoldenCase("c", "q", "p", "r", genres_any=("pop",))
    passed, _ = score_relevance(case, _result(songs=[_song(genre="indie pop")]))
    assert passed


def test_genre_assertion_rejects_an_unrelated_genre():
    case = GoldenCase("c", "q", "p", "r", genres_any=("pop",))
    passed, failures = score_relevance(case, _result(songs=[_song(genre="metal")]))
    assert not passed
    assert "metal" in failures[0]


def test_mood_assertion_is_exact_not_substring():
    """Moods are a small controlled vocabulary, so substring matching would be
    sloppy - 'chill' should not satisfy an expectation of 'chilling'."""
    case = GoldenCase("c", "q", "p", "r", moods_any=("happy",))
    assert score_relevance(case, _result(songs=[_song(mood="happy")]))[0]
    assert not score_relevance(case, _result(songs=[_song(mood="unhappy")]))[0]


def test_top_pick_energy_bounds():
    case = GoldenCase("c", "q", "p", "r", energy_at_least=0.7)
    assert not score_relevance(case, _result(songs=[_song(energy=0.5)]))[0]
    assert score_relevance(case, _result(songs=[_song(energy=0.8)]))[0]


def test_all_energy_bound_checks_every_pick_not_just_the_first():
    case = GoldenCase("c", "q", "p", "r", all_energy_at_most=0.5)
    songs = [_song(1, energy=0.2), _song(2, "Loud", energy=0.9)]
    passed, failures = score_relevance(case, _result(songs=songs))
    assert not passed
    assert "Loud" in failures[0]


def test_multiple_failures_are_all_reported():
    case = GoldenCase("c", "q", "p", "r", genres_any=("jazz",), energy_at_least=0.9)
    passed, failures = score_relevance(case, _result(songs=[_song(genre="pop", energy=0.1)]))
    assert not passed
    assert len(failures) == 2


def test_case_with_no_assertions_passes_on_any_recommendation():
    case = GoldenCase("c", "q", "p", "r")
    assert score_relevance(case, _result(songs=[_song()]))[0]


# ============================================================================
# Aggregation
# ============================================================================

def test_aggregate_of_empty_input_is_all_zero():
    summary = aggregate([])
    assert summary.cases == 0
    assert summary.relevance == 0.0


def test_groundedness_is_picks_weighted_not_case_weighted():
    """
    A case returning five picks should count five times toward groundedness.
    Averaging per case would let a single-pick failure outweigh five successes.
    """
    results = [
        CaseResult("a", "q", "ok", False, grounded_picks=5, total_picks=5),
        CaseResult("b", "q", "ok", False, grounded_picks=0, total_picks=1),
    ]
    assert aggregate(results).groundedness == pytest.approx(5 / 6)


def test_no_picks_at_all_is_vacuously_grounded():
    """Nothing was claimed, so nothing was unsupported."""
    results = [CaseResult("a", "q", "refused", True, grounded_picks=0, total_picks=0)]
    assert aggregate(results).groundedness == 1.0


def test_relevance_is_the_fraction_of_passing_cases():
    results = [
        CaseResult("a", "q", "ok", False, relevance_pass=True),
        CaseResult("b", "q", "ok", False, relevance_pass=False),
        CaseResult("c", "q", "ok", False, relevance_pass=True),
        CaseResult("d", "q", "ok", False, relevance_pass=True),
    ]
    assert aggregate(results).relevance == pytest.approx(0.75)


def test_failed_case_ids_are_listed_for_the_scorecard():
    results = [
        CaseResult("good", "q", "ok", False, relevance_pass=True),
        CaseResult("bad", "q", "ok", False, relevance_pass=False),
    ]
    assert aggregate(results).failed_case_ids == ["bad"]


def test_refusal_accuracy_counts_only_refusal_cases():
    results = [
        CaseResult("r1", "q", "refused", True),
        CaseResult("r2", "q", "ok", True),          # should have refused
        CaseResult("n1", "q", "ok", False),         # not a refusal case
    ]
    assert aggregate(results).refusal_accuracy == pytest.approx(0.5)


def test_consistency_ignores_cases_that_did_not_measure_it():
    results = [
        CaseResult("a", "q", "ok", False, consistency=1.0),
        CaseResult("b", "q", "ok", False, consistency=0.0),
        CaseResult("c", "q", "ok", False, consistency=None),
    ]
    assert aggregate(results).consistency == pytest.approx(0.5)


# ============================================================================
# Golden-set integrity
# ============================================================================

def test_case_ids_are_unique():
    ids = [case.id for case in GOLDEN_CASES]
    assert len(set(ids)) == len(ids)


def test_every_case_has_a_distinct_paraphrase():
    """A paraphrase equal to the query would make consistency trivially 1.0."""
    for case in GOLDEN_CASES:
        assert case.query != case.paraphrase, case.id


def test_every_case_documents_why_a_human_expects_it():
    for case in GOLDEN_CASES:
        assert len(case.rationale) > 20, case.id


def test_set_contains_both_refusal_and_recommendation_cases():
    assert len(refusal_cases()) >= 3
    assert len(recommendation_cases()) >= 15


def test_recommendation_cases_assert_something():
    """A case with no assertions cannot fail, so it measures nothing."""
    for case in recommendation_cases():
        has_assertion = any(
            [
                case.genres_any,
                case.moods_any,
                case.energy_at_least is not None,
                case.energy_at_most is not None,
                case.all_energy_at_least is not None,
                case.all_energy_at_most is not None,
            ]
        )
        assert has_assertion, f"{case.id} asserts nothing"


def test_near_genre_cases_exist():
    """These are the cases that justify semantic retrieval over exact matching."""
    assert any("near-genre" in case.id for case in GOLDEN_CASES)


def test_headline_groundedness_uses_selected_generated_strategy():
    """Retrieval-only results must not publish vacuous 100% groundedness."""
    summary = Aggregate(2, 1.0, 0.5, 0.5, 1.0, 10.0, 12.0)
    strategies = [
        {"strategy": "terse", "groundedness": 0.85},
        {"strategy": "explanatory", "groundedness": 0.80},
    ]

    result = apply_selected_strategy_metrics(summary, strategies, "terse")

    assert result.groundedness == 0.85

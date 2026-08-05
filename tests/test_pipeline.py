"""
End-to-end pipeline behaviour, using stub models.

These tests assert on the contract every caller depends on: that a request
always returns a Recommendation rather than raising, that each status means what
it says, and that the guardrails are actually wired into the path rather than
merely existing.
"""

from __future__ import annotations

import json

import pytest

from src.generate import TemplateGenerator
from src.obs import RunLogger, read_events
from src.pipeline import CrateDigger, Recommendation


# ============================================================================
# Status contract
# ============================================================================

def test_valid_query_returns_ok_with_picks(engine):
    result = engine.recommend("quiet music for studying", k=2, floor=0.0)
    assert result.status == "ok"
    assert result.ok
    assert 1 <= len(result.picks) <= 2


def test_empty_query_is_rejected_not_raised(engine):
    result = engine.recommend("", k=2)
    assert result.status == "rejected"
    assert result.picks == []
    assert result.message


def test_rejection_happens_before_retrieval(engine):
    """A rejected request must not do similarity work at all."""
    result = engine.recommend("", k=2)
    assert result.best_similarity == 0.0


def test_impossible_floor_produces_a_refusal(engine):
    result = engine.recommend("quiet music", k=2, floor=1.01)
    assert result.status == "refused"
    assert result.picks == []
    assert "Nothing in this catalog" in result.message


def test_refusal_still_reports_confidence(engine):
    """A refusal is a considered answer, so it carries a confidence band too."""
    result = engine.recommend("quiet music", k=2, floor=1.01)
    assert result.confidence is not None
    assert result.confidence.band == "low"


@pytest.mark.parametrize("query", ["", "   ", "x"])
def test_all_bad_inputs_return_a_recommendation_object(engine, query):
    assert isinstance(engine.recommend(query), Recommendation)


# ============================================================================
# Guardrail integration
# ============================================================================

def test_out_of_range_energy_is_clamped_and_surfaced(engine):
    result = engine.recommend("loud music", k=2, target_energy=5.0, floor=0.0)
    assert result.status == "ok"
    assert any("clamped" in w for w in result.warnings)


def test_k_is_capped_and_surfaced(engine):
    result = engine.recommend("music", k=500, floor=0.0)
    assert any("exceeds the cap" in w for w in result.warnings)


def test_fabricated_song_title_is_stripped_from_output(tiny_catalog, tiny_index, tmp_path):
    """
    The core hallucination test.

    A generator that invents a song must not be able to put that song in front
    of a user. This is not hypothetical - the real model rendered "Studio 55"
    as "Studio 45" during development.
    """

    class HallucinatingGenerator:
        name = "hallucinator"

        def generate(self, system, user, max_new_tokens=48):
            return 'You should listen to "Completely Invented Song" instead.'

    engine = CrateDigger(
        songs=tiny_catalog,
        index=tiny_index,
        generator=HallucinatingGenerator(),
        logger=RunLogger(log_dir=tmp_path / "logs"),
    )
    result = engine.recommend("quiet music", k=1, floor=0.0)

    assert result.status == "ok"
    assert not result.picks[0].grounded
    assert "Completely Invented Song" not in result.picks[0].reason
    assert result.guardrail_notes


def test_stripped_output_falls_back_to_catalog_facts(tiny_catalog, tiny_index, tmp_path):
    """When nothing survives the guardrail, the user still gets something true."""

    class HallucinatingGenerator:
        name = "hallucinator"

        def generate(self, system, user, max_new_tokens=48):
            return 'Try "Nonexistent Track" for this.'

    engine = CrateDigger(
        songs=tiny_catalog, index=tiny_index, generator=HallucinatingGenerator(),
        logger=RunLogger(log_dir=tmp_path / "logs"),
    )
    pick = engine.recommend("quiet music", k=1, floor=0.0).picks[0]
    assert pick.reason
    # The fallback is built from catalog fields, so it names the real genre.
    assert pick.song.genre in pick.reason


def test_generator_failure_does_not_crash_the_request(tiny_catalog, tiny_index, tmp_path):
    class BrokenGenerator:
        name = "broken"

        def generate(self, system, user, max_new_tokens=48):
            raise RuntimeError("model exploded")

    engine = CrateDigger(
        songs=tiny_catalog, index=tiny_index, generator=BrokenGenerator(),
        logger=RunLogger(log_dir=tmp_path / "logs"),
    )
    result = engine.recommend("quiet music", k=2, floor=0.0)
    assert result.status == "ok"          # degraded, not failed
    assert all(pick.reason for pick in result.picks)


# ============================================================================
# explain=False
# ============================================================================

def test_explain_false_skips_generation_but_still_returns_reasons(
    tiny_catalog, tiny_index, tmp_path
):
    class ExplodingGenerator:
        name = "should-never-be-called"

        def generate(self, system, user, max_new_tokens=48):
            raise AssertionError("generation ran despite explain=False")

    engine = CrateDigger(
        songs=tiny_catalog, index=tiny_index, generator=ExplodingGenerator(),
        logger=RunLogger(log_dir=tmp_path / "logs"),
    )
    result = engine.recommend("quiet music", k=2, floor=0.0, explain=False)
    assert result.status == "ok"
    assert all(pick.reason for pick in result.picks)


# ============================================================================
# Energy filtering
# ============================================================================

def test_energy_filter_narrows_results(engine):
    high = engine.recommend("music", k=4, target_energy=0.95, floor=0.0)
    assert all(pick.song.energy >= 0.5 for pick in high.picks)


def test_energy_filter_never_returns_nothing(engine):
    """A narrow filter must degrade to the closest song, not to silence."""
    result = engine.recommend("music", k=4, target_energy=0.5, floor=0.0)
    assert result.picks


# ============================================================================
# Serialisation and logging
# ============================================================================

def test_result_is_json_serialisable(engine):
    payload = engine.recommend("quiet music", k=2, floor=0.0).as_dict()
    json.dumps(payload)  # must not raise
    assert payload["status"] == "ok"
    assert "picks" in payload


def test_run_is_logged_as_jsonl(tiny_catalog, tiny_index, tmp_path):
    logger = RunLogger(log_dir=tmp_path / "logs")
    engine = CrateDigger(
        songs=tiny_catalog, index=tiny_index,
        generator=TemplateGenerator(), logger=logger,
    )
    engine.recommend("quiet music", k=2, floor=0.0)

    events = read_events(logger.path)
    kinds = {event["event"] for event in events}
    assert "engine_ready" in kinds
    assert "retrieval" in kinds
    assert "recommended" in kinds


def test_refusal_is_logged(tiny_catalog, tiny_index, tmp_path):
    logger = RunLogger(log_dir=tmp_path / "logs")
    engine = CrateDigger(
        songs=tiny_catalog, index=tiny_index,
        generator=TemplateGenerator(), logger=logger,
    )
    engine.recommend("quiet music", k=2, floor=1.01)
    assert "refused" in {event["event"] for event in read_events(logger.path)}


def test_log_records_retrieved_ids_for_auditing(tiny_catalog, tiny_index, tmp_path):
    logger = RunLogger(log_dir=tmp_path / "logs")
    engine = CrateDigger(
        songs=tiny_catalog, index=tiny_index,
        generator=TemplateGenerator(), logger=logger,
    )
    engine.recommend("quiet music", k=2, floor=0.0)
    retrieval = next(
        e for e in read_events(logger.path) if e["event"] == "retrieval"
    )
    assert retrieval["hits"], "retrieved songs must be recoverable from the log"
    assert "similarity" in retrieval["hits"][0]

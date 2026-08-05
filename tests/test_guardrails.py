"""
Guardrail tests.

These are the highest-value tests in the suite. The guardrails are what stop the
system making claims it cannot support, so each one gets an explicit test with a
named failure mode - including the two bugs the Module 3 adversarial pass found.
"""

from __future__ import annotations

import pytest

from src.guardrails import (
    check_citations,
    refusal_message,
    strip_unsupported,
    validate_request,
)
from src.recommender import Song


# ============================================================================
# Input validation
# ============================================================================

def test_valid_request_passes_and_normalises_whitespace():
    result = validate_request("  chill   lofi   beats  ", k=3)
    assert result.ok
    assert result.query == "chill lofi beats"   # collapsed, stripped
    assert result.k == 3
    assert result.errors == []


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t ", "a"])
def test_empty_or_too_short_query_is_rejected(bad_query):
    result = validate_request(bad_query)
    assert not result.ok
    assert result.reason  # a human-readable reason is always present


def test_none_query_is_rejected_without_raising():
    result = validate_request(None)
    assert not result.ok


def test_overlong_query_is_rejected():
    result = validate_request("x" * 5000)
    assert not result.ok
    assert "too long" in result.reason


def test_k_below_one_is_rejected():
    assert not validate_request("jazz", k=0).ok
    assert not validate_request("jazz", k=-5).ok


def test_k_above_cap_is_clamped_with_a_warning():
    result = validate_request("jazz", k=999)
    assert result.ok
    assert result.k == 20
    assert any("exceeds the cap" in w for w in result.warnings)


def test_non_numeric_k_is_rejected():
    assert not validate_request("jazz", k="lots").ok


# --- the two bugs the README's adversarial pass documented -------------------

def test_energy_above_range_is_clamped_not_passed_through():
    """
    README adversarial profile 2: target energy 2.0 drove scores negative
    because the scorer never validated its input domain.
    """
    result = validate_request("upbeat pop", target_energy=2.0)
    assert result.ok
    assert result.target_energy == 1.0
    assert any("clamped" in w for w in result.warnings)


def test_negative_energy_is_clamped_not_passed_through():
    """README adversarial profile 5: energy -1.0 was accepted silently."""
    result = validate_request("lofi chill", target_energy=-1.0)
    assert result.ok
    assert result.target_energy == 0.0
    assert any("clamped" in w for w in result.warnings)


def test_energy_inside_range_is_untouched_and_unwarned():
    result = validate_request("lofi chill", target_energy=0.4)
    assert result.target_energy == pytest.approx(0.4)
    assert result.warnings == []


def test_nan_energy_is_rejected():
    result = validate_request("lofi", target_energy=float("nan"))
    assert not result.ok


def test_non_numeric_energy_is_rejected():
    assert not validate_request("lofi", target_energy="loud").ok


# ============================================================================
# Citation / groundedness checking
# ============================================================================

def _song(song_id: int, title: str) -> Song:
    return Song(
        id=song_id, title=title, artist="A", genre="pop", mood="happy",
        energy=0.5, tempo_bpm=100, valence=0.5, danceability=0.5,
        acousticness=0.5, description="d",
    )


ALL_SONGS = [_song(1, "Alpha Track"), _song(2, "Beta Track"), _song(3, "Gamma Track")]
RETRIEVED = [ALL_SONGS[0], ALL_SONGS[1]]


def test_prose_about_retrieved_songs_is_grounded():
    report = check_citations(
        "Alpha Track is a good fit because it is upbeat.", RETRIEVED, ALL_SONGS
    )
    assert report.grounded
    assert "Alpha Track" in report.cited
    assert report.out_of_context == []
    assert report.fabricated == []


def test_naming_a_real_song_that_was_not_retrieved_is_a_failure():
    """
    The model pulled a real catalog song from memory instead of from the
    supplied context. The song exists, but it was not evidence for this query,
    so it is still a grounding failure.
    """
    report = check_citations(
        "You should try Gamma Track instead.", RETRIEVED, ALL_SONGS
    )
    assert not report.grounded
    assert report.out_of_context == ["Gamma Track"]


def test_quoted_invented_title_is_flagged_as_fabricated():
    report = check_citations(
        'I recommend "Total Invention" for this mood.', RETRIEVED, ALL_SONGS
    )
    assert not report.grounded
    assert "Total Invention" in report.fabricated


def test_title_by_artist_pattern_catches_fabrication():
    report = check_citations(
        "Midnight Hologram by Fakeband suits this request.", RETRIEVED, ALL_SONGS
    )
    assert not report.grounded
    assert report.fabricated


def test_empty_text_is_vacuously_grounded():
    assert check_citations("", RETRIEVED, ALL_SONGS).grounded
    assert check_citations("   ", RETRIEVED, ALL_SONGS).grounded


def test_failures_are_described_in_human_readable_terms():
    report = check_citations("Try Gamma Track.", RETRIEVED, ALL_SONGS)
    assert report.failures
    assert "Gamma Track" in report.failures[0]


# ============================================================================
# Stripping unsupported claims
# ============================================================================

def test_strip_removes_only_the_offending_sentence():
    text = "Alpha Track is upbeat. Gamma Track is also good."
    report = check_citations(text, RETRIEVED, ALL_SONGS)
    cleaned = strip_unsupported(text, report)
    assert "Alpha Track is upbeat." in cleaned
    assert "Gamma Track" not in cleaned


def test_strip_is_a_no_op_when_already_grounded():
    text = "Alpha Track is upbeat."
    report = check_citations(text, RETRIEVED, ALL_SONGS)
    assert strip_unsupported(text, report) == text


def test_strip_can_return_empty_when_everything_is_unsupported():
    text = "Gamma Track is the one."
    report = check_citations(text, RETRIEVED, ALL_SONGS)
    assert strip_unsupported(text, report) == ""


# ============================================================================
# Refusal message
# ============================================================================

def test_refusal_names_the_numbers_and_suggests_nothing():
    message = refusal_message("polka metal opera", 0.11, 0.20)
    assert "polka metal opera" in message
    assert "0.11" in message
    assert "0.20" in message
    # A refusal must not sneak in a recommendation.
    assert "recommend" not in message.lower().replace("recommendation", "")

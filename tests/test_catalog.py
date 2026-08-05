"""
Catalog loading and embedding-text construction.

The catalog is the system's only source of truth, so these tests check both that
a good file loads correctly and that a bad file fails loudly rather than being
silently half-loaded.
"""

from __future__ import annotations

import pytest

from src.catalog import (
    CatalogError,
    REQUIRED_COLUMNS,
    catalog_titles,
    load_catalog,
    searchable_text,
)

HEADER = (
    "id,title,artist,genre,mood,energy,tempo_bpm,valence,"
    "danceability,acousticness,description"
)
GOOD_ROW = "1,T,A,pop,happy,0.5,100,0.5,0.5,0.5,A description here"


def _write(tmp_path, text):
    path = tmp_path / "songs.csv"
    path.write_text(text, encoding="utf-8")
    return path


# ============================================================================
# Happy path against the real catalog
# ============================================================================

def test_real_catalog_loads(catalog):
    assert len(catalog) >= 70, "catalog should hold the expanded song set"


def test_real_catalog_ids_are_unique(catalog):
    ids = [song.id for song in catalog]
    assert len(set(ids)) == len(ids)


def test_real_catalog_every_song_has_a_description(catalog):
    assert all(song.description.strip() for song in catalog)


def test_real_catalog_energy_is_always_in_range(catalog):
    assert all(0.0 <= song.energy <= 1.0 for song in catalog)


def test_real_catalog_numerics_are_parsed_not_strings(catalog):
    song = catalog[0]
    assert isinstance(song.energy, float)
    assert isinstance(song.tempo_bpm, int)


def test_real_catalog_contains_near_genre_variants(catalog):
    """
    The near-genre entries are what demonstrate semantic retrieval beating the
    original exact-match rule. If they disappear, the RAG demonstration is gone
    and the golden set's near-genre cases become meaningless.
    """
    genres = {song.genre for song in catalog}
    assert "pop" in genres
    assert any(g != "pop" and "pop" in g for g in genres), (
        "expected at least one near-genre such as 'indie pop' or 'dream pop'"
    )


# ============================================================================
# Failure modes
# ============================================================================

def test_missing_file_raises_catalog_error(tmp_path):
    with pytest.raises(CatalogError, match="not found"):
        load_catalog(tmp_path / "nope.csv")


def test_missing_column_raises_and_names_the_column(tmp_path):
    path = _write(tmp_path, "id,title\n1,T\n")
    with pytest.raises(CatalogError, match="missing required column"):
        load_catalog(path)


def test_header_only_file_raises(tmp_path):
    path = _write(tmp_path, HEADER + "\n")
    with pytest.raises(CatalogError, match="no rows"):
        load_catalog(path)


def test_non_numeric_energy_raises_with_line_number(tmp_path):
    bad = "2,T2,A,pop,happy,NOT_A_NUMBER,100,0.5,0.5,0.5,desc"
    path = _write(tmp_path, f"{HEADER}\n{GOOD_ROW}\n{bad}\n")
    with pytest.raises(CatalogError, match="line 3"):
        load_catalog(path)


def test_duplicate_ids_raise(tmp_path):
    path = _write(tmp_path, f"{HEADER}\n{GOOD_ROW}\n{GOOD_ROW}\n")
    with pytest.raises(CatalogError, match="Duplicate song id"):
        load_catalog(path)


def test_required_columns_includes_description():
    assert "description" in REQUIRED_COLUMNS


# ============================================================================
# Searchable text
# ============================================================================

def test_searchable_text_includes_every_retrievable_field(tiny_catalog):
    song = tiny_catalog[1]  # Quiet Study
    text = searchable_text(song)
    for expected in (song.title, song.artist, song.genre, song.mood, song.description):
        assert expected in text


def test_searchable_text_converts_energy_to_words(tiny_catalog):
    """
    A sentence embedder gets nothing from the literal string "0.95", so energy
    is bucketed into language it can actually use.
    """
    loud = searchable_text(tiny_catalog[0])   # energy 0.95
    quiet = searchable_text(tiny_catalog[1])  # energy 0.20
    assert "Very high energy" in loud
    assert "Low energy" in quiet or "Very low energy" in quiet


def test_searchable_text_converts_tempo_to_words(tiny_catalog):
    assert "very fast tempo" in searchable_text(tiny_catalog[0])  # 160 BPM
    assert "very slow tempo" in searchable_text(tiny_catalog[3])  # 60 BPM


def test_catalog_titles_are_lowercased(tiny_catalog):
    titles = catalog_titles(tiny_catalog)
    assert "loud anthem" in titles
    assert "Loud Anthem" not in titles

"""
Retrieval tests.

Semantic quality is not tested here - that needs the real embedding model and is
measured by src/evals/run_eval.py. What is tested is the machinery around it:
ranking order, the similarity floor, and above all cache invalidation, which is
the failure mode most likely to cause a confusing bug in normal use.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recommender import Song
from src.retrieval import HashingEmbedder, Hit, RetrievalIndex, build_index


# ============================================================================
# Construction
# ============================================================================

def test_empty_catalog_is_rejected():
    with pytest.raises(ValueError, match="empty catalog"):
        RetrievalIndex([], HashingEmbedder())


def test_build_produces_one_normalised_vector_per_song(tiny_catalog):
    index = RetrievalIndex(tiny_catalog, HashingEmbedder()).build()
    assert index.vectors.shape[0] == len(tiny_catalog)
    norms = np.linalg.norm(index.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "vectors must be L2-normalised"


def test_search_before_build_raises(tiny_catalog):
    index = RetrievalIndex(tiny_catalog, HashingEmbedder())
    with pytest.raises(RuntimeError, match="not built"):
        index.search("anything")


# ============================================================================
# Search behaviour
# ============================================================================

def test_search_returns_k_hits_sorted_descending(tiny_index):
    hits = tiny_index.search("quiet music", k=3)
    assert len(hits) == 3
    sims = [hit.similarity for hit in hits]
    assert sims == sorted(sims, reverse=True)


def test_k_larger_than_catalog_returns_whole_catalog(tiny_index, tiny_catalog):
    hits = tiny_index.search("music", k=999)
    assert len(hits) == len(tiny_catalog)


def test_k_of_zero_is_coerced_to_one(tiny_index):
    assert len(tiny_index.search("music", k=0)) == 1


def test_search_is_deterministic(tiny_index):
    """Reproducibility is a graded requirement; identical input, identical order."""
    first = [h.song.id for h in tiny_index.search("quiet study music", k=4)]
    second = [h.song.id for h in tiny_index.search("quiet study music", k=4)]
    assert first == second


def test_hit_as_dict_is_log_safe(tiny_index):
    payload = tiny_index.search("music", k=1)[0].as_dict()
    assert set(payload) == {"id", "title", "artist", "genre", "similarity"}


# ============================================================================
# Similarity floor - the refusal mechanism
# ============================================================================

def test_floor_of_zero_keeps_everything(tiny_index):
    hits, best = tiny_index.search_with_floor("music", k=4, floor=0.0)
    assert len(hits) == 4
    assert best >= 0.0


def test_impossible_floor_returns_nothing_but_still_reports_best(tiny_index):
    """
    The empty list is what triggers a refusal, and best_similarity is what the
    refusal message quotes, so both halves of the return value matter.
    """
    hits, best = tiny_index.search_with_floor("music", k=4, floor=1.01)
    assert hits == []
    assert best > 0.0


def test_floor_filters_only_below_threshold(tiny_index):
    all_hits, _ = tiny_index.search_with_floor("quiet piano", k=4, floor=0.0)
    cutoff = all_hits[1].similarity
    kept, _ = tiny_index.search_with_floor("quiet piano", k=4, floor=cutoff)
    assert all(hit.similarity >= cutoff for hit in kept)
    assert len(kept) <= len(all_hits)


# ============================================================================
# Cache invalidation
# ============================================================================
# This is the section that earns its keep. Without a fingerprint check, adding a
# song to songs.csv would leave a stale index in place and the new song would be
# permanently unfindable - a bug that presents as bad retrieval, not bad caching.

def test_save_then_load_round_trips(tiny_catalog, tmp_path):
    path = tmp_path / "index.npz"
    original = RetrievalIndex(tiny_catalog, HashingEmbedder()).build()
    original.save(path)

    reloaded = RetrievalIndex(tiny_catalog, HashingEmbedder())
    assert reloaded.try_load(path)
    assert np.allclose(reloaded.vectors, original.vectors)


def test_save_before_build_raises(tiny_catalog, tmp_path):
    index = RetrievalIndex(tiny_catalog, HashingEmbedder())
    with pytest.raises(RuntimeError, match="build"):
        index.save(tmp_path / "index.npz")


def test_missing_cache_file_returns_false_not_an_error(tiny_catalog, tmp_path):
    index = RetrievalIndex(tiny_catalog, HashingEmbedder())
    assert index.try_load(tmp_path / "absent.npz") is False


def test_adding_a_song_invalidates_the_cache(tiny_catalog, tmp_path):
    path = tmp_path / "index.npz"
    RetrievalIndex(tiny_catalog, HashingEmbedder()).build().save(path)

    extended = list(tiny_catalog) + [
        Song(
            id=99, title="New Arrival", artist="Testers", genre="jazz",
            mood="relaxed", energy=0.4, tempo_bpm=90, valence=0.6,
            danceability=0.5, acousticness=0.7,
            description="A brand new song added after the index was built.",
        )
    ]
    assert RetrievalIndex(extended, HashingEmbedder()).try_load(path) is False


def test_editing_a_description_invalidates_the_cache(tiny_catalog, tmp_path):
    path = tmp_path / "index.npz"
    RetrievalIndex(tiny_catalog, HashingEmbedder()).build().save(path)

    edited = [
        Song(**{**song.__dict__, "description": "completely rewritten text"})
        if song.id == 1 else song
        for song in tiny_catalog
    ]
    assert RetrievalIndex(edited, HashingEmbedder()).try_load(path) is False


def test_changing_embedder_invalidates_the_cache(tiny_catalog, tmp_path):
    """Vectors from one model are meaningless to another."""
    path = tmp_path / "index.npz"
    RetrievalIndex(tiny_catalog, HashingEmbedder(dims=256)).build().save(path)
    assert RetrievalIndex(tiny_catalog, HashingEmbedder(dims=128)).try_load(path) is False


def test_corrupt_cache_returns_false_rather_than_crashing(tiny_catalog, tmp_path):
    path = tmp_path / "index.npz"
    path.write_bytes(b"this is not a valid npz archive")
    index = RetrievalIndex(tiny_catalog, HashingEmbedder())
    assert index.try_load(path) is False


def test_ensure_ready_builds_when_no_cache_exists(tiny_catalog, tmp_path):
    index = build_index(tiny_catalog, HashingEmbedder(), tmp_path / "index.npz")
    assert index.vectors is not None
    assert (tmp_path / "index.npz").exists()


def test_ensure_ready_is_idempotent(tiny_catalog, tmp_path):
    path = tmp_path / "index.npz"
    index = build_index(tiny_catalog, HashingEmbedder(), path)
    vectors = index.vectors
    index.ensure_ready(path)
    assert index.vectors is vectors  # no rebuild, same object

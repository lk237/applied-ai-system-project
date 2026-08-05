"""
Shared fixtures.

Every fixture here is offline. The suite must pass on a fresh clone with no
model downloads and no network, because a test suite that requires a 1.2 GB
download before it can tell you anything is a test suite people skip.

The real models are exercised by src/evals/run_eval.py instead, which is where
model quality belongs. These tests verify plumbing, guardrails, and contracts.
"""

from __future__ import annotations

import pytest

from src.catalog import load_catalog
from src.generate import TemplateGenerator
from src.obs import RunLogger
from src.pipeline import CrateDigger
from src.recommender import Song
from src.retrieval import HashingEmbedder, RetrievalIndex


@pytest.fixture(scope="session")
def catalog() -> list[Song]:
    """The real catalog from data/songs.csv."""
    return load_catalog()


@pytest.fixture()
def tiny_catalog() -> list[Song]:
    """
    Four hand-built songs with known properties.

    Deliberately small so assertions can name exact expected outcomes rather
    than asserting something vague about a 72-row file.
    """
    return [
        Song(
            id=1, title="Loud Anthem", artist="Testers", genre="rock",
            mood="intense", energy=0.95, tempo_bpm=160, valence=0.5,
            danceability=0.6, acousticness=0.1,
            description="Very loud guitar music for a workout.",
        ),
        Song(
            id=2, title="Quiet Study", artist="Testers", genre="lofi",
            mood="chill", energy=0.20, tempo_bpm=70, valence=0.6,
            danceability=0.4, acousticness=0.9,
            description="Soft quiet beats for studying and concentration.",
        ),
        Song(
            id=3, title="Middle Ground", artist="Testers", genre="indie pop",
            mood="happy", energy=0.55, tempo_bpm=110, valence=0.8,
            danceability=0.7, acousticness=0.4,
            description="Cheerful mid tempo guitar pop for a sunny afternoon.",
        ),
        Song(
            id=4, title="Sad Piano", artist="Testers", genre="classical",
            mood="melancholic", energy=0.15, tempo_bpm=60, valence=0.2,
            danceability=0.1, acousticness=0.98,
            description="Slow mournful solo piano in an empty room.",
        ),
    ]


@pytest.fixture()
def tiny_index(tiny_catalog) -> RetrievalIndex:
    """Built index over the tiny catalog, using the offline embedder."""
    return RetrievalIndex(tiny_catalog, HashingEmbedder()).build()


@pytest.fixture()
def engine(tiny_catalog, tiny_index, tmp_path) -> CrateDigger:
    """
    A fully wired engine with no model dependencies.

    Logs go to tmp_path so tests never append to the project's real logs.
    """
    return CrateDigger(
        songs=tiny_catalog,
        index=tiny_index,
        generator=TemplateGenerator(),
        logger=RunLogger(log_dir=tmp_path / "logs"),
    )

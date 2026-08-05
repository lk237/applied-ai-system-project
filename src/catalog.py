"""
Catalog loading and the text that gets embedded.

This module owns one job: turn data/songs.csv into Song objects, and decide what
text represents a song to the retriever. That second decision matters more than
it looks — the embedded string is the entire basis on which a song can be found,
so anything left out of it is effectively invisible to search.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.config import CATALOG_PATH
from src.recommender import Song

# Numeric columns, so downstream maths does not silently operate on strings.
_INT_FIELDS = {"id", "tempo_bpm"}
_FLOAT_FIELDS = {"energy", "valence", "danceability", "acousticness"}

REQUIRED_COLUMNS = {
    "id", "title", "artist", "genre", "mood",
    "energy", "tempo_bpm", "valence", "danceability", "acousticness",
    "description",
}


class CatalogError(RuntimeError):
    """Raised when the catalog file is missing or structurally wrong."""


def load_catalog(path: Path | str | None = None) -> List[Song]:
    """
    Read the catalog CSV into Song objects.

    Fails loudly on a malformed catalog rather than limping along with partial
    data — a recommender silently running on half a catalog is worse than one
    that refuses to start.
    """
    csv_path = Path(path) if path is not None else CATALOG_PATH

    if not csv_path.exists():
        raise CatalogError(
            f"Catalog not found at {csv_path}. "
            "Expected data/songs.csv relative to the project root."
        )

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        header = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise CatalogError(
                f"Catalog {csv_path} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

        songs: List[Song] = []
        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            try:
                values = {
                    key: (
                        int(value) if key in _INT_FIELDS
                        else float(value) if key in _FLOAT_FIELDS
                        else value.strip()
                    )
                    for key, value in row.items()
                    if key in REQUIRED_COLUMNS
                }
            except (TypeError, ValueError) as exc:
                raise CatalogError(
                    f"Bad numeric value in {csv_path} on line {line_no}: {exc}"
                ) from exc

            songs.append(Song(**values))

    if not songs:
        raise CatalogError(f"Catalog {csv_path} contains a header but no rows.")

    ids = [song.id for song in songs]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise CatalogError(f"Duplicate song id(s) in {csv_path}: {duplicates}")

    return songs


def searchable_text(song: Song) -> str:
    """
    The string handed to the embedding model for this song.

    Structured fields are folded into the same sentence as the prose blurb so a
    query like "high energy pop for the gym" can match on genre, mood, and the
    description at once. Energy is bucketed into a word because "0.93" carries
    no semantic weight to a sentence embedder, whereas "high energy" does.
    """
    return (
        f"{song.title} by {song.artist}. "
        f"Genre: {song.genre}. Mood: {song.mood}. "
        f"{_energy_word(song.energy)} energy, {_tempo_word(song.tempo_bpm)} tempo. "
        f"{song.description}"
    )


def _energy_word(energy: float) -> str:
    if energy >= 0.80:
        return "Very high"
    if energy >= 0.60:
        return "High"
    if energy >= 0.40:
        return "Moderate"
    if energy >= 0.25:
        return "Low"
    return "Very low"


def _tempo_word(bpm: int) -> str:
    if bpm >= 150:
        return "very fast"
    if bpm >= 120:
        return "fast"
    if bpm >= 95:
        return "medium"
    if bpm >= 75:
        return "slow"
    return "very slow"


def catalog_titles(songs: List[Song]) -> set[str]:
    """Lowercased titles, used by the citation guardrail to detect inventions."""
    return {song.title.lower() for song in songs}

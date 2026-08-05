"""
The retriever - the "R" in RAG.

Embeds every song's searchable text once, caches the vectors to disk, and
answers queries by cosine similarity. This is what replaces the original
recommender's exact-match categorical test: "indie pop" and "pop" land near each
other in embedding space, so a request for one can surface the other.

Two design points worth calling out:

  * The embedding model is loaded lazily. Importing this module costs nothing,
    so the test suite and the input guardrails can run with no model download.
    Tests inject a fake embedder through the same interface.

  * The cached index stores a fingerprint of the catalog text and the model
    name. If either changes the index rebuilds itself. Without this, adding a
    song to songs.csv would silently leave a stale index in place and the new
    song would be permanently unfindable - a bug that looks like bad retrieval
    rather than bad caching.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

from src.catalog import searchable_text
from src.config import EMBED_MODEL, INDEX_PATH, SIMILARITY_FLOOR, TOP_K
from src.recommender import Song


# ============================================================================
# Embedder interface
# ============================================================================

class Embedder(Protocol):
    """Minimal contract so tests can substitute a deterministic stand-in."""

    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """
    Wraps sentence-transformers. The model is fetched on first use and then
    reused, so a process that never retrieves never pays for the load.
    """

    def __init__(self, model_name: str = EMBED_MODEL):
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment issue
                raise RuntimeError(
                    "sentence-transformers is not installed. Run:\n"
                    "    pip install -r requirements.txt"
                ) from exc
            self._model = SentenceTransformer(self.name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,  # so a dot product IS cosine similarity
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """
    Dependency-free deterministic embedder used by the offline test suite.

    Not semantically meaningful - it hashes character trigrams into a fixed
    vector space. It exists so guardrail and plumbing tests can exercise the
    real retrieval code path without a 90 MB model download. It is never used
    by the application or the eval harness.
    """

    def __init__(self, dims: int = 256):
        self.name = f"hashing-{dims}d"
        self.dims = dims

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = text.lower()
            for i in range(len(lowered) - 2):
                trigram = lowered[i : i + 3]
                bucket = int(hashlib.md5(trigram.encode()).hexdigest(), 16) % self.dims
                out[row, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ============================================================================
# Results
# ============================================================================

@dataclass
class Hit:
    song: Song
    similarity: float

    def as_dict(self) -> dict:
        return {
            "id": self.song.id,
            "title": self.song.title,
            "artist": self.song.artist,
            "genre": self.song.genre,
            "similarity": round(self.similarity, 4),
        }


# ============================================================================
# Index
# ============================================================================

def _fingerprint(texts: Sequence[str], model_name: str) -> str:
    """Content hash of the corpus plus model, used to detect a stale cache."""
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for text in texts:
        digest.update(b"\x00")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:32]


class RetrievalIndex:
    """Embedded catalog with cosine search."""

    def __init__(self, songs: Sequence[Song], embedder: Optional[Embedder] = None):
        if not songs:
            raise ValueError("Cannot build a retrieval index over an empty catalog.")
        self.songs: List[Song] = list(songs)
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder()
        self.texts: List[str] = [searchable_text(song) for song in self.songs]
        self.fingerprint: str = _fingerprint(self.texts, self.embedder.name)
        self.vectors: Optional[np.ndarray] = None

    # -- build / cache ------------------------------------------------------

    def build(self) -> "RetrievalIndex":
        self.vectors = self.embedder.encode(self.texts)
        return self

    def save(self, path: Optional[Path] = None) -> Path:
        if self.vectors is None:
            raise RuntimeError("build() must run before save().")
        target = Path(path) if path else INDEX_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            vectors=self.vectors,
            fingerprint=np.array(self.fingerprint),
            model=np.array(self.embedder.name),
            ids=np.array([song.id for song in self.songs]),
        )
        return target

    def try_load(self, path: Optional[Path] = None) -> bool:
        """
        Load cached vectors if they match this catalog and model.

        Returns False - rather than raising - on any mismatch or corruption, so
        the caller simply rebuilds. A cache is an optimisation; it should never
        be able to break the system.
        """
        source = Path(path) if path else INDEX_PATH
        if not source.exists():
            return False
        try:
            with np.load(source, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self.fingerprint:
                    return False
                vectors = data["vectors"]
                if vectors.shape[0] != len(self.songs):
                    return False
                self.vectors = np.asarray(vectors, dtype=np.float32)
                return True
        except (OSError, KeyError, ValueError):
            return False

    def ensure_ready(self, path: Optional[Path] = None) -> "RetrievalIndex":
        """Load from cache, or build and cache. The normal entry point."""
        if self.vectors is not None:
            return self
        if self.try_load(path):
            return self
        self.build()
        try:
            self.save(path)
        except OSError:
            pass  # read-only filesystem: continue with in-memory vectors
        return self

    # -- search ------------------------------------------------------------

    def search(self, query: str, k: int = TOP_K) -> List[Hit]:
        """Top-k by cosine similarity, highest first."""
        if self.vectors is None:
            raise RuntimeError("Index is not built. Call ensure_ready() first.")

        query_vector = self.embedder.encode([query])[0]
        scores = self.vectors @ query_vector  # both sides are L2-normalised

        k = max(1, min(int(k), len(self.songs)))
        # argpartition finds the top k without fully sorting the catalog, then
        # only those k are sorted. Ties break on lower id for reproducibility.
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = sorted(top, key=lambda i: (-float(scores[i]), self.songs[i].id))

        return [Hit(song=self.songs[i], similarity=float(scores[i])) for i in ordered]

    def search_with_floor(
        self,
        query: str,
        k: int = TOP_K,
        floor: float = SIMILARITY_FLOOR,
    ) -> Tuple[List[Hit], float]:
        """
        Search, then drop anything below the similarity floor.

        Returns (kept_hits, best_similarity). An empty list means the catalog
        has nothing defensible to offer, and the caller must refuse rather than
        recommend the least-bad option - which is what the original scorer did,
        and why a request for "kpop" returned five unrelated mid-energy tracks.
        """
        hits = self.search(query, k)
        best = hits[0].similarity if hits else 0.0
        return [hit for hit in hits if hit.similarity >= floor], best


def build_index(
    songs: Sequence[Song],
    embedder: Optional[Embedder] = None,
    path: Optional[Path] = None,
) -> RetrievalIndex:
    """Convenience constructor: create, load-or-build, return."""
    return RetrievalIndex(songs, embedder).ensure_ready(path)

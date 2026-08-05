"""
Orchestration: the path a request actually takes.

    validate -> retrieve -> floor check -> generate -> citation check
             -> confidence -> log -> return

This module is the single place that sequence is written down, so the CLI, the
Streamlit app, and the eval harness all exercise the identical code path. That
matters for the reliability claims: the numbers in the scorecard describe the
same pipeline a user hits, not a test-only approximation of it.

Every exit is a Recommendation object. Nothing here raises for ordinary bad
input - refusals are values, not exceptions - so the callers can render one
shape regardless of outcome.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.catalog import load_catalog
from src.confidence import Confidence, EvalSummary, active_strategy, assess, load_eval_summary
from src.config import SIMILARITY_FLOOR, TOP_K
from src.generate import (
    TextGenerator,
    default_generator,
    generate_reasons,
    generate_summary,
)
from src.guardrails import (
    CitationReport,
    check_citations,
    refusal_message,
    strip_unsupported,
    validate_request,
)
from src.obs import RunLogger
from src.recommender import Song
from src.retrieval import Hit, RetrievalIndex, build_index


# ============================================================================
# Result types
# ============================================================================

@dataclass
class Pick:
    """One recommended song plus the prose justifying it."""
    song: Song
    similarity: float
    reason: str
    grounded: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.song.id,
            "title": self.song.title,
            "artist": self.song.artist,
            "genre": self.song.genre,
            "mood": self.song.mood,
            "energy": self.song.energy,
            "similarity": round(self.similarity, 4),
            "reason": self.reason,
            "grounded": self.grounded,
        }


@dataclass
class Recommendation:
    """
    The single return shape for every outcome.

    status is one of:
      ok       - picks were produced
      rejected - input failed validation; no model ran
      refused  - retrieval found nothing above the similarity floor
    """
    status: str
    query: str
    picks: List[Pick] = field(default_factory=list)
    summary: str = ""
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    confidence: Optional[Confidence] = None
    strategy: str = ""
    best_similarity: float = 0.0
    elapsed_ms: float = 0.0
    guardrail_notes: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "query": self.query,
            "summary": self.summary,
            "message": self.message,
            "warnings": self.warnings,
            "strategy": self.strategy,
            "best_similarity": round(self.best_similarity, 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
            "confidence": self.confidence.as_dict() if self.confidence else None,
            "guardrail_notes": self.guardrail_notes,
            "picks": [pick.as_dict() for pick in self.picks],
        }


# ============================================================================
# Engine
# ============================================================================

class CrateDigger:
    """
    Holds the loaded catalog, index, generator, and eval record.

    Construct once and reuse: the models are the expensive part, and Streamlit
    in particular re-runs its script on every interaction, so this object is
    cached there rather than rebuilt.
    """

    def __init__(
        self,
        songs: Optional[Sequence[Song]] = None,
        index: Optional[RetrievalIndex] = None,
        generator: Optional[TextGenerator] = None,
        logger: Optional[RunLogger] = None,
        eval_summary: Optional[EvalSummary] = None,
        offline: bool = False,
    ):
        self.log = logger or RunLogger()
        self.songs: List[Song] = list(songs) if songs else load_catalog()
        self.index = index or build_index(self.songs)
        self.generator = generator or default_generator(offline=offline)
        self.eval_summary = eval_summary or load_eval_summary()
        self.strategy = active_strategy(self.eval_summary)

        self.log.event(
            "engine_ready",
            catalog_size=len(self.songs),
            embedder=self.index.embedder.name,
            generator=self.generator.name,
            strategy=self.strategy,
            eval_on_record=self.eval_summary.exists,
        )

    # ------------------------------------------------------------------ main

    def recommend(
        self,
        query: str,
        k: int = TOP_K,
        target_energy: Optional[float] = None,
        floor: float = SIMILARITY_FLOOR,
        explain: bool = True,
        strategy: Optional[str] = None,
    ) -> Recommendation:
        """
        Run one request end to end.

        explain=False skips generation and uses deterministic reasons built from
        catalog fields. The eval harness uses this for paraphrase runs, where
        only the returned song ids matter - generating prose that is never scored
        would double the harness runtime for no information gain.

        strategy overrides the eval-selected prompt strategy. Used by the
        harness to score each strategy in turn; normal callers leave it None so
        the winner recorded in evals/results.json is honoured.
        """
        started = time.perf_counter()
        strategy = strategy or self.strategy

        # --- stage 2: input guardrail ------------------------------------
        request = validate_request(query, k=k, target_energy=target_energy)
        if not request.ok:
            self.log.event(
                "rejected",
                query=str(query)[:200],
                errors=request.errors,
            )
            return Recommendation(
                status="rejected",
                query=str(query),
                message=request.reason,
                warnings=request.warnings,
                strategy=strategy,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        # --- stage 3: retrieval ------------------------------------------
        with self.log.stage("retrieval", query=request.query, k=request.k) as st:
            hits, best = self.index.search_with_floor(
                request.query, k=request.k, floor=floor
            )
            st["hits"] = [hit.as_dict() for hit in hits]
            st["best_similarity"] = round(best, 4)

        # Optional structured filter, applied after semantic retrieval so it
        # narrows a relevant set rather than defining it.
        if request.target_energy is not None and hits:
            hits = self._filter_by_energy(hits, request.target_energy)

        # --- refusal path -------------------------------------------------
        if not hits:
            self.log.event(
                "refused",
                query=request.query,
                best_similarity=round(best, 4),
                floor=floor,
            )
            return Recommendation(
                status="refused",
                query=request.query,
                message=refusal_message(request.query, best, floor),
                warnings=request.warnings,
                confidence=assess(best, self.eval_summary),
                strategy=strategy,
                best_similarity=best,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        # --- stage 4: generation -----------------------------------------
        if explain:
            with self.log.stage("generation", n=len(hits), strategy=strategy) as st:
                reasons = generate_reasons(
                    request.query, hits, self.generator, strategy
                )
                summary = generate_summary(
                    request.query, hits, self.generator, strategy
                )
                st["chars"] = sum(len(r) for r in reasons) + len(summary)
        else:
            # Retrieval-only mode: deterministic reasons, no model call.
            reasons = [self._fallback_reason(hit) for hit in hits]
            summary = ""

        # --- stage 5: output guardrail -----------------------------------
        picks: List[Pick] = []
        notes: List[str] = []
        # The guardrail compares against Song objects, not Hits.
        retrieved_songs = [hit.song for hit in hits]

        for hit, reason in zip(hits, reasons):
            report = check_citations(reason, retrieved_songs, self.songs)
            text = reason
            if not report.grounded:
                text = strip_unsupported(reason, report)
                notes.extend(f"{hit.song.title}: {f}" for f in report.failures)

            # Fall back to catalog facts whenever there is no usable text -
            # whether the guardrail stripped everything, or generation failed
            # and returned nothing at all. An empty string is vacuously
            # "grounded", so this must be checked separately from the report.
            if not text.strip():
                text = self._fallback_reason(hit)
            picks.append(
                Pick(
                    song=hit.song,
                    similarity=hit.similarity,
                    reason=text,
                    grounded=report.grounded,
                )
            )

        summary_report = check_citations(summary, retrieved_songs, self.songs)
        if not summary_report.grounded:
            notes.extend(f"summary: {f}" for f in summary_report.failures)
            summary = strip_unsupported(summary, summary_report)

        if notes:
            self.log.event("guardrail_citation_failures", notes=notes)

        elapsed = (time.perf_counter() - started) * 1000
        result = Recommendation(
            status="ok",
            query=request.query,
            picks=picks,
            summary=summary,
            warnings=request.warnings,
            confidence=assess(best, self.eval_summary),
            strategy=strategy,
            best_similarity=best,
            elapsed_ms=elapsed,
            guardrail_notes=notes,
        )

        self.log.event(
            "recommended",
            query=request.query,
            n=len(picks),
            grounded=all(p.grounded for p in picks),
            best_similarity=round(best, 4),
            confidence=result.confidence.band if result.confidence else None,
            ms=round(elapsed, 1),
        )
        return result

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _filter_by_energy(
        hits: Sequence[Hit], target: float, tolerance: float = 0.25
    ) -> List[Hit]:
        """
        Keep hits near the requested energy, but never return nothing.

        If the filter would empty the list, the closest single hit is kept
        instead. Silently returning zero results for a valid request would look
        like a system fault rather than a narrow filter.
        """
        near = [h for h in hits if abs(h.song.energy - target) <= tolerance]
        if near:
            return near
        return [min(hits, key=lambda h: abs(h.song.energy - target))]

    @staticmethod
    def _fallback_reason(hit: Hit) -> str:
        """Deterministic reason built only from catalog fields."""
        song = hit.song
        return (
            f"{song.genre} track with a {song.mood} feel, "
            f"energy {song.energy:.2f} at {song.tempo_bpm} BPM."
        )

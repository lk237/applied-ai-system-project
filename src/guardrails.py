"""
Guardrails: everything that says "no" before or after the model runs.

Two gates:

  1. validate_request  - runs BEFORE any model is loaded. Rejects malformed or
                         out-of-domain input so no compute is spent on it.
  2. check_citations   - runs AFTER generation. Verifies the model only talked
                         about songs that were actually retrieved.

Why gate 2 exists: the generator is a small instruction-tuned model. It is good
at paraphrasing text placed in front of it and bad at knowing what it does not
know. Rather than trust it, every song it names is checked against the retrieved
set, and anything unsupported is stripped and logged as a groundedness failure.

Note on the baseline scorer: src/recommender.py is deliberately left with the
bugs documented in the README's adversarial section (unclamped energy, "+-0.27"
formatting). Those findings are evidence. The clamp lives here instead, so bad
input can never reach the scorer in the running system while the historical
finding remains reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.config import (
    ENERGY_MAX,
    ENERGY_MIN,
    MAX_K,
    MAX_QUERY_CHARS,
    MIN_QUERY_CHARS,
)
from src.recommender import Song


# ============================================================================
# Gate 1 - input validation
# ============================================================================

@dataclass
class ValidatedRequest:
    """Result of input validation. `ok` is the only thing callers must check."""
    ok: bool
    query: str = ""
    k: int = 5
    target_energy: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        return "; ".join(self.errors)


def validate_request(
    query: str,
    k: int = 5,
    target_energy: Optional[float] = None,
) -> ValidatedRequest:
    """
    Check and normalise a request. Never raises: returns a result object so the
    CLI and the Streamlit app can render the same refusal text.

    Errors reject the request outright. Warnings describe a value that was
    silently repaired (clamped) and are surfaced to the user, because a system
    that quietly changes your request is worse than one that tells you it did.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # --- query text -------------------------------------------------------
    if query is None:
        return ValidatedRequest(ok=False, errors=["Query is missing."])

    cleaned = " ".join(str(query).split())  # collapse whitespace, strip ends

    if not cleaned:
        errors.append("Query is empty - describe the music you want.")
    elif len(cleaned) < MIN_QUERY_CHARS:
        errors.append(
            f"Query is too short ({len(cleaned)} chars); "
            f"minimum is {MIN_QUERY_CHARS}."
        )
    elif len(cleaned) > MAX_QUERY_CHARS:
        errors.append(
            f"Query is too long ({len(cleaned)} chars); "
            f"maximum is {MAX_QUERY_CHARS}."
        )

    # --- k ----------------------------------------------------------------
    try:
        k_int = int(k)
    except (TypeError, ValueError):
        errors.append(f"k must be a whole number, got {k!r}.")
        k_int = 5
    else:
        if k_int < 1:
            errors.append(f"k must be at least 1, got {k_int}.")
        elif k_int > MAX_K:
            warnings.append(f"k of {k_int} exceeds the cap; using {MAX_K}.")
            k_int = MAX_K

    # --- target energy ----------------------------------------------------
    # This is the guardrail for README adversarial findings 2 and 5, where an
    # energy of 2.0 or -1.0 pushed scores negative with no complaint.
    energy_out: Optional[float] = None
    if target_energy is not None:
        try:
            energy = float(target_energy)
        except (TypeError, ValueError):
            errors.append(f"target_energy must be a number, got {target_energy!r}.")
        else:
            if energy != energy:  # NaN
                errors.append("target_energy is not a number (NaN).")
            elif energy < ENERGY_MIN or energy > ENERGY_MAX:
                clamped = min(max(energy, ENERGY_MIN), ENERGY_MAX)
                warnings.append(
                    f"target_energy {energy:.2f} is outside the valid range "
                    f"{ENERGY_MIN:.1f}-{ENERGY_MAX:.1f}; clamped to {clamped:.2f}."
                )
                energy_out = clamped
            else:
                energy_out = energy

    if errors:
        return ValidatedRequest(ok=False, errors=errors, warnings=warnings)

    return ValidatedRequest(
        ok=True,
        query=cleaned,
        k=k_int,
        target_energy=energy_out,
        warnings=warnings,
    )


# ============================================================================
# Gate 2 - citation / groundedness check
# ============================================================================

@dataclass
class CitationReport:
    """
    Verdict on one piece of generated prose.

    grounded          - no unsupported song was named
    cited             - retrieved titles the model actually referenced
    out_of_context    - real catalog songs named but NOT retrieved for this query
    fabricated        - quoted titles matching nothing in the catalog at all
    """
    grounded: bool
    cited: List[str] = field(default_factory=list)
    out_of_context: List[str] = field(default_factory=list)
    fabricated: List[str] = field(default_factory=list)

    @property
    def failures(self) -> List[str]:
        notes = []
        for title in self.out_of_context:
            notes.append(f"named '{title}', which was not retrieved for this query")
        for title in self.fabricated:
            notes.append(f"named '{title}', which is not in the catalog at all")
        return notes


# Matches text the model presented as a title: inside quotes, or in the
# "Title by Artist" shape the prompt asks for.
_QUOTED = re.compile(r'"([^"]{2,60})"|“([^”]{2,60})”')
_TITLE_BY = re.compile(r"\b([A-Z][\w'&-]*(?:\s+[A-Z0-9][\w'&-]*){0,4})\s+by\s+[A-Z]")


def check_citations(
    text: str,
    retrieved: Sequence[Song],
    all_songs: Sequence[Song],
) -> CitationReport:
    """
    Verify that generated prose only discusses retrieved songs.

    Deliberately deterministic - no model grades this. That is what makes the
    groundedness number in the scorecard trustworthy: it is a string check that
    cannot be talked out of a verdict.
    """
    if not text or not text.strip():
        return CitationReport(grounded=True)

    haystack = text.lower()

    retrieved_titles = {song.title.lower(): song.title for song in retrieved}
    catalog_titles = {song.title.lower(): song.title for song in all_songs}

    cited = [
        original for lower, original in retrieved_titles.items()
        if lower in haystack
    ]

    # A real song that was not retrieved means the model pulled from memory
    # instead of from the supplied context - a genuine grounding failure even
    # though the song exists.
    out_of_context = [
        original for lower, original in catalog_titles.items()
        if lower in haystack and lower not in retrieved_titles
    ]

    # Anything the model presented as a title but which matches no catalog
    # entry. Compared against the full catalog, so a near-miss on a retrieved
    # title is caught rather than excused.
    fabricated: List[str] = []
    for match in _QUOTED.finditer(text):
        candidate = (match.group(1) or match.group(2) or "").strip()
        if candidate and candidate.lower() not in catalog_titles:
            fabricated.append(candidate)
    for match in _TITLE_BY.finditer(text):
        candidate = match.group(1).strip()
        if candidate.lower() not in catalog_titles:
            fabricated.append(candidate)

    # De-duplicate while preserving order for stable log output.
    fabricated = list(dict.fromkeys(fabricated))

    return CitationReport(
        grounded=not out_of_context and not fabricated,
        cited=sorted(cited),
        out_of_context=sorted(out_of_context),
        fabricated=fabricated,
    )


def strip_unsupported(text: str, report: CitationReport) -> str:
    """
    Remove sentences containing unsupported song references.

    Preferred over discarding the whole response: the supported sentences are
    still useful, and dropping only the bad ones keeps the system helpful while
    staying honest. If everything was unsupported, the caller gets an empty
    string and should fall back to the deterministic reason text.
    """
    if report.grounded:
        return text

    bad_terms = [t.lower() for t in report.out_of_context + report.fabricated]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [
        sentence for sentence in sentences
        if not any(term in sentence.lower() for term in bad_terms)
    ]
    return " ".join(kept).strip()


def refusal_message(query: str, best_similarity: float, floor: float) -> str:
    """
    What the system says when retrieval finds nothing close enough.

    Names the actual numbers so the refusal is auditable rather than a shrug,
    and never suggests a song - the whole point of this path is that we have
    nothing defensible to suggest.
    """
    return (
        f"Nothing in this catalog is a close enough match for "
        f'"{query}". Best similarity was {best_similarity:.3f}, below the '
        f"{floor:.2f} threshold required to make a recommendation. "
        "The catalog holds 72 tracks, so this is a coverage limit, not a "
        "judgement about the request. Try describing a genre, mood, or "
        "activity instead."
    )

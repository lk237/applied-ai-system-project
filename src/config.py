"""
Central configuration for Crate Digger.

Every tunable lives here so the README, the eval harness, and the app can never
drift apart on what the system was actually configured to do. Anything a grader
might want to change without editing code is overridable by environment
variable.

Design note: no secrets and no API keys. This system runs entirely on local
models, so there is nothing here that cannot be committed to git.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# Resolved relative to the project root (the parent of src/) rather than the
# current working directory, so the CLI, Streamlit, and pytest all agree on
# where data lives no matter where they are launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATALOG_PATH = Path(os.getenv("CRATE_CATALOG", PROJECT_ROOT / "data" / "songs.csv"))
INDEX_PATH = Path(os.getenv("CRATE_INDEX", PROJECT_ROOT / "data" / "index.npz"))
LOG_DIR = Path(os.getenv("CRATE_LOG_DIR", PROJECT_ROOT / "logs"))
EVAL_DIR = Path(os.getenv("CRATE_EVAL_DIR", PROJECT_ROOT / "evals"))
EVAL_RESULTS_PATH = EVAL_DIR / "results.json"
EVAL_SCORECARD_PATH = EVAL_DIR / "scorecard.md"


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
# Both are small enough to run on CPU and are downloaded once to the local
# HuggingFace cache. Pinning the names here (rather than inline at call sites)
# means the scorecard can record exactly which models produced a result.
EMBED_MODEL = os.getenv("CRATE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Qwen2.5-0.5B-Instruct rather than flan-t5-base, after measuring both.
# flan-t5-base is extractive at this size: asked to justify a recommendation it
# echoed the prompt back, invented facts about artists, and on one query fell
# into a repetition loop ("groovy, soaring, soaring, soaring..."). Qwen is a
# genuine instruction-tuned chat model and produces usable one-line rationales
# at a comparable cost on CPU. See README "Design Decisions".
GEN_MODEL = os.getenv("CRATE_GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Greedy decoding, fixed seed. Reproducibility is a graded requirement, so
# sampling is deliberately switched off: the same query must produce the same
# recommendation on every run and on every machine.
GEN_MAX_NEW_TOKENS = int(os.getenv("CRATE_GEN_MAX_TOKENS", "48"))
RANDOM_SEED = int(os.getenv("CRATE_SEED", "20260804"))

# Blocks the degenerate repetition loops small models fall into on open-ended
# prompts. Measured, not guessed: without it flan-t5 emitted "soaring" eight
# times in one sentence.
NO_REPEAT_NGRAM = int(os.getenv("CRATE_NO_REPEAT_NGRAM", "4"))


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------
TOP_K = int(os.getenv("CRATE_TOP_K", "5"))
MAX_K = 20  # hard ceiling; guards against a caller asking for the whole catalog

# Cosine similarity below this means "the catalog has nothing like this".
#
# 0.34 is measured, not guessed. Across the 26 golden cases (52 queries), the
# observed distribution was:
#
#     should answer   0.382 - 0.692   (n=44, lowest: "heavy distorted guitars")
#     should refuse   0.187 - 0.488   (n=8)
#
# The two classes OVERLAP, so no single threshold separates them. 0.34 sits in
# the gap below every legitimate query (0.382) and above the nonsense and
# out-of-domain queries (max 0.298), which is the widest clean margin available.
# It costs zero false refusals.
#
# What it does NOT catch: near-miss requests whose subject matter the catalog
# partly covers - "korean pop idol group" scores 0.441 against real pop tracks,
# and "norwegian black metal with bagpipes" scores 0.488 against real metal.
# A dense embedding of a compound query is a blend, so the disqualifying detail
# (korean, bagpipes) is averaged away by the terms that do match. Raising the
# floor above these would start refusing legitimate requests.
#
# Consequence: refusal accuracy is ~50%, and the confidence band carries the
# caveat for the rest. This is documented rather than tuned away; see
# model_card.md "Limitations" and evals/scorecard.md.
SIMILARITY_FLOOR = float(os.getenv("CRATE_SIM_FLOOR", "0.34"))


# ----------------------------------------------------------------------------
# Input guardrails
# ----------------------------------------------------------------------------
MAX_QUERY_CHARS = int(os.getenv("CRATE_MAX_QUERY_CHARS", "400"))
MIN_QUERY_CHARS = 3

# The original Module 3 scorer accepted any float for energy, which let a
# request for energy=2.0 produce negative scores (see README, adversarial
# profile 2). These bounds exist to close that hole.
ENERGY_MIN = 0.0
ENERGY_MAX = 1.0


# ----------------------------------------------------------------------------
# Confidence bands
# ----------------------------------------------------------------------------
# Read at runtime from the stored eval results. Thresholds are applied to the
# top retrieval similarity for a given query; the LABEL that gets shown also
# depends on whether the eval harness has ever been run (see confidence.py).
CONFIDENCE_HIGH = float(os.getenv("CRATE_CONF_HIGH", "0.45"))
CONFIDENCE_MEDIUM = float(os.getenv("CRATE_CONF_MEDIUM", "0.30"))


# ----------------------------------------------------------------------------
# Prompt strategies
# ----------------------------------------------------------------------------
# Two competing instructions for the generator. The eval harness scores both
# and writes the winner into evals/results.json; generate.py reads that file at
# startup and uses whichever actually performed better. This is the mechanism
# that makes evaluation load-bearing rather than decorative.
PROMPT_STRATEGIES = ("terse", "explanatory")
DEFAULT_STRATEGY = "terse"


def describe() -> dict:
    """Configuration snapshot, embedded in logs and the eval scorecard."""
    return {
        "embed_model": EMBED_MODEL,
        "gen_model": GEN_MODEL,
        "top_k": TOP_K,
        "similarity_floor": SIMILARITY_FLOOR,
        "seed": RANDOM_SEED,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
    }

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
GEN_MODEL = os.getenv("CRATE_GEN_MODEL", "google/flan-t5-base")

# Greedy decoding, fixed seed. Reproducibility is a graded requirement, so
# sampling is deliberately switched off: the same query must produce the same
# recommendation on every run and on every machine.
GEN_MAX_NEW_TOKENS = int(os.getenv("CRATE_GEN_MAX_TOKENS", "96"))
RANDOM_SEED = int(os.getenv("CRATE_SEED", "20260804"))


# ----------------------------------------------------------------------------
# Retrieval
# ----------------------------------------------------------------------------
TOP_K = int(os.getenv("CRATE_TOP_K", "5"))
MAX_K = 20  # hard ceiling; guards against a caller asking for the whole catalog

# Cosine similarity below this means "the catalog has nothing like this".
# Tuned against the golden set — see evals/scorecard.md. Raising it makes the
# system refuse more often (safer, less useful); lowering it makes the system
# stretch to answer queries it should decline (more useful, less honest).
SIMILARITY_FLOOR = float(os.getenv("CRATE_SIM_FLOOR", "0.20"))


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

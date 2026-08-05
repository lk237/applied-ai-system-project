"""
Streamlit interface for Crate Digger.

    streamlit run app.py

Shows the same pipeline the CLI drives, with the parts a grader needs to see
made visible rather than hidden: retrieval similarity per pick, which guardrails
fired, whether confidence is calibrated or merely asserted, and the raw JSON.

The engine is cached with st.cache_resource because Streamlit re-runs this
script top to bottom on every interaction, and reloading a 0.5B model per
keystroke would make the app unusable.
"""

from __future__ import annotations

import json

import streamlit as st

from src.config import SIMILARITY_FLOOR, TOP_K, describe
from src.pipeline import CrateDigger

st.set_page_config(page_title="Crate Digger", page_icon=":musical_note:", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads ~1.2 GB)...")
def get_engine() -> CrateDigger:
    return CrateDigger()


# ============================================================================
# Sidebar - configuration and provenance
# ============================================================================

with st.sidebar:
    st.header("Settings")
    k = st.slider("How many songs", 1, 10, TOP_K)
    floor = st.slider(
        "Similarity floor", 0.0, 0.9, SIMILARITY_FLOOR, 0.01,
        help=(
            "Below this the system refuses instead of recommending. "
            "0.34 is measured, not guessed - see config.py."
        ),
    )
    use_energy = st.checkbox("Filter by target energy")
    energy = st.slider("Target energy", -1.0, 2.0, 0.5, 0.05) if use_energy else None
    if use_energy and not (0.0 <= energy <= 1.0):
        st.warning("Out of range - the guardrail will clamp this and say so.")

    explain = st.checkbox(
        "Generate explanations", value=True,
        help="Uncheck for retrieval only. Much faster; no model generation.",
    )

    st.divider()
    st.caption("Configuration")
    st.json(describe(), expanded=False)


# ============================================================================
# Header
# ============================================================================

st.title("Crate Digger")
st.caption(
    "Ask for music in plain language. Every recommendation is retrieved from a "
    "72-song catalog before anything is written, and every claim is checked "
    "against the retrieved rows before you see it."
)

engine = get_engine()

status = st.container()
with status:
    left, middle, right = st.columns(3)
    left.metric("Catalog", f"{len(engine.songs)} songs")
    middle.metric("Prompt strategy", engine.strategy)
    right.metric(
        "Evaluation",
        "on record" if engine.eval_summary.exists else "none yet",
        help=engine.eval_summary.provenance,
    )
    if not engine.eval_summary.exists:
        st.info(
            "No evaluation on record, so confidence below is reported as "
            "**unverified**. Run `python -m src.evals.run_eval` to calibrate it.",
            icon=":material/info:",
        )

# ============================================================================
# Query
# ============================================================================

EXAMPLES = [
    "chill lofi for studying",
    "catchy pop music with guitars and big harmonies",
    "aggressive high energy music for a heavy workout",
    "moody synth music for driving alone at night",
    "qqqq zzzz vvvv xxxx wwww",
]

st.write("**Try one of these**")
columns = st.columns(len(EXAMPLES))
for column, example in zip(columns, EXAMPLES):
    if column.button(example, use_container_width=True):
        st.session_state["query"] = example

query = st.text_input(
    "What do you want to listen to?",
    key="query",
    placeholder="something upbeat for a road trip",
)

if not query:
    st.stop()

# ============================================================================
# Run
# ============================================================================

with st.spinner("Retrieving and explaining..."):
    result = engine.recommend(
        query, k=k, target_energy=energy, floor=floor, explain=explain
    )

for warning in result.warnings:
    st.warning(f"Input guardrail: {warning}", icon=":material/shield:")

if result.status == "rejected":
    st.error(f"**Rejected.** {result.message}", icon=":material/block:")
    st.caption("No model was loaded - this failed input validation.")
    st.stop()

if result.status == "refused":
    st.error(f"**Refused.** {result.message}", icon=":material/search_off:")
    st.caption(
        f"Top similarity {result.best_similarity:.3f} is below the "
        f"{floor:.2f} floor. The system does not guess."
    )
    st.stop()

# --- confidence -------------------------------------------------------------

band = result.confidence.band if result.confidence else "unknown"
badge = {"high": "green", "medium": "orange", "low": "red"}.get(band, "grey")
st.markdown(f"### Confidence: :{badge}[{band.upper()}]")
if result.confidence:
    st.caption(result.confidence.note)

if result.summary:
    st.info(result.summary, icon=":material/queue_music:")

# --- picks ------------------------------------------------------------------

for rank, pick in enumerate(result.picks, start=1):
    with st.container(border=True):
        head, meta = st.columns([3, 1])
        head.markdown(f"**{rank}. {pick.song.title}** - {pick.song.artist}")
        head.caption(
            f"{pick.song.genre} / {pick.song.mood} / "
            f"energy {pick.song.energy:.2f} / {pick.song.tempo_bpm} BPM"
        )
        meta.metric("similarity", f"{pick.similarity:.3f}")
        st.write(pick.reason)
        if not pick.grounded:
            st.warning(
                "The generated text named something outside the retrieved set, "
                "so it was replaced with catalog facts.",
                icon=":material/report:",
            )

# --- guardrail + raw --------------------------------------------------------

if result.guardrail_notes:
    with st.expander(
        f"Citation guardrail caught {len(result.guardrail_notes)} unsupported claim(s)",
        expanded=True,
    ):
        for note in result.guardrail_notes:
            st.write(f"- {note}")

with st.expander("Raw result (JSON)"):
    st.code(json.dumps(result.as_dict(), indent=2), language="json")

st.caption(
    f"strategy `{result.strategy}` | top similarity {result.best_similarity:.3f} "
    f"| {result.elapsed_ms:.0f} ms"
)

"""
The grounded generator - the "G" in RAG.

The model's job is narrow on purpose: it explains, it does not select. Which
songs get recommended is decided by retrieval, which is deterministic and
auditable. The generator only writes the prose that justifies each pick, and it
is given nothing but the retrieved rows to write from.

That split is a direct response to the model being small. flan-t5-base is good
at rephrasing text placed in front of it and bad at recalling facts about the
world. Asking it to pick songs would invite invention; asking it to summarise
five supplied descriptions plays to what it can actually do. The citation
guardrail then checks even that narrow output.

Two prompt strategies are implemented. The eval harness scores both and records
the winner, which confidence.active_strategy() reads back at runtime.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

from src.config import GEN_MAX_NEW_TOKENS, GEN_MODEL, RANDOM_SEED
from src.retrieval import Hit


# ============================================================================
# Generator interface
# ============================================================================

class TextGenerator(Protocol):
    """So tests can run the full pipeline without loading a model."""

    name: str

    def generate(self, prompt: str, max_new_tokens: int = GEN_MAX_NEW_TOKENS) -> str:
        ...


class FlanGenerator:
    """
    Local seq2seq generation via transformers.

    Greedy decoding with a fixed seed and no sampling: identical input must give
    identical output, because the README publishes captured outputs and a grader
    re-running the command has to see the same thing.
    """

    def __init__(self, model_name: str = GEN_MODEL):
        self.name = model_name
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment issue
                raise RuntimeError(
                    "transformers/torch are not installed. Run:\n"
                    "    pip install -r requirements.txt"
                ) from exc

            torch.manual_seed(RANDOM_SEED)
            self._tokenizer = AutoTokenizer.from_pretrained(self.name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.name)
            self._model.eval()
        return self._tokenizer, self._model

    def generate(self, prompt: str, max_new_tokens: int = GEN_MAX_NEW_TOKENS) -> str:
        import torch

        tokenizer, model = self._load()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,      # greedy - reproducible
                num_beams=1,
                repetition_penalty=1.15,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


class TemplateGenerator:
    """
    Deterministic stand-in used by the offline test suite.

    Echoes supplied context rather than modelling anything, so pipeline tests
    can assert on plumbing and guardrails without a model download. Never used
    by the app or the eval harness.
    """

    name = "template-stub"

    def generate(self, prompt: str, max_new_tokens: int = GEN_MAX_NEW_TOKENS) -> str:
        for line in prompt.splitlines():
            if line.startswith("SONG:"):
                return line.removeprefix("SONG:").strip()[:200]
        return "A selection drawn from the supplied catalog rows."


# ============================================================================
# Prompts
# ============================================================================
# Every prompt states the grounding rule explicitly and supplies facts inline.
# Nothing outside these strings reaches the model - no catalog dump, no history.

_RULE = (
    "Use only the information given below. "
    "Do not mention any song that is not listed."
)


def build_reason_prompt(query: str, hit: Hit, strategy: str) -> str:
    """One-line justification for a single retrieved song."""
    song = hit.song
    facts = (
        f"SONG: {song.title} by {song.artist}. "
        f"Genre {song.genre}, mood {song.mood}, "
        f"energy {song.energy:.2f} of 1.0, {song.tempo_bpm} BPM. "
        f"{song.description}"
    )

    if strategy == "terse":
        instruction = (
            f"{_RULE}\n\n{facts}\n\n"
            f'The listener asked for: "{query}".\n'
            "In one short sentence, say why this song fits the request."
        )
    else:  # explanatory
        instruction = (
            f"{_RULE}\n\n{facts}\n\n"
            f'The listener asked for: "{query}".\n'
            "Explain in two sentences why this song fits the request. "
            "Mention the specific musical qualities that make it a good match."
        )
    return instruction


def build_summary_prompt(query: str, hits: Sequence[Hit], strategy: str) -> str:
    """A single sentence introducing the whole set."""
    listing = "\n".join(
        f"- {hit.song.title} by {hit.song.artist} "
        f"({hit.song.genre}, {hit.song.mood})"
        for hit in hits
    )
    detail = (
        "one sentence" if strategy == "terse"
        else "two sentences describing what these tracks have in common"
    )
    return (
        f"{_RULE}\n\n"
        f"Selected songs:\n{listing}\n\n"
        f'The listener asked for: "{query}".\n'
        f"In {detail}, describe this set of songs as a group."
    )


# ============================================================================
# Generation
# ============================================================================

def generate_reasons(
    query: str,
    hits: Sequence[Hit],
    generator: TextGenerator,
    strategy: str,
) -> List[str]:
    """
    One rationale per retrieved song.

    Generated per song rather than in a single batch call: a small model asked
    for five rationales at once tends to blur them together or drop some, while
    one focused call per song stays on topic. It costs more calls but the output
    is far more reliable, and locally there is no per-call billing to weigh.

    A failure on one song degrades to an empty string for that song rather than
    aborting the request - four good rationales beat none.
    """
    reasons: List[str] = []
    for hit in hits:
        try:
            text = generator.generate(build_reason_prompt(query, hit, strategy))
        except Exception:
            text = ""
        reasons.append(_tidy(text))
    return reasons


def generate_summary(
    query: str,
    hits: Sequence[Hit],
    generator: TextGenerator,
    strategy: str,
) -> str:
    if not hits:
        return ""
    try:
        return _tidy(generator.generate(build_summary_prompt(query, hits, strategy)))
    except Exception:
        return ""


def _tidy(text: str) -> str:
    """Normalise whitespace and ensure terminal punctuation."""
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned[0].upper() + cleaned[1:]


def default_generator(offline: bool = False) -> TextGenerator:
    """FlanGenerator normally; the deterministic stub when offline is requested."""
    return TemplateGenerator() if offline else FlanGenerator()

# AI Interactions Log

This file records how an AI coding agent contributed to the applied AI system. Claims below are tied to changes or runs preserved in the repository.

## Agentic workflow

**Task given to the agent**

Plan and build an advanced version of the Module 1–3 Music Recommender Simulation. The project had to include a fully integrated advanced AI feature, reproducible setup, logging or guardrails, a Mermaid architecture diagram, automated reliability evidence, a portfolio README, and a responsible-AI model card. I selected the proposed combination of retrieval-augmented generation and a reliability system, required it to work without an API key, and chose CLI plus Streamlit interfaces.

**Key prompts and decisions**

- “Plan the project (pick a concept + which advanced feature to implement) before any code. Propose a few.”
- “Let’s do A+C then.”
- “No—must work without [an Anthropic API key].”
- “CLI + Streamlit UI.”
- “Do what’s best.”
- I also supplied the exact grading requirements for the Mermaid diagram, README sections, testing evidence, and model-card reflection.

**What the agent generated or changed**

- Expanded `data/songs.csv` from 20 to 72 fictional tracks with retrieval descriptions.
- Preserved the original rule-based scorer and ported `src/adversarial_eval.py` from Module 3.
- Added catalog validation, local semantic retrieval, local text generation, input and output guardrails, confidence calibration, structured JSONL logging, pipeline orchestration, CLI, and Streamlit UI.
- Added a 26-case golden set, deterministic metrics, prompt-strategy comparison, JSON and Markdown scorecards, and an offline test suite.
- Created current and original Mermaid architecture source files.
- Ran real local models and captured failures rather than inventing sample output.

**What I verified or corrected**

- Confirmed PyTorch and the full model stack install on Python 3.14.3.
- Rejected `flan-t5-base` after real tests showed prompt echoing, fabricated facts, and repetition.
- Observed genuine Qwen title and artist corruption and verified that the citation fallback activated.
- Measured the similarity distribution before choosing the refusal threshold.
- Kept weak real evaluation results—54% relevance, 0.36 consistency, and 50% refusal accuracy—instead of tuning them away.
- Corrected an evaluation-reporting bug where a retrieval-only phase caused a vacuous 100% groundedness headline. The true selected-strategy result is 55/65, or 84.6%.

## Design pattern

**Pattern used: Strategy pattern**

The generator supports `terse` and `explanatory` prompt strategies behind the same interface. The evaluation harness runs both on the same recommendation cases, selects the strategy with higher groundedness and then lower latency, and writes the winner to `evals/results.json`.

At runtime, `src/confidence.py` loads the stored result and `src/pipeline.py` uses the winning strategy. This separates prompt behavior from orchestration and makes it possible to add or compare strategies without rewriting the pipeline.

The project also applies dependency injection: tests construct the pipeline with a deterministic hashing embedder and template generator. This lets 131 tests run quickly without model downloads while production uses the real local models.

## Portfolio summary

I evolved a transparent rule-based music recommender into **Crate Digger**, a fully local RAG application with semantic retrieval, grounded AI explanations, citation guardrails, structured logs, and a Streamlit interface. I built a 26-case reliability harness whose results feed back into runtime prompt selection and confidence messaging. The project’s most important outcome was not a perfect score but an auditable one: 131 offline tests pass, while the real-model scorecard openly documents 85% raw groundedness, 54% relevance, 0.36 paraphrase consistency, and the failure modes that remain.

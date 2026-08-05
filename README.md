# Crate Digger

Crate Digger is a local, retrieval-augmented music recommender. A listener describes a situation in ordinary language, the system retrieves matching songs from a 72-track catalog, and a small instruction-tuned model explains each recommendation using only the retrieved records. It matters because recommendations that sound persuasive are easy to mistake for reliable ones; this project measures groundedness, relevance, paraphrase consistency, refusal behavior, and latency instead of relying on a polished demo alone.

No API key or paid service is required. The embedding and generation models run locally after a one-time download.

## Original project from Modules 1–3

The original project was **Music Recommender Simulation (VibeFinder 1.0)**. It accepted a structured profile—genre, mood, and target energy—and ranked 20 songs with a transparent weighted formula. Its goal was to demonstrate how simple recommendation rules work and expose weaknesses such as exact-match genre rigidity, invalid numeric inputs, and confident results for preferences absent from the catalog.

Crate Digger keeps that scorer as a baseline (`python -m src.main --baseline`) while adding semantic retrieval, grounded local generation, guardrails, structured logging, and a reliability harness.

## Architecture overview

The source diagram is [diagrams/architecture.mmd](diagrams/architecture.mmd); the original Module 3 system is preserved in [diagrams/architecture-original.mmd](diagrams/architecture-original.mmd).

The request path is:

1. The CLI or Streamlit UI accepts a natural-language request.
2. Input guardrails reject empty or oversized requests, clamp energy to `0.0–1.0`, and cap the requested result count.
3. `all-MiniLM-L6-v2` embeds the request and retrieves top candidates from `data/songs.csv` by cosine similarity.
4. Requests below the measured similarity floor are refused. Otherwise, `Qwen2.5-0.5B-Instruct` writes short reasons from retrieved song facts.
5. A citation guardrail checks generated song names. Unsupported output is replaced with a deterministic explanation and logged.
6. The response includes recommendations and a confidence band calibrated from stored evaluation results.
7. The evaluator compares two prompt strategies and writes the winner and scores to `evals/results.json`. The application reads that file at startup, so testing changes runtime behavior.

Human involvement is explicit: a person authors the golden expectations, reviews `evals/scorecard.md` and JSONL logs, adjusts thresholds or prompts, and signs off before release. Automated unit tests separately verify scoring, retrieval plumbing, guardrails, metrics, and failure handling without downloading models.

## Project organization

```text
app.py                         Streamlit interface
data/songs.csv                 72-song catalog and retrieval descriptions
diagrams/architecture.mmd      Current system diagram (Mermaid source)
evals/results.json             Machine-readable real-model evaluation
evals/scorecard.md             Human-readable reliability report
src/catalog.py                 Validated catalog loading
src/retrieval.py               Embedding index and cosine retrieval
src/generate.py                Local Qwen generator and prompt strategies
src/guardrails.py              Input and citation checks
src/pipeline.py                End-to-end RAG orchestration
src/confidence.py              Runtime evaluation feedback loop
src/evals/                     Golden cases, metrics, and eval runner
src/recommender.py             Original deterministic baseline
src/adversarial_eval.py        Original edge-case evaluation
tests/                         Offline automated test suite
```

## Setup instructions

Tested on Windows 11 with Python 3.14.3. Python 3.11 or newer is recommended. The first real-model run downloads model weights from Hugging Face and may take several minutes; later runs use the local cache.

```bash
git clone <your-repository-url>
cd applied-ai-system-project
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install pinned dependencies and run the fast offline tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Run a recommendation from the command line:

```bash
python -m src.main "chill lofi for studying" --k 3
```

Launch the portfolio UI:

```bash
streamlit run app.py
```

Useful alternatives:

```bash
# Exercise the complete pipeline with deterministic stub models and no downloads
python -m src.main "chill lofi for studying" --offline

# Run retrieval without generated explanations
python -m src.main "heavy distorted guitars" --no-explain

# Compare against the original Module 3 scorer
python -m src.main --baseline

# Run the real-model reliability sweep (about 16 minutes on the tested CPU)
python -m src.evals.run_eval --k 3
```

Environment variables such as `CRATE_TOP_K`, `CRATE_SIM_FLOOR`, `CRATE_EMBED_MODEL`, and `CRATE_GEN_MODEL` override defaults; all supported settings are documented in `src/config.py`.

## Sample interactions

These are captured from real local-model runs, not hand-written examples. Small local models can mangle prose; the guardrail behavior shown below is part of the result.

### 1. Strong semantic match

**Input**

```text
chill lofi for studying
```

**Output excerpt**

```text
Status: ok | best similarity: 0.466 | confidence: high

Midnight Coding — LoRoom (lofi, chill)
Library Rain — Paper Lanterns (lofi, chill)
Slow Sunday Loop — LoRoom (lofi, relaxed)
```

The model changed `Slow Sunday Loop` to an unsupported shortened title, `Sunday Loop`. The citation guardrail detected it and replaced that explanation with the deterministic fallback: `lofi track with a relaxed feel, energy 0.37 at 76 BPM.`

### 2. A relevant first result with imperfect ranking

**Input**

```text
something upbeat for a road trip
```

**Output excerpt**

```text
Status: ok | best similarity: 0.487 | confidence: high

Dust Road Home — The Ember Hollow (country, nostalgic)
Sable Coast — Lune Ordinaire (downtempo, moody)
Sequin Boulevard — Studio 55 (disco, groovy)
```

The first and third tracks are plausible, but the moody downtempo second result shows that semantic similarity does not guarantee that every retrieved item satisfies every word in a compound request.

### 3. Honest refusal

**Input**

```text
zzzz qqqq xxxx vvvv
```

**Output excerpt**

```text
Status: refused
Nothing in this catalog is a close enough match for "zzzz qqqq xxxx vvvv".
Best similarity was 0.201.
```

The evaluator also found an important counterexample: near-miss out-of-catalog requests such as K-pop or metal with bagpipes can overlap strongly with existing pop or metal songs and are not always refused.

## Design decisions and trade-offs

### Local models instead of an API

The system uses `all-MiniLM-L6-v2` for specialized semantic retrieval and `Qwen2.5-0.5B-Instruct` for explanation generation. This removes keys, billing, rate limits, and network calls after download. The trade-off is CPU latency—about 18.4 seconds per three-pick generated response during the strategy evaluation—and lower prose reliability than a larger hosted model.

### Retrieval chooses; generation explains

The model never chooses arbitrary songs. Deterministic top-k retrieval selects catalog rows, then the generator explains those choices. This sharply limits hallucination impact and lets the system replace unsupported prose without changing the retrieved recommendation.

### Qwen replaced FLAN-T5

The first implementation used `flan-t5-base`. Measured trials showed prompt echoing, fabricated artist facts, and repetition loops such as “soaring, soaring, soaring.” Qwen followed instructions much better, though it still altered titles and artist names. The failed model choice is retained as a documented lesson rather than hidden.

### A measured refusal threshold

The similarity floor is `0.34`. Across 52 golden queries, answerable requests scored `0.382–0.692`, while requests intended for refusal scored `0.187–0.488`. Because those ranges overlap, no single threshold can perfectly separate them. The chosen floor rejects nonsense and clearly out-of-domain requests without falsely refusing measured answerable cases, but it misses some partial catalog matches.

### Evaluation changes the application

Two prompt strategies are evaluated on groundedness and latency. The selected `terse` strategy is written to `evals/results.json` and loaded on application startup. The same result file informs the confidence note, satisfying the requirement that the reliability feature be integrated into main application logic.

## Testing summary

The automated suite currently passes **131 tests** offline. It covers CSV validation, baseline scoring, deterministic test retrieval, input normalization, refusal paths, fabricated-title detection, generator failure fallback, evaluation arithmetic, and the runtime feedback loop.

The real-model run evaluated **26 golden cases** and two generation strategies:

| Measure | Result | Interpretation |
|---|---:|---|
| Selected-strategy groundedness | **85% (55/65 picks)** | Unsupported generated names occurred, but output guardrails detected and replaced them |
| Relevance | **54% (14/26 cases)** | Twelve cases missed at least one human-authored property |
| Paraphrase consistency | **0.36 Jaccard** | Rewording often changed the retrieved songs substantially |
| Refusal accuracy | **50% (4/8 cases)** | Clear nonsense was refused; partial genre matches were difficult |
| Selected prompt strategy | **terse** | Better groundedness (85% vs. 82%) and lower latency (18.4s vs. 22.3s) |

What worked: strong literal and semantic matches such as study lofi, deterministic error handling, real citation-guardrail catches, and reproducible offline unit tests. What did not: compound requests, paraphrase stability, near-miss refusals, and occasional title or artist corruption by the small generator. Full case-level evidence is in [evals/scorecard.md](evals/scorecard.md) and [evals/results.json](evals/results.json).

The original rule-based system remains reproducible through `python -m src.adversarial_eval`; its latest 72-song output is stored in `evals/adversarial_output.txt`.

## Reflection

This project taught me that adding an AI model is easier than defining evidence that the model helps. Retrieval improved natural-language matching, but the evaluation revealed failures that a few attractive examples would have hidden. The most useful engineering pattern was to make failures observable and recoverable: validate inputs, constrain the model’s role, verify its output, log what happened, and let measured results control the runtime strategy.

The graded responsible-AI reflection—including AI collaboration, one helpful suggestion, one flawed suggestion, misuse, and limitations—is in [model_card.md](model_card.md).

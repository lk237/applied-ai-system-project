# Model Card: Crate Digger

## System summary

Crate Digger is a classroom and portfolio demonstration of retrieval-augmented music recommendation. It retrieves from a synthetic 72-song catalog with `all-MiniLM-L6-v2`, asks `Qwen2.5-0.5B-Instruct` to explain the retrieved results, verifies generated song names, and falls back to deterministic explanations when verification fails. It runs locally and requires no API key.

The system is intended for learning, prototyping, and low-stakes music discovery. It is not a production recommender, a source of factual information about real artists, or a substitute for licensed music metadata.

## Data and models

The catalog contains 72 fictional songs across 53 genre labels and 25 mood labels. Each row includes title, artist, genre, mood, energy, tempo, valence, danceability, acousticness, and a short retrieval description. The hand-authored descriptions make the experiment reproducible but may encode the author’s assumptions about which sounds, moods, and activities belong together.

- Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Generator: `Qwen/Qwen2.5-0.5B-Instruct`
- Decoding: greedy, fixed seed, four-token no-repeat n-gram
- Evaluator: deterministic code plus 26 human-authored golden cases; no model grades another model

## Evaluation results

The real-model evaluation was run on Windows 11 with Python 3.14.3. The selected `terse` prompt produced 55 grounded explanations across 65 recommendations (**84.6%**); every detected unsupported explanation was replaced before display. Relevance passed 14 of 26 cases (**53.8%**), paraphrase consistency averaged **0.36 Jaccard**, and refusal accuracy was 4 of 8 cases (**50%**). The full parseable evidence is in `evals/results.json`; the readable report is `evals/scorecard.md`.

These metrics measure different things. Groundedness does not prove relevance, and the post-generation fallback means displayed output is safer than raw model output. A “high” confidence label describes retrieval similarity, not an 85% probability that the listener will enjoy a song.

## 1. How I collaborated with AI

I used an AI coding agent as a pair programmer for architecture, implementation, testing, and documentation. I supplied the assignment requirements, selected the combined RAG and reliability direction, approved a local no-key design, and reviewed consequential choices. The agent inspected the existing Module 3 project, preserved the original scorer and adversarial evaluation, built the new pipeline, ran models and tests, and surfaced unexpected failures while I controlled scope and approved continued work.

I did not treat generated claims as evidence. We ran the actual code, captured real outputs, measured similarity distributions, and kept failures in the final scorecard. When the first headline incorrectly reported 100% groundedness from a retrieval-only phase, review of the result schema exposed the error; the report was corrected to the selected strategy’s measured 84.6%, and a regression test was added.

## 2. One helpful AI suggestion and one flawed AI suggestion

**Helpful suggestion:** the agent proposed making evaluation load-bearing by reading `evals/results.json` at runtime. That idea became a central design feature: the application selects the better measured prompt strategy and adds evaluation context to its confidence note. This is stronger than a standalone test script because evaluation meaningfully changes the application’s behavior.

**Flawed suggestion:** the agent initially recommended `flan-t5-base` as a CPU-friendly grounded generator. Real trials showed that it echoed prompts, invented artist facts, and entered repetition loops. The design was changed to `Qwen2.5-0.5B-Instruct`, which followed instructions better but remained imperfect. The agent also initially claimed PyTorch lacked Python 3.14 wheels; a package-index dry run disproved that assumption. These mistakes reinforced the need to test environment and model claims instead of accepting plausible advice.

## 3. Potential misuse and responsible use

Although this catalog is fictional, the same architecture could be misused in a real service to create opaque filter bubbles, promote paid placements without disclosure, infer sensitive moods from listening requests, or present generated artist descriptions as facts. A system operator could also manipulate descriptions so sponsored tracks appear semantically relevant while claiming that retrieval is neutral.

Responsible deployment would require licensed and representative data, clear sponsorship labels, privacy controls, user-visible reasons, opt-out and correction mechanisms, diversity constraints, monitoring by genre and user segment, and a documented appeals process for artists. Query logs may reveal health, relationship, religious, or political context; production logs should minimize content, redact identifiers, enforce retention limits, and restrict access.

## 4. Limitations and what surprised me during testing

- The 72-song synthetic catalog is too small and subjective to represent music culture fairly.
- Dense retrieval blends compound requests. “Norwegian black metal with bagpipes” matches existing metal strongly enough to pass the threshold even though bagpipes are absent.
- No single similarity threshold separated all answerable and refusable golden queries: answerable scores ranged from 0.382 to 0.692, while refusal cases ranged from 0.187 to 0.488.
- Paraphrase consistency was only 0.36. For example, changing the wording of a dinner-party or dance-floor request could replace all three results.
- Relevance was 54%; a semantically similar result may still violate an important constraint such as energy, instrumentation, or genre.
- Raw generated prose was grounded only 85% of the time. The model shortened or altered titles and artist names, so deterministic fallback is essential.
- Citation checking focuses on song names. Artist-name corruption and subtler factual changes may escape the current checker.
- CPU generation is slow: the selected strategy averaged 18.4 seconds for three generated recommendations in the evaluation.
- The confidence band is a retrieval-strength label calibrated with aggregate evaluation context, not a user-enjoyment probability.

The most surprising result was that an apparently strong RAG design could fail for opposite reasons at once: lexical details such as “midnight” could dominate an irrelevant dance-floor query, while a dense embedding could average away decisive details such as “bagpipes.” I also learned that a reliability harness can contain its own silent bug—the original 100% groundedness headline—which means evaluation code needs tests and human review just as much as application code.

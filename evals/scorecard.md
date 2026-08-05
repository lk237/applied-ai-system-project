# Reliability Scorecard

Generated `2026-08-05T03:54:07+00:00` in 947s.

Produced by `python -m src.evals.run_eval`. Every metric below is a
deterministic check - no model grades another model's output.

## Headline

| Metric | Score | What it measures |
|---|---|---|
| Groundedness | **85%** | Share of selected-strategy recommendations whose prose named only retrieved songs |
| Relevance | **54%** | Golden cases meeting every human-authored expectation |
| Consistency | **0.36** | Mean Jaccard overlap between a query and its paraphrase |
| Refusal accuracy | **50%** | Out-of-catalog requests correctly declined |
| Latency (mean) | 2.0s | Retrieval-only, per case (two queries) |
| Latency (p95) | 0.1s | Slowest 5% |

Cases: **26**. Configuration: `{"embed_model": "sentence-transformers/all-MiniLM-L6-v2", "gen_model": "Qwen/Qwen2.5-0.5B-Instruct", "top_k": 5, "similarity_floor": 0.34, "seed": 20260804, "max_new_tokens": 48, "k": 3, "floor": 0.34}`

## Failures

12 of 26 cases failed their expectation.

| Case | Query | Why it failed |
|---|---|---|
| `study-focus` | quiet instrumental music for studying withou | top pick genre 'classical' is not one of ['lofi', 'ambient', 'drone', 'chillhop', 'neoclassical', 'piano'] |
| `gym-hard` | aggressive high energy music for a heavy wor | energy below 0.70 for: Spacewalk Thoughts (0.28) |
| `road-trip` | something upbeat for a long road trip with t | top pick energy 0.41 < required 0.55 |
| `sleep` | calm music with no drums to fall asleep to | top pick genre 'synthwave' is not one of ['ambient', 'drone', 'neoclassical', 'piano']; energy above 0.40 for: Night Drive Loop (0.75) |
| `dancefloor` | music that will fill a dancefloor at midnigh | top pick genre 'lofi' is not one of ['edm', 'house', 'disco', 'funk', 'techno', 'trance']; top pick energy 0.42 < required 0.75 |
| `near-genre-soul` | smooth soulful singing over warm keys | top pick genre 'ambient' is not one of ['soul', 'r&b'] |
| `heavy-guitars` | really heavy distorted guitars and screaming | top pick genre 'dream pop' is not one of ['metal', 'punk', 'rock']; top pick energy 0.52 < required 0.65 |
| `brass-horns` | music with a big brass horn section | top pick genre 'drum and bass' is not one of ['salsa', 'funk', 'afrobeat', 'gospel', 'disco', 'reggae'] |
| `anxious` | tense uneasy music that feels claustrophobic | top pick mood 'bittersweet' is not one of ['anxious', 'restless', 'intense', 'moody'] |
| `jamaican` | laid back jamaican music with offbeat guitar | top pick genre 'afrobeat' is not one of ['reggae', 'dub'] |
| `refuse-kpop` | korean pop idol group with rap verses and a  | expected a refusal but got ok (best similarity 0.441; returned Lagos Morning, Concrete Kingdom, Gym Hero) |
| `refuse-bagpipes` | norwegian black metal recorded with bagpipes | expected a refusal but got ok (best similarity 0.488; returned Kerosene Hymn, Hydroplane, Blacktop Prayer) |

## Weakest paraphrase robustness

Low overlap means rewording the same request returns different songs.

| Case | Overlap | Query result | Paraphrase result |
|---|---|---|---|
| `dinner-party` | 0.00 | Coffee Shop Stories, Late Set, Organ Grinder Blues | Bedroom Ceiling, Aftertaste, Kettle on the Stove |
| `dancefloor` | 0.00 | Midnight Coding, Night Drive Loop, Marble Steps | Gym Hero, Sequin Boulevard, Chalk Lines |
| `near-genre-soul` | 0.00 | Tidal Lull, Testify, Cold Open | Velvet Hours, Organ Grinder Blues, Prelude in Grey |
| `heartbreak` | 0.00 | Salt and Sorrow, Delta Blue Morning, Slow Collapse | Dos Copas, Aftertaste, Last Call Disco |
| `very-slow` | 0.00 | Moonlit Sonata Drift, Ceiling Fan Slow, Coffee Shop Stories | Panorama Wide, Signal Decay, Tidal Lull |

## Prompt strategy bake-off

Retrieval is identical across strategies, so relevance and consistency
cannot distinguish them. Groundedness and latency can.

| Strategy | Groundedness | Picks checked | Mean latency | Selected |
|---|---|---|---|---|
| `terse` | 85% | 65 | 18.4s | **yes** |
| `explanatory` | 82% | 65 | 22.3s | |

`terse` is written to `evals/results.json`; the running system reads it on start.

## Guardrail activity

**terse** - 12 caught:

- study-focus: Coffee Shop Stories: named 'quiet instrumental', which is not in the catalog at all
- gym-hard: Spacewalk Thoughts: named 'aggressive, high-energy music for a workout,', which is not in the catalog at all
- sleep: Slow Sunday Loop: named 'Sunday Loop', which is not in the catalog at all
- banjo-fiddle: Pocket Full of Nickels: named 'Pocket Full of Nickles', which is not in the catalog at all
- banjo-fiddle: Green Isle Reel: named 'playful', which is not in the catalog at all
- heavy-guitars: Static Bloom: named 'really heavy, distorted guitars and screamed vocals.', which is not in the catalog at all
- heavy-guitars: Static Bloom: named 'screaming', which is not in the catalog at all
- heavy-guitars: Static Bloom: named 'whispering,', which is not in the catalog at all
- heavy-guitars: Kerosene Hymn: named 'Kerosene Hmng', which is not in the catalog at all
- romantic: Moonlit Sonata Drift: named 'The Moonlit Sonata', which is not in the catalog at all

**explanatory** - 12 caught:

- study-focus: Moonlit Sonata Drift: named 'The Moonlit Sonata', which is not in the catalog at all
- late-night-drive: Night Drive Loop: named 'moody synths', which is not in the catalog at all
- late-night-drive: Sunrise City: named 'moody synths', which is not in the catalog at all
- late-night-drive: Amber Interchange: named 'moody synths', which is not in the catalog at all
- dancefloor: Night Drive Loop: named 'music that fills a dancefloor,', which is not in the catalog at all
- heavy-guitars: Slow Collapse: named 'really heavy', which is not in the catalog at all
- heartbreak: Salt and Sorrow: named 'sad songs', which is not in the catalog at all
- heartbreak: Delta Blue Morning: named 'sad songs', which is not in the catalog at all
- romantic: Moonlit Sonata Drift: named 'The Moonlit Sonata', which is not in the catalog at all
- very-slow: Moonlit Sonata Drift: named 'The Moonlit Sonata', which is not in the catalog at all

## Sample generated output

**`study-focus`** - quiet instrumental music for studying without distraction

> A group of classical and jazz tracks are selected to create a quiet instrumental background for studying without distractions.
>
> *Moonlit Sonata Drift*: The quiet instrumental music by Clara Vecht perfectly complements the listener's desire for a tranquil study environment, as it features a somber yet soothing atmosphere, suitable for reducing distractions while focusing on their studies.

**`gym-hard`** - aggressive high energy music for a heavy workout

> The selected songs form an aggressive high-energy duo, each contributing to a dynamic and energetic workout environment.
>
> *Gym Hero*: Gym Hero's aggressive high-energy pop, featuring a hard four on the floor kick and shouted lyrics, perfectly matches their demanding workout needs.

**`road-trip`** - something upbeat for a long road trip with the windows down

> The selected songs create an uplifting and reflective atmosphere suitable for a long car journey.
>
> *Sable Coast*: Sable Coast's downtempo, moody, and dark sound perfectly complements the listener's desire for an upbeat, relaxing experience on a long roadtrip with rain-soaked windows.


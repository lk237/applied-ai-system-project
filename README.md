# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Real-world platforms like Spotify and YouTube predict what you'll enjoy by watching your behavior — the songs you like, skip, replay, and add to playlists — and by analyzing the attributes of the music itself, such as genre, mood, energy, and tempo. They usually blend two approaches: **collaborative filtering** (recommending what people with similar taste enjoyed) and **content-based filtering** (recommending songs whose features match what you already like). My version is a small, transparent **content-based** recommender. It has no crowd of users to learn from, so instead of guessing from behavior it compares each song's measurable features against a user's stated taste profile and scores how closely they match. My priority is **explainability over accuracy**: every recommendation comes from a clear scoring rule I can inspect and tune, so I can always answer *why* a song was suggested.

### Data flow

The system is a simple five-stage pipeline. Scoring is separated from ranking so I can tune the recipe without touching how results are ordered:

```
INPUT          User preferences: favorite genre, favorite mood, target energy (0.0–1.0), and K
   │
   ▼
READ DATA      Load every song from data/songs.csv
   │
   ▼
PROCESS        For each individual song:
               1. Start score at 0
               2. Check genre  → add points on exact match
               3. Check mood   → add points on exact match
               4. Energy similarity → add partial points for closeness
               5. Save the song's total score
   │
   ▼
RANKING        Sort all songs from highest score to lowest
   │
   ▼
OUTPUT         Return the top K songs
```

Each song is scored in isolation, so loop order never affects results and any single song can be tested on its own.

### Algorithm Recipe

Every song's score is the sum of three independent parts (starting from 0):

| Feature | Type | Rule | Points |
|---------|------|------|--------|
| **Genre** | categorical | exact match → `+2.0`, else `0` | up to **2.0** |
| **Mood** | categorical | exact match → `+1.0`, else `0` | up to **1.0** |
| **Energy** | numerical `0.0`–`1.0` | closeness: `1.5 × (1 − |song.energy − user.energy|)` | up to **1.5** |

```
score(song) = genre_points + mood_points + energy_points
            = (2.0 if genre matches else 0)
            + (1.0 if mood matches  else 0)
            + 1.5 × (1 − |song.energy − user.energy|)
```

**Why these weights:** Genre is the most stable signal of long-term taste, so it's the anchor at `2.0`. Mood is more situational, so it's worth half of genre at `1.0`. Energy is a *gradient* rather than a pass/fail — the closer a song's energy is to the target, the more of its `1.5` it earns — which lets it break ties without overruling genre. This keeps the priority order **Genre → Energy → Mood**, though a near-perfect energy + mood fit (`1.5 + 1.0 = 2.5`) can still outrank a lone genre match (`2.0`), which is usually desirable.

### Features used

**`Song` features:**
- `id`, `title`, `artist` — identification (labels only, not scored)
- `genre` — categorical (exact match)
- `mood` — categorical (exact match)
- `energy` — numerical, `0.0`–`1.0` (closeness)
- `tempo_bpm`, `valence`, `danceability`, `acousticness` — present in the data but **not scored** in this recipe (candidate tie-breakers for later experiments)

**`UserProfile` preferences (the features songs are scored against):**
- `genre` — preferred genre
- `mood` — preferred mood
- `energy` — preferred energy level, `0.0`–`1.0`

### Potential biases I expect

- **Over-prioritizes genre.** With genre worth `2.0`, a great song that nails the user's *mood* and *energy* but sits in a neighboring genre can be buried beneath a same-genre song that's a worse fit overall. The system may never surface a genuinely delightful cross-genre track.
- **Exact-match rigidity for categories.** Genre and mood are all-or-nothing. "indie pop" earns nothing against a user who asked for "pop," even though they're close — and "chill" vs. "relaxed" are treated as totally unrelated. Semantically similar tags get zero credit.
- **Popularity/coverage blindness.** The recipe only rewards similarity to stated taste, so it will keep recommending the same close matches and never introduce variety or discovery — a filter-bubble effect in miniature.
- **Numeric features ignored.** Tempo, valence, danceability, and acousticness are unused, so two songs identical on genre/mood/energy are ranked arbitrarily even if one is a far better fit on rhythm or feel.

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Sample Recommendation Output

Generated by running `python -m src.main` with the default profile
(`genre=pop, mood=happy, energy=0.8`):

```
Loading songs from data/songs.csv...
Loaded songs: 20

============================================================
Top 5 recommendations for genre=pop, mood=happy, energy=0.8
============================================================

1. Sunrise City - Neon Echo
   Score: 4.47
   Reasons:
     - genre match (pop) (+2.0)
     - mood match (happy) (+1.0)
     - energy closeness (song 0.82 vs target 0.80) (+1.47)

2. Gym Hero - Max Pulse
   Score: 3.30
   Reasons:
     - genre match (pop) (+2.0)
     - energy closeness (song 0.93 vs target 0.80) (+1.30)

3. Rooftop Lights - Indigo Parade
   Score: 2.44
   Reasons:
     - mood match (happy) (+1.0)
     - energy closeness (song 0.76 vs target 0.80) (+1.44)

4. Concrete Kingdom - Vince Marrow
   Score: 1.48
   Reasons:
     - energy closeness (song 0.79 vs target 0.80) (+1.48)

5. Groove Machine - Funkadelphia
   Score: 1.46
   Reasons:
     - energy closeness (song 0.83 vs target 0.80) (+1.46)
```

**Why these results make sense:** *Sunrise City* is a perfect fit — it hits
the genre (pop), the mood (happy), and sits almost exactly on the target
energy, earning nearly the full 4.5 points. *Gym Hero* is also pop and
high-energy but not "happy", so it ranks second. *Rooftop Lights* is "indie
pop" (no exact genre match) but happy and energy-close, showing the
exact-match rigidity noted in the biases above.

---

## System Evaluation — Adversarial / Edge-Case Profiles

To probe whether the scoring logic can be *tricked* or produce surprising
results, I ran six deliberately hostile user profiles — conflicting, out-of-range,
and nonexistent preferences — against the recommender. These are defined in
[`src/adversarial_eval.py`](src/adversarial_eval.py) and run with:

```bash
python -m src.adversarial_eval
```

Each block below is the **raw terminal output** for that profile's top 5, followed
by what it reveals about the scoring recipe.

### 1. Contradiction — high energy + a "sad" mood

A user who wants `energy: 0.9` **and** `mood: sad`. No song in the catalog is
tagged "sad," so the mood term silently contributes nothing and **energy quietly
becomes the deciding factor** behind the single genre match.

```
================================================================
[Contradiction (energy 0.9 + mood sad)]
  genre=rock, mood=sad, energy=0.9
================================================================

1. Storm Runner - Voltline
   Score: 3.48
   Reasons:
     - genre match (rock) (+2.0)
     - energy closeness (song 0.91 vs target 0.90) (+1.48)

2. Gym Hero - Max Pulse
   Score: 1.46
   Reasons:
     - energy closeness (song 0.93 vs target 0.90) (+1.46)

3. Voltage Drop - Kilobyte
   Score: 1.41
   Reasons:
     - energy closeness (song 0.96 vs target 0.90) (+1.41)

4. Groove Machine - Funkadelphia
   Score: 1.40
   Reasons:
     - energy closeness (song 0.83 vs target 0.90) (+1.40)

5. Iron Verdict - Ashen Throne
   Score: 1.38
   Reasons:
     - energy closeness (song 0.98 vs target 0.90) (+1.38)
```

**Finding:** The system does *not* flag the contradiction — it just drops the
impossible term and ranks on what's left. Ranks 2–5 are pure energy matches with
zero connection to the stated mood, and the user is never told that "sad" matched
nothing.

### 2. Out-of-range energy (`2.0`)

Energy is documented as `0.0–1.0`, but nothing validates it. With a target of
`2.0`, the term `1.5 × (1 − |song − 2.0|)` **goes negative**, dragging totals
below zero.

```
================================================================
[Out-of-range energy (2.0)]
  genre=pop, mood=happy, energy=2.0
================================================================

1. Sunrise City - Neon Echo
   Score: 2.73
   Reasons:
     - genre match (pop) (+2.0)
     - mood match (happy) (+1.0)
     - energy closeness (song 0.82 vs target 2.00) (+-0.27)

2. Gym Hero - Max Pulse
   Score: 1.90
   Reasons:
     - genre match (pop) (+2.0)
     - energy closeness (song 0.93 vs target 2.00) (+-0.10)

3. Rooftop Lights - Indigo Parade
   Score: 0.64
   Reasons:
     - mood match (happy) (+1.0)
     - energy closeness (song 0.76 vs target 2.00) (+-0.36)

4. Iron Verdict - Ashen Throne
   Score: -0.03
   Reasons:
     - energy closeness (song 0.98 vs target 2.00) (+-0.03)

5. Voltage Drop - Kilobyte
   Score: -0.06
   Reasons:
     - energy closeness (song 0.96 vs target 2.00) (+-0.06)
```

**Finding:** Two bugs surface at once. (a) **Scores can be negative** — the energy
term has no `max(0, …)` clamp, so a song can be *penalized* for existing. (b) A
**cosmetic formatting bug**: the explanation prints `(+-0.27)` because the
template hard-codes a `+` in front of a value that is already negative.

### 3. Ghost preferences — genre & mood that don't exist

`genre: kpop` and `mood: ecstatic` appear nowhere in the catalog, so both
categorical terms score `0` for *every* song and the entire top 5 is decided by a
single feature — energy.

```
================================================================
[Ghost preferences (genre kpop + mood ecstatic)]
  genre=kpop, mood=ecstatic, energy=0.5
================================================================

1. Island Time - Sol Marley
   Score: 1.48
   Reasons:
     - energy closeness (song 0.51 vs target 0.50) (+1.48)

2. Velvet Hours - Simone Ray
   Score: 1.47
   Reasons:
     - energy closeness (song 0.48 vs target 0.50) (+1.47)

3. Dust Road Home - The Ember Hollow
   Score: 1.42
   Reasons:
     - energy closeness (song 0.55 vs target 0.50) (+1.42)

4. Paper Boats - Wren & Willow
   Score: 1.41
   Reasons:
     - energy closeness (song 0.44 vs target 0.50) (+1.41)

5. Midnight Coding - LoRoom
   Score: 1.38
   Reasons:
     - energy closeness (song 0.42 vs target 0.50) (+1.38)
```

**Finding:** A typo or unsupported taste produces **confident but meaningless
recommendations** — five mid-energy songs from five unrelated genres. The system
gives no signal that it understood *none* of the categorical request.

### 4. Impossible combo — metal at near-silent energy

Metal is the loudest genre in the catalog (`Iron Verdict`, energy `0.98`), but the
user asks for `energy: 0.05`. This pits the `+2.0` genre anchor directly against
the energy penalty.

```
================================================================
[Impossible combo (genre metal + energy 0.05)]
  genre=metal, mood=calm, energy=0.05
================================================================

1. Iron Verdict - Ashen Throne
   Score: 2.10
   Reasons:
     - genre match (metal) (+2.0)
     - energy closeness (song 0.98 vs target 0.05) (+0.11)

2. Moonlit Sonata Drift - Clara Vetch
   Score: 1.22
   Reasons:
     - energy closeness (song 0.24 vs target 0.05) (+1.22)

3. Spacewalk Thoughts - Orbit Bloom
   Score: 1.16
   Reasons:
     - energy closeness (song 0.28 vs target 0.05) (+1.16)

4. Library Rain - Paper Lanterns
   Score: 1.05
   Reasons:
     - energy closeness (song 0.35 vs target 0.05) (+1.05)

5. Coffee Shop Stories - Slow Stereo
   Score: 1.02
   Reasons:
     - energy closeness (song 0.37 vs target 0.05) (+1.02)
```

**Finding:** The genre anchor is strong enough that a maximally *wrong-energy*
metal track still wins rank 1 — the `+2.0` swamps the near-zero energy term. But
ranks 2–5 are quiet classical/ambient/lofi tracks that ignore the genre request
entirely, so the top 5 is internally contradictory: one screaming metal song
followed by four lullabies.

### 5. Negative energy (`-1.0`)

Pushing even further out of the valid domain confirms there is no lower bound
either — the energy penalty just grows.

```
================================================================
[Negative energy (-1.0)]
  genre=lofi, mood=chill, energy=-1.0
================================================================

1. Library Rain - Paper Lanterns
   Score: 2.47
   Reasons:
     - genre match (lofi) (+2.0)
     - mood match (chill) (+1.0)
     - energy closeness (song 0.35 vs target -1.00) (+-0.53)

2. Midnight Coding - LoRoom
   Score: 2.37
   Reasons:
     - genre match (lofi) (+2.0)
     - mood match (chill) (+1.0)
     - energy closeness (song 0.42 vs target -1.00) (+-0.63)

3. Focus Flow - LoRoom
   Score: 1.40
   Reasons:
     - genre match (lofi) (+2.0)
     - energy closeness (song 0.40 vs target -1.00) (+-0.60)

4. Spacewalk Thoughts - Orbit Bloom
   Score: 0.58
   Reasons:
     - mood match (chill) (+1.0)
     - energy closeness (song 0.28 vs target -1.00) (+-0.42)

5. Moonlit Sonata Drift - Clara Vetch
   Score: -0.36
   Reasons:
     - energy closeness (song 0.24 vs target -1.00) (+-0.42)
```

**Finding:** Genre + mood matches (`+3.0`) are large enough to survive a `-0.6`
energy penalty, so the ranking is *still* reasonable at the top — but the negative
energy points and the `(+-0.53)` formatting bug persist, and rank 5 is again a
negative-scored song.

### 6. Mood + energy vs. genre — does the priority order flip?

The README claims a near-perfect mood + energy fit (`1.0 + 1.5 = 2.5`) can outrank
a lone genre match (`2.0`). This profile tests it: `genre: jazz` (matches only one
low-energy song) against a `happy` + `energy 0.82` combo.

```
================================================================
[Mood+energy vs genre (genre jazz, mood happy, energy 0.82)]
  genre=jazz, mood=happy, energy=0.82
================================================================

1. Coffee Shop Stories - Slow Stereo
   Score: 2.83
   Reasons:
     - genre match (jazz) (+2.0)
     - energy closeness (song 0.37 vs target 0.82) (+0.83)

2. Sunrise City - Neon Echo
   Score: 2.50
   Reasons:
     - mood match (happy) (+1.0)
     - energy closeness (song 0.82 vs target 0.82) (+1.50)

3. Rooftop Lights - Indigo Parade
   Score: 2.41
   Reasons:
     - mood match (happy) (+1.0)
     - energy closeness (song 0.76 vs target 0.82) (+1.41)

4. Groove Machine - Funkadelphia
   Score: 1.48
   Reasons:
     - energy closeness (song 0.83 vs target 0.82) (+1.48)

5. Concrete Kingdom - Vince Marrow
   Score: 1.46
   Reasons:
     - energy closeness (song 0.79 vs target 0.82) (+1.46)
```

**Finding:** The genre song still wins here (its `0.83` energy points were enough
to keep it at `2.83`), but ranks 2 and 3 — cross-genre `happy` + energy-perfect
songs at `2.50`/`2.41` — confirm the documented behavior: a strong mood + energy
fit **out-scores every non-matching genre** and lands directly behind the anchor.
The ordering is defensible, but note the top result is a *calm jazz* track being
recommended to someone who asked for *high-energy happy* music.

### Summary of what the adversarial pass exposed

| # | Profile | Weakness revealed |
|---|---------|-------------------|
| 1 | Contradiction (energy 0.9 + mood sad) | Impossible/absent categories are silently dropped, not flagged |
| 2 | Out-of-range energy (2.0) | **Scores go negative** (no energy clamp) + `(+-0.27)` formatting bug |
| 3 | Ghost preferences (kpop / ecstatic) | Unknown genre & mood → confident but meaningless energy-only ranking |
| 4 | Impossible combo (metal + energy 0.05) | Genre anchor produces an internally contradictory top 5 |
| 5 | Negative energy (-1.0) | No lower-bound validation; negative points persist |
| 6 | Mood+energy vs genre (jazz + happy) | Priority order *can* flip below rank 1, as documented |

**Recommended fixes:** clamp the energy term with `max(0.0, …)`, validate that
`energy ∈ [0, 1]` on input, warn (or zero-weight) when a requested genre/mood is
absent from the catalog, and fix the `(+{value})` template so it doesn't emit
`+-`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this




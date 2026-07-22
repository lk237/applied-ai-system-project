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

Paste a sample of your recommender's output here as a text block so a reader can see what it produces:

```
# e.g.:
# User profile: genre=indie, mood=chill, energy=low
# Recommendations:
#   1. ...
#   2. ...
#   3. ...
```

**Screenshot or video** *(optional)*: <!-- Insert a screenshot or demo video link here -->

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




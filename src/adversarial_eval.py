"""
Adversarial / edge-case evaluation for the Music Recommender.

Goal: probe whether the scoring logic in recommender.py can be "tricked" or
produces surprising results when given user profiles with conflicting,
out-of-range, or nonexistent preferences.

Run with:  python -m src.adversarial_eval   (or)   python src/adversarial_eval.py
"""

try:
    from src.recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    from recommender import load_songs, recommend_songs


# Each profile is crafted to stress a specific weakness in the scoring recipe:
#   score = genre(+2 exact) + mood(+1 exact) + 1.5*(1 - |song.energy - user.energy|)
ADVERSARIAL_PROFILES = {
    # 1. Classic contradiction: wants high energy AND a sad mood. No "sad" song
    #    exists, so mood never matches -> energy silently dominates the ranking.
    "Contradiction (energy 0.9 + mood sad)":
        {"genre": "rock", "mood": "sad", "energy": 0.9},

    # 2. Out-of-range energy (>1.0). The energy term has no clamp, so it can go
    #    NEGATIVE, dragging total scores below zero. Ranking degrades to "whoever
    #    is loudest" and reasons show negative point values.
    "Out-of-range energy (2.0)":
        {"genre": "pop", "mood": "happy", "energy": 2.0},

    # 3. Ghost preferences: genre + mood that don't exist in the catalog. Both
    #    categorical terms score 0 for every song, so a single unused feature
    #    (energy) decides the entire top 5.
    "Ghost preferences (genre kpop + mood ecstatic)":
        {"genre": "kpop", "mood": "ecstatic", "energy": 0.5},

    # 4. Impossible combo: metal is the highest-energy genre in the catalog, but
    #    the user asks for near-silent energy. The genre anchor (+2.0) gets
    #    overruled by the energy penalty, so non-metal calm songs win instead.
    "Impossible combo (genre metal + energy 0.05)":
        {"genre": "metal", "mood": "calm", "energy": 0.05},

    # 5. Negative energy (below 0.0). Even further out of range than #2, to show
    #    the formula never validates its input domain.
    "Negative energy (-1.0)":
        {"genre": "lofi", "mood": "chill", "energy": -1.0},

    # 6. Mood/energy fight genre: user's genre matches almost nothing useful, but
    #    mood + a perfect energy fit (1.0 + 1.5 = 2.5) can outrank a lone genre
    #    match (2.0) -- probing the documented "priority order can flip" edge.
    "Mood+energy vs genre (genre jazz, mood happy, energy 0.82)":
        {"genre": "jazz", "mood": "happy", "energy": 0.82},
}


def print_recommendations(name, user_prefs, songs):
    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 64)
    print(f"[{name}]")
    print(f"  genre={user_prefs['genre']}, mood={user_prefs['mood']}, "
          f"energy={user_prefs['energy']}")
    print("=" * 64)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Score: {score:.2f}")
        print("   Reasons:")
        for reason in explanation.split("; "):
            print(f"     - {reason}")
    print()


def main():
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, user_prefs in ADVERSARIAL_PROFILES.items():
        print_recommendations(name, user_prefs, songs)


if __name__ == "__main__":
    main()

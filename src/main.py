"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

try:
    # Works when run as a module: `python -m src.main`
    from src.recommender import load_songs, recommend_songs
except ModuleNotFoundError:
    # Works when run as a script: `python src/main.py`
    from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\n" + "=" * 60)
    print(f"Top {len(recommendations)} recommendations "
          f"for genre={user_prefs['genre']}, mood={user_prefs['mood']}, "
          f"energy={user_prefs['energy']}")
    print("=" * 60)

    for rank, rec in enumerate(recommendations, start=1):
        # Each item is (song, score, explanation).
        song, score, explanation = rec
        print(f"\n{rank}. {song['title']} - {song['artist']}")
        print(f"   Score: {score:.2f}")
        print("   Reasons:")
        for reason in explanation.split("; "):
            print(f"     - {reason}")

    print()


if __name__ == "__main__":
    main()

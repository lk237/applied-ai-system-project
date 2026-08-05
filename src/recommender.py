import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Prose blurb used by the retriever (see catalog.searchable_text). Defaults
    # to empty so existing callers and tests that build a Song without one keep
    # working, and so the rule-based scorer below is unaffected by its presence.
    description: str = ""

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Rank songs for this profile, highest score first.

        Delegates to the module-level score_song so the OOP and functional APIs
        can never disagree about what a score means. The acoustic preference is
        applied as a tie-breaker only: it nudges ordering without overruling the
        genre/mood/energy recipe documented in the README.
        """
        scored = []
        for song in self.songs:
            score, _ = score_song(self._prefs(user), self._as_dict(song))
            if user.likes_acoustic:
                score += 0.25 * song.acousticness
            scored.append((score, song))

        # Sort by score descending, then by id ascending so equal scores produce
        # a stable, reproducible order instead of depending on input ordering.
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Human-readable reason string for why this song scored as it did."""
        score, reasons = score_song(self._prefs(user), self._as_dict(song))
        if user.likes_acoustic:
            bonus = 0.25 * song.acousticness
            score += bonus
            reasons.append(f"acoustic preference (+{bonus:.2f})")

        detail = "; ".join(reasons) if reasons else "no strong matches"
        return f"{song.title} scored {score:.2f} - {detail}"

    @staticmethod
    def _prefs(user: UserProfile) -> Dict:
        """Adapt a UserProfile to the dict shape score_song expects."""
        return {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
        }

    @staticmethod
    def _as_dict(song: Song) -> Dict:
        """Adapt a Song to the dict shape score_song expects."""
        return {
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
        }

def load_songs(csv_path: str) -> List[Dict]:
    """Read the CSV at csv_path and return a list of song dicts with numeric fields cast to int/float."""
    print(f"Loading songs from {csv_path}...")

    # Columns that must be numeric so we can do math on them later.
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            song = dict(row)
            for field in int_fields:
                if field in song:
                    song[field] = int(song[field])
            for field in float_fields:
                if field in song:
                    song[field] = float(song[field])
            songs.append(song)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user_prefs (genre + mood + energy) and return (score, list-of-reasons)."""
    # Algorithm Recipe (see README.md):
    #   score = genre_points + mood_points + energy_points
    #     Genre  (categorical): exact match -> +2.0, else 0
    #     Mood   (categorical): exact match -> +1.0, else 0
    #     Energy (numerical):   1.5 * (1 - |song.energy - user.energy|)
    score = 0.0
    reasons: List[str] = []

    # Genre: exact-match, worth the most because it's the most stable taste signal.
    if song.get("genre") == user_prefs.get("genre"):
        score += 2.0
        reasons.append(f"genre match ({song['genre']}) (+2.0)")

    # Mood: exact-match, worth half of genre because it's more situational.
    if song.get("mood") == user_prefs.get("mood"):
        score += 1.0
        reasons.append(f"mood match ({song['mood']}) (+1.0)")

    # Energy: a gradient, not pass/fail. The closer the song's energy is to the
    # target, the more of its 1.5 it earns.
    energy_points = 1.5 * (1 - abs(song["energy"] - user_prefs["energy"]))
    score += energy_points
    reasons.append(
        f"energy closeness (song {song['energy']:.2f} vs target "
        f"{user_prefs['energy']:.2f}) (+{energy_points:.2f})"
    )

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song and return the top k as (song, score, explanation), highest score first."""
    # Score every song, pairing each with its (score, reasons).
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons) if reasons else "no strong matches"
        scored.append((song, score, explanation))

    # Sort highest score first. sorted() returns a NEW list and leaves the
    # input `songs` untouched; key picks the score (index 1) from each tuple.
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)

    # Return only the top k.
    return ranked[:k]

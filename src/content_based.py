# src/content_based.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

GENRE_NAMES = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


class ContentBasedRecommender:
    """
    Content-based similarity using MovieLens's real genre flags (u.item),
    plus a lightly-weighted normalized release year.

    Earlier version of this class only vectorized movie titles with TF-IDF
    ("fake genre bag for demo") which meant two movies with completely
    different genres but similar-sounding titles looked "similar", and two
    movies in the exact same genre with unrelated titles looked unrelated.
    Using the actual genre one-hot vectors is the standard, defensible
    approach for content-based filtering on MovieLens.
    """

    def __init__(self, year_weight: float = 0.15):
        self.year_weight = year_weight
        self.item_profiles = None          # (n_movies, n_features) matrix
        self.movie_id_to_idx = None
        self.similarity_matrix = None
        self.movies_df = None

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame = None):
        """Build item profiles from real genre flags + normalized year."""
        self.movies_df = movies_df.reset_index(drop=True).copy()

        missing_genres = [g for g in GENRE_NAMES if g not in self.movies_df.columns]
        if missing_genres:
            raise ValueError(
                f"movies_df is missing genre columns {missing_genres}. "
                "Re-run data/preprocess.py to regenerate movies.parquet with genres."
            )

        genre_matrix = self.movies_df[GENRE_NAMES].to_numpy(dtype=float)

        # Normalize year into [0, 1] and weight it down relative to genres,
        # so two movies from the same era get a small similarity nudge
        # without swamping the (much more informative) genre signal.
        if 'year' in self.movies_df.columns and self.movies_df['year'].notna().any():
            year = self.movies_df['year'].to_numpy(dtype=float)
            year_filled = np.nan_to_num(year, nan=np.nanmedian(year))
            y_min, y_max = year_filled.min(), year_filled.max()
            year_norm = (year_filled - y_min) / (y_max - y_min + 1e-9)
            year_feature = (year_norm * self.year_weight).reshape(-1, 1)
            self.item_profiles = np.hstack([genre_matrix, year_feature])
        else:
            self.item_profiles = genre_matrix

        self.similarity_matrix = cosine_similarity(self.item_profiles)
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movies_df['movie_id'])}

        print(f"Built content profiles for {len(self.movies_df)} movies "
              f"({genre_matrix.shape[1]} genres + year)")
        return self

    def get_similar_movies(self, movie_id: int, top_k: int = 10):
        """Return top similar movies by content"""
        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        similar = []
        for i, score in sim_scores[1:top_k + 1]:  # skip self
            similar.append({
                'movie_id': self.movies_df.iloc[i]['movie_id'],
                'title': self.movies_df.iloc[i]['title'],
                'similarity': float(score)
            })
        return similar

    def get_item_vector(self, movie_id: int):
        """Raw content feature vector for a movie (used by the hybrid model)."""
        if movie_id not in self.movie_id_to_idx:
            return None
        return self.item_profiles[self.movie_id_to_idx[movie_id]]

    def build_user_profile(self, user_id: int, ratings_df: pd.DataFrame, liked_threshold: float = 4.0):
        """
        Build a user's content taste profile as the average content vector of
        movies they rated >= liked_threshold. Returns None if the user has no
        qualifying ratings (falls back to no content signal for that user).
        """
        user_ratings = ratings_df[(ratings_df['user_id'] == user_id) &
                                   (ratings_df['rating'] >= liked_threshold)]
        return self.build_profile_from_movies(user_ratings['movie_id'].tolist())

    def build_profile_from_movies(self, movie_ids: list):
        """Average content vector over a list of movie_ids (used for cold-start too)."""
        vectors = [self.get_item_vector(mid) for mid in movie_ids]
        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def score_movie_against_profile(self, movie_id: int, profile_vector: np.ndarray):
        """Cosine similarity between a movie's content vector and a user profile vector."""
        item_vector = self.get_item_vector(movie_id)
        if item_vector is None or profile_vector is None:
            return 0.0
        denom = (np.linalg.norm(item_vector) * np.linalg.norm(profile_vector))
        if denom == 0:
            return 0.0
        return float(np.dot(item_vector, profile_vector) / denom)

    def recommend_for_new_user(self, liked_movie_ids: list, top_k: int = 10):
        """Cold-start: recommend based on liked movies"""
        if not liked_movie_ids:
            return []

        scores = np.zeros(len(self.movies_df))

        for mid in liked_movie_ids:
            if mid in self.movie_id_to_idx:
                idx = self.movie_id_to_idx[mid]
                scores += self.similarity_matrix[idx]

        # Get top movies not in liked set
        scores = pd.Series(scores, index=self.movies_df['movie_id'])
        for mid in liked_movie_ids:
            if mid in scores.index:
                scores[mid] = -np.inf  # exclude already liked

        top_ids = scores.nlargest(top_k).index.tolist()

        return [{'movie_id': mid, 'title': self.movies_df[self.movies_df['movie_id'] == mid]['title'].values[0]}
                for mid in top_ids]

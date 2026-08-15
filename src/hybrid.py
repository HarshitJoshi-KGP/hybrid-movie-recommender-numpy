# src/hybrid.py
import numpy as np
import pandas as pd
from src.model import MatrixFactorizationSGD
from src.content_based import ContentBasedRecommender, GENRE_NAMES


class HybridRecommender:
    """
    Blends the collaborative MF score with a content-based score.

    Earlier version of this class always set content_boost = 0.0 ("future
    enhancement"), so despite the confidence-weighting logic below, the
    hybrid score reduced to `weight * collab_score` -- i.e. it was never
    actually blending two signals. content_boost is now computed from
    cosine similarity between a candidate movie's genre vector and the
    user's taste profile (average genre vector of their highly-rated
    movies), rescaled onto the 1-5 rating scale so it's comparable to the
    MF collaborative score.
    """

    def __init__(self, mf_model: MatrixFactorizationSGD, content_model: ContentBasedRecommender,
                 ratings_df: pd.DataFrame, movies_path: str = "data/movies.parquet"):
        self.mf_model = mf_model
        self.content_model = content_model
        self.ratings_df = ratings_df
        self.movies_path = movies_path

        # Precompute interaction counts for confidence
        self.user_interaction_count = ratings_df.groupby('user_id').size().to_dict()
        self.item_interaction_count = ratings_df.groupby('movie_id').size().to_dict()

        # Precompute each user's content taste profile once (average genre
        # vector of movies they rated >= 4) instead of recomputing it inside
        # the per-item loop.
        self._user_profile_cache = {}

        # Load movie titles once instead of re-reading the parquet file on
        # every recommend() call.
        self._movies_df = pd.read_parquet(self.movies_path)[['movie_id', 'title']]

    def _get_user_profile(self, user_id):
        if user_id not in self._user_profile_cache:
            self._user_profile_cache[user_id] = self.content_model.build_user_profile(
                user_id, self.ratings_df
            )
        return self._user_profile_cache[user_id]

    def _get_confidence_weight(self, user_id, movie_id, alpha=0.5):
        """Higher interaction count = higher confidence in collaborative signal"""
        user_cnt = self.user_interaction_count.get(user_id, 0)
        item_cnt = self.item_interaction_count.get(movie_id, 0)

        # Normalize roughly
        user_conf = min(user_cnt / 50.0, 1.0)  # 50+ ratings = full trust
        item_conf = min(item_cnt / 100.0, 1.0)

        collab_weight = (user_conf + item_conf) / 2
        return collab_weight ** alpha  # tunable curve

    def _content_boost(self, user_id, movie_id, profile_vector):
        """
        Cosine similarity in [-1, 1] between the movie's genre vector and the
        user's taste profile, rescaled to roughly the 1-5 rating scale so it's
        on the same footing as the MF collaborative score.
        """
        if profile_vector is None:
            return None  # no content signal available for this user
        sim = self.content_model.score_movie_against_profile(movie_id, profile_vector)
        return 3.0 + 2.0 * sim  # cosine in [-1,1] -> rating-like scale [1,5]

    def explain_recommendation(self, user_id, movie_id, collab_score, content_score, weight,
                                profile_vector):
        """
        Build a human-readable explanation for one recommended movie:
        - the raw collaborative/content scores and the confidence weight
          used to blend them
        - the single latent factor that contributed most to the MF score
          (interviewable: "the model leaned heaviest on latent factor 7 for
          this pairing"), when the user/item exist in the MF model
        - which of the user's liked genres this movie actually shares, when
          a content profile is available

        This does not claim the latent factors are individually
        human-interpretable (they generally aren't) -- it surfaces which one
        dominated the dot product, which is honest about what the number
        means.
        """
        explanation = {
            'collaborative_score': round(float(collab_score), 3),
            'content_score': round(float(content_score), 3) if content_score is not None else None,
            'collaborative_weight': round(float(weight), 3),
        }

        if user_id in self.mf_model.user_map and movie_id in self.mf_model.item_map:
            u = self.mf_model.user_map[user_id]
            i = self.mf_model.item_map[movie_id]
            factor_contributions = self.mf_model.user_factors[u] * self.mf_model.item_factors[i]
            dominant_idx = int(np.argmax(np.abs(factor_contributions)))
            explanation['dominant_latent_factor'] = {
                'index': dominant_idx,
                'contribution': round(float(factor_contributions[dominant_idx]), 3),
            }

        if profile_vector is not None:
            item_vector = self.content_model.get_item_vector(movie_id)
            if item_vector is not None:
                n_genres = len(GENRE_NAMES)
                item_genres = item_vector[:n_genres]
                profile_genres = profile_vector[:n_genres]
                explanation['matching_genres'] = [
                    GENRE_NAMES[g] for g in range(n_genres)
                    if item_genres[g] > 0 and profile_genres[g] > 0
                ]

        return explanation

    def predict_rating(self, user_id, movie_id):
        """
        Point-wise hybrid rating prediction (used for RMSE/MAE evaluation).
        Falls back cleanly to pure MF when there's no usable content profile
        for this user (e.g. they have no ratings >= the liked threshold yet).
        """
        collab_score = self.mf_model.predict(user_id, movie_id)
        weight = self._get_confidence_weight(user_id, movie_id)
        profile_vector = self._get_user_profile(user_id)
        content_score = self._content_boost(user_id, movie_id, profile_vector)

        if content_score is None:
            return collab_score
        return weight * collab_score + (1 - weight) * content_score

    def recommend(self, user_id: int, top_k: int = 10, liked_movies_for_coldstart: list = None,
                  explain: bool = False):
        """Main recommendation method with hybrid cold-start.

        explain: when True, each result also carries an 'explanation' dict
        (see explain_recommendation) covering the collab/content scores,
        the blend weight, the dominant latent factor, and any shared
        genres. Off by default so normal callers don't pay for it.
        """
        # Case 1: New user (cold-start)
        if user_id not in self.mf_model.user_map:
            return self.content_model.recommend_for_new_user(liked_movies_for_coldstart or [], top_k)

        profile_vector = self._get_user_profile(user_id)

        # Get collaborative candidates
        candidates = []
        for movie_id in self.mf_model.item_map.keys():
            pred = self.mf_model.predict(user_id, movie_id)
            candidates.append((movie_id, pred))

        # Sort by predicted rating
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Hybrid scoring
        final_recs = []  # (movie_id, hybrid_score, collab_score, content_score, weight)
        for movie_id, collab_score in candidates[:top_k * 3]:  # oversample
            weight = self._get_confidence_weight(user_id, movie_id)
            content_score = self._content_boost(user_id, movie_id, profile_vector)

            if content_score is None:
                hybrid_score = collab_score
            else:
                hybrid_score = weight * collab_score + (1 - weight) * content_score

            final_recs.append((movie_id, hybrid_score, collab_score, content_score, weight))

        final_recs.sort(key=lambda x: x[1], reverse=True)
        top_recs = final_recs[:top_k]

        # Return with titles (uses the titles loaded once in __init__)
        result = []
        for mid, hybrid_score, collab_score, content_score, weight in top_recs:
            title = self._movies_df.loc[self._movies_df['movie_id'] == mid, 'title'].values[0]
            entry = {'movie_id': mid, 'title': title}
            if explain:
                entry['explanation'] = self.explain_recommendation(
                    user_id, mid, collab_score, content_score, weight, profile_vector
                )
            result.append(entry)

        return result

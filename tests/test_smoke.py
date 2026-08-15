# tests/test_smoke.py
"""
End-to-end smoke test: preprocess -> (load or train) MF model -> content
model -> hybrid recommend. Converted from the original root-level
`test.py` script into a real pytest test (assertions instead of prints),
kept in the `tests/` package like the rest of the suite.

Run with: pytest tests/test_smoke.py -v
(Requires `data/ml-100k` to be present -- run `python -m data.preprocess`
first, see README.)
"""
from pathlib import Path

import pytest

from data.preprocess import load_data
from src.model import MatrixFactorizationSGD
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender

MODEL_PATH = Path("src/mf_model.pkl")


@pytest.fixture(scope="module")
def data_bundle():
    ratings, movies = load_data()
    return ratings, movies


@pytest.fixture(scope="module")
def mf_model(data_bundle):
    ratings, _ = data_bundle
    if MODEL_PATH.exists():
        return MatrixFactorizationSGD.load(str(MODEL_PATH))
    # Small/fast config for CI -- full training config lives in train.py
    model = MatrixFactorizationSGD(n_factors=10, lr=0.01, reg=0.02, n_epochs=3)
    model.fit(ratings)
    return model


@pytest.fixture(scope="module")
def hybrid(data_bundle, mf_model):
    ratings, movies = data_bundle
    content_model = ContentBasedRecommender()
    content_model.fit(movies)
    return HybridRecommender(mf_model, content_model, ratings)


def test_data_loads(data_bundle):
    ratings, movies = data_bundle
    assert len(ratings) > 0
    assert len(movies) > 0
    assert {"user_id", "movie_id", "rating"}.issubset(ratings.columns)


def test_mf_predicts_within_rating_bounds(mf_model, data_bundle):
    ratings, _ = data_bundle
    user_id = ratings["user_id"].iloc[0]
    movie_id = ratings["movie_id"].iloc[0]
    pred = mf_model.predict(user_id, movie_id)
    assert 1.0 <= pred <= 5.0


def test_mf_cold_start_returns_global_bias(mf_model):
    # A user_id/movie_id that can't exist in MovieLens-100K
    pred = mf_model.predict(user_id=-1, movie_id=-1)
    assert pred == mf_model.global_bias


def test_hybrid_recommend_returns_top_k(hybrid, data_bundle):
    ratings, _ = data_bundle
    known_user = ratings["user_id"].iloc[0]
    recs = hybrid.recommend(user_id=known_user, top_k=5)
    assert len(recs) == 5
    assert all({"movie_id", "title"}.issubset(r.keys()) for r in recs)


def test_hybrid_cold_start_fallback(hybrid):
    # user_id 0 does not exist in MovieLens-100K (ids start at 1) -> cold start
    recs = hybrid.recommend(user_id=0, top_k=5, liked_movies_for_coldstart=[1, 2, 50])
    assert len(recs) <= 5
    assert all("movie_id" in r for r in recs)


def test_hybrid_recommend_with_explain(hybrid, data_bundle):
    ratings, _ = data_bundle
    known_user = ratings["user_id"].iloc[0]
    recs = hybrid.recommend(user_id=known_user, top_k=3, explain=True)
    assert len(recs) == 3
    for r in recs:
        assert "explanation" in r
        exp = r["explanation"]
        assert {"collaborative_score", "content_score", "collaborative_weight"}.issubset(exp.keys())
        # dominant_latent_factor should be present for a known user/item pair
        assert "dominant_latent_factor" in exp
        assert {"index", "contribution"}.issubset(exp["dominant_latent_factor"].keys())


def test_hybrid_recommend_without_explain_has_no_explanation_key(hybrid, data_bundle):
    ratings, _ = data_bundle
    known_user = ratings["user_id"].iloc[0]
    recs = hybrid.recommend(user_id=known_user, top_k=3)  # explain defaults to False
    assert all("explanation" not in r for r in recs)


def test_content_model_similar_movies(data_bundle):
    _, movies = data_bundle
    content_model = ContentBasedRecommender()
    content_model.fit(movies)
    some_movie_id = int(movies["movie_id"].iloc[0])
    similar = content_model.get_similar_movies(some_movie_id, top_k=5)
    assert len(similar) == 5
    assert all({"movie_id", "title", "similarity"}.issubset(s.keys()) for s in similar)
    # A movie should never be "similar to itself" in its own results
    assert all(s["movie_id"] != some_movie_id for s in similar)


def test_content_model_similar_movies_unknown_id_returns_empty(data_bundle):
    _, movies = data_bundle
    content_model = ContentBasedRecommender()
    content_model.fit(movies)
    assert content_model.get_similar_movies(movie_id=-999, top_k=5) == []

# src/api.py
import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))   # Add project root
import pickle
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import uvicorn
import pandas as pd

from src.model import MatrixFactorizationSGD
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender

# Global models (loaded once at startup)
mf_model = None
content_model = None
hybrid_recommender = None
movies_df = None


class TTLLRUCache:
    """Bounded in-memory cache with LRU eviction *and* TTL expiry.

    README previously advertised an "Online: FastAPI inference + caching
    pattern" but the original cache was a plain dict with oldest-inserted
    (not actually least-recently-*used*) eviction and no expiry at all --
    stale entries would be served forever. This version:
      - moves an entry to the front on every get/set (true LRU, not FIFO)
      - expires entries after `ttl_seconds` so results eventually refresh
        (e.g. after retraining and swapping mf_model.pkl)

    This is still a single-process, in-memory cache -- not a substitute for
    Redis in a multi-instance deployment (see README "Known Limitations"),
    but it is a real, correct LRU+TTL rather than a FIFO dict.
    """

    def __init__(self, max_size: int = 2048, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[Any, tuple]" = OrderedDict()

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def set(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time() + self.ttl_seconds)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self):
        return len(self._store)


_recommendation_cache = TTLLRUCache(
    max_size=int(os.environ.get("RECS_CACHE_MAX_SIZE", 2048)),
    ttl_seconds=int(os.environ.get("RECS_CACHE_TTL_SECONDS", 300)),
)


def _cache_key(user_id: int, top_k: int, liked_movie_ids: Optional[List[int]], explain: bool):
    liked_key = tuple(sorted(liked_movie_ids)) if liked_movie_ids else None
    return (user_id, top_k, liked_key, explain)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    global mf_model, content_model, hybrid_recommender, movies_df

    print("Loading models for API...")

    data_dir = Path("data")
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    movies_df = pd.read_parquet(data_dir / "movies.parquet")

    # Load MF model
    mf_path = Path("src/mf_model.pkl")
    if mf_path.exists():
        with open(mf_path, "rb") as f:
            mf_model = pickle.load(f)
    else:
        print("Warning: MF model not found. Train first (python -m src.model).")

    # Content model
    content_model = ContentBasedRecommender()
    content_model.fit(movies_df)

    # Hybrid
    if mf_model:
        hybrid_recommender = HybridRecommender(mf_model, content_model, ratings)

    print("API ready with models loaded.")
    yield
    # --- shutdown --- (nothing to clean up)


app = FastAPI(
    title="Recommendation System from Scratch",
    description="Matrix Factorization + Hybrid Cold-Start Recommender",
    version="1.0",
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10
    liked_movie_ids: Optional[List[int]] = None  # for cold-start demo
    explain: bool = False  # attach a per-item explanation (scores, weight, dominant factor, shared genres)


class Recommendation(BaseModel):
    movie_id: int
    title: str
    explanation: Optional[Dict[str, Any]] = None


class SimilarMovie(BaseModel):
    movie_id: int
    title: str
    similarity: float


class BatchRecommendRequest(BaseModel):
    requests: List[RecommendRequest]


class BatchRecommendResponseItem(BaseModel):
    user_id: int
    recommendations: List[Recommendation]
    error: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "Recommendation System API is running. Go to /docs for Swagger UI."}


def _compute_recommendations(user_id: int, request: RecommendRequest) -> List[dict]:
    """Shared cache-then-compute path used by both the single and batch
    recommend endpoints, so they can't drift out of sync."""
    key = _cache_key(user_id, request.top_k, request.liked_movie_ids, request.explain)
    cached = _recommendation_cache.get(key)
    if cached is not None:
        return cached

    recs = hybrid_recommender.recommend(
        user_id=user_id,
        top_k=request.top_k,
        liked_movies_for_coldstart=request.liked_movie_ids,
        explain=request.explain,
    )
    _recommendation_cache.set(key, recs)
    return recs


@app.post("/recommend/batch", response_model=List[BatchRecommendResponseItem])
async def recommend_batch(batch: BatchRecommendRequest):
    """Recommend for many users in a single call, instead of one HTTP
    round-trip per user. Each user's failure is isolated -- one bad
    user_id/liked_movie_ids in the batch returns an `error` for that item
    rather than failing the whole batch.

    NOTE: this route is registered *before* /recommend/{user_id} below.
    FastAPI/Starlette match routes in registration order, and {user_id} is
    typed as int -- but route matching happens before type coercion, so if
    /recommend/{user_id} were registered first it would still capture a
    request to /recommend/batch's URL and only fail once it tried (and
    failed) to parse "batch" as an int, returning a confusing 422 instead
    of reaching this handler."""
    if not hybrid_recommender:
        raise HTTPException(status_code=500, detail="Models not loaded")

    if len(batch.requests) > 200:
        raise HTTPException(status_code=400, detail="Batch size limited to 200 requests per call")

    results = []
    for req in batch.requests:
        try:
            recs = _compute_recommendations(req.user_id, req)
            results.append({"user_id": req.user_id, "recommendations": recs, "error": None})
        except Exception as e:
            results.append({"user_id": req.user_id, "recommendations": [], "error": str(e)})

    return results


@app.post("/recommend/{user_id}", response_model=List[Recommendation])
async def recommend(user_id: int, request: RecommendRequest):
    if not hybrid_recommender:
        raise HTTPException(status_code=500, detail="Models not loaded")

    try:
        return _compute_recommendations(user_id, request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/{movie_id}", response_model=List[SimilarMovie])
async def similar_movies(movie_id: int, top_k: int = 10):
    """Item-to-item content similarity (genres + year), independent of any
    user -- useful for a 'more like this' widget on a movie's own page."""
    if not content_model:
        raise HTTPException(status_code=500, detail="Content model not loaded")

    similar = content_model.get_similar_movies(movie_id, top_k)
    if not similar and movie_id not in content_model.movie_id_to_idx:
        raise HTTPException(status_code=404, detail=f"movie_id {movie_id} not found")
    return similar


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": mf_model is not None,
        "cache_size": len(_recommendation_cache),
    }


# Run with: uvicorn src.api:app --reload --port 8000
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

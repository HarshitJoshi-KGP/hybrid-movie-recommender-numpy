# tests/test_api_integration.py
"""
Integration test against a *running* FastAPI server. Converted from the
original root-level `test_api.py` script into a pytest test.

This is an integration test, not a unit test -- it needs a live server:

    uvicorn src.api:app --port 8000   # in one terminal
    pytest tests/test_api_integration.py -v   # in another

It's skipped automatically (not failed) if the server isn't reachable,
so `pytest tests/` still passes in CI without a running API.
"""
import requests
import pytest

API_URL = "http://127.0.0.1:8000"


def _api_is_up() -> bool:
    try:
        requests.get(f"{API_URL}/health", timeout=1)
        return True
    except requests.exceptions.ConnectionError:
        return False


pytestmark = pytest.mark.skipif(
    not _api_is_up(),
    reason="FastAPI server not running on 127.0.0.1:8000 (see docstring)",
)


def test_health_endpoint():
    response = requests.get(f"{API_URL}/health", timeout=5)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"


def test_recommend_endpoint_returns_top_k():
    response = requests.post(
        f"{API_URL}/recommend/196",
        json={"user_id": 196, "top_k": 10},
        timeout=10,
    )
    assert response.status_code == 200
    recs = response.json()
    assert len(recs) == 10
    assert all({"movie_id", "title"}.issubset(r.keys()) for r in recs)


def test_recommend_endpoint_with_explain():
    response = requests.post(
        f"{API_URL}/recommend/196",
        json={"user_id": 196, "top_k": 3, "explain": True},
        timeout=10,
    )
    assert response.status_code == 200
    recs = response.json()
    assert len(recs) == 3
    for r in recs:
        assert r["explanation"] is not None
        assert "collaborative_score" in r["explanation"]


def test_recommend_batch_endpoint():
    response = requests.post(
        f"{API_URL}/recommend/batch",
        json={"requests": [
            {"user_id": 196, "top_k": 5},
            {"user_id": 1, "top_k": 5},
        ]},
        timeout=10,
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {r["user_id"] for r in body} == {196, 1}
    for item in body:
        assert item["error"] is None
        assert len(item["recommendations"]) == 5


def test_similar_movies_endpoint():
    response = requests.get(f"{API_URL}/similar/1", params={"top_k": 5}, timeout=10)
    assert response.status_code == 200
    similar = response.json()
    assert len(similar) == 5
    assert all({"movie_id", "title", "similarity"}.issubset(m.keys()) for m in similar)


def test_similar_movies_endpoint_unknown_id():
    response = requests.get(f"{API_URL}/similar/999999", params={"top_k": 5}, timeout=10)
    assert response.status_code == 404

"""Tests for FastAPI endpoints and contracts."""

import pytest


def test_api_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["app_name"] == "PulseRank"
    assert "models_ready" in data
    assert data["dataset_loaded"] is True


def test_api_list_users(client):
    res = client.get("/api/users?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "user_id" in data[0]
    assert "history_count" in data[0]


def test_api_user_profile(client):
    res = client.get("/api/users/U1001/profile")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == "U1001"
    assert "category_distribution" in data
    assert "top_entities" in data
    assert "semantic_summary" in data


def test_api_feed_all_models(client):
    models = ["ranker", "hybrid", "content", "collaborative", "popularity"]
    for m in models:
        res = client.get(f"/api/feed/U1001?model={m}&limit=5")
        assert res.status_code == 200, f"Model {m} feed request failed: {res.text}"
        data = res.json()
        assert data["model_used"] == m
        assert len(data["items"]) > 0

        first = data["items"][0]
        assert "recommendation_id" in first
        assert "article_id" in first
        assert "title" in first
        assert "score" in first
        assert "sources" in first


def test_api_recommendation_explanation(client):
    # Fetch feed to get recommendation_id
    feed_res = client.get("/api/feed/U1001?model=ranker&limit=2")
    assert feed_res.status_code == 200
    items = feed_res.json()["items"]
    assert len(items) > 0
    rec_id = items[0]["recommendation_id"]

    # Query explanation
    exp_res = client.get(f"/api/recommendations/{rec_id}/explanation")
    assert exp_res.status_code == 200
    data = exp_res.json()
    assert data["recommendation_id"] == rec_id
    assert "component_scores" in data
    assert "feature_breakdown" in data
    assert "primary_reason" in data


def test_api_models_status(client):
    res = client.get("/api/models")
    assert res.status_code == 200
    models = res.json()
    assert len(models) == 5
    model_types = [m["model_type"] for m in models]
    assert "popularity" in model_types
    assert "content_based" in model_types
    assert "collaborative" in model_types
    assert "hybrid" in model_types
    assert "ranker" in model_types


def test_api_evaluations(client):
    res = client.get("/api/evaluations")
    assert res.status_code == 200
    evals = res.json()
    assert isinstance(evals, list)
    if evals:
        first = evals[0]
        assert "auc" in first
        assert "mrr" in first
        assert "ndcg_at_5" in first
        assert "ndcg_at_10" in first


def test_api_interaction_feedback(client):
    payload = {
        "user_id": "U1001",
        "article_id": "N10003",
        "interaction_type": "like",
    }
    res = client.post("/api/interactions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "recorded"
    assert data["interaction_type"] == "like"


def test_interaction_commit_failure_raises_500(client):
    """Verify transactional integrity: failure on commit must rollback and raise 500."""
    from unittest.mock import MagicMock
    from pulserank.api.app import app
    from pulserank.db.session import get_db

    mock_session = MagicMock()
    mock_session.commit.side_effect = RuntimeError("Disk full / Simulated write failure")

    app.dependency_overrides[get_db] = lambda: mock_session
    try:
        payload = {
            "user_id": "U1001",
            "article_id": "N10003",
            "interaction_type": "like",
        }
        res = client.post("/api/interactions", json=payload)
        assert res.status_code == 500
        data = res.json()
        assert "detail" in data
        assert "Database commit failed" in data["detail"]
        mock_session.rollback.assert_called_once()
    finally:
        app.dependency_overrides.pop(get_db, None)


"""Tests verifying dynamic experiment policy resolution and validation."""

import pytest


def test_experiments_policy_resolution(client):
    """Verify that experiments route dynamically scores different candidate policies."""
    # Test valid policies: popularity vs ranker
    payload_valid = {
        "experiment_id": "exp_pop_vs_ranker",
        "user_id": "U1001",
        "candidate_article_ids": ["N1001", "N1002", "N1003", "N1004"],
        "model_a": "popularity",
        "model_b": "ranker",
        "limit": 3,
    }
    res = client.post("/api/experiments/evaluate-candidates", json=payload_valid)
    assert res.status_code == 200
    data = res.json()
    assert data["assigned_variant"] in ["A", "B"]
    assert len(data["slate"]) > 0

    # Test RRF vs MMR
    payload_rrf = {
        "experiment_id": "exp_rrf_vs_mmr",
        "user_id": "U1001",
        "candidate_article_ids": ["N1001", "N1002", "N1003"],
        "model_a": "rrf",
        "model_b": "ranker_mmr",
        "limit": 2,
    }
    res2 = client.post("/api/experiments/evaluate-candidates", json=payload_rrf)
    assert res2.status_code == 200

    # Test unknown policy fails with 400
    payload_invalid = {
        "experiment_id": "exp_invalid",
        "user_id": "U1001",
        "candidate_article_ids": ["N1001"],
        "model_a": "non_existent_deep_net",
        "model_b": "popularity",
        "limit": 1,
    }
    res_bad = client.post("/api/experiments/evaluate-candidates", json=payload_invalid)
    assert res_bad.status_code == 400
    assert "Unknown policy" in res_bad.json()["detail"]

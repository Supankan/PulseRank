"""Tests for interpretable user interest profile generation."""

import math
from pulserank.ml.profiles import ProfileBuilder


def test_user_profile_generation(processed_dataset):
    builder = ProfileBuilder(processed_dataset.articles)

    history = ["N10001", "N10002", "N10011"]  # All technology articles
    profile = builder.build_profile("U_TECH", history)

    assert profile.user_id == "U_TECH"
    assert profile.history_size == 3
    assert "technology" in profile.category_distribution
    # Technology should be 1.0 since all 3 articles are in technology
    assert math.isclose(profile.category_distribution["technology"], 1.0, rel_tol=1e-3)
    assert len(profile.top_entities) > 0
    assert len(profile.recent_articles) == 3


def test_cold_start_profile(processed_dataset):
    builder = ProfileBuilder(processed_dataset.articles)
    profile = builder.build_profile("U_COLD", [])

    assert profile.user_id == "U_COLD"
    assert profile.history_size == 0
    assert profile.category_distribution == {}
    assert profile.top_entities == []

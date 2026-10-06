"""Tests for recency-aware popularity baseline recommender."""

from datetime import datetime, timezone, timedelta
from pulserank.ml.popularity import PopularityRecommender


def test_popularity_decay_ordering():
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Article A: 2 clicks, but 7 days old
    # Article B: 1 click, but 10 minutes old
    old_time = now - timedelta(days=7)
    recent_time = now - timedelta(minutes=10)

    interactions = [
        ("A", old_time),
        ("A", old_time),
        ("B", recent_time),
    ]

    model = PopularityRecommender(decay_half_life_hours=24.0)
    model.fit(interactions, reference_time=now)

    recs = model.recommend(k=2)
    assert len(recs) == 2
    # B should rank higher than A because 7 days decay (7 half-lives = 1/128) reduces 2 clicks to ~0.015
    assert recs[0][0] == "B"
    assert recs[1][0] == "A"


def test_popularity_exclude_consumed():
    interactions = [("A", None), ("B", None), ("C", None)]
    model = PopularityRecommender()
    model.fit(interactions)

    recs = model.recommend(k=5, exclude_article_ids={"A"})
    rec_ids = [r[0] for r in recs]
    assert "A" not in rec_ids
    assert "B" in rec_ids
    assert "C" in rec_ids


def test_popularity_save_load(temp_dir):
    interactions = [("A", None), ("B", None)]
    model = PopularityRecommender()
    model.fit(interactions)

    save_path = temp_dir / "pop_test.pkl"
    model.save(save_path)

    loaded = PopularityRecommender.load(save_path)
    assert loaded.get_score("A") == model.get_score("A")
    assert len(loaded.ranked_articles) == 2

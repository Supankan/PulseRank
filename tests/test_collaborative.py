"""Tests for implicit collaborative filtering recommender."""

from pulserank.ml.collaborative import ImplicitALSRecommender


def test_collaborative_fitting():
    interactions = [
        ("U1", "A1", 2.0),
        ("U1", "A2", 1.0),
        ("U2", "A2", 2.0),
        ("U2", "A3", 1.0),
        ("U3", "A1", 1.0),
        ("U3", "A3", 2.0),
    ]

    model = ImplicitALSRecommender(factors=16, iterations=5)
    model.fit(interactions)

    assert model.user_factors is not None
    assert model.item_factors is not None
    assert model.is_user_known("U1")
    assert not model.is_user_known("U999")  # Cold start user


def test_collaborative_recommend_and_exclude():
    interactions = [
        ("U1", "A1", 2.0),
        ("U1", "A2", 1.0),
        ("U2", "A2", 2.0),
        ("U2", "A3", 2.0),
    ]
    model = ImplicitALSRecommender(factors=16, iterations=5)
    model.fit(interactions)

    recs = model.recommend(user_id="U1", k=5)
    rec_ids = [r[0] for r in recs]
    # Consumed items A1, A2 must not be in recommendations
    assert "A1" not in rec_ids
    assert "A2" not in rec_ids


def test_collaborative_cold_start():
    model = ImplicitALSRecommender(factors=16, iterations=5)
    model.fit([("U1", "A1", 1.0)])

    # Unknown user returns empty list without error
    recs = model.recommend(user_id="U_UNKNOWN", k=5)
    assert recs == []

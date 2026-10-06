"""Tests for multi-source hybrid candidate generator."""

from pulserank.ml.popularity import PopularityRecommender
from pulserank.ml.content_based import ContentBasedRecommender
from pulserank.ml.collaborative import ImplicitALSRecommender
from pulserank.ml.hybrid import HybridCandidateGenerator


def test_hybrid_candidate_fusion(processed_dataset):
    pop_model = PopularityRecommender()
    pop_model.fit([("N10003", None), ("N10004", None), ("N10005", None)])

    content_model = ContentBasedRecommender(embedding_dim=64)
    content_model.build_index(processed_dataset.articles)

    collab_model = ImplicitALSRecommender(factors=16, iterations=5)
    collab_model.fit([("U1001", "N10001", 1.0), ("U1001", "N10003", 2.0)])

    hybrid = HybridCandidateGenerator(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=collab_model,
    )

    user_history = ["N10001", "N10002"]
    candidates = hybrid.generate_candidates(
        user_id="U1001",
        user_history=user_history,
        content_k=10,
        collab_k=10,
        popularity_k=10,
    )

    assert len(candidates) > 0

    # User history must be strictly excluded
    cand_ids = [c.article_id for c in candidates]
    for aid in user_history:
        assert aid not in cand_ids

    # Check that duplicates are merged and sources preserved
    for c in candidates:
        assert len(c.sources) >= 1
        for src in c.sources:
            assert src in ["collaborative", "content", "popularity"]
            assert src in c.source_scores
        # Sources list should be deterministically sorted
        assert c.sources == sorted(c.sources)


def test_hybrid_pointwise_scoring(processed_dataset):
    pop_model = PopularityRecommender()
    pop_model.fit([("N10003", None), ("N10004", None)])

    content_model = ContentBasedRecommender(embedding_dim=64)
    content_model.build_index(processed_dataset.articles)

    collab_model = ImplicitALSRecommender(factors=16, iterations=5)
    collab_model.fit([("U1001", "N10001", 1.0)])

    hybrid = HybridCandidateGenerator(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=collab_model,
    )

    candidate_ids = ["N10003", "N10004", "N10005"]
    scores = hybrid.score_candidates_pointwise(
        user_id="U1001",
        candidate_ids=candidate_ids,
        user_history=["N10001"],
    )

    assert len(scores) == 3
    for s in scores:
        assert isinstance(s, float)
        assert s >= 0.0

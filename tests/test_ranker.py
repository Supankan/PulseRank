"""Tests for LightGBM LambdaMART ranker and feature extraction."""

from pulserank.ml.ranker import LambdaMARTRanker, FeatureExtractor, FEATURE_NAMES
from pulserank.ml.hybrid import CandidateItem


def test_feature_extractor(processed_dataset):
    extractor = FeatureExtractor(
        articles=processed_dataset.articles,
        user_histories=processed_dataset.user_histories,
    )

    feat = extractor.extract_features(user_id="U1001", article_id="N10003")
    assert len(feat) == len(FEATURE_NAMES)
    # Check that user history length feature is accurately populated
    hist_len_idx = FEATURE_NAMES.index("user_history_length")
    assert feat[hist_len_idx] == float(len(processed_dataset.user_histories["U1001"]))


def test_ranker_training_and_reranking(processed_dataset):
    extractor = FeatureExtractor(
        articles=processed_dataset.articles,
        user_histories=processed_dataset.user_histories,
    )

    ranker = LambdaMARTRanker(n_estimators=10, num_leaves=15)
    ranker.fit_from_behaviors(
        behaviors=processed_dataset.train_behaviors,
        feature_extractor=extractor,
    )

    assert ranker.model is not None

    candidates = [
        CandidateItem(
            article_id="N10003",
            user_id="U1001",
            sources=["popularity"],
            source_scores={"popularity": 0.5},
            popularity_score=0.5,
            hybrid_score=0.5,
        ),
        CandidateItem(
            article_id="N10011",
            user_id="U1001",
            sources=["content"],
            source_scores={"content": 0.9},
            content_score=0.9,
            hybrid_score=0.9,
        ),
    ]

    reranked = ranker.rerank_candidates(candidates, extractor)
    assert len(reranked) == 2
    assert isinstance(reranked[0][1], float)
    # Ordered descending by predicted score
    assert reranked[0][1] >= reranked[1][1]


def test_ranker_graceful_fallback(processed_dataset):
    extractor = FeatureExtractor(
        articles=processed_dataset.articles,
        user_histories=processed_dataset.user_histories,
    )

    untrained_ranker = LambdaMARTRanker()
    candidates = [
        CandidateItem(
            article_id="A",
            user_id="U1",
            sources=["pop"],
            source_scores={},
            hybrid_score=0.3,
        ),
        CandidateItem(
            article_id="B",
            user_id="U1",
            sources=["content"],
            source_scores={},
            hybrid_score=0.8,
        ),
    ]

    reranked = untrained_ranker.rerank_candidates(candidates, extractor)
    assert reranked[0][0].article_id == "B"
    assert reranked[1][0].article_id == "A"


def test_ranker_group_alignment(processed_dataset):
    """Ensure sum(groups) strictly equals len(y) even with mixed impressions."""
    from pulserank.data.mind_parser import BehaviorRecord, ImpressionItem

    behaviors = [
        # Valid impression 1: 1 click, 2 non-clicks -> group size 3
        BehaviorRecord(1, "U1", "2026-01-01 10:00:00", ["N10001"], [
            ImpressionItem("N10002", 1),
            ImpressionItem("N10003", 0),
            ImpressionItem("N10004", 0),
        ]),
        # Invalid impression: 0 clicks -> should be skipped entirely
        BehaviorRecord(2, "U2", "2026-01-01 10:00:00", ["N10001"], [
            ImpressionItem("N10002", 0),
            ImpressionItem("N10003", 0),
        ]),
        # Valid impression 2: 2 clicks, 1 non-click -> group size 3
        BehaviorRecord(3, "U3", "2026-01-01 10:00:00", ["N10002"], [
            ImpressionItem("N10001", 1),
            ImpressionItem("N10003", 1),
            ImpressionItem("N10004", 0),
        ]),
    ]

    extractor = FeatureExtractor(
        articles=processed_dataset.articles,
        user_histories=processed_dataset.user_histories,
    )

    ranker = LambdaMARTRanker(n_estimators=5, num_leaves=10)
    ranker.fit_from_behaviors(behaviors, extractor)

    assert ranker.model is not None


def test_feature_extractor_no_leakage(processed_dataset):
    """Ensure passing user_history explicitly isolates category counts to that history."""
    extractor = FeatureExtractor(
        articles=processed_dataset.articles,
        user_histories=processed_dataset.user_histories,
    )

    # Article N10003 is sports
    target_aid = "N10003"
    target_cat = processed_dataset.articles[target_aid].category

    # If user history has 0 sports articles, affinity is 0.0
    feat_zero = extractor.extract_features("U1001", target_aid, user_history=[])
    cat_aff_idx = FEATURE_NAMES.index("category_affinity")
    assert feat_zero[cat_aff_idx] == 0.0

    # If user history has sports articles, affinity is > 0.0
    feat_with_hist = extractor.extract_features("U1001", target_aid, user_history=[target_aid])
    assert feat_with_hist[cat_aff_idx] == 1.0

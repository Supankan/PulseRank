"""Tests verifying strict train/serve mathematical parity in LTR feature extraction and scoring."""

import pytest
import numpy as np
from pulserank.data.mind_parser import NewsArticleRecord
from pulserank.ml.ranker import FeatureExtractor, compute_ltr_features, FEATURE_NAMES
from pulserank.ml.hybrid import CandidateItem


class MockPopularity:
    def get_score(self, aid: str) -> float:
        return 0.75 if aid == "A1" else 0.25

    def recommend(self, k: int = 10, exclude_article_ids=None):
        return [("A1", 0.75), ("A2", 0.25)]


class MockCollaborative:
    def get_score(self, uid: str, aid: str) -> float:
        return 0.85 if aid == "A1" else 0.15

    def recommend(self, user_id: str, k: int = 10, exclude_article_ids=None):
        return [("A1", 0.85), ("A2", 0.15)]


class MockContent:
    def compute_similarity(self, v, aid: str) -> float:
        return 0.65 if aid == "A1" else 0.10

    def get_user_vector(self, hist):
        return np.ones(8, dtype=np.float32)

    def recommend(self, history_article_ids, k: int = 10, exclude_article_ids=None):
        return [("A1", 0.65), ("A2", 0.10)]


def test_feature_extraction_exact_parity():
    """Verify that extract_features matches compute_ltr_features identically."""
    articles = {
        "A1": NewsArticleRecord(
            news_id="A1",
            category="tech",
            subcategory="ai",
            title="AI Revolution",
            abstract="Summary",
            title_entities=[{"WikidataId": "Q1"}],
        ),
        "A2": NewsArticleRecord(
            news_id="A2",
            category="sports",
            subcategory="football",
            title="Football Match",
            abstract="Match summary",
            title_entities=[{"WikidataId": "Q2"}],
        ),
    }
    user_histories = {"U1": ["A1"]}

    pop = MockPopularity()
    collab = MockCollaborative()
    content = MockContent()

    extractor = FeatureExtractor(
        articles=articles,
        user_histories=user_histories,
        content_model=content,
        collab_model=collab,
        popularity_model=pop,
    )

    # 1. Direct function call
    feat_direct = compute_ltr_features(
        user_id="U1",
        article_id="A1",
        user_history=["A1"],
        articles=articles,
        content_model=content,
        collab_model=collab,
        popularity_model=pop,
    )

    # 2. Extractor call
    feat_extractor = extractor.extract_features(
        user_id="U1",
        article_id="A1",
        user_history=["A1"],
    )

    # Assert exact float equality
    np.testing.assert_allclose(feat_direct, feat_extractor, rtol=1e-7, atol=1e-7)
    assert len(feat_direct) == len(FEATURE_NAMES) == 7


def test_candidate_item_does_not_override_canonical_features():
    """Verify that CandidateItem scores (such as RRF ranks) do not mutate ranking features."""
    articles = {
        "A1": NewsArticleRecord(
            news_id="A1",
            category="tech",
            subcategory="ai",
            title="AI Revolution",
            abstract="Summary",
        ),
    }
    user_histories = {"U1": ["A1"]}
    collab = MockCollaborative()
    pop = MockPopularity()

    extractor = FeatureExtractor(
        articles=articles,
        user_histories=user_histories,
        collab_model=collab,
        popularity_model=pop,
    )

    # Extract without candidate_item
    feat_base = extractor.extract_features("U1", "A1", user_history=["A1"])

    # Create candidate item with arbitrary distorted scores
    cand = CandidateItem(
        article_id="A1",
        user_id="U1",
        sources=["als"],
        source_scores={"als": 999.0},
        collaborative_score=999.0,
    )

    # Extract with candidate_item passed
    feat_with_cand = extractor.extract_features("U1", "A1", user_history=["A1"], candidate_item=cand)

    # Must be 100% identical; candidate_item score must NEVER overwrite features!
    np.testing.assert_allclose(feat_base, feat_with_cand, rtol=1e-7, atol=1e-7)


def test_dynamic_profile_fail_closed_when_dyn_ranker_absent():
    """
    Verify that when dedicated dyn_ranker cannot be trained/loaded, Config 7 and Config 9
    strictly fall back to non-LTR scoring and NEVER invoke baseline ranker on dynamic features.
    """
    from unittest.mock import MagicMock
    from pulserank.data.mind_parser import BehaviorRecord, ImpressionItem
    from pulserank.ml.evaluation import Evaluator
    from pulserank.ml.diversity import DiversityEngine
    from pulserank.ml.ablation import AblationFramework

    articles = {
        "A1": NewsArticleRecord(news_id="A1", category="tech", subcategory="ai", title="AI", abstract="A"),
        "A2": NewsArticleRecord(news_id="A2", category="sports", subcategory="fb", title="FB", abstract="B"),
    }
    behaviors = [
        BehaviorRecord(
            impression_id="1",
            user_id="U1",
            time="11/11/2019 9:00:00 AM",
            history=["A1"],
            impressions=[
                ImpressionItem("A1", clicked=1),
                ImpressionItem("A2", clicked=0),
            ],
        )
    ]
    evaluator = Evaluator(behaviors)
    div_engine = DiversityEngine()

    mock_ranker = MagicMock()
    mock_ranker.model = MagicMock()
    mock_ranker.model.predict.return_value = np.array([0.9, 0.1])

    pop = MockPopularity()
    collab = MockCollaborative()
    content = MockContent()

    framework = AblationFramework(
        evaluator=evaluator,
        diversity_engine=div_engine,
        popularity_model=pop,
        content_model=content,
        collaborative_model=collab,
        ranker_model=mock_ranker,
        articles=articles,
        train_behaviors=None,  # No training data -> dyn_ranker is None
    )

    results = framework.run_all_ablations(max_impressions=1)
    results_map = {r.config_name: r for r in results}

    assert "7. LTR + Dynamic Profiles" in results_map
    assert "9. Experimental Full Policy" in results_map

    r7 = results_map["7. LTR + Dynamic Profiles"]
    r9 = results_map["9. Experimental Full Policy"]

    assert r7.status == "fallback_non_ltr"
    assert r9.status == "fallback_non_ltr"
    assert r9.config_name == "9. Experimental Full Policy"
    assert "Dynamic-profile LambdaMART + MMR + principled catalog exploration" in r9.description


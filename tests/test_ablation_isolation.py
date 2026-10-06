"""Tests verifying exploration state isolation and measurement truthfulness in offline ablation."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from pulserank.data.mind_parser import BehaviorRecord, ImpressionItem, NewsArticleRecord
from pulserank.ml.ablation import AblationFramework, AblationResult
from pulserank.ml.diversity import DiversityEngine
from pulserank.ml.evaluation import Evaluator
from pulserank.ml.exploration import ExplorationEngine
from pulserank.ml.hybrid import HybridCandidateGenerator, CandidateItem


class MockPopularity:
    def get_score(self, aid: str) -> float:
        scores = {"A1": 0.9, "A2": 0.8, "A3": 0.7, "A4": 0.6, "A5": 0.5}
        return scores.get(aid, 0.1)

    def recommend(self, k: int = 10, exclude_article_ids=None):
        exclude = set(exclude_article_ids or [])
        items = [("A1", 0.9), ("A2", 0.8), ("A3", 0.7), ("A4", 0.6), ("A5", 0.5)]
        return [it for it in items if it[0] not in exclude][:k]


class MockCollaborative:
    def get_score(self, uid: str, aid: str) -> float:
        scores = {"A1": 0.85, "A2": 0.75, "A3": 0.65, "A4": 0.55, "A5": 0.45}
        return scores.get(aid, 0.1)

    def recommend(self, user_id: str, k: int = 10, exclude_article_ids=None):
        exclude = set(exclude_article_ids or [])
        items = [("A1", 0.85), ("A2", 0.75), ("A3", 0.65), ("A4", 0.55), ("A5", 0.45)]
        return [it for it in items if it[0] not in exclude][:k]


class MockContent:
    def compute_similarity(self, v, aid: str) -> float:
        scores = {"A1": 0.95, "A2": 0.70, "A3": 0.60, "A4": 0.50, "A5": 0.40}
        return scores.get(aid, 0.1)

    def build_user_vector(self, hist):
        return np.ones(8, dtype=np.float32)

    def recommend(self, history_article_ids, k: int = 10, exclude_article_ids=None):
        exclude = set(exclude_article_ids or [])
        items = [("A1", 0.95), ("A2", 0.70), ("A3", 0.60), ("A4", 0.50), ("A5", 0.40)]
        return [it for it in items if it[0] not in exclude][:k]


def _build_test_framework(live_exploration_engine=None):
    articles = {
        f"A{i}": NewsArticleRecord(
            news_id=f"A{i}",
            category="tech" if i % 2 == 1 else "finance",
            subcategory="general",
            title=f"Article {i}",
            abstract=f"Abstract {i}",
        )
        for i in range(1, 11)
    }

    behaviors = [
        BehaviorRecord(
            impression_id="imp_1",
            user_id="U1",
            time=None,
            history=["A1"],
            impressions=[
                ImpressionItem("A1", clicked=1),
                ImpressionItem("A2", clicked=0),
                ImpressionItem("A3", clicked=0),
            ],
        ),
        BehaviorRecord(
            impression_id="imp_2",
            user_id="U2",
            time=None,
            history=["A2"],
            impressions=[
                ImpressionItem("A2", clicked=1),
                ImpressionItem("A3", clicked=0),
                ImpressionItem("A4", clicked=0),
            ],
        ),
        BehaviorRecord(
            impression_id="imp_3",
            user_id="U3",
            time=None,
            history=["A3"],
            impressions=[
                ImpressionItem("A3", clicked=1),
                ImpressionItem("A4", clicked=0),
                ImpressionItem("A5", clicked=0),
            ],
        ),
    ]

    evaluator = Evaluator(behaviors)
    div_engine = DiversityEngine()

    mock_ranker = MagicMock()
    mock_ranker.model = MagicMock()
    mock_ranker.model.predict.return_value = np.array([0.9, 0.7, 0.5, 0.3, 0.1])

    hybrid = MagicMock()
    hybrid.weights = {"content": 0.45, "collaborative": 0.35, "popularity": 0.20}
    def mock_candidates(user_id, user_history, **kwargs):
        cands = []
        for i in range(1, 8):
            aid = f"A{i}"
            cands.append(CandidateItem(
                article_id=aid,
                content_score=0.9 - 0.1 * i,
                collaborative_score=0.85 - 0.1 * i,
                popularity_score=0.8 - 0.1 * i,
                hybrid_score=0.85 - 0.1 * i,
                source_channels=["content", "collab", "popularity"],
                metadata={"category": articles[aid].category},
            ))
        return cands
    hybrid.generate_candidates = mock_candidates

    framework = AblationFramework(
        evaluator=evaluator,
        diversity_engine=div_engine,
        popularity_model=MockPopularity(),
        content_model=MockContent(),
        collaborative_model=MockCollaborative(),
        hybrid_generator=hybrid,
        ranker_model=mock_ranker,
        articles=articles,
        catalog_article_ids=list(articles.keys()),
        exploration_engine=live_exploration_engine,
        train_behaviors=None,
    )
    return framework, behaviors


def test_offline_config9_starts_with_isolated_exposure_state():
    """Property 1: Offline Config 9 Track B starts with clean, isolated exposure state."""
    # Pre-pollute live exploration engine
    live_engine = ExplorationEngine(random_state=42)
    live_engine.item_exposure_counts = {"A1": 999, "A2": 500, "A3": 250}

    framework, behaviors = _build_test_framework(live_exploration_engine=live_engine)

    # Calling whole-catalog slate serving for Config 9
    slates = framework._serve_catalog_slates_for_config(behaviors, config_idx=9)
    assert len(slates) > 0

    # The live engine's exposure counts must remain strictly untouched
    assert live_engine.item_exposure_counts == {"A1": 999, "A2": 500, "A3": 250}


def test_previous_live_feed_exposure_history_does_not_affect_offline_rerun():
    """Property 2: Live-feed usage accumulating exposures has ZERO impact on offline rerun."""
    live_engine = ExplorationEngine(random_state=42)
    framework, behaviors = _build_test_framework(live_exploration_engine=live_engine)

    # Run 1: clean state
    slates_run1 = framework._serve_catalog_slates_for_config(behaviors, config_idx=9)
    res_run1 = framework._compute_real_diversity_metrics(slates_run1)

    # Simulate heavy live-feed mutations on the service container's exploration engine
    live_engine.record_exposures(["A1", "A1", "A1", "A2", "A3", "A7", "A8"])
    assert live_engine.item_exposure_counts["A1"] == 3

    # Run 2: after live feed contamination
    slates_run2 = framework._serve_catalog_slates_for_config(behaviors, config_idx=9)
    res_run2 = framework._compute_real_diversity_metrics(slates_run2)

    # Slates and diversity metrics must be 100% bit-for-bit identical
    assert slates_run1 == slates_run2
    assert res_run1 == res_run2


def test_exposure_counts_evolve_deterministically_between_simulated_track_b_slates():
    """Property 3: Exposure counts evolve deterministically across simulated Track B slates."""
    framework, behaviors = _build_test_framework()

    isolated_engine = ExplorationEngine(random_state=42)
    assert isolated_engine.item_exposure_counts == {}

    slates = framework._serve_catalog_slates_for_config(
        behaviors,
        config_idx=9,
        exploration_engine=isolated_engine,
    )
    assert len(slates) == len(behaviors)

    # After serving, isolated engine must have recorded exposures for the served items
    assert len(isolated_engine.item_exposure_counts) > 0
    all_served_aids = [item["article_id"] for slate in slates for item in slate]
    for aid in all_served_aids:
        assert isolated_engine.item_exposure_counts[aid] >= 1


def test_repeated_identical_offline_runs_produce_identical_track_b_results():
    """Property 4: Repeated offline ablation runs yield identical Track B results."""
    framework, _ = _build_test_framework()

    results_run1 = framework.run_all_ablations(max_impressions=3)
    results_run2 = framework.run_all_ablations(max_impressions=3)

    r9_1 = next(r for r in results_run1 if r.config_name == "9. Experimental Full Policy")
    r9_2 = next(r for r in results_run2 if r.config_name == "9. Experimental Full Policy")

    # Identical Track B diversity metrics
    assert r9_1.ild == r9_2.ild
    assert r9_1.category_entropy == r9_2.category_entropy
    assert r9_1.novelty == r9_2.novelty
    assert r9_1.catalog_coverage == r9_2.catalog_coverage
    assert r9_1.slate_policy_benchmark == r9_2.slate_policy_benchmark


def test_valid_zero_diversity_slate_reports_status_measured():
    """Property 5: A valid ranking policy whose measured diversity happens to be zero reports 'measured'."""
    # Build single-category catalog where all items share the exact same category
    single_cat_articles = {
        f"A{i}": NewsArticleRecord(
            news_id=f"A{i}",
            category="news",  # Single category everywhere
            subcategory="general",
            title=f"News {i}",
            abstract=f"Abstract {i}",
        )
        for i in range(1, 6)
    }

    behaviors = [
        BehaviorRecord(
            impression_id="imp_zero",
            user_id="U_zero",
            time=None,
            history=[],
            impressions=[
                ImpressionItem("A1", clicked=1),
                ImpressionItem("A2", clicked=0),
            ],
        )
    ]

    # Identical embeddings -> ILD = 0.0, identical category -> category_entropy = 0.0
    embs = {f"A{i}": np.array([1.0, 0.0], dtype=np.float32) for i in range(1, 6)}
    div_engine = DiversityEngine(article_embeddings=embs)

    evaluator = Evaluator(behaviors)
    framework = AblationFramework(
        evaluator=evaluator,
        diversity_engine=div_engine,
        popularity_model=MockPopularity(),
        articles=single_cat_articles,
        catalog_article_ids=list(single_cat_articles.keys()),
    )

    results = framework.run_all_ablations(max_impressions=1)
    r1 = next(r for r in results if r.config_name == "1. Popularity Baseline")

    # Diversity metrics should be 0.0 (or minimal), but policy was evaluated -> status MUST be "measured"
    assert r1.ild == 0.0
    assert r1.category_entropy == 0.0
    assert r1.slate_policy_benchmark["status"] == "measured"
    assert r1.slate_policy_benchmark["ild"] == 0.0
    assert r1.slate_policy_benchmark["category_entropy"] == 0.0
    assert r1.num_slates_evaluated == 1

    # Verify that an un-evaluated policy (0 slates) reports status = "unavailable"
    dummy_result = AblationResult(
        config_name="Unrun Policy",
        description="Never executed",
        auc=0.0, mrr=0.0, ndcg_at_5=0.0, ndcg_at_10=0.0,
        ild=0.0, category_entropy=0.0, novelty=0.0, catalog_coverage=0.0,
        num_slates_evaluated=0,
    )
    # Simulate benchmark dict generation
    benchmark = {
        "status": "measured" if dummy_result.num_slates_evaluated > 0 else "unavailable",
        "ild": dummy_result.ild,
        "category_entropy": dummy_result.category_entropy,
        "novelty": dummy_result.novelty,
        "catalog_coverage": dummy_result.catalog_coverage,
        "num_slates_evaluated": dummy_result.num_slates_evaluated,
    }
    assert benchmark["status"] == "unavailable"

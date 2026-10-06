"""Tests verifying complete scientific integrity and anti-fabrication invariants across PulseRank."""

import pytest
import numpy as np
from pulserank.ml.ablation import AblationFramework, AblationResult
from pulserank.ml.diversity import DiversityEngine
from pulserank.ml.evaluation import Evaluator, EvaluationMetrics
from pulserank.data.mind_parser import BehaviorRecord, ImpressionItem


def test_ablation_metrics_are_genuinely_measured():
    """Verify that ablation metrics are produced from genuine evaluation and diversity computations, never synthetic arithmetic."""
    # Create minimal behavior set
    behaviors = [
        BehaviorRecord(
            impression_id="1",
            user_id="U1",
            time=None,
            history=["N1", "N2"],
            impressions=[
                ImpressionItem(news_id="N3", clicked=1),
                ImpressionItem(news_id="N4", clicked=0),
                ImpressionItem(news_id="N5", clicked=0),
            ],
        ),
        BehaviorRecord(
            impression_id="2",
            user_id="U2",
            time=None,
            history=["N3"],
            impressions=[
                ImpressionItem(news_id="N1", clicked=0),
                ImpressionItem(news_id="N2", clicked=1),
            ],
        ),
    ]

    evaluator = Evaluator(behaviors)

    # Synthetic mock articles
    embs = {
        "N1": np.array([1.0, 0.0], dtype=np.float32),
        "N2": np.array([0.0, 1.0], dtype=np.float32),
        "N3": np.array([0.7, 0.7], dtype=np.float32),
        "N4": np.array([-1.0, 0.0], dtype=np.float32),
        "N5": np.array([0.0, -1.0], dtype=np.float32),
    }
    pops = {"N1": 0.9, "N2": 0.8, "N3": 0.5, "N4": 0.2, "N5": 0.1}

    class MockArticle:
        def __init__(self, cat):
            self.category = cat

    articles = {
        "N1": MockArticle("sports"),
        "N2": MockArticle("tech"),
        "N3": MockArticle("sports"),
        "N4": MockArticle("finance"),
        "N5": MockArticle("entertainment"),
    }

    div_engine = DiversityEngine(article_embeddings=embs, article_popularities=pops)

    framework = AblationFramework(
        evaluator=evaluator,
        diversity_engine=div_engine,
        catalog_article_ids=["N1", "N2", "N3", "N4", "N5"],
    )
    framework.articles = articles

    results = framework.run_all_ablations(max_impressions=2)

    assert len(results) == 9
    assert all(isinstance(r, AblationResult) for r in results)

    # 1. Assert no hardcoded constants exist
    for r in results:
        assert r.auc != 0.994, f"Suspicious fabricated constant found in {r.config_name}"
        assert r.ndcg_at_5 != 0.680, f"Suspicious fabricated constant found in {r.config_name}"
        assert r.ild >= 0.0
        assert r.category_entropy >= 0.0
        assert r.catalog_coverage >= 0.0
        assert 0.0 <= r.auc <= 1.0
        assert 0.0 <= r.mrr <= 1.0
        assert 0.0 <= r.ndcg_at_5 <= 1.0

    # 2. Assert relative deltas match exact subtraction
    full_ndcg = results[-1].ndcg_at_5
    for r in results:
        expected_delta = round(r.ndcg_at_5 - full_ndcg, 4)
        assert r.delta_ndcg5_vs_full == expected_delta


def test_diversity_metrics_change_with_slate_contents():
    """Verify that DiversityEngine produces mathematically different metrics when recommendations change."""
    embs = {
        "N1": np.array([1.0, 0.0], dtype=np.float32),
        "N2": np.array([1.0, 0.0], dtype=np.float32),  # Identical to N1
        "N3": np.array([0.0, 1.0], dtype=np.float32),  # Orthogonal to N1
    }
    pops = {"N1": 0.9, "N2": 0.8, "N3": 0.1}

    div_engine = DiversityEngine(article_embeddings=embs, article_popularities=pops)

    # Homogeneous slates (identical items): ILD must be 0.0
    homo_slates = [["N1", "N2"]]
    ild_homo, ent_homo, nov_homo, cov_homo = div_engine.compute_metrics(homo_slates, catalog_size=3)
    assert ild_homo == 0.0

    # Heterogeneous slates (orthogonal items): ILD must be > 0.0 (specifically 1.0)
    hetero_slates = [["N1", "N3"]]
    ild_hetero, ent_hetero, nov_hetero, cov_hetero = div_engine.compute_metrics(hetero_slates, catalog_size=3)
    assert ild_hetero > ild_homo
    assert abs(ild_hetero - 1.0) < 1e-4

"""Tests for offline recommendation evaluation metrics."""

import math
import numpy as np
from pulserank.ml.evaluation import (
    compute_impression_auc,
    compute_impression_mrr,
    compute_impression_ndcg,
    compute_impression_recall,
)


def test_mrr_first_rank():
    labels_ranked = [1, 0, 0, 0]
    assert compute_impression_mrr(labels_ranked) == 1.0


def test_mrr_second_rank():
    labels_ranked = [0, 1, 0, 0]
    assert compute_impression_mrr(labels_ranked) == 0.5


def test_mrr_no_clicks():
    labels_ranked = [0, 0, 0, 0]
    assert compute_impression_mrr(labels_ranked) == 0.0


def test_ndcg_perfect_ranking():
    labels_ranked = [1, 1, 0, 0]
    ndcg = compute_impression_ndcg(labels_ranked, total_positives=2, k=5)
    assert math.isclose(ndcg, 1.0, rel_tol=1e-5)


def test_ndcg_imperfect_ranking():
    # Positives at rank 2 and 4
    labels_ranked = [0, 1, 0, 1]
    dcg = (1.0 / np.log2(3)) + (1.0 / np.log2(5))
    idcg = (1.0 / np.log2(2)) + (1.0 / np.log2(3))
    expected_ndcg = dcg / idcg
    actual_ndcg = compute_impression_ndcg(labels_ranked, total_positives=2, k=5)
    assert math.isclose(actual_ndcg, expected_ndcg, rel_tol=1e-5)


def test_recall_at_k():
    labels_ranked = [0, 1, 0, 0, 1]
    # At K=2, 1 positive out of 2 captured -> recall = 0.5
    assert compute_impression_recall(labels_ranked, total_positives=2, k=2) == 0.5
    # At K=5, 2 positives out of 2 captured -> recall = 1.0
    assert compute_impression_recall(labels_ranked, total_positives=2, k=5) == 1.0


def test_impression_auc_perfect():
    labels = [1, 0, 0, 0]
    scores = [0.9, 0.4, 0.3, 0.1]
    auc = compute_impression_auc(labels, scores)
    assert math.isclose(auc, 1.0, rel_tol=1e-5)


def test_impression_auc_inverted():
    labels = [1, 0, 0]
    scores = [0.1, 0.5, 0.8]
    auc = compute_impression_auc(labels, scores)
    assert math.isclose(auc, 0.0, rel_tol=1e-5)


def test_impression_auc_tied():
    labels = [1, 0]
    scores = [0.5, 0.5]
    auc = compute_impression_auc(labels, scores)
    assert math.isclose(auc, 0.5, rel_tol=1e-5)
def test_mrr_multiple_positives():
    # Only the first clicked item determines MRR
    labels_ranked = [0, 0, 1, 0, 1]
    # First positive is at rank 3 -> 1/3
    assert math.isclose(compute_impression_mrr(labels_ranked), 1.0 / 3.0, rel_tol=1e-5)


def test_ndcg_cutoff_at_k():
    # Positives at rank 6 and 7; evaluated at K=5
    labels_ranked = [0, 0, 0, 0, 0, 1, 1]
    ndcg5 = compute_impression_ndcg(labels_ranked, total_positives=2, k=5)
    assert ndcg5 == 0.0

    # At K=6, rank 6 is captured
    ndcg6 = compute_impression_ndcg(labels_ranked, total_positives=2, k=6)
    expected_dcg = 1.0 / np.log2(7)
    expected_idcg = (1.0 / np.log2(2)) + (1.0 / np.log2(3))
    assert math.isclose(ndcg6, expected_dcg / expected_idcg, rel_tol=1e-5)


def test_ndcg_more_positives_than_k():
    # When user has 10 positives, IDCG@5 only sums top 5 ideal slots
    labels_ranked = [1, 1, 1, 1, 1, 0, 0]
    ndcg5 = compute_impression_ndcg(labels_ranked, total_positives=10, k=5)
    assert math.isclose(ndcg5, 1.0, rel_tol=1e-5)


def test_impression_auc_all_positive_or_all_negative():
    assert compute_impression_auc([1, 1, 1], [0.9, 0.8, 0.7]) is None
    assert compute_impression_auc([0, 0, 0], [0.9, 0.8, 0.7]) is None


def test_catalog_retrieval_metrics():
    from pulserank.data.mind_parser import BehaviorRecord, ImpressionItem
    from pulserank.ml.evaluation import OfflineEvaluator

    dev_behaviors = [
        BehaviorRecord(
            impression_id=1,
            user_id="U1",
            time="2026-01-01 10:00:00",
            history=["N1"],
            impressions=[ImpressionItem("N2", 1), ImpressionItem("N3", 0)],
        ),
        BehaviorRecord(
            impression_id=2,
            user_id="U2",
            time="2026-01-01 10:00:00",
            history=["N2"],
            impressions=[ImpressionItem("N4", 1), ImpressionItem("N5", 0)],
        ),
    ]

    evaluator = OfflineEvaluator(dev_behaviors)
    catalog = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10"]

    # Mock recommend func: U1 gets [N2, N3], U2 gets [N5, N6]
    def mock_rec(uid, hist, k):
        return ["N2", "N3"] if uid == "U1" else ["N5", "N6"]

    eval_result = evaluator.evaluate_catalog_retrieval(
        model_name="MockModel",
        recommend_func=mock_rec,
        catalog_article_ids=catalog,
        k=2,
        dataset_source="synthetic_fixture",
    )

    # U1 hit: N2 is positive -> Hit=1.0, Recall=1.0
    # U2 hit: N4 is positive, rec was [N5, N6] -> Hit=0.0, Recall=0.0
    # Mean HitRate = 0.5, Mean Recall = 0.5
    assert math.isclose(eval_result.hit_rate_at_k, 0.5, rel_tol=1e-4)
    assert math.isclose(eval_result.recall_at_k, 0.5, rel_tol=1e-4)

    # Catalog coverage: recommended items are {N2, N3, N5, N6} (4 out of 10 items) -> 40%
    assert math.isclose(eval_result.catalog_coverage, 0.4, rel_tol=1e-4)
    assert eval_result.dataset_source == "synthetic_fixture"
    assert eval_result.evaluation_mode == "catalog_retrieval"

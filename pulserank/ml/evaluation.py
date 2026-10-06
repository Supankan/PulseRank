"""Offline evaluation framework computing impression-level AUC, MRR, nDCG@K, Recall@K, and Coverage."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from sklearn.metrics import roc_auc_score

from pulserank.data.mind_parser import BehaviorRecord

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    model_name: str
    auc: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    recall_at_k: float  # K=5 default for impression recall
    coverage: float  # Backwards compatibility: = impression_coverage in impression mode, catalog_coverage in retrieval mode
    num_impressions: int
    num_scored_impressions: int
    evaluated_at: str
    dataset_source: str = "mind-small"  # "mind-small" vs "synthetic_fixture"
    evaluation_mode: str = "impression"  # "impression" vs "catalog_retrieval"
    impression_coverage: float = 1.0  # fraction of valid dev impressions scored
    catalog_coverage: Optional[float] = None  # fraction of unique catalog items recommended across users
    hit_rate_at_k: Optional[float] = None  # catalog-level hit rate at K


def compute_impression_dcg(labels: List[int], k: int) -> float:
    """Discounted cumulative gain at k for binary labels."""
    dcg = 0.0
    for idx, label in enumerate(labels[:k]):
        if label > 0:
            dcg += label / np.log2(idx + 2)
    return dcg


def compute_impression_ndcg(labels_ranked: List[int], total_positives: int, k: int) -> float:
    """Normalized discounted cumulative gain at k."""
    if total_positives == 0:
        return 0.0
    dcg = compute_impression_dcg(labels_ranked, k)
    ideal_labels = [1] * min(k, total_positives)
    idcg = compute_impression_dcg(ideal_labels, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_impression_mrr(labels_ranked: List[int]) -> float:
    """Mean reciprocal rank of first clicked item."""
    for idx, label in enumerate(labels_ranked):
        if label == 1:
            return 1.0 / (idx + 1)
    return 0.0


def compute_impression_recall(labels_ranked: List[int], total_positives: int, k: int) -> float:
    """Recall@K: fraction of total positives captured in top-K."""
    if total_positives == 0:
        return 0.0
    positives_in_top_k = sum(labels_ranked[:k])
    return positives_in_top_k / total_positives


def compute_impression_auc(labels: List[int], scores: List[float]) -> Optional[float]:
    """ROC AUC for a single impression list."""
    y = np.array(labels)
    s = np.array(scores)
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    if pos_count == 0 or neg_count == 0:
        return None
    # If all scores are identical, AUC is 0.5
    if np.all(s == s[0]):
        return 0.5
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return 0.5


class OfflineEvaluator:
    """Evaluates candidate scoring models against validation behavior logs."""

    def __init__(self, dev_behaviors: List[BehaviorRecord]):
        self.dev_behaviors = dev_behaviors

    def evaluate_model(
        self,
        model_name: str,
        scoring_func: Any,
        k: int = 5,
        dataset_source: str = "mind-small",
    ) -> EvaluationMetrics:
        """
        Evaluate scoring_func(user_id: str, candidate_news_ids: List[str], [user_history]) -> List[float].
        Strictly respects MIND impression grouping without future interaction leakage.
        """
        auc_list = []
        mrr_list = []
        ndcg5_list = []
        ndcg10_list = []
        recall_list = []
        scored_impressions = 0
        total_impressions = len(self.dev_behaviors)

        for b in self.dev_behaviors:
            if not b.impressions:
                continue

            candidate_ids = [item.news_id for item in b.impressions]
            labels = [item.clicked for item in b.impressions]
            num_pos = sum(labels)
            if num_pos == 0:
                continue

            try:
                try:
                    scores = scoring_func(b.user_id, candidate_ids, user_history=b.history)
                except TypeError:
                    scores = scoring_func(b.user_id, candidate_ids)

                if len(scores) != len(candidate_ids):
                    continue
                scored_impressions += 1
            except Exception as e:
                logger.debug(f"Scoring error for user {b.user_id} in {model_name}: {e}")
                continue

            # Compute AUC
            auc = compute_impression_auc(labels, scores)
            if auc is not None:
                auc_list.append(auc)

            # Rank candidates descending by score
            sorted_indices = np.argsort(-np.array(scores))
            ranked_labels = [labels[idx] for idx in sorted_indices]

            # MRR
            mrr_list.append(compute_impression_mrr(ranked_labels))

            # nDCG@5 & nDCG@10
            ndcg5_list.append(compute_impression_ndcg(ranked_labels, num_pos, 5))
            ndcg10_list.append(compute_impression_ndcg(ranked_labels, num_pos, 10))

            # Recall@K
            recall_list.append(compute_impression_recall(ranked_labels, num_pos, k))

        impression_cov = scored_impressions / total_impressions if total_impressions > 0 else 0.0

        return EvaluationMetrics(
            model_name=model_name,
            auc=round(float(np.mean(auc_list)) if auc_list else 0.5, 4),
            mrr=round(float(np.mean(mrr_list)) if mrr_list else 0.0, 4),
            ndcg_at_5=round(float(np.mean(ndcg5_list)) if ndcg5_list else 0.0, 4),
            ndcg_at_10=round(float(np.mean(ndcg10_list)) if ndcg10_list else 0.0, 4),
            recall_at_k=round(float(np.mean(recall_list)) if recall_list else 0.0, 4),
            coverage=round(impression_cov, 4),
            num_impressions=total_impressions,
            num_scored_impressions=scored_impressions,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            dataset_source=dataset_source,
            evaluation_mode="impression",
            impression_coverage=round(impression_cov, 4),
            catalog_coverage=None,
            hit_rate_at_k=None,
        )

    def evaluate_all(
        self,
        popularity_model: Any,
        content_model: Any,
        collaborative_model: Any,
        hybrid_generator: Any,
        ranker_model: Any,
        feature_extractor: Any,
        dataset_source: str = "mind-small",
    ) -> List[EvaluationMetrics]:
        """
        Evaluate and compare all 5 pipeline models on MIND impression benchmark:
        1. Popularity Baseline
        2. Content-Based
        3. Collaborative Filtering
        4. Hybrid
        5. Hybrid + LightGBM LambdaMART Ranker
        """
        results: List[EvaluationMetrics] = []

        # 1. Popularity Baseline
        def score_popularity(user_id: str, candidate_ids: List[str], user_history: Optional[List[str]] = None) -> List[float]:
            return [popularity_model.get_score(aid) for aid in candidate_ids]

        results.append(self.evaluate_model("Popularity Baseline", score_popularity, dataset_source=dataset_source))

        # 2. Content-Based Model
        def score_content(user_id: str, candidate_ids: List[str], user_history: Optional[List[str]] = None) -> List[float]:
            history = user_history if user_history is not None else feature_extractor.user_histories.get(user_id, [])
            if history and hasattr(content_model, "build_user_vector"):
                u_vec = content_model.build_user_vector(history)
            else:
                u_vec = feature_extractor.user_vectors.get(user_id)
            if u_vec is None:
                return [0.0] * len(candidate_ids)
            return [content_model.compute_similarity(u_vec, aid) for aid in candidate_ids]

        results.append(self.evaluate_model("Content-Based (Sentence-Transformers)", score_content, dataset_source=dataset_source))

        # 3. Collaborative Filtering
        def score_collab(user_id: str, candidate_ids: List[str], user_history: Optional[List[str]] = None) -> List[float]:
            return [collaborative_model.get_score(user_id, aid) for aid in candidate_ids]

        results.append(self.evaluate_model("Collaborative Filtering (Implicit ALS)", score_collab, dataset_source=dataset_source))

        # 4. Hybrid System (Pointwise ensemble to prevent whole-catalog truncation)
        def score_hybrid(user_id: str, candidate_ids: List[str], user_history: Optional[List[str]] = None) -> List[float]:
            history = user_history if user_history is not None else feature_extractor.user_histories.get(user_id, [])
            return hybrid_generator.score_candidates_pointwise(
                user_id=user_id,
                candidate_ids=candidate_ids,
                user_history=history,
            )

        results.append(self.evaluate_model("Hybrid Pointwise Ensemble", score_hybrid, dataset_source=dataset_source))

        # 5. Hybrid + LightGBM Ranker
        def score_ranker(user_id: str, candidate_ids: List[str], user_history: Optional[List[str]] = None) -> List[float]:
            if ranker_model.model is None:
                return score_hybrid(user_id, candidate_ids, user_history=user_history)
            history = user_history if user_history is not None else feature_extractor.user_histories.get(user_id, [])
            feats = [
                feature_extractor.extract_features(user_id, aid, user_history=history)
                for aid in candidate_ids
            ]
            X = np.vstack(feats)
            preds = ranker_model.model.predict(X)
            return [float(p) for p in preds]

        results.append(self.evaluate_model("Hybrid + LightGBM LambdaMART", score_ranker, dataset_source=dataset_source))

        return results

    def evaluate_catalog_retrieval(
        self,
        model_name: str,
        recommend_func: Any,
        catalog_article_ids: List[str],
        k: int = 10,
        dataset_source: str = "mind-small",
    ) -> EvaluationMetrics:
        """
        Evaluate full-catalog recommendation per impression record:
        recommend_func(user_id: str, user_history: List[str], top_k: int) -> List[str]
        Computes HitRate@K, Recall@K, and Catalog Coverage (|unique recommended| / |catalog|).
        Ground truth positives are the specific impression's clicked items, preserving temporal sequence
        without future session contamination.
        """
        valid_behaviors = [
            b for b in self.dev_behaviors
            if any(item.clicked == 1 for item in b.impressions)
        ]

        if not valid_behaviors or not catalog_article_ids:
            return EvaluationMetrics(
                model_name=model_name,
                auc=0.0,
                mrr=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                recall_at_k=0.0,
                coverage=0.0,
                num_impressions=len(self.dev_behaviors),
                num_scored_impressions=0,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
                dataset_source=dataset_source,
                evaluation_mode="catalog_retrieval",
                impression_coverage=0.0,
                catalog_coverage=0.0,
                hit_rate_at_k=0.0,
            )

        hits_list = []
        recall_list = []
        all_recommended_ids: Set[str] = set()
        scored_impressions = 0
        catalog_set = set(catalog_article_ids)

        for b in valid_behaviors:
            clicks = {item.news_id for item in b.impressions if item.clicked == 1}
            try:
                rec_ids = recommend_func(b.user_id, b.history, k)
                if not rec_ids:
                    continue
                rec_top = rec_ids[:k]
                scored_impressions += 1
                all_recommended_ids.update(rec_top)

                hits = set(rec_top) & clicks
                hits_list.append(1.0 if hits else 0.0)
                recall_list.append(len(hits) / len(clicks))
            except Exception as e:
                logger.debug(f"Catalog retrieval scoring error for user {b.user_id} in {model_name}: {e}")
                continue

        hit_rate = float(np.mean(hits_list)) if hits_list else 0.0
        mean_recall = float(np.mean(recall_list)) if recall_list else 0.0
        catalog_cov = len(all_recommended_ids & catalog_set) / len(catalog_set) if catalog_set else 0.0
        imp_cov = scored_impressions / len(valid_behaviors) if valid_behaviors else 0.0

        return EvaluationMetrics(
            model_name=model_name,
            auc=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            ndcg_at_10=0.0,
            recall_at_k=round(mean_recall, 4),
            coverage=round(catalog_cov, 4),
            num_impressions=len(valid_behaviors),
            num_scored_impressions=scored_impressions,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            dataset_source=dataset_source,
            evaluation_mode="catalog_retrieval",
            impression_coverage=round(imp_cov, 4),
            catalog_coverage=round(catalog_cov, 4),
            hit_rate_at_k=round(hit_rate, 4),
        )

    def evaluate_catalog_retrieval_all(
        self,
        popularity_model: Any,
        content_model: Any,
        collaborative_model: Any,
        hybrid_generator: Any,
        ranker_model: Any,
        feature_extractor: Any,
        catalog_article_ids: List[str],
        k: int = 10,
        dataset_source: str = "mind-small",
    ) -> List[EvaluationMetrics]:
        """
        Evaluate full-catalog candidate retrieval across all 5 models.
        """
        results: List[EvaluationMetrics] = []

        # 1. Popularity: Top-K globally popular items
        def rec_pop(user_id: str, history: List[str], top_k: int) -> List[str]:
            return [aid for aid, _ in popularity_model.recommend(k=top_k)]

        results.append(self.evaluate_catalog_retrieval("Popularity Baseline", rec_pop, catalog_article_ids, k=k, dataset_source=dataset_source))

        # 2. Content-Based: Top-K content neighbors
        def rec_content(user_id: str, history: List[str], top_k: int) -> List[str]:
            return [aid for aid, _ in content_model.recommend(history_article_ids=history, k=top_k)]

        results.append(self.evaluate_catalog_retrieval("Content-Based (Sentence-Transformers)", rec_content, catalog_article_ids, k=k, dataset_source=dataset_source))

        # 3. Collaborative Filtering: Top-K ALS items
        def rec_collab(user_id: str, history: List[str], top_k: int) -> List[str]:
            return [aid for aid, _ in collaborative_model.recommend(user_id=user_id, k=top_k)]

        results.append(self.evaluate_catalog_retrieval("Collaborative Filtering (Implicit ALS)", rec_collab, catalog_article_ids, k=k, dataset_source=dataset_source))

        # 4. Hybrid (RRF Candidate Fusion): Top-K fused candidates
        def rec_hybrid(user_id: str, history: List[str], top_k: int) -> List[str]:
            cands = hybrid_generator.generate_candidates(
                user_id=user_id,
                user_history=history,
                content_k=top_k * 2,
                collab_k=top_k * 2,
                popularity_k=top_k * 2,
            )
            return [c.article_id for c in cands[:top_k]]

        results.append(self.evaluate_catalog_retrieval("Hybrid Candidate Fusion (RRF)", rec_hybrid, catalog_article_ids, k=k, dataset_source=dataset_source))

        # 5. Hybrid + Ranker: Top-K re-ranked candidates
        def rec_ranker(user_id: str, history: List[str], top_k: int) -> List[str]:
            cands = hybrid_generator.generate_candidates(
                user_id=user_id,
                user_history=history,
                content_k=top_k * 2,
                collab_k=top_k * 2,
                popularity_k=top_k * 2,
            )
            if not cands or ranker_model.model is None:
                return [c.article_id for c in cands[:top_k]]
            ranked = ranker_model.rerank(cands, feature_extractor=feature_extractor, user_history=history)
            return [c.article_id for c, _ in ranked[:top_k]]

        results.append(self.evaluate_catalog_retrieval("Hybrid + LightGBM LambdaMART", rec_ranker, catalog_article_ids, k=k, dataset_source=dataset_source))

        return results


# Alias for clean readability across testing and experimentation engines
Evaluator = OfflineEvaluator


def save_evaluation_results(results: List[EvaluationMetrics], output_path: Path) -> None:

    """Persist evaluation results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")

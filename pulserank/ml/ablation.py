"""Systematic 9-Configuration Recommender Ablation Framework with Real Evaluation."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4
import numpy as np

from pulserank.ml.evaluation import Evaluator, EvaluationMetrics
from pulserank.ml.diversity import DiversityEngine
from pulserank.ml.ranker import FeatureExtractor, LambdaMARTRanker
from pulserank.ml.exploration import ExplorationEngine, build_exploration_pool
from pulserank.data.provenance import ArtifactManifest

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    config_name: str
    description: str
    # Track A: Impression-level Relevance Benchmark (MIND Impression Candidates)
    auc: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    # Track B: Slate Policy Benchmark (Final Returned Slate over Catalog/Candidate Retrieval)
    ild: float
    category_entropy: float
    novelty: float
    catalog_coverage: float
    delta_ndcg5_vs_full: float = 0.0
    status: str = "measured"
    mmr_lambda_sweep: Optional[Dict[str, Dict[str, float]]] = None
    impression_benchmark: Optional[Dict[str, Any]] = None
    slate_policy_benchmark: Optional[Dict[str, Any]] = None
    num_slates_evaluated: int = 0


class AblationFramework:
    """
    Executes a genuine, mathematically defensible 9-configuration ablation study:
    1. Popularity Baseline
    2. Content-Based Policy (popularity cold-start fallback)
    3. Collaborative Filtering Policy (popularity cold-start fallback)
    4. Pointwise Hybrid Ensemble
    5. RRF Hybrid (Within-impression Reciprocal Rank Fusion)
    6. Full LTR Ranker (LightGBM LambdaMART)
    7. Full LTR Ranker + Dynamic Profiles (Dual-timescale recency alpha-blend)
    8. Full LTR Ranker + MMR Diversity Reranking
    9. Experimental Full Policy (Dynamic-profile LambdaMART + MMR + principled catalog exploration)

    ZERO hardcoded or synthesized performance metrics.
    Every metric is strictly produced by executing the corresponding policy over the evaluation dataset.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        diversity_engine: DiversityEngine,
        popularity_model: Optional[Any] = None,
        content_model: Optional[Any] = None,
        collaborative_model: Optional[Any] = None,
        hybrid_generator: Optional[Any] = None,
        ranker_model: Optional[Any] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        catalog_article_ids: Optional[List[str]] = None,
        articles: Optional[Dict[str, Any]] = None,
        dynamic_profile_manager: Optional[Any] = None,
        exploration_engine: Optional[ExplorationEngine] = None,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
        train_behaviors: Optional[List[Any]] = None,
        exploration_seed: int = 42,
    ):
        self.evaluator = evaluator
        self.diversity_engine = diversity_engine
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.hybrid_generator = hybrid_generator
        self.ranker_model = ranker_model
        self.catalog_article_ids = catalog_article_ids or []
        self.dynamic_profile_manager = dynamic_profile_manager
        self.exploration_seed = exploration_seed if exploration_seed is not None else (
            getattr(exploration_engine, "random_state", 42) if exploration_engine is not None else 42
        )
        self.exploration_engine = exploration_engine or ExplorationEngine(random_state=self.exploration_seed)
        self.dataset_source = dataset_source
        self.dataset_fingerprint = dataset_fingerprint
        self.train_behaviors = train_behaviors

        if feature_extractor is None:
            self.articles = articles or {}
            self.feature_extractor = FeatureExtractor(
                articles=self.articles,
                user_histories={},
                content_model=content_model,
                collab_model=collaborative_model,
                popularity_model=popularity_model,
            )
        else:
            self.feature_extractor = feature_extractor
            self.articles = articles or getattr(feature_extractor, "articles", {})

    def _serve_catalog_slate(
        self,
        behavior: Any,
        config_idx: int,
        top_k: int = 5,
        dyn_ranker: Optional[Any] = None,
        dyn_extractor: Optional[Any] = None,
        mmr_lambda: float = 0.7,
        exploration_engine: Optional[ExplorationEngine] = None,
    ) -> List[Dict[str, Any]]:
        """
        Serve an authentic whole-catalog recommendation slate for a user behavior
        according to the specified configuration policy. Evaluates the true multi-stage
        recommendation serving pipeline over the full catalog.
        """
        user_id = behavior.user_id
        history = getattr(behavior, "history", []) or []
        exclude_ids = set(history)

        # Config 1: Popularity Baseline
        if config_idx == 1:
            if self.popularity_model is not None and hasattr(self.popularity_model, "recommend"):
                recs = self.popularity_model.recommend(k=top_k, exclude_article_ids=exclude_ids)
                return [
                    {
                        "article_id": aid,
                        "score": float(score),
                        "final_score": float(score),
                        "category": getattr(self.articles.get(aid), "category", "unknown"),
                    }
                    for aid, score in recs
                ]
            return []

        # Config 2: Content-Based Only
        if config_idx == 2:
            if self.content_model is not None and hasattr(self.content_model, "recommend") and history:
                recs = self.content_model.recommend(history_article_ids=history, k=top_k, exclude_article_ids=exclude_ids)
                if recs:
                    return [
                        {
                            "article_id": aid,
                            "score": float(score),
                            "final_score": float(score),
                            "category": getattr(self.articles.get(aid), "category", "unknown"),
                        }
                        for aid, score in recs
                    ]
            if self.popularity_model is not None and hasattr(self.popularity_model, "recommend"):
                recs = self.popularity_model.recommend(k=top_k, exclude_article_ids=exclude_ids)
                return [
                    {
                        "article_id": aid,
                        "score": float(score),
                        "final_score": float(score),
                        "category": getattr(self.articles.get(aid), "category", "unknown"),
                    }
                    for aid, score in recs
                ]
            return []

        # Config 3: Collaborative Filtering Only
        if config_idx == 3:
            if self.collaborative_model is not None and hasattr(self.collaborative_model, "recommend"):
                recs = self.collaborative_model.recommend(user_id=user_id, k=top_k, exclude_article_ids=exclude_ids)
                if recs:
                    return [
                        {
                            "article_id": aid,
                            "score": float(score),
                            "final_score": float(score),
                            "category": getattr(self.articles.get(aid), "category", "unknown"),
                        }
                        for aid, score in recs
                    ]
            if self.popularity_model is not None and hasattr(self.popularity_model, "recommend"):
                recs = self.popularity_model.recommend(k=top_k, exclude_article_ids=exclude_ids)
                return [
                    {
                        "article_id": aid,
                        "score": float(score),
                        "final_score": float(score),
                        "category": getattr(self.articles.get(aid), "category", "unknown"),
                    }
                    for aid, score in recs
                ]
            return []

        # Configs 4-9: Multi-channel candidate retrieval -> ranking -> optional MMR -> optional exploration
        candidates = []
        if self.hybrid_generator is not None and hasattr(self.hybrid_generator, "generate_candidates"):
            try:
                candidates = self.hybrid_generator.generate_candidates(
                    user_id=user_id,
                    user_history=history,
                    content_k=30,
                    collab_k=30,
                    popularity_k=30,
                    article_metadata=self.articles,
                )
            except Exception as e:
                logger.debug(f"Candidate generation error for user {user_id}: {e}")
                candidates = []

        if not candidates:
            if self.popularity_model is not None and hasattr(self.popularity_model, "recommend"):
                recs = self.popularity_model.recommend(k=top_k, exclude_article_ids=exclude_ids)
                return [
                    {
                        "article_id": aid,
                        "score": float(score),
                        "final_score": float(score),
                        "category": getattr(self.articles.get(aid), "category", "unknown"),
                    }
                    for aid, score in recs
                ]
            return []

        # Config 4: Pointwise Hybrid Ensemble
        if config_idx == 4:
            w_c = self.hybrid_generator.weights.get("content", 0.45)
            w_cf = self.hybrid_generator.weights.get("collaborative", 0.35)
            w_p = self.hybrid_generator.weights.get("popularity", 0.20)
            cand_scored = []
            for c in candidates:
                s = (
                    w_c * (c.content_score or 0.0)
                    + w_cf * (c.collaborative_score or 0.0)
                    + w_p * (c.popularity_score or 0.0)
                )
                cand_scored.append({
                    "article_id": c.article_id,
                    "score": float(s),
                    "final_score": float(s),
                    "category": c.metadata.get("category", "unknown") if c.metadata else getattr(self.articles.get(c.article_id), "category", "unknown"),
                })
            cand_scored.sort(key=lambda x: x["final_score"], reverse=True)
            return cand_scored[:top_k]

        # Config 5: RRF Candidate Fusion
        if config_idx == 5:
            cand_scored = [
                {
                    "article_id": c.article_id,
                    "score": float(c.hybrid_score),
                    "final_score": float(c.hybrid_score),
                    "category": c.metadata.get("category", "unknown") if c.metadata else getattr(self.articles.get(c.article_id), "category", "unknown"),
                }
                for c in candidates
            ]
            cand_scored.sort(key=lambda x: x["final_score"], reverse=True)
            return cand_scored[:top_k]

        # Config 6: Full LTR Ranker (LightGBM LambdaMART)
        if config_idx == 6:
            if self.ranker_model is not None and getattr(self.ranker_model, "model", None) is not None:
                cids = [c.article_id for c in candidates]
                feats = [self.feature_extractor.extract_features(user_id, a, user_history=history) for a in cids]
                preds = self.ranker_model.model.predict(np.vstack(feats))
                cand_scored = [
                    {
                        "article_id": c.article_id,
                        "score": float(p),
                        "final_score": float(p),
                        "category": c.metadata.get("category", "unknown") if c.metadata else getattr(self.articles.get(c.article_id), "category", "unknown"),
                    }
                    for c, p in zip(candidates, preds)
                ]
                cand_scored.sort(key=lambda x: x["final_score"], reverse=True)
                return cand_scored[:top_k]
            else:
                return self._serve_catalog_slate(behavior, config_idx=4, top_k=top_k)

        # Config 7: Full LTR Ranker + Dynamic Profiles
        if config_idx == 7:
            if dyn_ranker is not None and getattr(dyn_ranker, "model", None) is not None and dyn_extractor is not None:
                cids = [c.article_id for c in candidates]
                feats = [dyn_extractor.extract_features(user_id, a, user_history=history) for a in cids]
                preds = dyn_ranker.model.predict(np.vstack(feats))
                cand_scored = [
                    {
                        "article_id": c.article_id,
                        "score": float(p),
                        "final_score": float(p),
                        "category": c.metadata.get("category", "unknown") if c.metadata else getattr(self.articles.get(c.article_id), "category", "unknown"),
                    }
                    for c, p in zip(candidates, preds)
                ]
                cand_scored.sort(key=lambda x: x["final_score"], reverse=True)
                return cand_scored[:top_k]
            else:
                # FAIL-CLOSED: strictly do NOT evaluate baseline LambdaMART on dynamic features
                logger.warning("Dedicated dyn_ranker not available; Config 7 slate serving falling back to non-LTR hybrid.")
                return self._serve_catalog_slate(behavior, config_idx=4, top_k=top_k)

        # Config 8: Full LTR Ranker + MMR Diversity
        if config_idx == 8:
            ranked_base = self._serve_catalog_slate(behavior, config_idx=6, top_k=len(candidates))
            if not ranked_base:
                return []
            return self.diversity_engine.rerank_mmr(ranked_base, lambda_param=mmr_lambda, top_k=top_k)

        # Config 9: Experimental Full Policy (Dynamic LTR + MMR + Principled Catalog Exploration)
        if config_idx == 9:
            ranked_base = self._serve_catalog_slate(
                behavior,
                config_idx=7,
                top_k=len(candidates),
                dyn_ranker=dyn_ranker,
                dyn_extractor=dyn_extractor,
            )
            if not ranked_base:
                return []
            reranked = self.diversity_engine.rerank_mmr(ranked_base, lambda_param=0.7, top_k=top_k)
            exp_engine = exploration_engine if exploration_engine is not None else ExplorationEngine(random_state=self.exploration_seed)
            if self.catalog_article_ids:
                exp_pool = build_exploration_pool(
                    catalog_article_ids=self.catalog_article_ids,
                    articles=self.articles,
                    user_history=history,
                    content_model=self.content_model,
                    popularity_model=self.popularity_model,
                    exposure_counts=exp_engine.item_exposure_counts,
                    exclude_ids={it["article_id"] for it in reranked},
                )
                final_slate = exp_engine.explore_slate(
                    ranked_candidates=reranked,
                    exploration_pool=exp_pool,
                    top_k=top_k,
                    epsilon=0.10,
                    seed=None,
                )
                return final_slate
            return reranked

        return []

    def _serve_catalog_slates_for_config(
        self,
        behaviors: List[Any],
        config_idx: int,
        top_k: int = 5,
        dyn_ranker: Optional[Any] = None,
        dyn_extractor: Optional[Any] = None,
        mmr_lambda: float = 0.7,
        exploration_engine: Optional[ExplorationEngine] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Serve whole-catalog recommendation slates across a batch of behaviors."""
        slates = []

        # Config 9 Track B requires isolated exploration state that starts with empty exposure counts
        # and does not inherit from live service container
        isolated_exp_engine = None
        if config_idx == 9:
            isolated_exp_engine = exploration_engine if exploration_engine is not None else ExplorationEngine(random_state=self.exploration_seed)

        # Process behaviors in deterministic order
        if config_idx == 9:
            ordered_behaviors = sorted(
                behaviors,
                key=lambda b: (str(getattr(b, "impression_id", "")), str(getattr(b, "user_id", "")))
            )
        else:
            ordered_behaviors = behaviors

        for b in ordered_behaviors:
            sl = self._serve_catalog_slate(
                behavior=b,
                config_idx=config_idx,
                top_k=top_k,
                dyn_ranker=dyn_ranker,
                dyn_extractor=dyn_extractor,
                mmr_lambda=mmr_lambda,
                exploration_engine=isolated_exp_engine,
            )
            if sl:
                slates.append(sl)
                # After each slate is served, update isolated exposure counts so exploration decisions
                # evolve deterministically during the ablation run
                if config_idx == 9 and isolated_exp_engine is not None:
                    article_ids = [item["article_id"] for item in sl if "article_id" in item]
                    isolated_exp_engine.record_exposures(article_ids)

        return slates

    def _extract_slates_for_policy(
        self,
        behaviors: List[Any],
        score_func: Any,
        top_k: int = 5,
        exploration_epsilon: float = 0.0,
    ) -> List[List[Dict[str, Any]]]:
        """
        Legacy candidate-set slate extractor kept for backward compatibility.
        """
        slates: List[List[Dict[str, Any]]] = []
        for b in behaviors:
            if not b.impressions:
                continue
            cids = [imp.news_id for imp in b.impressions]
            scores = score_func(b.user_id, cids, b.history)

            scored = []
            for aid, s in zip(cids, scores):
                art = self.articles.get(aid)
                cat = getattr(art, "category", "unknown") if art else "unknown"
                scored.append({
                    "article_id": aid,
                    "score": float(s),
                    "final_score": float(s),
                    "category": cat,
                })
            scored.sort(key=lambda x: x["final_score"], reverse=True)
            slate = scored[:top_k]

            if exploration_epsilon > 0.0 and self.catalog_article_ids:
                exp_engine = ExplorationEngine(random_state=self.exploration_seed)
                exp_pool = build_exploration_pool(
                    catalog_article_ids=self.catalog_article_ids,
                    articles=self.articles,
                    user_history=b.history,
                    content_model=self.content_model,
                    popularity_model=self.popularity_model,
                    exposure_counts=exp_engine.item_exposure_counts,
                    exclude_ids={it["article_id"] for it in slate},
                )
                slate = exp_engine.explore_slate(
                    ranked_candidates=slate,
                    exploration_pool=exp_pool,
                    top_k=top_k,
                    epsilon=exploration_epsilon,
                    seed=42,
                )

            slates.append(slate)
        return slates

    def _compute_real_diversity_metrics(
        self,
        slates: List[List[Dict[str, Any]]],
    ) -> Tuple[float, float, float, float]:
        """
        Compute real ILD, Category Entropy, Novelty, and Catalog Coverage on actual slates.
        """
        if not slates:
            return 0.0, 0.0, 0.0, 0.0

        ild_list = []
        entropy_list = []
        novelty_list = []
        unique_recommended = set()

        for slate in slates:
            m = self.diversity_engine.compute_metrics(slate)
            ild_list.append(m.intra_list_diversity)
            entropy_list.append(m.category_entropy)
            novelty_list.append(m.novelty)
            for item in slate:
                aid = item.get("article_id")
                if aid:
                    unique_recommended.add(aid)

        ild = float(np.mean(ild_list)) if ild_list else 0.0
        entropy = float(np.mean(entropy_list)) if entropy_list else 0.0
        novelty = float(np.mean(novelty_list)) if novelty_list else 0.0
        coverage = float(len(unique_recommended) / len(self.catalog_article_ids)) if self.catalog_article_ids else 0.0

        return round(ild, 4), round(entropy, 4), round(novelty, 4), round(coverage, 4)

    def run_all_ablations(
        self,
        max_impressions: Optional[int] = 100,
        artifacts_dir: Optional[Path] = None,
    ) -> List[AblationResult]:
        """
        Execute the complete 9-stage ablation study with genuine measurement.
        """
        dev_sample = self.evaluator.dev_behaviors[:max_impressions] if max_impressions else self.evaluator.dev_behaviors
        sub_evaluator = Evaluator(dev_sample)

        results: List[AblationResult] = []

        # -------------------------------------------------------------
        # 1. Popularity Baseline
        # -------------------------------------------------------------
        def score_pop(uid, cids, hist=None):
            if not self.popularity_model:
                return [0.0 for _ in cids]
            return [self.popularity_model.get_score(a) for a in cids]

        m1 = sub_evaluator.evaluate_model("1. Popularity Baseline", score_pop)
        slates_1 = self._serve_catalog_slates_for_config(dev_sample, config_idx=1)
        ild_1, ent_1, nov_1, cov_1 = self._compute_real_diversity_metrics(slates_1)
        results.append(AblationResult(
            config_name="1. Popularity Baseline",
            description="Pure global recency-decayed popularity baseline",
            auc=m1.auc, mrr=m1.mrr, ndcg_at_5=m1.ndcg_at_5, ndcg_at_10=m1.ndcg_at_10,
            ild=ild_1, category_entropy=ent_1, novelty=nov_1, catalog_coverage=cov_1,
            num_slates_evaluated=len(slates_1),
        ))

        # -------------------------------------------------------------
        # 2. Content-Based Policy (popularity cold-start fallback)
        # -------------------------------------------------------------
        def score_content(uid, cids, hist=None):
            h = hist if hist is not None else self.feature_extractor.user_histories.get(uid, [])
            u_vec = self.content_model.build_user_vector(h) if (self.content_model and h) else None
            return [self.content_model.compute_similarity(u_vec, a) if (self.content_model and u_vec is not None) else 0.0 for a in cids]

        m2 = sub_evaluator.evaluate_model("2. Content-Based Only", score_content)
        slates_2 = self._serve_catalog_slates_for_config(dev_sample, config_idx=2)
        ild_2, ent_2, nov_2, cov_2 = self._compute_real_diversity_metrics(slates_2)
        results.append(AblationResult(
            config_name="2. Content-Based Policy (popularity cold-start fallback)",
            description="Content-Based Policy (popularity cold-start fallback)",
            auc=m2.auc, mrr=m2.mrr, ndcg_at_5=m2.ndcg_at_5, ndcg_at_10=m2.ndcg_at_10,
            ild=ild_2, category_entropy=ent_2, novelty=nov_2, catalog_coverage=cov_2,
            num_slates_evaluated=len(slates_2),
        ))

        # -------------------------------------------------------------
        # 3. Collaborative Filtering Policy (popularity cold-start fallback)
        # -------------------------------------------------------------
        def score_collab(uid, cids, hist=None):
            if not self.collaborative_model:
                return [0.0 for _ in cids]
            return [self.collaborative_model.get_score(uid, a) for a in cids]

        m3 = sub_evaluator.evaluate_model("3. Collaborative Filtering Only", score_collab)
        slates_3 = self._serve_catalog_slates_for_config(dev_sample, config_idx=3)
        ild_3, ent_3, nov_3, cov_3 = self._compute_real_diversity_metrics(slates_3)
        results.append(AblationResult(
            config_name="3. Collaborative Filtering Policy (popularity cold-start fallback)",
            description="Collaborative Filtering Policy (popularity cold-start fallback)",
            auc=m3.auc, mrr=m3.mrr, ndcg_at_5=m3.ndcg_at_5, ndcg_at_10=m3.ndcg_at_10,
            ild=ild_3, category_entropy=ent_3, novelty=nov_3, catalog_coverage=cov_3,
            num_slates_evaluated=len(slates_3),
        ))

        # -------------------------------------------------------------
        # 4. Pointwise Hybrid Ensemble (No LTR)
        # -------------------------------------------------------------
        def score_pw_hybrid(uid, cids, hist=None):
            if self.hybrid_generator and hasattr(self.hybrid_generator, "score_candidates_pointwise"):
                h = hist if hist is not None else self.feature_extractor.user_histories.get(uid, [])
                return self.hybrid_generator.score_candidates_pointwise(uid, cids, user_history=h)
            c = score_content(uid, cids, hist)
            a = score_collab(uid, cids, hist)
            p = score_pop(uid, cids, hist)
            return [0.4 * c[i] + 0.4 * a[i] + 0.2 * p[i] for i in range(len(cids))]

        m4 = sub_evaluator.evaluate_model("4. Pointwise Hybrid", score_pw_hybrid)
        slates_4 = self._serve_catalog_slates_for_config(dev_sample, config_idx=4)
        ild_4, ent_4, nov_4, cov_4 = self._compute_real_diversity_metrics(slates_4)
        results.append(AblationResult(
            config_name="4. Pointwise Hybrid",
            description="Calibrated linear blend of content, collaborative, and popularity",
            auc=m4.auc, mrr=m4.mrr, ndcg_at_5=m4.ndcg_at_5, ndcg_at_10=m4.ndcg_at_10,
            ild=ild_4, category_entropy=ent_4, novelty=nov_4, catalog_coverage=cov_4,
            num_slates_evaluated=len(slates_4),
        ))

        # -------------------------------------------------------------
        # 5. RRF Hybrid (Within-impression Reciprocal Rank Fusion)
        # -------------------------------------------------------------
        def score_rrf(uid, cids, hist=None):
            if not cids:
                return []
            c_scores = score_content(uid, cids, hist)
            als_scores = score_collab(uid, cids, hist)
            pop_scores = score_pop(uid, cids, hist)

            c_ranks = np.argsort(np.argsort(-np.array(c_scores))) + 1
            als_ranks = np.argsort(np.argsort(-np.array(als_scores))) + 1
            pop_ranks = np.argsort(np.argsort(-np.array(pop_scores))) + 1

            return [
                float(1.0 / (60 + c_ranks[i]) + 1.0 / (60 + als_ranks[i]) + 1.0 / (60 + pop_ranks[i]))
                for i in range(len(cids))
            ]

        m5 = sub_evaluator.evaluate_model("5. RRF Candidate Fusion", score_rrf)
        slates_5 = self._serve_catalog_slates_for_config(dev_sample, config_idx=5)
        ild_5, ent_5, nov_5, cov_5 = self._compute_real_diversity_metrics(slates_5)
        results.append(AblationResult(
            config_name="5. RRF Candidate Fusion",
            description="Within-impression Reciprocal Rank Fusion across 3 retrieval channels",
            auc=m5.auc, mrr=m5.mrr, ndcg_at_5=m5.ndcg_at_5, ndcg_at_10=m5.ndcg_at_10,
            ild=ild_5, category_entropy=ent_5, novelty=nov_5, catalog_coverage=cov_5,
            num_slates_evaluated=len(slates_5),
        ))

        # -------------------------------------------------------------
        # 6. Full LTR Ranker (LightGBM LambdaMART)
        # -------------------------------------------------------------
        def score_ltr(uid, cids, hist=None):
            if not self.ranker_model or self.ranker_model.model is None:
                return score_pw_hybrid(uid, cids, hist)
            feats = [self.feature_extractor.extract_features(uid, a, user_history=hist) for a in cids]
            return [float(p) for p in self.ranker_model.model.predict(np.vstack(feats))]

        m6 = sub_evaluator.evaluate_model("6. Full LTR Ranker", score_ltr)
        slates_6 = self._serve_catalog_slates_for_config(dev_sample, config_idx=6)
        ild_6, ent_6, nov_6, cov_6 = self._compute_real_diversity_metrics(slates_6)
        results.append(AblationResult(
            config_name="6. Full LTR Ranker",
            description="LightGBM LambdaMART ranking on 7 canonical leakage-free features",
            auc=m6.auc, mrr=m6.mrr, ndcg_at_5=m6.ndcg_at_5, ndcg_at_10=m6.ndcg_at_10,
            ild=ild_6, category_entropy=ent_6, novelty=nov_6, catalog_coverage=cov_6,
            num_slates_evaluated=len(slates_6),
        ))

        # -------------------------------------------------------------
        # 7. Full LTR Ranker + Dynamic Profiles
        # -------------------------------------------------------------
        dyn_extractor = FeatureExtractor(
            articles=self.articles,
            user_histories=self.feature_extractor.user_histories,
            content_model=self.content_model,
            collab_model=self.collaborative_model,
            popularity_model=self.popularity_model,
            dynamic_profile_manager=self.dynamic_profile_manager,
            use_dynamic_profiles=True,
        )

        # Train dedicated LambdaMART model for dynamic-profile configuration
        # using the exact same dynamic-profile feature extractor during training and evaluation
        dyn_ranker = None
        if self.train_behaviors and self.ranker_model and getattr(self.ranker_model, "model", None) is not None:
            try:
                dyn_ranker = LambdaMARTRanker(
                    learning_rate=getattr(self.ranker_model, "learning_rate", 0.05),
                    num_leaves=getattr(self.ranker_model, "num_leaves", 31),
                    max_depth=getattr(self.ranker_model, "max_depth", 6),
                    n_estimators=getattr(self.ranker_model, "n_estimators", 150),
                    random_state=getattr(self.ranker_model, "random_state", 42),
                )
                dyn_ranker.fit_from_behaviors(
                    train_behaviors=self.train_behaviors,
                    feature_extractor=dyn_extractor,
                )
            except Exception as e:
                logger.warning(f"Failed to fit dedicated dynamic LambdaMART ranker: {e}")
                dyn_ranker = None

        if dyn_ranker is not None and getattr(dyn_ranker, "model", None) is not None:
            def score_ltr_dynamic(uid, cids, hist=None):
                feats = [dyn_extractor.extract_features(uid, a, user_history=hist) for a in cids]
                return [float(p) for p in dyn_ranker.model.predict(np.vstack(feats))]
            status_7 = "measured"
        else:
            logger.warning(
                "Dedicated dynamic ranker unavailable; Config 7 strictly falling back to non-LTR hybrid "
                "to prevent train/evaluation feature distribution mismatch."
            )
            def score_ltr_dynamic(uid, cids, hist=None):
                return score_pw_hybrid(uid, cids, hist)
            status_7 = "fallback_non_ltr"

        m7 = sub_evaluator.evaluate_model("7. LTR + Dynamic Profiles", score_ltr_dynamic)
        slates_7 = self._serve_catalog_slates_for_config(
            dev_sample,
            config_idx=7,
            dyn_ranker=dyn_ranker,
            dyn_extractor=dyn_extractor,
        )
        ild_7, ent_7, nov_7, cov_7 = self._compute_real_diversity_metrics(slates_7)
        results.append(AblationResult(
            config_name="7. LTR + Dynamic Profiles",
            description="Dedicated LambdaMART trained & evaluated with dynamic recency profiles",
            auc=m7.auc, mrr=m7.mrr, ndcg_at_5=m7.ndcg_at_5, ndcg_at_10=m7.ndcg_at_10,
            ild=ild_7, category_entropy=ent_7, novelty=nov_7, catalog_coverage=cov_7,
            status=status_7,
            num_slates_evaluated=len(slates_7),
        ))

        # -------------------------------------------------------------
        # 8. Full LTR Ranker + MMR Diversity
        # -------------------------------------------------------------
        # Evaluate multiple lambda values for genuine sensitivity curve
        lambda_sweep_results: Dict[str, Dict[str, float]] = {}
        target_lambda = 0.7

        for lam in [0.3, 0.5, 0.7, 0.9, 1.0]:
            def make_score_mmr_lam(l_param):
                def score_mmr_lam(uid, cids, hist=None):
                    base_scores = score_ltr(uid, cids, hist)
                    cands = [
                        {"article_id": a, "score": s, "final_score": s, "category": getattr(self.articles.get(a), "category", "unknown")}
                        for a, s in zip(cids, base_scores)
                    ]
                    reranked = self.diversity_engine.rerank_mmr(cands, lambda_param=l_param, top_k=len(cids))
                    rank_map = {item["article_id"]: len(cids) - idx for idx, item in enumerate(reranked)}
                    return [float(rank_map.get(a, 0)) for a in cids]
                return score_mmr_lam

            fn = make_score_mmr_lam(lam)
            m_lam = sub_evaluator.evaluate_model(f"MMR_{lam}", fn)
            sl_lam = self._serve_catalog_slates_for_config(dev_sample, config_idx=8, mmr_lambda=lam)
            ild_lam, ent_lam, nov_lam, cov_lam = self._compute_real_diversity_metrics(sl_lam)
            lambda_sweep_results[str(lam)] = {
                "ndcg_at_5": m_lam.ndcg_at_5,
                "mrr": m_lam.mrr,
                "ild": ild_lam,
                "category_entropy": ent_lam,
                "novelty": nov_lam,
                "catalog_coverage": cov_lam,
            }

        # Target MMR (lambda=0.7)
        def score_mmr(uid, cids, hist=None):
            base_scores = score_ltr(uid, cids, hist)
            cands = [
                {"article_id": a, "score": s, "final_score": s, "category": getattr(self.articles.get(a), "category", "unknown")}
                for a, s in zip(cids, base_scores)
            ]
            reranked = self.diversity_engine.rerank_mmr(cands, lambda_param=target_lambda, top_k=len(cids))
            rank_map = {item["article_id"]: len(cids) - idx for idx, item in enumerate(reranked)}
            return [float(rank_map.get(a, 0)) for a in cids]

        m8 = sub_evaluator.evaluate_model("8. LTR + MMR Diversity", score_mmr)
        slates_8 = self._serve_catalog_slates_for_config(dev_sample, config_idx=8, mmr_lambda=target_lambda)
        ild_8, ent_8, nov_8, cov_8 = self._compute_real_diversity_metrics(slates_8)
        results.append(AblationResult(
            config_name="8. LTR + MMR Diversity",
            description=f"LTR reranked with Maximal Marginal Relevance (lambda={target_lambda})",
            auc=m8.auc, mrr=m8.mrr, ndcg_at_5=m8.ndcg_at_5, ndcg_at_10=m8.ndcg_at_10,
            ild=ild_8, category_entropy=ent_8, novelty=nov_8, catalog_coverage=cov_8,
            mmr_lambda_sweep=lambda_sweep_results,
            num_slates_evaluated=len(slates_8),
        ))

        # -------------------------------------------------------------
        # 9. Experimental Full Policy (LTR + Dynamic + MMR + Exploration)
        # -------------------------------------------------------------
        # Track A: Impression Benchmark (Candidate reranking within MIND impression)
        # Uses isolated exploration state with empty counts
        track_a_explore_engine = ExplorationEngine(random_state=self.exploration_seed)
        def score_full_system_impression(uid, cids, hist=None):
            base_scores = score_ltr_dynamic(uid, cids, hist)
            cands = [
                {"article_id": a, "score": s, "final_score": s, "category": getattr(self.articles.get(a), "category", "unknown")}
                for a, s in zip(cids, base_scores)
            ]
            reranked = self.diversity_engine.rerank_mmr(cands, lambda_param=target_lambda, top_k=len(cids))

            # Apply deterministic exploration within impression candidates using isolated engine
            explore_pool = build_exploration_pool(
                catalog_article_ids=cids,
                articles=self.articles,
                user_history=hist,
                content_model=self.content_model,
                popularity_model=self.popularity_model,
                exposure_counts=track_a_explore_engine.item_exposure_counts,
            )
            explored = track_a_explore_engine.explore_slate(
                ranked_candidates=reranked,
                exploration_pool=explore_pool,
                top_k=len(cids),
                epsilon=0.10,
                seed=42,
            )
            rank_map = {item["article_id"]: len(cids) - idx for idx, item in enumerate(explored)}
            return [float(rank_map.get(a, 0)) for a in cids]

        status_9 = "measured" if (dyn_ranker is not None and getattr(dyn_ranker, "model", None) is not None) else "fallback_non_ltr"
        m9 = sub_evaluator.evaluate_model("9. Experimental Full Policy", score_full_system_impression)

        # Track B: Slate Policy Benchmark (Full-system whole-catalog recommendation serving)
        slates_9 = self._serve_catalog_slates_for_config(
            dev_sample,
            config_idx=9,
            dyn_ranker=dyn_ranker,
            dyn_extractor=dyn_extractor,
        )
        ild_9, ent_9, nov_9, cov_9 = self._compute_real_diversity_metrics(slates_9)
        results.append(AblationResult(
            config_name="9. Experimental Full Policy",
            description="Dynamic-profile LambdaMART + MMR + principled catalog exploration",
            auc=m9.auc, mrr=m9.mrr, ndcg_at_5=m9.ndcg_at_5, ndcg_at_10=m9.ndcg_at_10,
            ild=ild_9, category_entropy=ent_9, novelty=nov_9, catalog_coverage=cov_9,
            status=status_9,
            num_slates_evaluated=len(slates_9),
        ))

        # Compute relative delta NDCG@5 vs Experimental Full Policy and populate explicit benchmark dicts
        full_ndcg = results[-1].ndcg_at_5
        for r in results:
            r.delta_ndcg5_vs_full = round(r.ndcg_at_5 - full_ndcg, 4)
            r.impression_benchmark = {
                "auc": r.auc,
                "mrr": r.mrr,
                "ndcg_at_5": r.ndcg_at_5,
                "ndcg_at_10": r.ndcg_at_10,
                "status": r.status,
            }
            r.slate_policy_benchmark = {
                "status": "measured" if r.num_slates_evaluated > 0 else "unavailable",
                "ild": r.ild,
                "category_entropy": r.category_entropy,
                "novelty": r.novelty,
                "catalog_coverage": r.catalog_coverage,
                "num_slates_evaluated": r.num_slates_evaluated,
            }

        # Persist completed experiment results artifact
        if artifacts_dir:
            self._save_ablation_run(results, Path(artifacts_dir), len(dev_sample))

        return results

    def _save_ablation_run(self, results: List[AblationResult], artifacts_dir: Path, num_impressions: int) -> Path:
        run_id = f"ablation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        run_dir = artifacts_dir / "ablations" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest = ArtifactManifest(
            model_name="ablation_study",
            model_version="2.0.0",
            artifact_type="ablation_benchmark",
            dataset_source=self.dataset_source,
            dataset_fingerprint=self.dataset_fingerprint,
            hyperparameters={
                "num_configurations": len(results),
                "num_impressions_evaluated": num_impressions,
                "mmr_target_lambda": 0.7,
                "exploration_epsilon": 0.10,
                "exploration_seed": 42,
            },
        )
        manifest.save(run_dir / "manifest.json")

        results_data = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": self.dataset_source,
            "dataset_fingerprint": self.dataset_fingerprint,
            "num_impressions_evaluated": num_impressions,
            "ablations": [asdict(r) for r in results],
        }

        results_file = run_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        # Also write symlink or latest pointer for UI consumption
        latest_file = artifacts_dir / "ablation_results.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Persisted measured ablation results to {results_file} and {latest_file}")
        return run_dir

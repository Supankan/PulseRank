"""Maximal Marginal Relevance (MMR) and Beyond-Accuracy Diversity Metrics."""

import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    intra_list_diversity: float  # ILD: mean pair-wise cosine distance
    category_entropy: float      # Shannon entropy over categories
    novelty: float               # Mean self-information (-log2(p_pop))
    relevance_score: float       # Mean predicted ranking relevance
    catalog_coverage: float = 0.0

    def __iter__(self):
        yield self.intra_list_diversity
        yield self.category_entropy
        yield self.novelty
        if self.catalog_coverage > 0.0:
            yield self.catalog_coverage
        else:
            yield self.relevance_score


class DiversityEngine:
    """
    Applies Maximal Marginal Relevance (MMR) reranking to balance precision and diversity,
    and computes beyond-accuracy evaluation metrics (ILD, Entropy, Novelty).
    """

    def __init__(
        self,
        article_embeddings: Optional[Dict[str, np.ndarray]] = None,
        article_popularities: Optional[Dict[str, int]] = None,
    ):
        self.article_embeddings = article_embeddings or {}
        self.article_popularities = article_popularities or {}
        self._total_popularity = max(sum(self.article_popularities.values()), 1)

    def rerank_mmr(
        self,
        candidates: List[Dict[str, Any]],
        lambda_param: float = 0.7,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance (MMR) greedy selection.
        MMR(d) = lambda * Rel(d) - (1 - lambda) * max_{s in S} Sim(d, s)
        """
        if not candidates:
            return []
        if len(candidates) <= 1 or lambda_param >= 1.0:
            return candidates[:top_k]

        selected: List[Dict[str, Any]] = []
        remaining = list(candidates)

        # First item is purely the highest relevance item
        remaining.sort(key=lambda x: x.get("final_score", x.get("score", 0.0)), reverse=True)
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_mmr_score = -float("inf")
            best_idx = 0

            for i, cand in enumerate(remaining):
                rel = float(cand.get("final_score", cand.get("score", 0.0)))
                cand_aid = cand.get("article_id", "")
                cand_vec = self.article_embeddings.get(cand_aid)

                # Compute maximum similarity to any currently selected article
                max_sim = 0.0
                if cand_vec is not None:
                    for sel in selected:
                        sel_aid = sel.get("article_id", "")
                        sel_vec = self.article_embeddings.get(sel_aid)
                        if sel_vec is not None:
                            cos_sim = float(np.dot(cand_vec, sel_vec))
                            if cos_sim > max_sim:
                                max_sim = cos_sim

                mmr_val = (lambda_param * rel) - ((1.0 - lambda_param) * max_sim)
                if mmr_val > best_mmr_score:
                    best_mmr_score = mmr_val
                    best_idx = i

            chosen = remaining.pop(best_idx)
            chosen["mmr_score"] = round(best_mmr_score, 4)
            selected.append(chosen)

        return selected

    def compute_metrics(
        self,
        slate: Any,
        catalog_size: Optional[int] = None,
    ) -> Any:
        """
        Compute ILD, Category Entropy, Novelty, and Mean Relevance for recommendation slate(s).
        Supports single slate (list of dicts or list of article_ids) or list of slates.
        """
        if not slate:
            if catalog_size is not None:
                return (0.0, 0.0, 0.0, 0.0)
            return DiversityMetrics(0.0, 0.0, 0.0, 0.0)

        # Case 1: Multiple slates passed (list of lists)
        if isinstance(slate, list) and slate and isinstance(slate[0], list):
            ild_list, ent_list, nov_list = [], [], []
            unique_articles = set()
            for sub_slate in slate:
                norm_sub = [
                    {"article_id": s, "score": 1.0} if isinstance(s, str) else s
                    for s in sub_slate
                ]
                m = self.compute_metrics(norm_sub)
                ild_list.append(m.intra_list_diversity)
                ent_list.append(m.category_entropy)
                nov_list.append(m.novelty)
                for item in norm_sub:
                    unique_articles.add(item.get("article_id", ""))
            ild = float(np.mean(ild_list)) if ild_list else 0.0
            entropy = float(np.mean(ent_list)) if ent_list else 0.0
            novelty = float(np.mean(nov_list)) if nov_list else 0.0
            cov = float(len(unique_articles) / catalog_size) if catalog_size else 0.0
            return (round(ild, 4), round(entropy, 4), round(novelty, 4), round(cov, 4))

        # Normalize single slate items (if strings, convert to dicts)
        norm_slate: List[Dict[str, Any]] = [
            {"article_id": s, "score": 1.0} if isinstance(s, str) else s
            for s in slate
        ]

        k = len(norm_slate)

        # 1. Mean relevance
        mean_rel = float(np.mean([s.get("final_score", s.get("score", 0.0)) for s in norm_slate]))

        # 2. Intra-List Diversity (ILD)
        ild = 0.0
        if k > 1:
            distances = []
            for i in range(k):
                vec_i = self.article_embeddings.get(norm_slate[i].get("article_id", ""))
                for j in range(i + 1, k):
                    vec_j = self.article_embeddings.get(norm_slate[j].get("article_id", ""))
                    if vec_i is not None and vec_j is not None:
                        cos = float(np.clip(np.dot(vec_i, vec_j), -1.0, 1.0))
                        distances.append(1.0 - cos)
                    else:
                        distances.append(0.5)  # neutral fallback distance
            ild = float(np.mean(distances)) if distances else 0.0

        # 3. Category Entropy
        cat_counts: Dict[str, int] = {}
        for s in norm_slate:
            cat = s.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        entropy = 0.0
        for count in cat_counts.values():
            p = count / k
            if p > 0:
                entropy -= p * math.log2(p)

        # 4. Novelty (Self-Information)
        novelty_vals = []
        for s in norm_slate:
            aid = s.get("article_id", "")
            pop = self.article_popularities.get(aid, 1)
            prob_pop = max(pop / self._total_popularity, 1e-7)
            novelty_vals.append(-math.log2(prob_pop))
        mean_novelty = float(np.mean(novelty_vals)) if novelty_vals else 0.0

        cov = float(len({s.get("article_id") for s in norm_slate}) / catalog_size) if catalog_size else 0.0

        metrics_obj = DiversityMetrics(
            intra_list_diversity=round(ild, 4),
            category_entropy=round(entropy, 4),
            novelty=round(mean_novelty, 4),
            relevance_score=round(mean_rel, 4),
            catalog_coverage=round(cov, 4),
        )

        if catalog_size is not None:
            return (round(ild, 4), round(entropy, 4), round(novelty, 4), round(cov, 4))
        return metrics_obj

    def generate_tradeoff_curve(
        self,
        candidates: List[Dict[str, Any]],
        lambda_steps: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate trade-off frontier points over lambda in [0.0, 1.0].
        """
        if lambda_steps is None:
            lambda_steps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        curve = []
        for lam in lambda_steps:
            slate = self.rerank_mmr(candidates, lambda_param=lam, top_k=top_k)
            metrics = self.compute_metrics(slate)
            curve.append({
                "lambda": lam,
                "relevance": metrics.relevance_score,
                "intra_list_diversity": metrics.intra_list_diversity,
                "category_entropy": metrics.category_entropy,
                "novelty": metrics.novelty,
            })
        return curve

"""Pairwise Cross-Encoder Reranking and Latency/Quality Benchmark."""

import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class PairwiseCrossEncoder:
    """
    Cross-Encoder for high-precision second-stage reranking.
    Jointly encodes (user_context, candidate_article) through full cross-attention.
    Benchmarked against Bi-Encoder retrieval for latency vs NDCG trade-offs.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._initialized = False

    def _load_model(self):
        if not self._initialized:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded CrossEncoder model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not load CrossEncoder ({e}). Using cosine cross-attention simulator.")
                self._model = None
            self._initialized = True

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score a list of (context_text, article_text) pairs."""
        if not pairs:
            return []
        self._load_model()

        if self._model is not None:
            try:
                raw_scores = self._model.predict(pairs, show_progress_bar=False)
                # Sigmoid normalization
                return [float(1.0 / (1.0 + np.exp(-float(s)))) for s in raw_scores]
            except Exception as e:
                logger.warning(f"CrossEncoder scoring error ({e}). Fallback to token similarity.")

        # Deterministic Jaccard/token overlap cross-score fallback
        scores = []
        for ctx, art in pairs:
            s_ctx = set(ctx.lower().split())
            s_art = set(art.lower().split())
            inter = len(s_ctx & s_art)
            union = len(s_ctx | s_art)
            scores.append(float(round(inter / union if union > 0 else 0.0, 4)))
        return scores

    def rerank_candidates(
        self,
        user_context: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Rerank a candidate list using the cross-encoder.
        Returns reranked candidates and measured inference latency in milliseconds.
        """
        if not candidates:
            return [], 0.0

        pairs = [(user_context, f"{c.get('title', '')} {c.get('abstract', '')}") for c in candidates]

        t0 = time.perf_counter()
        scores = self.score_pairs(pairs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        reranked = []
        for c, score in zip(candidates, scores):
            c_copy = dict(c)
            c_copy["cross_encoder_score"] = round(score, 4)
            # Blend initial bi-encoder score with cross-encoder score
            bi_score = c.get("hybrid_score", c.get("score", 0.5))
            c_copy["final_score"] = round(0.4 * bi_score + 0.6 * score, 4)
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k], round(latency_ms, 2)

    def benchmark_tradeoff(
        self,
        sample_queries: List[Tuple[str, List[Dict[str, Any]]]],
    ) -> Dict[str, Any]:
        """
        Benchmark Bi-Encoder vs Cross-Encoder latency and ranking displacement.
        """
        total_latency = 0.0
        total_items = 0
        rank_displacements = []

        for query_ctx, cands in sample_queries:
            if not cands:
                continue
            reranked, lat = self.rerank_candidates(query_ctx, cands, top_k=len(cands))
            total_latency += lat
            total_items += len(cands)

            # Compute rank displacement
            orig_order = {c["article_id"]: idx for idx, c in enumerate(cands)}
            for new_idx, c in enumerate(reranked):
                aid = c["article_id"]
                if aid in orig_order:
                    rank_displacements.append(abs(new_idx - orig_order[aid]))

        mean_latency_per_query = total_latency / max(1, len(sample_queries))
        mean_latency_per_item = total_latency / max(1, total_items)
        mean_displacement = float(np.mean(rank_displacements)) if rank_displacements else 0.0

        return {
            "mean_query_latency_ms": round(mean_latency_per_query, 2),
            "mean_item_latency_ms": round(mean_latency_per_item, 3),
            "mean_rank_displacement": round(mean_displacement, 2),
            "num_queries_benchmarked": len(sample_queries),
            "num_candidates_benchmarked": total_items,
        }

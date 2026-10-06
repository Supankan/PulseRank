"""Hybrid candidate generator and rank fusion engine."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import logging

from pulserank.ml.popularity import PopularityRecommender
from pulserank.ml.content_based import ContentBasedRecommender
from pulserank.ml.collaborative import ImplicitALSRecommender

logger = logging.getLogger(__name__)


@dataclass
class CandidateItem:
    article_id: str
    user_id: str
    sources: List[str]  # e.g. ["collaborative", "content"]
    source_scores: Dict[str, float]  # raw per-source scores: {"content": 0.82, "popularity": 0.65}
    content_score: float = 0.0
    collaborative_score: float = 0.0
    popularity_score: float = 0.0
    hybrid_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridCandidateGenerator:
    """
    Multi-source candidate fusion engine:
    - Combines Content-Based, Collaborative, and Popularity retrieval streams.
    - Implements Reciprocal Rank Fusion (RRF) to prevent scale domination.
    - Accurately tracks multi-source provenance and consensus bonuses.
    - Supports both full-catalog candidate generation and direct impression candidate scoring.
    """

    def __init__(
        self,
        popularity_model: Optional[PopularityRecommender] = None,
        content_model: Optional[ContentBasedRecommender] = None,
        collaborative_model: Optional[ImplicitALSRecommender] = None,
        weights: Optional[Dict[str, float]] = None,
        rrf_k: int = 60,
    ):
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.weights = weights or {
            "content": 0.45,
            "collaborative": 0.35,
            "popularity": 0.20,
        }
        self.rrf_k = rrf_k

    def generate_candidates(
        self,
        user_id: str,
        user_history: List[str],
        content_k: int = 50,
        collab_k: int = 50,
        popularity_k: int = 50,
        exclude_article_ids: Optional[Set[str]] = None,
        article_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateItem]:
        """
        Generate, merge, and deduplicate candidates across retrieval channels
        using Reciprocal Rank Fusion (RRF).
        """
        exclude: Set[str] = set(user_history)
        if exclude_article_ids:
            exclude.update(exclude_article_ids)

        candidates_map: Dict[str, CandidateItem] = {}
        channel_ranks: Dict[str, Dict[str, int]] = {
            "content": {},
            "collaborative": {},
            "popularity": {},
        }

        # 1. Content-based retrieval
        if self.content_model is not None and user_history:
            try:
                content_recs = self.content_model.recommend(
                    history_article_ids=user_history,
                    k=content_k,
                    exclude_article_ids=exclude,
                )
                for rank_idx, (aid, score) in enumerate(content_recs, 1):
                    channel_ranks["content"][aid] = rank_idx
                    if aid not in candidates_map:
                        candidates_map[aid] = CandidateItem(
                            article_id=aid,
                            user_id=user_id,
                            sources=["content"],
                            source_scores={"content": score},
                            content_score=score,
                        )
                    else:
                        if "content" not in candidates_map[aid].sources:
                            candidates_map[aid].sources.append("content")
                        candidates_map[aid].source_scores["content"] = score
                        candidates_map[aid].content_score = score
            except Exception as e:
                logger.warning(f"Content retrieval failed for user {user_id}: {e}")

        # 2. Collaborative filtering retrieval
        if self.collaborative_model is not None:
            try:
                collab_recs = self.collaborative_model.recommend(
                    user_id=user_id,
                    k=collab_k,
                    exclude_article_ids=exclude,
                )
                for rank_idx, (aid, score) in enumerate(collab_recs, 1):
                    channel_ranks["collaborative"][aid] = rank_idx
                    if aid not in candidates_map:
                        candidates_map[aid] = CandidateItem(
                            article_id=aid,
                            user_id=user_id,
                            sources=["collaborative"],
                            source_scores={"collaborative": score},
                            collaborative_score=score,
                        )
                    else:
                        if "collaborative" not in candidates_map[aid].sources:
                            candidates_map[aid].sources.append("collaborative")
                        candidates_map[aid].source_scores["collaborative"] = score
                        candidates_map[aid].collaborative_score = score
            except Exception as e:
                logger.warning(f"Collaborative retrieval failed for user {user_id}: {e}")

        # 3. Popularity baseline retrieval
        if self.popularity_model is not None:
            try:
                pop_recs = self.popularity_model.recommend(
                    k=popularity_k,
                    exclude_article_ids=exclude,
                )
                for rank_idx, (aid, score) in enumerate(pop_recs, 1):
                    channel_ranks["popularity"][aid] = rank_idx
                    if aid not in candidates_map:
                        candidates_map[aid] = CandidateItem(
                            article_id=aid,
                            user_id=user_id,
                            sources=["popularity"],
                            source_scores={"popularity": score},
                            popularity_score=score,
                        )
                    else:
                        if "popularity" not in candidates_map[aid].sources:
                            candidates_map[aid].sources.append("popularity")
                        candidates_map[aid].source_scores["popularity"] = score
                        candidates_map[aid].popularity_score = score
            except Exception as e:
                logger.warning(f"Popularity retrieval failed for user {user_id}: {e}")

        w_c = self.weights.get("content", 0.45)
        w_cf = self.weights.get("collaborative", 0.35)
        w_p = self.weights.get("popularity", 0.20)

        results: List[CandidateItem] = []
        for aid, item in candidates_map.items():
            # Canonical sort of sources for order invariance
            item.sources = sorted(list(set(item.sources)))

            # Populate missing point scores if unobserved in specific channels
            if "popularity" not in item.sources and self.popularity_model is not None:
                item.popularity_score = self.popularity_model.get_score(aid)
            if "collaborative" not in item.sources and self.collaborative_model is not None:
                item.collaborative_score = self.collaborative_model.get_score(user_id, aid)

            # Reciprocal Rank Fusion (RRF) calculation
            # For each channel that retrieved this item, add w / (k + rank)
            rrf_score = 0.0
            if aid in channel_ranks["content"]:
                rrf_score += w_c / (self.rrf_k + channel_ranks["content"][aid])
            if aid in channel_ranks["collaborative"]:
                rrf_score += w_cf / (self.rrf_k + channel_ranks["collaborative"][aid])
            if aid in channel_ranks["popularity"]:
                rrf_score += w_p / (self.rrf_k + channel_ranks["popularity"][aid])

            # Multi-channel consensus bonus
            consensus_multiplier = 1.0 + 0.15 * (len(item.sources) - 1)
            item.hybrid_score = rrf_score * consensus_multiplier

            if article_metadata and aid in article_metadata:
                art = article_metadata[aid]
                item.metadata = {
                    "category": getattr(art, "category", "unknown"),
                    "subcategory": getattr(art, "subcategory", "unknown"),
                    "title": getattr(art, "title", ""),
                }

            results.append(item)

        # Normalize hybrid_score to [0.0, 1.0] across candidate pool for interpretability
        if results:
            max_s = max(r.hybrid_score for r in results)
            min_s = min(r.hybrid_score for r in results)
            span = max_s - min_s if max_s > min_s else 1.0
            for r in results:
                r.hybrid_score = round(float((r.hybrid_score - min_s) / span), 4)

        # Sort descending by hybrid score
        results.sort(key=lambda x: x.hybrid_score, reverse=True)
        return results

    def score_candidates_pointwise(
        self,
        user_id: str,
        candidate_ids: List[str],
        user_history: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Directly score a specified list of candidate articles for a user
        without whole-catalog retrieval truncation. Used for impression evaluation.
        """
        w_c = self.weights.get("content", 0.45)
        w_cf = self.weights.get("collaborative", 0.35)
        w_p = self.weights.get("popularity", 0.20)

        # Build user vector if content model is present
        user_vec = None
        if self.content_model is not None and user_history:
            user_vec = self.content_model.build_user_vector(user_history)

        scores = []
        for aid in candidate_ids:
            # Content cosine similarity
            s_content = 0.0
            if self.content_model is not None and user_vec is not None:
                s_content = self.content_model.compute_similarity(user_vec, aid)

            # Collaborative ALS score
            s_collab = 0.5  # Neutral default for unseen items/users
            if self.collaborative_model is not None:
                s_collab = self.collaborative_model.get_score(user_id, aid)

            # Popularity score
            s_pop = 0.0
            if self.popularity_model is not None:
                s_pop = self.popularity_model.get_score(aid)

            # Calibrated ensemble
            combined = (w_c * s_content) + (w_cf * s_collab) + (w_p * s_pop)
            scores.append(float(combined))

        return scores

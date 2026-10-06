"""Short-term vs. Long-term User Interest Modeling with Recency Weighting."""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EntityInterest:
    name: str
    wikidata_id: Optional[str] = None
    count: int = 1
    category: Optional[str] = None


@dataclass
class UserInterestProfile:
    user_id: str
    history_size: int
    category_distribution: Dict[str, float]
    subcategory_distribution: Dict[str, float]
    top_entities: List[EntityInterest]
    top_categories: List[str]
    recent_articles: List[Dict[str, Any]]
    semantic_summary: str


class ProfileBuilder:
    """Builds interpretable user interest profiles from reading history."""

    def __init__(self, articles: Dict[str, Any]):
        self.articles = articles

    def build_profile(self, user_id: str, history_article_ids: List[str]) -> UserInterestProfile:
        history_size = len(history_article_ids)
        if history_size == 0:
            return UserInterestProfile(
                user_id=user_id,
                history_size=0,
                category_distribution={},
                subcategory_distribution={},
                top_entities=[],
                top_categories=[],
                recent_articles=[],
                semantic_summary="Cold-start user with no prior reading history.",
            )

        cat_counts: Dict[str, int] = {}
        subcat_counts: Dict[str, int] = {}
        entity_counts: Dict[str, Dict[str, Any]] = {}
        recent = []

        for aid in history_article_ids:
            art = self.articles.get(aid)
            if not art:
                continue
            cat = getattr(art, "category", "unknown")
            subcat = getattr(art, "subcategory", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            subcat_counts[subcat] = subcat_counts.get(subcat, 0) + 1

            for ent_list in [getattr(art, "title_entities", []), getattr(art, "abstract_entities", [])]:
                if isinstance(ent_list, list):
                    for ent in ent_list:
                        lbl = ent.get("Label") if isinstance(ent, dict) else None
                        wid = ent.get("WikidataId") if isinstance(ent, dict) else None
                        if lbl:
                            if lbl not in entity_counts:
                                entity_counts[lbl] = {"count": 0, "wikidata_id": wid, "category": cat}
                            entity_counts[lbl]["count"] += 1

            recent.append({
                "article_id": aid,
                "title": getattr(art, "title", ""),
                "category": cat,
                "subcategory": subcat,
            })

        total_valid = sum(cat_counts.values()) or 1
        cat_dist = {k: round(v / total_valid, 4) for k, v in cat_counts.items()}
        subcat_dist = {k: round(v / total_valid, 4) for k, v in subcat_counts.items()}
        top_cats = [c[0] for c in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]]

        sorted_entities = [
            EntityInterest(
                name=ent_name,
                wikidata_id=ent_info.get("wikidata_id"),
                count=ent_info["count"],
                category=ent_info.get("category"),
            )
            for ent_name, ent_info in sorted(entity_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:10]
        ]

        summary = f"User engaged with {history_size} articles, primarily in {', '.join(top_cats)}." if top_cats else "User reading profile."

        return UserInterestProfile(
            user_id=user_id,
            history_size=history_size,
            category_distribution=cat_dist,
            subcategory_distribution=subcat_dist,
            top_entities=sorted_entities,
            top_categories=top_cats,
            recent_articles=recent[-10:],
            semantic_summary=summary,
        )



@dataclass
class CombinedUserProfile:
    user_id: str
    alpha: float  # weight on short-term profile vs (1 - alpha) on long-term
    short_term_vector: np.ndarray
    long_term_vector: np.ndarray
    combined_vector: np.ndarray
    short_term_categories: Dict[str, float]
    long_term_categories: Dict[str, float]
    category_drift_score: float  # 1 - cosine similarity of category distributions



class DynamicProfileManager:
    """
    Constructs dual-timescale user representations:
    - Short-term profile: Recency-decayed window over the last K interactions.
    - Long-term profile: Cumulative interest distribution across all history.
    - Alpha-blend: user_vec = alpha * short_vec + (1 - alpha) * long_vec.
    """

    def __init__(
        self,
        default_alpha: float = 0.65,
        short_term_window: int = 5,
        recency_decay_lambda: float = 0.2,
        embedding_dim: int = 384,
    ):
        self.default_alpha = default_alpha
        self.short_term_window = short_term_window
        self.recency_decay_lambda = recency_decay_lambda
        self.embedding_dim = embedding_dim

    def build_profile(
        self,
        user_id: str,
        user_history: List[str],
        article_embeddings: Dict[str, np.ndarray],
        article_categories: Dict[str, str],
        alpha: Optional[float] = None,
    ) -> CombinedUserProfile:
        """
        Build short-term, long-term, and combined user profiles.
        """
        if alpha is None:
            alpha = self.default_alpha

        valid_articles = [aid for aid in user_history if aid in article_embeddings]

        if not valid_articles:
            zero_vec = np.zeros(self.embedding_dim, dtype=np.float32)
            return CombinedUserProfile(
                user_id=user_id,
                alpha=alpha,
                short_term_vector=zero_vec,
                long_term_vector=zero_vec,
                combined_vector=zero_vec,
                short_term_categories={},
                long_term_categories={},
                category_drift_score=0.0,
            )

        # 1. Long-Term Profile: Uniform or unweighted mean across full history
        long_vecs = [article_embeddings[aid] for aid in valid_articles]
        long_term_vec = np.mean(long_vecs, axis=0)
        norm_l = np.linalg.norm(long_term_vec)
        if norm_l > 0:
            long_term_vec = long_term_vec / norm_l

        # Long-term category distribution
        long_cat_counts: Dict[str, int] = {}
        for aid in valid_articles:
            cat = article_categories.get(aid, "unknown")
            long_cat_counts[cat] = long_cat_counts.get(cat, 0) + 1
        total_long = len(valid_articles)
        long_cat_dist = {k: round(v / total_long, 4) for k, v in long_cat_counts.items()}

        # 2. Short-Term Profile: Last K articles with exponential recency decay
        short_articles = valid_articles[-self.short_term_window:]
        short_vecs = []
        short_weights = []
        short_cat_counts: Dict[str, float] = {}

        for pos, aid in enumerate(reversed(short_articles)):
            # pos=0 is most recent
            weight = math.exp(-self.recency_decay_lambda * pos)
            short_vecs.append(article_embeddings[aid] * weight)
            short_weights.append(weight)

            cat = article_categories.get(aid, "unknown")
            short_cat_counts[cat] = short_cat_counts.get(cat, 0.0) + weight

        short_term_vec = np.sum(short_vecs, axis=0) / max(sum(short_weights), 1e-9)
        norm_s = np.linalg.norm(short_term_vec)
        if norm_s > 0:
            short_term_vec = short_term_vec / norm_s

        total_short_w = sum(short_cat_counts.values())
        short_cat_dist = {
            k: round(v / total_short_w, 4) for k, v in short_cat_counts.items()
        } if total_short_w > 0 else {}

        # 3. Dynamic Alpha Combination
        combined_vec = (alpha * short_term_vec) + ((1.0 - alpha) * long_term_vec)
        norm_c = np.linalg.norm(combined_vec)
        if norm_c > 0:
            combined_vec = combined_vec / norm_c

        # 4. Interest Drift Metric
        drift_score = self.compute_distribution_drift(long_cat_dist, short_cat_dist)

        return CombinedUserProfile(
            user_id=user_id,
            alpha=alpha,
            short_term_vector=short_term_vec,
            long_term_vector=long_term_vec,
            combined_vector=combined_vec,
            short_term_categories=short_cat_dist,
            long_term_categories=long_cat_dist,
            category_drift_score=drift_score,
        )

    @staticmethod
    def compute_distribution_drift(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
        """Cosine distance between two category distributions in [0, 1]."""
        all_cats = sorted(list(set(dist_a.keys()) | set(dist_b.keys())))
        if not all_cats:
            return 0.0
        v_a = np.array([dist_a.get(c, 0.0) for c in all_cats], dtype=np.float32)
        v_b = np.array([dist_b.get(c, 0.0) for c in all_cats], dtype=np.float32)
        norm_a = np.linalg.norm(v_a)
        norm_b = np.linalg.norm(v_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        cos_sim = float(np.dot(v_a, v_b) / (norm_a * norm_b))
        return round(float(np.clip(1.0 - cos_sim, 0.0, 1.0)), 4)

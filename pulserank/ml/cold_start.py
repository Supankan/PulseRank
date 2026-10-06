"""Cold-Start Recommendations, Onboarding Persona Engine, and Cold-Item Surfacing."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OnboardingPersona:
    selected_categories: List[str]
    selected_subcategories: Optional[List[str]] = None
    seed_article_ids: Optional[List[str]] = None


class ColdStartEngine:
    """
    Handles user cold-start via guided onboarding personas,
    and item cold-start via exploration boosts for newly published articles.
    """

    def __init__(
        self,
        category_embeddings: Optional[Dict[str, np.ndarray]] = None,
        article_embeddings: Optional[Dict[str, np.ndarray]] = None,
        article_metadata: Optional[Dict[str, Any]] = None,
        cold_item_threshold: int = 5,
        exploration_boost: float = 0.15,
    ):
        self.category_embeddings = category_embeddings or {}
        self.article_embeddings = article_embeddings or {}
        self.article_metadata = article_metadata or {}
        self.cold_item_threshold = cold_item_threshold
        self.exploration_boost = exploration_boost

    def create_onboarding_profile(
        self,
        persona: OnboardingPersona,
        embedding_dim: int = 384,
        return_subcategories: bool = False,
    ) -> Any:
        """
        Synthesize user embedding vector, category priors, and subcategory priors from onboarding input.
        """
        vecs = []

        # 1. Seed article vectors
        if persona.seed_article_ids:
            for aid in persona.seed_article_ids:
                if aid in self.article_embeddings:
                    vecs.append(self.article_embeddings[aid])

        # 2. Category centroid vectors
        if persona.selected_categories:
            for cat in persona.selected_categories:
                cat_lower = cat.lower()
                if cat_lower in self.category_embeddings:
                    vecs.append(self.category_embeddings[cat_lower])

        if vecs:
            synthetic_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(synthetic_vec)
            if norm > 0:
                synthetic_vec = synthetic_vec / norm
        else:
            synthetic_vec = np.zeros(embedding_dim, dtype=np.float32)

        # Build category priors
        num_cats = len(persona.selected_categories) if persona.selected_categories else 1
        cat_priors = {cat.lower(): round(1.0 / num_cats, 4) for cat in (persona.selected_categories or [])}

        # Build subcategory priors
        subcat_priors = {}
        if persona.selected_subcategories:
            num_subcats = len(persona.selected_subcategories)
            subcat_priors = {sc.lower(): round(1.0 / num_subcats, 4) for sc in persona.selected_subcategories}

        if return_subcategories:
            return synthetic_vec, cat_priors, subcat_priors
        return synthetic_vec, cat_priors

    def score_onboarding_candidates(
        self,
        synthetic_vec: np.ndarray,
        cat_priors: Dict[str, float],
        subcat_priors: Optional[Dict[str, float]] = None,
        articles: Optional[Dict[str, Any]] = None,
        content_model: Optional[Any] = None,
        popularity_model: Optional[Any] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank catalog candidates for a cold user onboarding persona.
        Principled formulation:
        score = 0.40 * cos_sim + 0.35 * cat_prior + 0.15 * subcat_prior + 0.10 * pop_score
        """
        candidates = []
        has_vec = np.any(synthetic_vec != 0)
        articles = articles or {}
        subcat_priors = subcat_priors or {}

        for aid, art in articles.items():
            cat = getattr(art, "category", "").lower()
            subcat = getattr(art, "subcategory", "").lower()

            c_prior = cat_priors.get(cat, 0.0)
            sc_prior = subcat_priors.get(subcat, 0.0)

            # Cosine semantic similarity if content model and valid vector
            cos_sim = 0.0
            if has_vec and content_model is not None and hasattr(content_model, "compute_similarity"):
                cos_sim = content_model.compute_similarity(synthetic_vec, aid)

            pop_score = popularity_model.get_score(aid) if popularity_model else 0.0

            final_score = (
                0.40 * cos_sim
                + 0.35 * c_prior
                + 0.15 * sc_prior
                + 0.10 * pop_score
            )

            # Filter for candidates matching at least some onboarding relevance or high semantic similarity
            if c_prior > 0 or sc_prior > 0 or cos_sim > 0.4:
                candidates.append({
                    "article_id": aid,
                    "title": getattr(art, "title", ""),
                    "category": getattr(art, "category", ""),
                    "subcategory": getattr(art, "subcategory", ""),
                    "prior_weight": round(c_prior, 4),
                    "semantic_similarity": round(cos_sim, 4),
                    "popularity_prior": round(pop_score, 4),
                    "composite_score": round(final_score, 4),
                })

        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        return candidates[:top_k]

    def identify_cold_items(self, item_interaction_counts: Dict[str, int]) -> Set[str]:
        """Identify articles with interaction count below cold-start threshold."""
        cold = set()
        for aid in self.article_metadata.keys():
            if item_interaction_counts.get(aid, 0) < self.cold_item_threshold:
                cold.add(aid)
        return cold

    def apply_cold_item_exploration(
        self,
        candidates: List[Dict[str, Any]],
        cold_items: Set[str],
        user_category_priors: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Apply exploration boost to cold items if they match user category preferences.
        Prevents popularity starvation for fresh content.
        """
        boosted = []
        for c in candidates:
            c_copy = dict(c)
            aid = c.get("article_id", "")
            is_cold = aid in cold_items
            c_copy["is_cold_item"] = is_cold

            if is_cold:
                cat = c.get("category", "").lower()
                affinity = user_category_priors.get(cat, 0.2)
                boost = self.exploration_boost * affinity
                original_score = c.get("hybrid_score", c.get("score", 0.0))
                c_copy["cold_start_boost"] = round(boost, 4)
                c_copy["final_score"] = round(original_score + boost, 4)
            else:
                c_copy["cold_start_boost"] = 0.0
                c_copy["final_score"] = round(c.get("hybrid_score", c.get("score", 0.0)), 4)

            boosted.append(c_copy)

        boosted.sort(key=lambda x: x["final_score"], reverse=True)
        return boosted

    def slice_evaluation_segments(
        self,
        dev_behaviors: List[Any],
        item_interaction_counts: Dict[str, int],
    ) -> Dict[str, List[Any]]:
        """
        Slice evaluation behaviors into granular history cohorts:
        - 0 history (pure cold user)
        - 1-2 history (emerging user)
        - 3-10 history (warm user)
        - 10+ history (established user)
        - cold items (impressions with at least one cold item)
        """
        zero_history = []
        history_1_to_2 = []
        history_3_to_10 = []
        history_10_plus = []
        cold_item_impressions = []

        cold_items_set = self.identify_cold_items(item_interaction_counts)

        for b in dev_behaviors:
            h_len = len(b.history)
            if h_len == 0:
                zero_history.append(b)
            elif h_len <= 2:
                history_1_to_2.append(b)
            elif h_len <= 10:
                history_3_to_10.append(b)
            else:
                history_10_plus.append(b)

            if any(item.news_id in cold_items_set for item in b.impressions):
                cold_item_impressions.append(b)

        return {
            "zero_history": zero_history,
            "history_1_to_2": history_1_to_2,
            "history_3_to_10": history_3_to_10,
            "history_10_plus": history_10_plus,
            "cold_users": zero_history + history_1_to_2,
            "warm_users": history_3_to_10 + history_10_plus,
            "cold_item_impressions": cold_item_impressions,
            "all": dev_behaviors,
        }

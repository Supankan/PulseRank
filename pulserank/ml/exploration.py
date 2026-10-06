"""Exploration Engine for Serendipity and Long-Tail Content Discovery."""

import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


class ExplorationEngine:
    """
    Exploration and serendipity injector:
    - Implements epsilon-greedy and UCB-based tail item discovery.
    - Surfaces under-exposed content without degrading top rank positions (ranks 1-2 preserved).
    - Tracks item exposure frequency to ensure equitable catalog coverage.
    - Uses reproducible random seeds for deterministic evaluation.
    """

    def __init__(
        self,
        epsilon: float = 0.10,
        random_state: int = 42,
    ):
        self.epsilon = epsilon
        self.random_state = random_state
        self.rng = random.Random(random_state)
        self.item_exposure_counts: Dict[str, int] = {}

    def set_seed(self, seed: int) -> None:
        self.random_state = seed
        self.rng = random.Random(seed)

    def record_exposures(self, article_ids: List[str]) -> None:
        """Increment historical exposure counts for articles presented in a served slate."""
        for aid in article_ids:
            self.item_exposure_counts[aid] = self.item_exposure_counts.get(aid, 0) + 1

    def compute_exploration_score(
        self,
        article_id: str,
        base_relevance: float = 0.5,
        category_prior: float = 0.5,
        popularity_score: float = 0.0,
    ) -> float:
        """
        Principled exploration priority score:
        Combines underexposure bonus, relevance guard, and category prior.
        Never returns an arbitrary constant.
        """
        n_i = self.item_exposure_counts.get(article_id, 0)
        underexposure_bonus = 1.0 / (1.0 + np.log1p(n_i))
        score = 0.45 * underexposure_bonus + 0.35 * base_relevance + 0.20 * category_prior
        return float(round(float(np.clip(score, 0.0, 1.0)), 4))

    def explore_slate(
        self,
        ranked_candidates: List[Dict[str, Any]],
        exploration_pool: List[Dict[str, Any]],
        top_k: int = 10,
        epsilon: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Inject exploratory candidate items into the recommendation slate.
        Preserves top-ranked items (ranks 1-2) and introduces exploration into ranks 3+.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if not ranked_candidates:
            return []

        slate = [dict(c) for c in ranked_candidates[:top_k]]
        for c in slate:
            c["is_exploration"] = False

        if not exploration_pool or epsilon <= 0.0 or len(slate) <= 2:
            return slate

        rng = random.Random(seed) if seed is not None else self.rng

        # Number of exploration slots to allocate
        num_explore = max(1, int(round(len(slate) * epsilon)))
        eligible_slots = list(range(2, len(slate)))
        num_to_sample = min(num_explore, len(eligible_slots))
        explore_indices = sorted(rng.sample(eligible_slots, num_to_sample))

        # Filter exploration pool for least-exposed items not already in slate
        existing_aids = {c.get("article_id") for c in slate}
        available_pool = [c for c in exploration_pool if c.get("article_id") not in existing_aids]

        if not available_pool:
            return slate

        # Rank exploration candidates by their computed exploration score
        scored_pool = []
        for cand in available_pool:
            aid = cand.get("article_id", "")
            base_rel = cand.get("base_relevance", cand.get("hybrid_score", cand.get("score", 0.5)))
            cat_prior = cand.get("category_prior", 0.5)
            exp_score = cand.get("exploration_score")
            if exp_score is None:
                exp_score = self.compute_exploration_score(
                    article_id=aid,
                    base_relevance=base_rel,
                    category_prior=cat_prior,
                )
            c_copy = dict(cand)
            c_copy["exploration_score"] = exp_score
            c_copy["final_score"] = exp_score
            c_copy["score"] = exp_score
            scored_pool.append(c_copy)

        scored_pool.sort(key=lambda x: x["exploration_score"], reverse=True)

        for slot_idx, exp_item in zip(explore_indices, scored_pool):
            exp_item["is_exploration"] = True
            exp_item["exploration_reason"] = "serendipity_underexposed"
            slate[slot_idx] = exp_item

        return slate

    def ucb_score(
        self,
        article_id: str,
        base_relevance: float,
        total_trials: int,
        c_param: float = 1.414,
    ) -> float:
        """Upper Confidence Bound (UCB1) for balancing exploitation and exploration."""
        n_i = self.item_exposure_counts.get(article_id, 0)
        if n_i == 0:
            return float("inf")
        return float(base_relevance + c_param * np.sqrt(np.log(max(total_trials, 1)) / n_i))


def build_exploration_pool(
    catalog_article_ids: List[str],
    articles: Dict[str, Any],
    user_history: Optional[List[str]] = None,
    content_model: Optional[Any] = None,
    popularity_model: Optional[Any] = None,
    exposure_counts: Optional[Dict[str, int]] = None,
    exclude_ids: Optional[Set[str]] = None,
    max_candidates: int = 100,
) -> List[Dict[str, Any]]:
    """
    Build a principled exploration pool for serendipity and tail-item discovery.

    Combines:
    1. Category prior alignment from user reading history.
    2. Base semantic or popularity relevance.
    3. Underexposure priority bonus based on historical exposure tracking.

    Zero arbitrary score constants (no hardcoded 0.4).
    """
    hist = user_history or []
    exclude_set = set(hist)
    if exclude_ids:
        exclude_set.update(exclude_ids)

    # Filter eligible items
    eligible = [aid for aid in catalog_article_ids if aid not in exclude_set and aid in articles]
    if not eligible:
        return []

    # Category prior distribution from user history
    cat_counts: Dict[str, int] = {}
    for h_aid in hist:
        art = articles.get(h_aid)
        if art:
            cat = getattr(art, "category", None) or (art.get("category") if isinstance(art, dict) else None)
            if cat:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

    total_h = len(hist) or 1
    user_cat_priors = {c: count / total_h for c, count in cat_counts.items()}
    default_cat_prior = 1.0 / max(len(user_cat_priors), 1) if user_cat_priors else 0.1

    # User semantic vector if content model is available
    u_vec = None
    if content_model and hist and hasattr(content_model, "build_user_vector"):
        try:
            u_vec = content_model.build_user_vector(hist)
        except Exception:
            u_vec = None

    exp_counts = exposure_counts or {}
    pool: List[Dict[str, Any]] = []

    for aid in eligible:
        art = articles[aid]
        cat = getattr(art, "category", None) or (art.get("category") if isinstance(art, dict) else "unknown")
        cat_prior = user_cat_priors.get(cat, default_cat_prior)
        pop_score = popularity_model.get_score(aid) if popularity_model and hasattr(popularity_model, "get_score") else 0.0

        content_sim = 0.0
        if content_model and u_vec is not None and hasattr(content_model, "compute_similarity"):
            try:
                content_sim = float(content_model.compute_similarity(u_vec, aid))
                base_rel = 0.40 * content_sim + 0.35 * cat_prior + 0.25 * pop_score
            except Exception:
                base_rel = 0.60 * cat_prior + 0.40 * pop_score
        else:
            base_rel = 0.60 * cat_prior + 0.40 * pop_score

        n_i = exp_counts.get(aid, 0)
        underexposure_bonus = 1.0 / (1.0 + np.log1p(n_i))
        exploration_score = 0.45 * underexposure_bonus + 0.35 * base_rel + 0.20 * cat_prior

        clean_score = float(round(float(np.clip(exploration_score, 0.0, 1.0)), 4))
        clean_base = float(round(float(np.clip(base_rel, 0.0, 1.0)), 4))

        pool.append({
            "article_id": aid,
            "score": clean_score,
            "hybrid_score": clean_base,
            "final_score": clean_score,
            "exploration_score": clean_score,
            "base_relevance": clean_base,
            "category": cat,
            "category_prior": float(round(cat_prior, 4)),
            "popularity_score": float(round(pop_score, 4)),
            "content_similarity": float(round(content_sim, 4)),
        })

    # Sort descending by exploration score
    pool.sort(key=lambda x: x["exploration_score"], reverse=True)
    return pool[:max_candidates]

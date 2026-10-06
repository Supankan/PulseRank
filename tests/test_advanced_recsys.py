"""Unit and integration tests for Advanced Recommender & Evaluation components."""

import pytest
import numpy as np
from pulserank.ml.profiles import DynamicProfileManager
from pulserank.ml.cold_start import ColdStartEngine, OnboardingPersona
from pulserank.ml.diversity import DiversityEngine
from pulserank.ml.exploration import ExplorationEngine
from pulserank.ml.ablation import AblationFramework
from pulserank.ml.segment_analysis import SegmentDiagnosticsEngine
from pulserank.ml.evaluation import Evaluator


def test_dynamic_profile_alpha_blend():
    manager = DynamicProfileManager(default_alpha=0.6, short_term_window=3, embedding_dim=4)
    embs = {
        "A1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "A2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "A3": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "A4": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    cats = {"A1": "tech", "A2": "tech", "A3": "sports", "A4": "sports"}

    profile = manager.build_profile(
        user_id="U1",
        user_history=["A1", "A2", "A3", "A4"],
        article_embeddings=embs,
        article_categories=cats,
        alpha=0.7,
    )

    assert profile.user_id == "U1"
    assert profile.alpha == 0.7
    assert profile.combined_vector.shape == (4,)
    assert np.isclose(np.linalg.norm(profile.combined_vector), 1.0, atol=1e-3)
    assert "sports" in profile.short_term_categories
    assert profile.category_drift_score >= 0.0


def test_cold_start_engine():
    art_meta = {
        "A1": type("Art", (), {"category": "sports", "title": "Game 1"})(),
        "A2": type("Art", (), {"category": "tech", "title": "Chip 1"})(),
        "A3": type("Art", (), {"category": "sports", "title": "Match 2"})(),
    }
    embs = {
        "A1": np.array([1.0, 0.0], dtype=np.float32),
        "A2": np.array([0.0, 1.0], dtype=np.float32),
    }
    engine = ColdStartEngine(article_embeddings=embs, article_metadata=art_meta, cold_item_threshold=5)

    persona = OnboardingPersona(selected_categories=["sports", "tech"], seed_article_ids=["A1"])
    vec, priors = engine.create_onboarding_profile(persona, embedding_dim=2)

    assert "sports" in priors
    assert "tech" in priors
    assert vec.shape == (2,)

    cold_items = engine.identify_cold_items({"A1": 10, "A2": 1, "A3": 0})
    assert "A2" in cold_items
    assert "A3" in cold_items
    assert "A1" not in cold_items

    candidates = [
        {"article_id": "A1", "category": "sports", "score": 0.8},
        {"article_id": "A2", "category": "tech", "score": 0.7},
    ]
    boosted = engine.apply_cold_item_exploration(candidates, cold_items, priors)
    assert len(boosted) == 2
    # Cold item A2 should have cold_start_boost > 0
    cold_c = next(c for c in boosted if c["article_id"] == "A2")
    assert cold_c["is_cold_item"] is True
    assert cold_c["cold_start_boost"] > 0.0


def test_diversity_engine_mmr_and_metrics():
    embs = {
        "A1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "A2": np.array([0.98, 0.05, 0.0], dtype=np.float32),  # very similar to A1
        "A3": np.array([0.0, 1.0, 0.0], dtype=np.float32),   # distinct from A1
        "A4": np.array([0.0, 0.0, 1.0], dtype=np.float32),   # distinct
    }
    pops = {"A1": 100, "A2": 80, "A3": 20, "A4": 5}
    engine = DiversityEngine(article_embeddings=embs, article_popularities=pops)

    candidates = [
        {"article_id": "A1", "score": 0.95, "final_score": 0.95, "category": "tech"},
        {"article_id": "A2", "score": 0.94, "final_score": 0.94, "category": "tech"},
        {"article_id": "A3", "score": 0.85, "final_score": 0.85, "category": "sports"},
        {"article_id": "A4", "score": 0.70, "final_score": 0.70, "category": "finance"},
    ]

    # MMR with lambda=0.5: A3 should be prioritized over near-duplicate A2 for diversity
    selected = engine.rerank_mmr(candidates, lambda_param=0.5, top_k=3)
    assert len(selected) == 3
    selected_ids = [s["article_id"] for s in selected]
    assert selected_ids[0] == "A1"
    assert "A3" in selected_ids

    # Metrics
    metrics = engine.compute_metrics(selected)
    assert metrics.intra_list_diversity > 0.0
    assert metrics.category_entropy > 0.0
    assert metrics.novelty > 0.0

    curve = engine.generate_tradeoff_curve(candidates, lambda_steps=[0.0, 0.5, 1.0], top_k=3)
    assert len(curve) == 3


def test_exploration_engine():
    engine = ExplorationEngine(epsilon=0.2, random_state=42)
    ranked = [
        {"article_id": f"R{i}", "score": 1.0 - i * 0.1, "category": "news"}
        for i in range(10)
    ]
    pool = [
        {"article_id": f"E{i}", "score": 0.4, "category": "tail"}
        for i in range(5)
    ]

    explored = engine.explore_slate(ranked, pool, top_k=8, epsilon=0.25)
    assert len(explored) == 8
    # Top 2 ranks should be preserved
    assert explored[0]["article_id"] == "R0"
    assert explored[1]["article_id"] == "R1"
    # An exploration item should be injected
    has_explore = any(c.get("is_exploration", False) for c in explored)
    assert has_explore

    # UCB score
    engine.record_exposures(["R0", "R0", "R0", "E1"])
    ucb_r0 = engine.ucb_score("R0", 0.5, total_trials=10)
    ucb_unseen = engine.ucb_score("UNSEEN", 0.5, total_trials=10)
    assert ucb_unseen > ucb_r0  # Unseen items get optimistic exploration boost

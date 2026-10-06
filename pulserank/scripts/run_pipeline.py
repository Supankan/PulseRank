"""Unified End-to-End Orchestration CLI for PulseRank Pipeline."""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

from pulserank.config import settings
from pulserank.data.preprocessor import ProcessedDataset, build_processed_dataset
from pulserank.data.fixtures import write_mind_tsv_fixtures, is_synthetic_fixture_dir
from pulserank.data.interactions import build_popularity_events, build_als_interactions
from pulserank.data.provenance import compute_dataset_fingerprint, ArtifactManifest
from pulserank.ml.popularity import PopularityRecommender
from pulserank.ml.content_based import ContentBasedRecommender
from pulserank.ml.collaborative import ImplicitALSRecommender
from pulserank.ml.hybrid import HybridCandidateGenerator
from pulserank.ml.ranker import LambdaMARTRanker, FeatureExtractor
from pulserank.ml.evaluation import Evaluator, save_evaluation_results
from pulserank.ml.diversity import DiversityEngine
from pulserank.ml.exploration import ExplorationEngine
from pulserank.ml.profiles import DynamicProfileManager
from pulserank.ml.ablation import AblationFramework
from pulserank.nlp.classifier_baseline import TFIDFTopicClassifier, train_test_split_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end PulseRank ML pipeline.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Pipeline execution mode: 'quick' for rapid verification or 'full' for complete MIND training.",
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"=== Starting PulseRank Orchestration Pipeline (Mode: {args.mode.upper()}) ===")

    settings.ensure_directories()
    stage_statuses = {}

    # 1. Prepare / verify dataset
    train_dir = settings.raw_data_dir / "MINDsmall_train"
    dev_dir = settings.raw_data_dir / "MINDsmall_dev"
    if not (train_dir / "news.tsv").exists():
        logger.info("Bootstrapping synthetic MIND fixtures...")
        write_mind_tsv_fixtures(settings.raw_data_dir)

    is_synth = is_synthetic_fixture_dir(settings.raw_data_dir)
    source_label = "synthetic_fixture" if is_synth else "mind-small"
    if is_synth:
        logger.warning("[Demo Fixture - Not a MIND Benchmark Result] Synthetic data will be used.")

    dataset_path = settings.processed_data_dir / "dataset.pkl"
    if not dataset_path.exists():
        logger.info("Building processed dataset from raw TSVs...")
        dataset = build_processed_dataset(
            train_news_path=train_dir / "news.tsv",
            train_behaviors_path=train_dir / "behaviors.tsv",
            dev_news_path=dev_dir / "news.tsv" if (dev_dir / "news.tsv").exists() else None,
            dev_behaviors_path=dev_dir / "behaviors.tsv" if (dev_dir / "behaviors.tsv").exists() else None,
            dataset_source=source_label,
        )
        dataset.save(settings.processed_data_dir)
    else:
        logger.info(f"Loading existing dataset from {dataset_path}...")
        dataset = ProcessedDataset.load(dataset_path)

    stage_statuses["dataset_preparation"] = "completed"
    fingerprint = compute_dataset_fingerprint(dataset)
    logger.info(f"Dataset Fingerprint (SHA-256): {fingerprint}")

    # 2. Train Popularity Baseline
    logger.info("Step 2/7: Fitting Popularity Recommender...")
    pop_interactions = build_popularity_events(dataset.train_behaviors, include_history_as_prior=True)
    pop_model = PopularityRecommender(decay_half_life_hours=settings.popularity_decay_hours)
    pop_model.fit(pop_interactions, reference_time=dataset.reference_time)
    pop_path = settings.artifacts_dir / "popularity.pkl"
    pop_model.save(pop_path, dataset_source=source_label, dataset_fingerprint=fingerprint)
    stage_statuses["popularity_baseline"] = "completed"

    # 3. Train Content-Based Model
    logger.info("Step 3/7: Fitting Content-Based Embeddings...")
    content_model = ContentBasedRecommender(
        model_name=settings.embedding_model_name,
        embedding_dim=settings.embedding_dim,
    )
    content_articles = list(dataset.articles.values())
    if args.mode == "quick" and len(content_articles) > 150:
        content_articles = content_articles[:150]
    content_model.build_index({getattr(a, "news_id", getattr(a, "article_id", "")): a for a in content_articles})
    content_dir = settings.artifacts_dir / "content"
    content_model.save(content_dir, dataset_source=source_label, dataset_fingerprint=fingerprint)
    stage_statuses["content_embeddings"] = "completed"

    # 4. Train Collaborative ALS Model
    logger.info("Step 4/7: Fitting Collaborative ALS Recommender...")
    als_interactions = build_als_interactions(dataset.train_behaviors)
    als_model = ImplicitALSRecommender(
        factors=32 if args.mode == "quick" else settings.als_factors,
        iterations=5 if args.mode == "quick" else settings.als_iterations,
    )
    als_model.fit(als_interactions)
    als_path = settings.artifacts_dir / "als.pkl"
    als_model.save(als_path, dataset_source=source_label, dataset_fingerprint=fingerprint)
    stage_statuses["collaborative_als"] = "completed"

    # 5. Dynamic Profiles and Feature Extractor
    dyn_profile_mgr = DynamicProfileManager(embedding_dim=settings.embedding_dim)
    feat_extractor = FeatureExtractor(
        articles=dataset.articles,
        user_histories=dataset.user_histories,
        content_model=content_model,
        collab_model=als_model,
        popularity_model=pop_model,
        dynamic_profile_manager=dyn_profile_mgr,
    )

    # 6. Train LightGBM LambdaMART Ranker
    logger.info("Step 5/7: Fitting LambdaMART Pointwise Ranker...")
    hybrid_gen = HybridCandidateGenerator(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=als_model,
    )
    ranker = LambdaMARTRanker(
        n_estimators=30 if args.mode == "quick" else settings.ranker_n_estimators,
        num_leaves=15 if args.mode == "quick" else settings.ranker_num_leaves,
    )
    train_beh = dataset.train_behaviors[:100] if args.mode == "quick" else dataset.train_behaviors
    ranker.fit_from_behaviors(train_beh, feat_extractor)
    ranker_path = settings.artifacts_dir / "ranker.pkl"
    ranker.save(ranker_path, dataset_source=source_label, dataset_fingerprint=fingerprint)
    stage_statuses["lambdamart_ranker"] = "completed"

    # 7. Fit and Evaluate Topic Classifier Baseline
    logger.info("Step 6/7: Fitting and Evaluating TF-IDF Topic Classifier...")
    topic_clf = TFIDFTopicClassifier()
    all_articles = list(dataset.articles.values())
    candidate_arts = all_articles[:300] if (args.mode == "quick" and len(all_articles) > 300) else all_articles

    train_arts, test_arts, eval_mode = train_test_split_articles(candidate_arts, test_size=0.2, random_state=42)
    train_texts = [f"{a.title} {a.abstract}" for a in train_arts]
    train_labels = [a.category for a in train_arts]
    test_texts = [f"{a.title} {a.abstract}" for a in test_arts]
    test_labels = [a.category for a in test_arts]

    topic_clf.fit(train_texts, train_labels)
    tfidf_eval_metrics = topic_clf.evaluate(
        test_texts,
        test_labels,
        evaluation_mode=eval_mode,
        train_sample_count=len(train_arts),
        test_sample_count=len(test_arts),
        split_seed=42,
    )
    if eval_mode == "demo_resubstitution_not_held_out":
        logger.info(
            f"TF-IDF Classifier Evaluation (DEMO RESUBSTITUTION - NOT HELD OUT): "
            f"Accuracy={tfidf_eval_metrics.get('accuracy')}, Macro-F1={tfidf_eval_metrics.get('macro_f1')}"
        )
    else:
        logger.info(
            f"TF-IDF Classifier Evaluation (HELD-OUT): Accuracy={tfidf_eval_metrics.get('accuracy')}, "
            f"Macro-F1={tfidf_eval_metrics.get('macro_f1')}, Brier={tfidf_eval_metrics.get('brier_score')}"
        )

    topic_clf.save(
        settings.artifacts_dir / "tfidf_topic_classifier.pkl",
        settings.artifacts_dir / "tfidf_topic_classifier_manifest.json",
        dataset_source=source_label,
        dataset_fingerprint=fingerprint,
    )
    stage_statuses["topic_classifier_baseline"] = "completed"
    stage_statuses["distilbert_classifier"] = "skipped_optional"
    stage_statuses["transformer_cross_encoder"] = "skipped_optional"

    # 8. Evaluate and Ablate
    logger.info("Step 7/7: Running Impression Benchmarks & 9-Stage Ablation...")
    dev_eval = dataset.dev_behaviors[:80] if args.mode == "quick" else dataset.dev_behaviors
    evaluator = Evaluator(dev_eval)
    eval_results = evaluator.evaluate_all(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=als_model,
        hybrid_generator=hybrid_gen,
        ranker_model=ranker,
        feature_extractor=feat_extractor,
        dataset_source=source_label,
    )
    save_evaluation_results(eval_results, settings.artifacts_dir / "evaluations.json")
    stage_statuses["impression_evaluation"] = "completed"

    div_engine = DiversityEngine(
        article_embeddings=content_model.article_embeddings,
        article_popularities=pop_model.scores,
    )
    explore_engine = ExplorationEngine()
    ablation_framework = AblationFramework(
        evaluator=evaluator,
        diversity_engine=div_engine,
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=als_model,
        hybrid_generator=hybrid_gen,
        ranker_model=ranker,
        feature_extractor=feat_extractor,
        dynamic_profile_manager=dyn_profile_mgr,
        exploration_engine=explore_engine,
        catalog_article_ids=list(dataset.articles.keys()),
        dataset_source=source_label,
        dataset_fingerprint=fingerprint,
        train_behaviors=dataset.train_behaviors,
    )
    ablation_results = ablation_framework.run_all_ablations(
        max_impressions=50 if args.mode == "quick" else 200,
        artifacts_dir=settings.artifacts_dir,
    )
    stage_statuses["ablation_framework"] = "completed"

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"=== Pipeline Execution Succeeded in {elapsed}s ===")
    logger.info(f"Ablation configurations tested: {len(ablation_results)}")
    for r in ablation_results:
        logger.info(f"  - {r.config_name}: nDCG@5={r.ndcg_at_5:.4f}, ILD={r.ild:.4f}, Delta={r.delta_ndcg5_vs_full:+.4f}")

    # Write truthful machine-readable execution summary
    summary = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": elapsed,
        "mode": args.mode,
        "dataset_source": source_label,
        "dataset_fingerprint": fingerprint,
        "num_articles": len(dataset.articles),
        "num_train_behaviors": len(dataset.train_behaviors),
        "num_dev_behaviors": len(dataset.dev_behaviors),
        "tfidf_classifier_evaluation": {
            "evaluation_mode": tfidf_eval_metrics.get("evaluation_mode"),
            "train_sample_count": tfidf_eval_metrics.get("train_sample_count"),
            "test_sample_count": tfidf_eval_metrics.get("test_sample_count"),
            "split_seed": tfidf_eval_metrics.get("split_seed"),
            "accuracy": tfidf_eval_metrics.get("accuracy"),
            "macro_f1": tfidf_eval_metrics.get("macro_f1"),
            "weighted_f1": tfidf_eval_metrics.get("weighted_f1"),
            "log_loss": tfidf_eval_metrics.get("log_loss"),
            "brier_score": tfidf_eval_metrics.get("brier_score"),
            "num_test_samples": tfidf_eval_metrics.get("num_test_samples"),
        },
        "stages": stage_statuses,
    }
    summary_path = settings.artifacts_dir / "pipeline_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote machine-readable pipeline run summary to {summary_path}")


if __name__ == "__main__":
    main()

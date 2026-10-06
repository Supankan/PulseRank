"""CLI Script: Execute offline evaluation across all recommendation models."""

import argparse
import logging
from pathlib import Path

from pulserank.config import settings
from pulserank.data.preprocessor import ProcessedDataset
from pulserank.ml.popularity import PopularityRecommender
from pulserank.ml.content_based import ContentBasedRecommender
from pulserank.ml.collaborative import ImplicitALSRecommender
from pulserank.ml.hybrid import HybridCandidateGenerator
from pulserank.ml.ranker import LambdaMARTRanker, FeatureExtractor
from pulserank.ml.evaluation import OfflineEvaluator, save_evaluation_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all recommendation models on validation split.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(settings.processed_data_dir / "dataset.pkl"),
        help="Path to preprocessed dataset.pkl.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(settings.artifacts_dir / "evaluation_results.json"),
        help="Path to save evaluation results JSON.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Run 'python -m pulserank.scripts.prepare_data' first.", dataset_path)
        return

    logger.info("Loading dataset from %s...", dataset_path)
    dataset = ProcessedDataset.load(dataset_path)

    if not dataset.dev_behaviors:
        logger.error("No validation behavior records available for offline evaluation.")
        return

    # Load artifacts
    pop_path = settings.artifacts_dir / "popularity.pkl"
    content_dir = settings.artifacts_dir / "content"
    als_path = settings.artifacts_dir / "als.pkl"
    ranker_path = settings.artifacts_dir / "ranker.pkl"

    logger.info("Loading trained models...")
    pop_model = PopularityRecommender.load(pop_path) if pop_path.exists() else None
    content_model = ContentBasedRecommender.load(content_dir) if (content_dir / "content_meta.pkl").exists() else None
    collab_model = ImplicitALSRecommender.load(als_path) if als_path.exists() else None
    ranker_model = LambdaMARTRanker.load(ranker_path) if ranker_path.exists() else None

    if not pop_model or not content_model or not collab_model or not ranker_model:
        logger.error(
            "Missing one or more model artifacts. Please ensure popularity, content, collaborative, and ranker are trained."
        )
        return

    hybrid_gen = HybridCandidateGenerator(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=collab_model,
    )

    extractor = FeatureExtractor(
        articles=dataset.articles,
        user_histories=dataset.user_histories,
        content_model=content_model,
        collab_model=collab_model,
        popularity_model=pop_model,
    )

    dataset_source = getattr(dataset, "dataset_source", "mind-small")
    is_fixture = dataset_source == "synthetic_fixture"

    if is_fixture:
        print("\n" + "*" * 80)
        print("*  NOTICE: [DEMO FIXTURE - NOT A MIND BENCHMARK RESULT]                        *")
        print("*  Evaluation running on synthetic test fixtures. Metrics reflect pipeline     *")
        print("*  smoke-test execution, NOT official Microsoft News Dataset benchmark scores. *")
        print("*" * 80 + "\n")

    logger.info("Evaluating impression ranking on %d validation impressions...", len(dataset.dev_behaviors))
    evaluator = OfflineEvaluator(dataset.dev_behaviors)
    impression_results = evaluator.evaluate_all(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=collab_model,
        hybrid_generator=hybrid_gen,
        ranker_model=ranker_model,
        feature_extractor=extractor,
        dataset_source=dataset_source,
    )

    logger.info("Evaluating whole-catalog retrieval on %d catalog articles...", len(dataset.articles))
    catalog_results = evaluator.evaluate_catalog_retrieval_all(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=collab_model,
        hybrid_generator=hybrid_gen,
        ranker_model=ranker_model,
        feature_extractor=extractor,
        catalog_article_ids=list(dataset.articles.keys()),
        k=10,
        dataset_source=dataset_source,
    )

    all_results = impression_results + catalog_results
    save_evaluation_results(all_results, output_path)

    # Display Impression-Level Benchmarks
    print("\n" + "=" * 85)
    provenance_tag = "[DEMO FIXTURE]" if is_fixture else "[MIND-SMALL BENCHMARK]"
    print(f"{'PULSERANK IMPRESSION RANKING BENCHMARKS ' + provenance_tag:^85}")
    print("=" * 85)
    header1 = f"{'Model':<35} | {'AUC':<6} | {'MRR':<6} | {'nDCG@5':<6} | {'nDCG@10':<7} | {'Recall@5':<8} | {'Imp. Cov':<8}"
    print(header1)
    print("-" * len(header1))
    for r in impression_results:
        print(
            f"{r.model_name:<35} | {r.auc:<6.4f} | {r.mrr:<6.4f} | {r.ndcg_at_5:<6.4f} | {r.ndcg_at_10:<7.4f} | {r.recall_at_k:<8.4f} | {r.impression_coverage:<8.2%}"
        )
    print("=" * 85)

    # Display Whole-Catalog Retrieval Benchmarks
    print("\n" + "=" * 85)
    print(f"{'PULSERANK WHOLE-CATALOG RETRIEVAL BENCHMARKS ' + provenance_tag:^85}")
    print("=" * 85)
    header2 = f"{'Model':<35} | {'HitRate@10':<10} | {'Recall@10':<10} | {'Catalog Coverage':<16} | {'User Cov':<8}"
    print(header2)
    print("-" * len(header2))
    for r in catalog_results:
        hr = r.hit_rate_at_k if r.hit_rate_at_k is not None else 0.0
        cc = r.catalog_coverage if r.catalog_coverage is not None else 0.0
        print(
            f"{r.model_name:<35} | {hr:<10.4f} | {r.recall_at_k:<10.4f} | {cc:<16.2%} | {r.impression_coverage:<8.2%}"
        )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()

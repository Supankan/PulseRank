"""CLI Script: Extract impression features and train LightGBM LambdaMART ranker."""

import argparse
import logging
from pathlib import Path

from pulserank.config import settings
from pulserank.data.preprocessor import ProcessedDataset
from pulserank.ml.popularity import PopularityRecommender
from pulserank.ml.content_based import ContentBasedRecommender
from pulserank.ml.collaborative import ImplicitALSRecommender
from pulserank.ml.ranker import LambdaMARTRanker, FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM LambdaMART ranking model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(settings.processed_data_dir / "dataset.pkl"),
        help="Path to preprocessed dataset.pkl.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(settings.artifacts_dir / "ranker.pkl"),
        help="Path to save trained ranker model artifact.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Run 'python -m pulserank.scripts.prepare_data' first.", dataset_path)
        return

    logger.info("Loading dataset from %s...", dataset_path)
    dataset = ProcessedDataset.load(dataset_path)

    # Load component models
    pop_path = settings.artifacts_dir / "popularity.pkl"
    content_dir = settings.artifacts_dir / "content"
    als_path = settings.artifacts_dir / "als.pkl"

    if pop_path.exists():
        popularity_model = PopularityRecommender.load(pop_path)
    else:
        from pulserank.data.interactions import build_popularity_events
        interactions = build_popularity_events(dataset.train_behaviors)
        popularity_model = PopularityRecommender(decay_half_life_hours=settings.popularity_decay_hours)
        popularity_model.fit(interactions, reference_time=dataset.reference_time)
        popularity_model.save(pop_path)

    content_model = ContentBasedRecommender.load(content_dir) if (content_dir / "content_meta.pkl").exists() else None
    collab_model = ImplicitALSRecommender.load(als_path) if als_path.exists() else None

    logger.info("Building feature extractor...")
    extractor = FeatureExtractor(
        articles=dataset.articles,
        user_histories=dataset.user_histories,
        content_model=content_model,
        collab_model=collab_model,
        popularity_model=popularity_model,
    )

    logger.info("Initializing and fitting LambdaMART ranker...")
    ranker = LambdaMARTRanker(
        learning_rate=settings.ranker_learning_rate,
        num_leaves=settings.ranker_num_leaves,
        max_depth=settings.ranker_max_depth,
        n_estimators=settings.ranker_n_estimators,
        random_state=settings.ranker_random_state,
    )
    ranker.fit_from_behaviors(
        behaviors=dataset.train_behaviors,
        feature_extractor=extractor,
    )

    from datetime import datetime, timezone
    from pulserank.data.provenance import ArtifactManifest, compute_dataset_fingerprint

    ds_source = getattr(dataset, "dataset_source", "synthetic_fixture")
    fingerprint = compute_dataset_fingerprint(
        news_path=settings.raw_data_dir / "MINDsmall_train" / "news.tsv",
        behaviors_path=settings.raw_data_dir / "MINDsmall_train" / "behaviors.tsv",
        dataset_source=ds_source,
    )

    ranker.manifest = ArtifactManifest(
        model_name="LightGBM LambdaMART",
        model_version="2.0.0",
        dataset_source=ds_source,
        dataset_fingerprint=fingerprint,
        hyperparameters={
            "learning_rate": settings.ranker_learning_rate,
            "num_leaves": settings.ranker_num_leaves,
            "max_depth": settings.ranker_max_depth,
            "n_estimators": settings.ranker_n_estimators,
            "features": ranker.feature_names,
        },
        random_seed=settings.ranker_random_state,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    ranker.save(output_path)
    logger.info("Ranker and manifest saved to %s", output_path)

    importances = ranker.get_feature_importances()
    logger.info("Relative Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %-25s: %.4f", feat, imp)


if __name__ == "__main__":
    main()

"""CLI Script: Train implicit feedback ALS collaborative recommender."""

import argparse
import logging
from pathlib import Path

from pulserank.config import settings
from pulserank.data.preprocessor import ProcessedDataset
from pulserank.ml.collaborative import ImplicitALSRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Implicit ALS collaborative model.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(settings.processed_data_dir / "dataset.pkl"),
        help="Path to preprocessed dataset.pkl.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(settings.artifacts_dir / "als.pkl"),
        help="Path to save trained ALS model artifact.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output_path)

    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Run 'python -m pulserank.scripts.prepare_data' first.", dataset_path)
        return

    logger.info("Loading dataset from %s...", dataset_path)
    dataset = ProcessedDataset.load(dataset_path)

    from pulserank.data.interactions import build_als_interactions
    interactions = build_als_interactions(dataset.train_behaviors)
    logger.info("Extracted %d unique interactions for collaborative training.", len(interactions))

    recommender = ImplicitALSRecommender(
        factors=settings.als_factors,
        regularization=settings.als_regularization,
        iterations=settings.als_iterations,
        alpha=settings.als_alpha,
    )
    recommender.fit(interactions)
    recommender.save(output_path)
    logger.info("Implicit ALS model saved to %s", output_path)

    # Save manifest
    from pulserank.data.provenance import ArtifactManifest, compute_dataset_fingerprint
    manifest = ArtifactManifest(
        artifact_type="collaborative_als",
        dataset_fingerprint=compute_dataset_fingerprint(dataset),
        feature_schema=["user_id", "news_id", "interaction_weight"],
        parameters={
            "factors": settings.als_factors,
            "regularization": settings.als_regularization,
            "iterations": settings.als_iterations,
            "alpha": settings.als_alpha,
            "num_users": len(recommender.user_to_idx),
            "num_items": len(recommender.item_to_idx),
            "num_interactions": len(interactions),
        },
    )
    manifest_path = output_path.parent / "als_manifest.json"
    manifest.save(manifest_path)
    logger.info("Collaborative manifest saved to %s", manifest_path)


if __name__ == "__main__":
    main()


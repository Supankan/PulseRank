"""CLI Script: Generate Sentence Transformer embeddings and build FAISS vector index."""

import argparse
import logging
from pathlib import Path

from pulserank.config import settings
from pulserank.data.preprocessor import ProcessedDataset
from pulserank.ml.content_based import ContentBasedRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate article embeddings and construct FAISS index.")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=str(settings.processed_data_dir / "dataset.pkl"),
        help="Path to preprocessed dataset.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.artifacts_dir / "content"),
        help="Directory to save content embeddings and FAISS index.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=settings.embedding_model_name,
        help="Sentence Transformers model identifier.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Run 'python -m pulserank.scripts.prepare_data' first.", dataset_path)
        return

    logger.info("Loading dataset from %s...", dataset_path)
    dataset = ProcessedDataset.load(dataset_path)

    logger.info("Building content index for %d articles using '%s'...", len(dataset.articles), args.model_name)
    recommender = ContentBasedRecommender(
        model_name=args.model_name,
        embedding_dim=settings.embedding_dim,
    )
    recommender.build_index(dataset.articles)
    recommender.save(output_dir)
    logger.info("Embeddings and FAISS index saved to %s", output_dir)


if __name__ == "__main__":
    main()

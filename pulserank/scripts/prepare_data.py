"""CLI Script: Ingest and preprocess MIND-small dataset."""

import argparse
import logging
import sys
from pathlib import Path

from pulserank.config import settings
from pulserank.data.preprocessor import build_processed_dataset
from pulserank.data.fixtures import write_mind_tsv_fixtures, is_synthetic_fixture_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess MIND-small dataset.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(settings.raw_data_dir),
        help="Directory containing MINDsmall_train and MINDsmall_dev folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(settings.processed_data_dir),
        help="Directory to store processed dataset artifacts.",
    )
    parser.add_argument(
        "--use-fixtures-if-missing",
        action="store_true",
        default=True,
        help="Generate synthetic authentic MIND fixtures if raw data is not present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick parsing testing.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    out_path = Path(args.output_dir)

    train_dir = raw_path / "MINDsmall_train"
    dev_dir = raw_path / "MINDsmall_dev"

    train_news = train_dir / "news.tsv"
    train_behaviors = train_dir / "behaviors.tsv"
    dev_news = dev_dir / "news.tsv"
    dev_behaviors = dev_dir / "behaviors.tsv"

    if not train_news.exists() or not train_behaviors.exists():
        if args.use_fixtures_if_missing:
            logger.warning("MIND-small raw dataset not detected in %s. Generating authentic MIND fixtures...", raw_path)
            write_mind_tsv_fixtures(raw_path)
        else:
            logger.error(
                "Missing expected MIND files at %s and %s.\n"
                "Please place MINDsmall_train.zip (extracted) and MINDsmall_dev.zip (extracted) into %s.",
                train_news,
                train_behaviors,
                raw_path,
            )
            sys.exit(1)

    dataset_source = "synthetic_fixture" if is_synthetic_fixture_dir(raw_path) else "mind-small"

    logger.info("Building processed dataset from %s (source: %s)...", raw_path, dataset_source)
    dataset = build_processed_dataset(
        train_news_path=train_news,
        train_behaviors_path=train_behaviors,
        dev_news_path=dev_news if dev_news.exists() else None,
        dev_behaviors_path=dev_behaviors if dev_behaviors.exists() else None,
        limit=args.limit,
        dataset_source=dataset_source,
    )

    dataset.save(out_path)
    logger.info(
        "Preprocessing complete! Articles: %d, Users: %d, Train Behaviors: %d, Dev Behaviors: %d",
        len(dataset.articles),
        len(dataset.user_histories),
        len(dataset.train_behaviors),
        len(dataset.dev_behaviors),
    )


if __name__ == "__main__":
    main()

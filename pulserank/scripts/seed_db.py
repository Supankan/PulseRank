"""CLI Script: Seed database with MIND metadata and evaluation runs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pulserank.config import settings
from pulserank.db.session import SessionLocal, init_db
from pulserank.db.models import User, Article, Interaction, ModelMetadata, EvaluationResult
from pulserank.data.preprocessor import ProcessedDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing database schema...")
    init_db()

    dataset_path = settings.processed_data_dir / "dataset.pkl"
    if not dataset_path.exists():
        logger.error("Dataset not found at %s. Please run prepare_data first.", dataset_path)
        return

    dataset = ProcessedDataset.load(dataset_path)
    db = SessionLocal()

    try:
        # 1. Seed Articles
        logger.info("Seeding articles into database...")
        for aid, art in dataset.articles.items():
            existing = db.query(Article).filter(Article.article_id == aid).first()
            if not existing:
                db_art = Article(
                    article_id=art.news_id,
                    category=art.category,
                    subcategory=art.subcategory,
                    title=art.title,
                    abstract=art.abstract,
                    url=art.url,
                    title_entities=art.title_entities,
                    abstract_entities=art.abstract_entities,
                    click_count=dataset.article_click_counts.get(aid, 0),
                )
                db.add(db_art)
        db.commit()

        # 2. Seed Users & History Interactions
        logger.info("Seeding users and interactions...")
        for uid, history in dataset.user_histories.items():
            existing_user = db.query(User).filter(User.user_id == uid).first()
            if not existing_user:
                db_user = User(
                    user_id=uid,
                    history_count=len(history),
                )
                db.add(db_user)
                for aid in history:
                    db_inter = Interaction(
                        user_id=uid,
                        article_id=aid,
                        interaction_type="history",
                        timestamp=dataset.reference_time,
                    )
                    db.add(db_inter)
        db.commit()

        # 3. Seed Evaluation Results if available
        eval_path = settings.artifacts_dir / "evaluation_results.json"
        if eval_path.exists():
            logger.info("Seeding evaluation metrics from %s...", eval_path)
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_records = json.load(f)

            for rec in eval_records:
                eval_res = EvaluationResult(
                    model_name=rec["model_name"],
                    auc=rec["auc"],
                    mrr=rec["mrr"],
                    ndcg_at_5=rec["ndcg_at_5"],
                    ndcg_at_10=rec["ndcg_at_10"],
                    recall_at_k=rec["recall_at_k"],
                    coverage=rec["coverage"],
                    num_impressions=rec["num_impressions"],
                    num_users=len(dataset.user_histories),
                    evaluated_at=datetime.now(timezone.utc),
                )
                db.add(eval_res)
            db.commit()

        logger.info("Database seeding completed successfully.")
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()

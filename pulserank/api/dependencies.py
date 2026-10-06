"""Application state, model registry, and dependency injection."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from pulserank.config import settings
from pulserank.data.mind_parser import NewsArticleRecord
from pulserank.data.preprocessor import ProcessedDataset, build_processed_dataset
from pulserank.data.fixtures import write_mind_tsv_fixtures, is_synthetic_fixture_dir
from pulserank.ml.popularity import PopularityRecommender
from pulserank.ml.content_based import ContentBasedRecommender
from pulserank.ml.collaborative import ImplicitALSRecommender
from pulserank.ml.hybrid import HybridCandidateGenerator, CandidateItem
from pulserank.ml.ranker import LambdaMARTRanker, FeatureExtractor
from pulserank.ml.profiles import ProfileBuilder
from pulserank.ml.explanations import ExplanationGenerator, RecommendationExplanation
from pulserank.data.interactions import build_popularity_events, build_als_interactions
from pulserank.data.provenance import compute_dataset_fingerprint, IncompatibleArtifactError

logger = logging.getLogger(__name__)


class ServiceContainer:
    """Singleton service container maintaining in-memory models and data caches."""

    def __init__(self):
        self.dataset: Optional[ProcessedDataset] = None
        self.popularity_model: Optional[PopularityRecommender] = None
        self.content_model: Optional[ContentBasedRecommender] = None
        self.collaborative_model: Optional[ImplicitALSRecommender] = None
        self.hybrid_generator: Optional[HybridCandidateGenerator] = None
        self.ranker_model: Optional[LambdaMARTRanker] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.profile_builder: Optional[ProfileBuilder] = None

        # Ephemeral cache for recommendation explanations: recommendation_id -> RecommendationExplanation
        self.explanation_cache: Dict[str, Dict[str, Any]] = {}

    def is_ready(self) -> bool:
        return self.dataset is not None

    def initialize(self) -> None:
        """Load artifacts from disk or initialize from sample data if empty."""
        logger.info("Initializing PulseRank service container...")
        settings.ensure_directories()

        # 1. Try loading processed dataset
        if self.dataset is None:
            dataset_path = settings.processed_data_dir / "dataset.pkl"
            if dataset_path.exists():
                try:
                    self.dataset = ProcessedDataset.load(dataset_path)
                    logger.info("Loaded processed dataset from cache.")
                except Exception as e:
                    logger.warning(f"Failed to load dataset cache: {e}")

        # If no dataset yet, check raw directory; if empty, create fixtures so the app runs out-of-the-box
        if self.dataset is None:
            train_dir = settings.raw_data_dir / "MINDsmall_train"
            dev_dir = settings.raw_data_dir / "MINDsmall_dev"
            if not (train_dir / "news.tsv").exists():
                logger.info("Raw MIND data not present; bootstrapping authentic MIND fixtures for dev...")
                write_mind_tsv_fixtures(settings.raw_data_dir)

            dataset_source = "synthetic_fixture" if is_synthetic_fixture_dir(settings.raw_data_dir) else "mind-small"

            self.dataset = build_processed_dataset(
                train_news_path=train_dir / "news.tsv",
                train_behaviors_path=train_dir / "behaviors.tsv",
                dev_news_path=dev_dir / "news.tsv" if (dev_dir / "news.tsv").exists() else None,
                dev_behaviors_path=dev_dir / "behaviors.tsv" if (dev_dir / "behaviors.tsv").exists() else None,
                dataset_source=dataset_source,
            )
            self.dataset.save(settings.processed_data_dir)

        dataset_source = getattr(self.dataset, "dataset_source", "mind-small")
        dataset_fingerprint = compute_dataset_fingerprint(self.dataset)

        self.profile_builder = ProfileBuilder(self.dataset.articles)

        # 2. Popularity Model
        pop_path = settings.artifacts_dir / "popularity.pkl"
        if pop_path.exists():
            try:
                self.popularity_model = PopularityRecommender.load(
                    pop_path,
                    expected_source=dataset_source,
                    expected_fingerprint=dataset_fingerprint,
                    fail_closed=True,
                )
                logger.info("Loaded compatible popularity artifact from disk.")
            except Exception as e:
                logger.warning(f"Could not load popularity artifact ({e}), refitting for provenance compatibility...")
                self.popularity_model = None

        if self.popularity_model is None:
            logger.info("Fitting popularity model...")
            interactions = build_popularity_events(self.dataset.train_behaviors, include_history_as_prior=True)
            self.popularity_model = PopularityRecommender(decay_half_life_hours=settings.popularity_decay_hours)
            self.popularity_model.fit(interactions, reference_time=self.dataset.reference_time)
            self.popularity_model.save(
                pop_path,
                dataset_source=dataset_source,
                dataset_fingerprint=dataset_fingerprint,
            )

        # 3. Content Model
        content_dir = settings.artifacts_dir / "content"
        if (content_dir / "content_meta.pkl").exists():
            try:
                self.content_model = ContentBasedRecommender.load(
                    content_dir,
                    expected_source=dataset_source,
                    expected_fingerprint=dataset_fingerprint,
                    fail_closed=True,
                )
                logger.info("Loaded compatible content artifact from disk.")
            except Exception as e:
                logger.warning(f"Could not load content artifact ({e}), refitting for provenance compatibility...")
                self.content_model = None

        if self.content_model is None:
            logger.info("Fitting content-based embeddings & vector index...")
            self.content_model = ContentBasedRecommender(
                model_name=settings.embedding_model_name,
                embedding_dim=settings.embedding_dim,
            )
            self.content_model.build_index(self.dataset.articles)
            self.content_model.save(
                content_dir,
                dataset_source=dataset_source,
                dataset_fingerprint=dataset_fingerprint,
            )

        # 4. Collaborative Model
        als_path = settings.artifacts_dir / "als.pkl"
        if als_path.exists():
            try:
                self.collaborative_model = ImplicitALSRecommender.load(
                    als_path,
                    expected_source=dataset_source,
                    expected_fingerprint=dataset_fingerprint,
                    fail_closed=True,
                )
                logger.info("Loaded compatible ALS artifact from disk.")
            except Exception as e:
                logger.warning(f"Could not load ALS artifact ({e}), refitting for provenance compatibility...")
                self.collaborative_model = None

        if self.collaborative_model is None:
            logger.info("Fitting collaborative ALS model...")
            collab_interactions = build_als_interactions(self.dataset.train_behaviors)
            self.collaborative_model = ImplicitALSRecommender(
                factors=settings.als_factors,
                regularization=settings.als_regularization,
                iterations=settings.als_iterations,
                alpha=settings.als_alpha,
            )
            self.collaborative_model.fit(collab_interactions)
            self.collaborative_model.save(
                als_path,
                dataset_source=dataset_source,
                dataset_fingerprint=dataset_fingerprint,
            )

        # 5. Hybrid Generator
        self.hybrid_generator = HybridCandidateGenerator(
            popularity_model=self.popularity_model,
            content_model=self.content_model,
            collaborative_model=self.collaborative_model,
        )

        # 6. Feature Extractor
        self.feature_extractor = FeatureExtractor(
            articles=self.dataset.articles,
            user_histories=self.dataset.user_histories,
            content_model=self.content_model,
            collab_model=self.collaborative_model,
            popularity_model=self.popularity_model,
        )

        # 7. LightGBM Ranker
        ranker_path = settings.artifacts_dir / "ranker.pkl"
        if ranker_path.exists():
            try:
                self.ranker_model = LambdaMARTRanker.load(
                    ranker_path,
                    expected_source=dataset_source,
                    expected_fingerprint=dataset_fingerprint,
                    fail_closed=True,
                )
                logger.info("Loaded compatible ranker artifact from disk.")
            except Exception as e:
                logger.warning(f"Could not load ranker artifact ({e}), refitting for provenance compatibility...")
                self.ranker_model = None

        if self.ranker_model is None:
            logger.info("Fitting LightGBM LambdaMART ranker...")
            self.ranker_model = LambdaMARTRanker(
                learning_rate=settings.ranker_learning_rate,
                num_leaves=settings.ranker_num_leaves,
                max_depth=settings.ranker_max_depth,
                n_estimators=settings.ranker_n_estimators,
                random_state=settings.ranker_random_state,
            )
            self.ranker_model.fit_from_behaviors(
                behaviors=self.dataset.train_behaviors,
                feature_extractor=self.feature_extractor,
            )
            self.ranker_model.save(
                ranker_path,
                dataset_source=dataset_source,
                dataset_fingerprint=dataset_fingerprint,
            )

        # 8. Pass 2 Advanced Engines
        from pulserank.ml.diversity import DiversityEngine
        from pulserank.ml.exploration import ExplorationEngine
        from pulserank.ml.cold_start import ColdStartEngine
        from pulserank.ml.profiles import DynamicProfileManager
        from pulserank.ml.monitoring import InferenceMonitor
        from pulserank.ml.registry import ModelRegistry
        from pulserank.ml.tracking import MLflowTracker
        from pulserank.nlp.ner import NamedEntityPipeline
        from pulserank.nlp.cross_encoder import PairwiseCrossEncoder
        from pulserank.nlp.annotation import ActiveLearningQueue
        from pulserank.nlp.classifier_baseline import TFIDFTopicClassifier

        art_embs = self.content_model.article_embeddings if self.content_model else {}
        art_pops = self.popularity_model.scores if self.popularity_model else {}
        art_cats = {aid: art.category for aid, art in self.dataset.articles.items()}

        self.diversity_engine = DiversityEngine(
            article_embeddings=art_embs,
            article_popularities=art_pops,
        )
        self.exploration_engine = ExplorationEngine()
        self.cold_start_engine = ColdStartEngine(
            article_embeddings=art_embs,
            article_metadata=self.dataset.articles,
        )
        self.dynamic_profile_manager = DynamicProfileManager(
            embedding_dim=settings.embedding_dim,
        )
        # Link dynamic profile manager into feature extractor for train/serve parity
        self.feature_extractor.dynamic_profile_manager = self.dynamic_profile_manager

        # Compute baseline category distribution from training corpus
        cat_counts = {}
        for art in self.dataset.articles.values():
            cat_counts[art.category] = cat_counts.get(art.category, 0) + 1
        tot = max(sum(cat_counts.values()), 1)
        base_dist = {k: v / tot for k, v in cat_counts.items()}
        self.inference_monitor = InferenceMonitor(baseline_category_dist=base_dist)

        self.model_registry = ModelRegistry()
        self.mlflow_tracker = MLflowTracker()
        self.named_entity_pipeline = NamedEntityPipeline()
        self.pairwise_cross_encoder = PairwiseCrossEncoder()
        self.active_learning_queue = ActiveLearningQueue()

        # Initialize TF-IDF topic classifier on dataset articles
        tfidf_path = settings.artifacts_dir / "tfidf_topic_classifier.pkl"
        if tfidf_path.exists():
            try:
                self.topic_classifier_baseline = TFIDFTopicClassifier.load(
                    tfidf_path,
                    expected_source=dataset_source,
                    expected_fingerprint=dataset_fingerprint,
                    fail_closed=True,
                )
                logger.info("Loaded compatible TF-IDF classifier artifact from disk.")
            except Exception as e:
                logger.warning(f"Could not load TF-IDF artifact ({e}), refitting...")
                self.topic_classifier_baseline = None
        else:
            self.topic_classifier_baseline = None

        if self.topic_classifier_baseline is None:
            self.topic_classifier_baseline = TFIDFTopicClassifier()
            sample_articles = list(self.dataset.articles.values())[:300]
            if sample_articles:
                texts = [f"{a.title} {a.abstract}" for a in sample_articles]
                labels = [a.category for a in sample_articles]
                self.topic_classifier_baseline.fit(texts, labels)
                try:
                    self.topic_classifier_baseline.save(
                        tfidf_path,
                        dataset_source=dataset_source,
                        dataset_fingerprint=dataset_fingerprint,
                    )
                except Exception as e:
                    logger.warning(f"Could not save TF-IDF classifier artifact: {e}")

        logger.info("PulseRank service container fully initialized.")


# Global singleton instance
container = ServiceContainer()


def get_service_container() -> ServiceContainer:
    """Dependency for obtaining the service container."""
    if not container.is_ready():
        container.initialize()
    return container

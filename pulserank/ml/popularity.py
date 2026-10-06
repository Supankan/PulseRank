"""Recency-aware popularity baseline recommender."""

import math
import pickle
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PopularityRecommender:
    """
    Recency-aware popularity baseline.
    Computes exponential time-decayed interaction scores:
    score(item) = sum_{t in clicks} exp(-lambda * (T_ref - t))
    """

    def __init__(self, decay_half_life_hours: float = 72.0):
        self.decay_half_life_hours = decay_half_life_hours
        self.lambda_param = math.log(2) / (decay_half_life_hours * 3600.0)
        self.article_scores: Dict[str, float] = {}
        self.raw_counts: Dict[str, int] = {}
        self.ranked_articles: List[Tuple[str, float]] = []
        self.max_score: float = 1.0

    def fit(
        self,
        article_interactions: List[Tuple[str, Optional[datetime]]],
        reference_time: Optional[datetime] = None,
    ) -> "PopularityRecommender":
        """
        Fit popularity model from interaction tuples (article_id, timestamp).
        """
        if not article_interactions:
            logger.warning("Empty interactions provided to PopularityRecommender.fit")
            return self

        ref = reference_time or datetime.now(timezone.utc)
        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        for article_id, ts in article_interactions:
            counts[article_id] = counts.get(article_id, 0) + 1
            if ts is not None:
                # Ensure timezone-naive or comparable timestamp
                delta_sec = max(0.0, (ref - ts).total_seconds()) if ref >= ts else 0.0
                decay_weight = math.exp(-self.lambda_param * delta_sec)
            else:
                decay_weight = 1.0
            scores[article_id] = scores.get(article_id, 0.0) + decay_weight

        self.raw_counts = counts
        self.max_score = max(scores.values()) if scores else 1.0

        # Normalize scores to [0.0, 1.0]
        self.article_scores = {
            aid: score / self.max_score for aid, score in scores.items()
        }

        # Sort descending
        self.ranked_articles = sorted(
            self.article_scores.items(), key=lambda x: x[1], reverse=True
        )
        logger.info(f"Popularity model fitted with {len(self.ranked_articles)} unique articles.")
        return self

    @property
    def scores(self) -> Dict[str, float]:
        return getattr(self, "article_scores", {})

    def recommend(

        self,
        k: int = 20,
        exclude_article_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return top-K popular articles excluding consumed items."""
        exclude = exclude_article_ids or set()
        recommendations: List[Tuple[str, float]] = []
        for aid, score in self.ranked_articles:
            if aid not in exclude:
                recommendations.append((aid, score))
            if len(recommendations) >= k:
                break
        return recommendations

    def get_score(self, article_id: str) -> float:
        """Get normalized popularity score for article (0.0 if unseen)."""
        return self.article_scores.get(article_id, 0.0)

    def save(
        self,
        output_path: Path,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
    ) -> None:
        """Persist model and fail-closed manifest to disk."""
        from pulserank.data.provenance import ArtifactManifest
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.manifest = ArtifactManifest(
            model_name="popularity_baseline",
            model_version="2.0.0",
            artifact_type="popularity",
            dataset_source=dataset_source,
            dataset_fingerprint=dataset_fingerprint,
            hyperparameters={
                "decay_half_life_hours": self.decay_half_life_hours,
                "lambda_param": self.lambda_param,
                "num_articles": len(self.article_scores),
            },
        )
        manifest_path = output_path.parent / (output_path.stem + "_manifest.json")
        self.manifest.save(manifest_path)

    @classmethod
    def load(
        cls,
        input_path: Path,
        expected_source: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
        fail_closed: bool = True,
    ) -> "PopularityRecommender":
        """Load model from disk and validate fail-closed provenance compatibility."""
        from pulserank.data.provenance import ArtifactManifest, enforce_artifact_compatibility, validate_artifact_compatibility
        with open(input_path, "rb") as f:
            instance = pickle.load(f)

        manifest_path = input_path.parent / (input_path.stem + "_manifest.json")
        if manifest_path.exists():
            instance.manifest = ArtifactManifest.load(manifest_path)
        else:
            instance.manifest = None

        if expected_source is not None:
            if fail_closed:
                enforce_artifact_compatibility(
                    instance.manifest,
                    expected_dataset_source=expected_source,
                    expected_fingerprint=expected_fingerprint,
                    artifact_name="PopularityRecommender",
                )
            else:
                compat, reason = validate_artifact_compatibility(
                    instance.manifest, expected_source, expected_fingerprint
                )
                if not compat:
                    logger.warning(f"Incompatible popularity artifact: {reason}")

        return instance


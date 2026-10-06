"""LightGBM LambdaMART Learning-to-Rank model with canonical impression-aware feature engineering."""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import lightgbm as lgb

from pulserank.config import settings
from pulserank.data.mind_parser import BehaviorRecord, NewsArticleRecord
from pulserank.data.provenance import (
    ArtifactManifest,
    enforce_artifact_compatibility,
    IncompatibleArtifactError,
)
from pulserank.ml.hybrid import CandidateItem

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "content_similarity",
    "collaborative_score",
    "popularity_score",
    "category_affinity",
    "subcategory_affinity",
    "entity_overlap_count",
    "user_history_length",
]


class FeatureExtractor:
    """
    Extracts canonical ranking features for (user_id, article_id, user_history) tuples.

    Guarantees strict train/serve mathematical parity:
    - Features are computed from identical functional definitions whether in training,
      offline evaluation, catalog reranking, live serving, or explanations.
    - CandidateItem scores (retrieval scores, RRF ranks) never silently override features.
    """

    def __init__(
        self,
        articles: Dict[str, NewsArticleRecord],
        user_histories: Dict[str, List[str]],
        content_model: Optional[Any] = None,
        collab_model: Optional[Any] = None,
        popularity_model: Optional[Any] = None,
        dynamic_profile_manager: Optional[Any] = None,
        use_dynamic_profiles: bool = False,
    ):
        self.articles = articles
        self.user_histories = user_histories
        self.content_model = content_model
        self.collab_model = collab_model
        self.popularity_model = popularity_model
        self.dynamic_profile_manager = dynamic_profile_manager
        self.use_dynamic_profiles = use_dynamic_profiles

        # Precompute user profile entity sets and category distributions for fast lookup when history is not passed
        self.user_entities: Dict[str, Set[str]] = {}
        self.user_cat_dist: Dict[str, Dict[str, float]] = {}
        self.user_subcat_dist: Dict[str, Dict[str, float]] = {}
        self.user_vectors: Dict[str, Optional[np.ndarray]] = {}

        self._build_user_profiles()

    def _build_user_profiles(self):
        for user_id, history in self.user_histories.items():
            cat_counts: Dict[str, int] = {}
            subcat_counts: Dict[str, int] = {}
            entities: Set[str] = set()

            for aid in history:
                art = self.articles.get(aid)
                if art:
                    cat_counts[art.category] = cat_counts.get(art.category, 0) + 1
                    subcat_counts[art.subcategory] = subcat_counts.get(art.subcategory, 0) + 1
                    for ent in getattr(art, "title_entities", []) + getattr(art, "abstract_entities", []):
                        wid = ent.get("WikidataId") or ent.get("Label")
                        if wid:
                            entities.add(wid)

            total = len(history) if history else 1
            self.user_cat_dist[user_id] = {k: v / total for k, v in cat_counts.items()}
            self.user_subcat_dist[user_id] = {k: v / total for k, v in subcat_counts.items()}
            self.user_entities[user_id] = entities

            if self.content_model is not None and hasattr(self.content_model, "build_user_vector"):
                self.user_vectors[user_id] = self.content_model.build_user_vector(history)
            else:
                self.user_vectors[user_id] = None

    def compute_ltr_features(
        self,
        user_id: str,
        article_id: str,
        user_history: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Canonical feature computation path:
        Evaluates the exact 7-feature contract for (user_id, article_id, user_history).
        """
        return self.extract_features(
            user_id=user_id,
            article_id=article_id,
            candidate_item=None,
            user_history=user_history,
        )

    def extract_features(
        self,
        user_id: str,
        article_id: str,
        candidate_item: Optional[CandidateItem] = None,
        user_history: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Extract fixed-length feature vector matching FEATURE_NAMES.

        Note: `candidate_item` parameter is preserved for API compatibility and metadata,
        but features are strictly calculated from canonical model functions to avoid train/serve skew.
        """
        art = self.articles.get(article_id)
        history = user_history if user_history is not None else self.user_histories.get(user_id, [])

        # User vector for content similarity
        user_vec = None
        if self.use_dynamic_profiles and self.dynamic_profile_manager is not None and self.content_model is not None:
            # Build dynamic profile vector blending short-term and long-term interest
            comb_prof = self.dynamic_profile_manager.build_profile(
                user_id=user_id,
                user_history=history,
                article_embeddings=self.content_model.article_embeddings,
                article_categories={aid: self.articles[aid].category for aid in history if aid in self.articles},
            )
            user_vec = comb_prof.combined_vector
        elif user_history is not None and self.content_model is not None and hasattr(self.content_model, "build_user_vector"):
            user_vec = self.content_model.build_user_vector(user_history)
        else:
            user_vec = self.user_vectors.get(user_id)

        # 1. Content similarity: strictly computed via content_model.compute_similarity
        content_sim = 0.0
        if self.content_model is not None and user_vec is not None:
            content_sim = float(self.content_model.compute_similarity(user_vec, article_id))

        # 2. Collaborative score: strictly computed via collab_model.get_score (canonical sigmoid)
        collab_score = 0.0
        if self.collab_model is not None:
            collab_score = float(self.collab_model.get_score(user_id, article_id))

        # 3. Popularity score: strictly computed via popularity_model.get_score
        pop_score = 0.0
        if self.popularity_model is not None:
            pop_score = float(self.popularity_model.get_score(article_id))

        # 4 & 5. Category and Subcategory affinities
        cat_aff = 0.0
        subcat_aff = 0.0
        entity_overlap = 0.0

        if art:
            if user_history is not None:
                # Dynamic compute strictly on impression history to prevent target leakage
                c_counts = 0
                sc_counts = 0
                u_ents: Set[str] = set()
                for aid in history:
                    h_art = self.articles.get(aid)
                    if h_art:
                        if h_art.category == art.category:
                            c_counts += 1
                        if h_art.subcategory == art.subcategory:
                            sc_counts += 1
                        for ent in getattr(h_art, "title_entities", []) + getattr(h_art, "abstract_entities", []):
                            wid = ent.get("WikidataId") or ent.get("Label")
                            if wid:
                                u_ents.add(wid)
                tot = len(history) if history else 1
                cat_aff = float(c_counts / tot)
                subcat_aff = float(sc_counts / tot)
                art_ents = {
                    ent.get("WikidataId") or ent.get("Label")
                    for ent in getattr(art, "title_entities", []) + getattr(art, "abstract_entities", [])
                    if ent.get("WikidataId") or ent.get("Label")
                }
                entity_overlap = float(len(u_ents.intersection(art_ents)))
            else:
                cat_aff = float(self.user_cat_dist.get(user_id, {}).get(art.category, 0.0))
                subcat_aff = float(self.user_subcat_dist.get(user_id, {}).get(art.subcategory, 0.0))
                user_ents = self.user_entities.get(user_id, set())
                art_ents = {
                    ent.get("WikidataId") or ent.get("Label")
                    for ent in getattr(art, "title_entities", []) + getattr(art, "abstract_entities", [])
                    if ent.get("WikidataId") or ent.get("Label")
                }
                entity_overlap = float(len(user_ents.intersection(art_ents)))

        # 7. User history length
        hist_len = float(len(history))

        return np.array([
            content_sim,
            collab_score,
            pop_score,
            cat_aff,
            subcat_aff,
            entity_overlap,
            hist_len,
        ], dtype=np.float32)


def compute_ltr_features(
    user_id: str,
    article_id: str,
    user_history: Optional[List[str]],
    articles: Dict[str, Any],
    content_model: Optional[Any] = None,
    collab_model: Optional[Any] = None,
    popularity_model: Optional[Any] = None,
) -> np.ndarray:
    """Standalone canonical feature computation verifying train/serve parity."""
    extractor = FeatureExtractor(
        articles=articles,
        user_histories={user_id: user_history or []},
        content_model=content_model,
        collab_model=collab_model,
        popularity_model=popularity_model,
    )
    return extractor.compute_ltr_features(user_id, article_id, user_history=user_history)


class LambdaMARTRanker:
    """
    LightGBM LambdaMART ranker:
    - Learns pairwise / listwise ranking directly from MIND impression logs.
    - Groups samples strictly by impression_id.
    - Gracefully degrades to hybrid scoring for unranked candidates or missing artifacts.
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = 6,
        n_estimators: int = 150,
        random_state: int = 42,
    ):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[lgb.LGBMRanker] = None
        self.feature_names = list(FEATURE_NAMES)
        self.manifest: Optional[ArtifactManifest] = None

    def fit_from_behaviors(
        self,
        train_behaviors: Optional[List[BehaviorRecord]] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        eval_behaviors: Optional[List[BehaviorRecord]] = None,
        behaviors: Optional[List[BehaviorRecord]] = None,
    ) -> "LambdaMARTRanker":
        """
        Train LambdaMART from BehaviorRecords.
        Samples are grouped strictly by impression.
        """
        if train_behaviors is None and behaviors is not None:
            train_behaviors = behaviors
        if train_behaviors is None:
            raise ValueError("train_behaviors or behaviors must be provided.")
        if feature_extractor is None:
            raise ValueError("feature_extractor must be provided.")

        logger.info(f"Extracting ranking features from {len(train_behaviors)} training impressions...")
        X_train, y_train, groups_train = self._build_dataset(train_behaviors, feature_extractor)

        if len(X_train) == 0:
            logger.warning("Empty training dataset extracted for LambdaMART.")
            return self

        assert sum(groups_train) == len(y_train), (
            f"Group alignment error: sum(groups)={sum(groups_train)} != len(y)={len(y_train)}"
        )

        eval_set = None
        eval_group = None
        if eval_behaviors:
            X_eval, y_eval, groups_eval = self._build_dataset(eval_behaviors, feature_extractor)
            if len(X_eval) > 0 and sum(groups_eval) == len(y_eval):
                eval_set = [(X_eval, y_eval)]
                eval_group = [groups_eval]

        logger.info(
            f"Training LightGBM LambdaMART on {len(groups_train)} impressions "
            f"({len(y_train)} samples, {len(FEATURE_NAMES)} features)..."
        )

        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=[5, 10],
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(
            X_train,
            y_train,
            group=groups_train,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)] if eval_set else None,
        )
        logger.info("LambdaMART ranker training complete.")
        return self

    def _build_dataset(
        self,
        behaviors: List[BehaviorRecord],
        feature_extractor: FeatureExtractor,
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Build feature matrix, labels, and group boundaries from behaviors."""
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        groups: List[int] = []

        for b in behaviors:
            if not b.impressions:
                continue

            # Need at least one positive and one negative for meaningful ranking gradient
            has_pos = any(imp.clicked == 1 for imp in b.impressions)
            has_neg = any(imp.clicked == 0 for imp in b.impressions)
            if not (has_pos and has_neg):
                continue

            group_size = 0
            for imp in b.impressions:
                features = feature_extractor.extract_features(
                    user_id=b.user_id,
                    article_id=imp.news_id,
                    user_history=b.history,
                )
                X_list.append(features)
                y_list.append(imp.clicked)
                group_size += 1

            if group_size > 0:
                groups.append(group_size)

        if not X_list:
            return np.empty((0, len(FEATURE_NAMES))), np.array([]), []

        return np.vstack(X_list), np.array(y_list, dtype=np.int32), groups

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict(features)

    def rerank(
        self,
        candidates: List[CandidateItem],
        feature_extractor: FeatureExtractor,
        user_history: Optional[List[str]] = None,
    ) -> List[Tuple[CandidateItem, float]]:
        """
        Rerank a pool of CandidateItem objects using the trained LambdaMART model.
        Falls back to candidate.hybrid_score if ranker is not trained or input is empty.
        """
        if not candidates:
            return []

        if self.model is None:
            return [(c, c.hybrid_score) for c in sorted(candidates, key=lambda x: x.hybrid_score, reverse=True)]

        X_cand = np.vstack([
            feature_extractor.extract_features(
                user_id=c.user_id,
                article_id=c.article_id,
                candidate_item=c,
                user_history=user_history,
            )
            for c in candidates
        ])

        pred_scores = self.model.predict(X_cand)
        scored_pairs = list(zip(candidates, [float(s) for s in pred_scores]))
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs

    rerank_candidates = rerank

    def get_feature_importances(self) -> Dict[str, float]:
        """Return relative feature importances."""
        if self.model is None:
            return {name: 0.0 for name in self.feature_names}
        importances = self.model.feature_importances_
        total = sum(importances) if sum(importances) > 0 else 1.0
        return {name: float(imp / total) for name, imp in zip(self.feature_names, importances)}

    def save(
        self,
        output_path: Path,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
    ) -> None:
        """Save ranker artifact and fail-closed manifest."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.manifest = ArtifactManifest(
            model_name="lambdamart_ranker",
            model_version="2.0.0",
            artifact_type="ranker",
            dataset_source=dataset_source,
            dataset_fingerprint=dataset_fingerprint,
            feature_schema=self.feature_names,
            hyperparameters={
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
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
    ) -> "LambdaMARTRanker":
        """
        Load ranker artifact and validate manifest compatibility.
        Fails closed by raising IncompatibleArtifactError if manifest is missing or mismatched.
        """
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
                    artifact_name="LambdaMARTRanker",
                )
            else:
                from pulserank.data.provenance import validate_artifact_compatibility
                compat, reason = validate_artifact_compatibility(
                    instance.manifest, expected_source, expected_fingerprint
                )
                if not compat:
                    logger.warning(f"Incompatible ranker artifact: {reason}")

        return instance

"""Collaborative filtering using Implicit Feedback Alternating Least Squares (ALS)."""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from scipy import sparse

try:
    import implicit
    from implicit.als import AlternatingLeastSquares
    IMPLICIT_LIB_AVAILABLE = True
except ImportError:
    IMPLICIT_LIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ImplicitALSRecommender:
    """
    Implicit Feedback Collaborative Filtering model.
    Constructs user-item interaction matrix from history and impression clicks,
    learning latent user and item factors via Alternating Least Squares.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        iterations: int = 15,
        alpha: float = 40.0,
        random_state: int = 42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_state = random_state

        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}

        self.user_factors: Optional[np.ndarray] = None  # (num_users, factors)
        self.item_factors: Optional[np.ndarray] = None  # (num_items, factors)
        self.user_consumed: Dict[str, Set[str]] = {}

    def fit(
        self,
        interactions: List[Tuple[str, str, float]],
    ) -> "ImplicitALSRecommender":
        """
        Fit ALS model on (user_id, article_id, weight) tuples.
        """
        if not interactions:
            logger.warning("Empty interactions passed to ImplicitALSRecommender.fit")
            return self

        # Build ID mappings
        unique_users = sorted(list({u for u, _, _ in interactions}))
        unique_items = sorted(list({i for _, i, _ in interactions}))

        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        self.user_consumed = {u: set() for u in unique_users}

        rows, cols, data = [], [], []
        for u, i, weight in interactions:
            u_idx = self.user_to_idx[u]
            i_idx = self.item_to_idx[i]
            rows.append(u_idx)
            cols.append(i_idx)
            data.append(weight)
            self.user_consumed[u].add(i)

        num_users = len(unique_users)
        num_items = len(unique_items)
        user_item = sparse.csr_matrix(
            (data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32
        )

        logger.info(
            f"Fitting ALS with {num_users} users, {num_items} items, {len(interactions)} interactions..."
        )

        if IMPLICIT_LIB_AVAILABLE:
            try:
                model = AlternatingLeastSquares(
                    factors=self.factors,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    random_state=self.random_state,
                )
                # implicit expects confidence matrix C = user_item * alpha
                confidence = (user_item * self.alpha).astype(np.float32)
                model.fit(confidence, show_progress=False)
                self.user_factors = model.user_factors
                self.item_factors = model.item_factors
                logger.info("Fitted implicit ALS via 'implicit' library.")
                return self
            except Exception as e:
                logger.warning(f"Failed using implicit library ({e}). Falling back to pure NumPy ALS.")

        # Robust vectorized NumPy / SciPy ALS solver fallback
        self._fit_numpy_als(user_item)
        return self

    def _fit_numpy_als(self, user_item: sparse.csr_matrix) -> None:
        """Standard implicit ALS solver (Hu, Koren, Volinsky 2008)."""
        rng = np.random.RandomState(self.random_state)
        num_users, num_items = user_item.shape
        d = self.factors

        X = rng.normal(0, 1.0 / np.sqrt(d), (num_users, d)).astype(np.float32)
        Y = rng.normal(0, 1.0 / np.sqrt(d), (num_items, d)).astype(np.float32)
        reg_I = self.regularization * np.eye(d, dtype=np.float32)

        # Confidence: C_ui = 1 + alpha * R_ui
        # Preference: P_ui = 1 if R_ui > 0 else 0
        user_item_csc = user_item.tocsc()

        for it in range(self.iterations):
            # Update user factors: X_u = (Y^T C_u Y + lambda I)^-1 Y^T C_u p_u
            YtY = np.dot(Y.T, Y)
            for u in range(num_users):
                row = user_item.getrow(u)
                if row.nnz == 0:
                    continue
                indices = row.indices
                r_u = row.data
                c_u = 1.0 + self.alpha * r_u
                # Y_u: factors for interacted items
                Y_u = Y[indices]
                # Y^T (C_u - I) Y + Y^T Y
                A = YtY + np.dot(Y_u.T, (c_u[:, None] - 1.0) * Y_u) + reg_I
                # Y^T C_u p_u
                b = np.dot(Y_u.T, c_u)
                X[u] = np.linalg.solve(A, b)

            # Update item factors: Y_i = (X^T C_i X + lambda I)^-1 X^T C_i p_i
            XtX = np.dot(X.T, X)
            for i in range(num_items):
                col = user_item_csc.getcol(i)
                if col.nnz == 0:
                    continue
                indices = col.indices
                r_i = col.data
                c_i = 1.0 + self.alpha * r_i
                X_i = X[indices]
                A = XtX + np.dot(X_i.T, (c_i[:, None] - 1.0) * X_i) + reg_I
                b = np.dot(X_i.T, c_i)
                Y[i] = np.linalg.solve(A, b)

        self.user_factors = X
        self.item_factors = Y
        logger.info(f"Fitted NumPy ALS model across {self.iterations} iterations.")

    def recommend(
        self,
        user_id: str,
        k: int = 20,
        exclude_article_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Recommend top-K collaborative candidates for user_id.
        Returns empty list if user is unseen/cold-start.
        """
        if (
            self.user_factors is None
            or self.item_factors is None
            or user_id not in self.user_to_idx
        ):
            return []

        u_idx = self.user_to_idx[user_id]
        u_vec = self.user_factors[u_idx]

        # Predicted scores: dot product with all items
        raw_scores = np.dot(self.item_factors, u_vec)

        # Exclude already consumed
        exclude = set(self.user_consumed.get(user_id, set()))
        if exclude_article_ids:
            exclude.update(exclude_article_ids)

        ranked_indices = np.argsort(-raw_scores)
        recommendations: List[Tuple[str, float]] = []

        # Find min and max for normalization
        valid_scores = [raw_scores[i] for i in ranked_indices if self.idx_to_item[i] not in exclude]
        if not valid_scores:
            return []

        s_min, s_max = min(valid_scores), max(valid_scores)
        range_val = s_max - s_min if s_max > s_min else 1.0

        for idx in ranked_indices:
            item_id = self.idx_to_item[idx]
            if item_id not in exclude:
                norm_score = float(np.clip((raw_scores[idx] - s_min) / range_val, 0.0, 1.0))
                recommendations.append((item_id, norm_score))
            if len(recommendations) >= k:
                break

        return recommendations

    def get_score(self, user_id: str, article_id: str) -> float:
        """Predict collaborative score for (user_id, article_id) in [0, 1]."""
        if (
            self.user_factors is None
            or self.item_factors is None
            or user_id not in self.user_to_idx
            or article_id not in self.item_to_idx
        ):
            return 0.0

        u_idx = self.user_to_idx[user_id]
        i_idx = self.item_to_idx[article_id]
        raw_score = float(np.dot(self.user_factors[u_idx], self.item_factors[i_idx]))
        # Sigmoidal mapping to [0, 1]
        return float(1.0 / (1.0 + np.exp(-raw_score)))

    def is_user_known(self, user_id: str) -> bool:
        return user_id in self.user_to_idx

    def save(
        self,
        output_path: Path,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
    ) -> None:
        """Save ALS model and fail-closed manifest to disk."""
        from pulserank.data.provenance import ArtifactManifest
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.manifest = ArtifactManifest(
            model_name="collaborative_als",
            model_version="2.0.0",
            artifact_type="collaborative",
            dataset_source=dataset_source,
            dataset_fingerprint=dataset_fingerprint,
            hyperparameters={
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
                "alpha": self.alpha,
                "num_users": len(self.user_to_idx),
                "num_items": len(self.item_to_idx),
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
    ) -> "ImplicitALSRecommender":
        """Load ALS model from disk and validate fail-closed provenance compatibility."""
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
                    artifact_name="ImplicitALSRecommender",
                )
            else:
                compat, reason = validate_artifact_compatibility(
                    instance.manifest, expected_source, expected_fingerprint
                )
                if not compat:
                    logger.warning(f"Incompatible ALS artifact: {reason}")

        return instance

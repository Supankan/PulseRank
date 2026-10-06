"""Classical TF-IDF + Calibrated Logistic Regression baseline topic classifier."""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss

from pulserank.data.provenance import (
    ArtifactManifest,
    enforce_artifact_compatibility,
    validate_artifact_compatibility,
)

logger = logging.getLogger(__name__)


class TFIDFTopicClassifier:
    """
    Topic classifier combining TF-IDF n-grams with calibrated L2-regularized Logistic Regression.
    Uses CalibratedClassifierCV (Platt scaling / sigmoid calibration) to produce authentic,
    probabilistically calibrated class probabilities.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        calibrate: bool = True,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.calibrate = calibrate
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self.base_model = LogisticRegression(
            C=C,
            max_iter=250,
            random_state=random_state,
            solver="lbfgs",
        )
        self.model: Any = None
        self.classes_: List[str] = []
        self.is_fitted = False
        self.is_calibrated = False
        self.manifest: Optional[ArtifactManifest] = None
        self.last_evaluation: Optional[Dict[str, Any]] = None

    def fit(self, texts: List[str], labels: List[str]) -> "TFIDFTopicClassifier":
        """
        Fit vectorizer and calibrated logistic regression classifier.
        Uses 3-fold cross-validation calibration if sufficient samples exist per class.
        """
        if not texts or not labels:
            raise ValueError("Empty texts or labels provided to TFIDFTopicClassifier.fit")

        logger.info(f"Fitting TF-IDF topic classifier on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        self.classes_ = sorted(list(set(labels)))

        # Check minimum class frequency for calibration
        class_counts = {c: labels.count(c) for c in self.classes_}
        min_count = min(class_counts.values()) if class_counts else 0

        if self.calibrate and min_count >= 3 and len(texts) >= 15:
            try:
                cv_folds = min(3, min_count)
                calibrated = CalibratedClassifierCV(
                    estimator=self.base_model,
                    method="sigmoid",
                    cv=cv_folds,
                )
                calibrated.fit(X, labels)
                self.model = calibrated
                self.is_calibrated = True
                logger.info(f"Fitted calibrated classifier with {cv_folds}-fold Platt scaling.")
            except Exception as e:
                logger.warning(f"Calibration failed ({e}); falling back to standard logistic regression.")
                self.base_model.fit(X, labels)
                self.model = self.base_model
                self.is_calibrated = False
        else:
            self.base_model.fit(X, labels)
            self.model = self.base_model
            self.is_calibrated = False

        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        logger.info(
            f"Fitted TF-IDF classifier with {len(self.classes_)} classes and {X.shape[1]} vocabulary features "
            f"(calibrated={self.is_calibrated})."
        )
        return self

    def predict(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted yet.")
        if not texts:
            return []
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        return [str(p) for p in preds]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted yet.")
        if not texts:
            return np.empty((0, len(self.classes_)))
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def predict_top_k(self, texts: List[str], k: int = 3) -> List[List[Tuple[str, float]]]:
        probs = self.predict_proba(texts)
        results = []
        for p in probs:
            top_indices = np.argsort(-p)[:k]
            results.append([(self.classes_[idx], float(round(p[idx], 4))) for idx in top_indices])
        return results

    def evaluate(
        self,
        texts: List[str],
        labels: List[str],
        evaluation_mode: str = "deterministic_held_out_split",
        train_sample_count: Optional[int] = None,
        test_sample_count: Optional[int] = None,
        split_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Compute multi-class accuracy, F1, log-loss, and multiclass Brier score.
        Explicitly tracks evaluation semantics (held-out vs demo resubstitution).
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted yet.")
        preds = self.predict(texts)
        probs = self.predict_proba(texts)

        acc = float(accuracy_score(labels, preds))
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

        cm = []
        report = {}
        try:
            eval_labels = [c for c in self.classes_ if c in set(labels)]
            if eval_labels:
                cm = confusion_matrix(labels, preds, labels=eval_labels).tolist()
                report = classification_report(labels, preds, labels=eval_labels, output_dict=True, zero_division=0)
        except Exception as e:
            logger.debug(f"Could not compute confusion matrix: {e}")

        # Log-loss and multiclass Brier score
        loss_val = None
        brier_score = None
        try:
            if len(set(labels).intersection(set(self.classes_))) == len(set(labels)):
                loss_val = float(round(log_loss(labels, probs, labels=self.classes_), 4))
            # Multiclass Brier score: mean squared error between one-hot labels and predicted probabilities
            y_one_hot = np.zeros_like(probs)
            for i, l in enumerate(labels):
                if l in self.classes_:
                    y_one_hot[i, self.classes_.index(l)] = 1.0
            brier_score = float(round(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)), 4))
        except Exception as e:
            logger.debug(f"Could not compute calibration loss metrics: {e}")

        effective_test_count = test_sample_count if test_sample_count is not None else len(texts)
        eval_result = {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "log_loss": loss_val,
            "brier_score": brier_score,
            "is_calibrated": self.is_calibrated,
            "classes": self.classes_,
            "confusion_matrix": cm,
            "classification_report": report,
            "evaluation_mode": evaluation_mode,
            "train_sample_count": train_sample_count,
            "test_sample_count": effective_test_count,
            "split_seed": split_seed,
            "num_test_samples": len(texts),
        }
        self.last_evaluation = eval_result
        return eval_result

    def save(
        self,
        path: Path,
        manifest_path: Optional[Path] = None,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        actual_manifest_path = manifest_path or (path.parent / (path.stem + "_manifest.json"))
        hyperparams = {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "C": self.C,
            "calibrate": self.calibrate,
            "is_calibrated": self.is_calibrated,
            "classes": self.classes_,
            "vocab_size": len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0,
        }
        if self.last_evaluation:
            hyperparams["evaluation_mode"] = self.last_evaluation.get("evaluation_mode")
            hyperparams["train_sample_count"] = self.last_evaluation.get("train_sample_count")
            hyperparams["test_sample_count"] = self.last_evaluation.get("test_sample_count")
            hyperparams["split_seed"] = self.last_evaluation.get("split_seed")

        self.manifest = ArtifactManifest(
            model_name="tfidf_topic_classifier",
            model_version="2.0.0",
            artifact_type="topic_classifier",
            dataset_source=dataset_source,
            dataset_fingerprint=dataset_fingerprint,
            hyperparameters=hyperparams,
        )
        self.manifest.save(actual_manifest_path)

    @classmethod
    def load(
        cls,
        path: Path,
        expected_source: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
        fail_closed: bool = True,
    ) -> "TFIDFTopicClassifier":
        """Load TF-IDF classifier and validate fail-closed provenance."""
        with open(path, "rb") as f:
            instance = pickle.load(f)

        manifest_path = path.parent / (path.stem + "_manifest.json")
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
                    artifact_name="TFIDFTopicClassifier",
                )
            else:
                compat, reason = validate_artifact_compatibility(
                    instance.manifest, expected_source, expected_fingerprint
                )
                if not compat:
                    logger.warning(f"Incompatible TF-IDF artifact: {reason}")

        return instance


def train_test_split_articles(
    articles: List[Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[Any], List[Any], str]:
    """
    Deterministically split articles into train and test sets.
    Preserves reproducibility with fixed random_state.
    If the dataset is small (< 30 articles), returns the full dataset for both train and test,
    explicitly tagged with evaluation_mode='demo_resubstitution_not_held_out'.
    For >= 30 articles, returns deterministic disjoint train/test splits tagged with
    evaluation_mode='deterministic_held_out_split'.
    """
    if len(articles) < 30:
        return list(articles), list(articles), "demo_resubstitution_not_held_out"

    import random
    rng = random.Random(random_state)
    shuffled = list(articles)
    rng.shuffle(shuffled)
    split_idx = max(int(len(shuffled) * (1.0 - test_size)), 1)
    train_arts = shuffled[:split_idx]
    test_arts = shuffled[split_idx:]
    if not test_arts:
        return list(articles), list(articles), "demo_resubstitution_not_held_out"
    return train_arts, test_arts, "deterministic_held_out_split"

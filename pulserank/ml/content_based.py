"""Content-based recommender using Sentence-Transformers and FAISS vector search."""

import hashlib
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from pulserank.config import settings
from pulserank.data.mind_parser import NewsArticleRecord

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Semantic content recommender:
    - Encodes article text (title + abstract) into dense embeddings.
    - Builds a FAISS vector index (IndexFlatIP for exact cosine similarity).
    - Aggregates user history into an interpretable user interest vector.
    - Retrieves top-K semantically relevant unseen candidates.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.encoder: Optional[Any] = None
        self.article_ids: List[str] = []
        self.article_id_to_idx: Dict[str, int] = {}
        self.embeddings: Optional[np.ndarray] = None  # shape: (N, D), L2-normalized
        self.index: Optional[Any] = None

    def _init_encoder(self):
        if self.encoder is None and ST_AVAILABLE:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.encoder = SentenceTransformer(self.model_name)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts into L2-normalized embeddings."""
        self._init_encoder()
        if self.encoder is not None:
            vecs = self.encoder.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.astype(np.float32)

        # Fallback deterministic pseudo-embedding for environments without model download access
        logger.warning("SentenceTransformer not available; using deterministic text projection fallback.")
        vecs = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for w in words:
                h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16) % self.embedding_dim
                vecs[i, h] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
            else:
                vecs[i, 0] = 1.0
        return vecs

    def build_index(self, articles: Dict[str, NewsArticleRecord]) -> "ContentBasedRecommender":
        """Generate embeddings and construct the FAISS retrieval index."""
        self.article_ids = list(articles.keys())
        self.article_id_to_idx = {aid: idx for idx, aid in enumerate(self.article_ids)}

        texts = [articles[aid].full_text for aid in self.article_ids]
        logger.info(f"Generating content embeddings for {len(texts)} articles...")
        self.embeddings = self.encode_texts(texts)
        self.embedding_dim = self.embeddings.shape[1]

        # Normalize explicitly
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (self.embeddings / norms).astype(np.float32)

        if FAISS_AVAILABLE:
            logger.info("Building FAISS IndexFlatIP (cosine similarity)...")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)
        else:
            logger.warning("FAISS not available; using NumPy matrix multiplication for similarity search.")
            self.index = None

        logger.info(f"Content index constructed for {len(self.article_ids)} articles.")
        return self

    def get_article_embedding(self, article_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding for article_id."""
        idx = self.article_id_to_idx.get(article_id)
        if idx is not None and self.embeddings is not None:
            return self.embeddings[idx]
        return None

    @property
    def article_embeddings(self) -> Dict[str, np.ndarray]:
        """Dictionary mapping article_id to its dense embedding."""
        if self.embeddings is None or not self.article_id_to_idx:
            return {}
        return {aid: self.embeddings[idx] for aid, idx in self.article_id_to_idx.items()}


    def build_user_vector(
        self,
        history_article_ids: List[str],
        recency_decay: float = 0.9,
    ) -> Optional[np.ndarray]:
        """
        Aggregate user interaction history into a normalized semantic profile vector:
        recent articles receive exponentially higher weight.
        """
        if not history_article_ids or self.embeddings is None:
            return None

        valid_vecs = []
        weights = []
        n = len(history_article_ids)
        for i, aid in enumerate(history_article_ids):
            vec = self.get_article_embedding(aid)
            if vec is not None:
                # i = 0 (oldest), i = n-1 (newest)
                w = recency_decay ** (n - 1 - i)
                valid_vecs.append(vec)
                weights.append(w)

        if not valid_vecs:
            return None

        weighted_sum = np.zeros(self.embedding_dim, dtype=np.float32)
        for vec, w in zip(valid_vecs, weights):
            weighted_sum += w * vec

        norm = np.linalg.norm(weighted_sum)
        if norm > 0:
            weighted_sum /= norm
        return weighted_sum

    def compute_similarity(self, user_vec: np.ndarray, article_id: str) -> float:
        """Compute cosine similarity between user vector and a specific article."""
        art_vec = self.get_article_embedding(article_id)
        if art_vec is None or user_vec is None:
            return 0.0
        dot = float(np.dot(user_vec, art_vec))
        # Map cosine [-1, 1] to [0, 1]
        return float(np.clip((dot + 1.0) / 2.0, 0.0, 1.0))

    def recommend(
        self,
        history_article_ids: List[str],
        k: int = 20,
        exclude_article_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-K semantically relevant candidates for user history.
        Filters out already-consumed items.
        """
        user_vec = self.build_user_vector(history_article_ids)
        if user_vec is None:
            return []

        exclude = set(history_article_ids)
        if exclude_article_ids:
            exclude.update(exclude_article_ids)

        fetch_k = min(len(self.article_ids), k + len(exclude) + 50)
        user_query = user_vec.reshape(1, -1).astype(np.float32)

        if self.index is not None and FAISS_AVAILABLE:
            scores, indices = self.index.search(user_query, fetch_k)
            indices = indices[0]
            scores = scores[0]
        else:
            # Fallback dot product
            scores_all = np.dot(self.embeddings, user_vec)
            indices = np.argsort(-scores_all)[:fetch_k]
            scores = scores_all[indices]

        recommendations: List[Tuple[str, float]] = []
        for idx, sim in zip(indices, scores):
            if idx < 0 or idx >= len(self.article_ids):
                continue
            aid = self.article_ids[idx]
            if aid not in exclude:
                # Normalize cosine to [0, 1]
                norm_score = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))
                recommendations.append((aid, norm_score))
            if len(recommendations) >= k:
                break

        return recommendations

    def save(
        self,
        output_dir: Path,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
    ) -> None:
        """Persist embeddings, mapping, index, and fail-closed manifest."""
        from pulserank.data.provenance import ArtifactManifest
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "article_ids": self.article_ids,
            "article_id_to_idx": self.article_id_to_idx,
        }
        with open(output_dir / "content_meta.pkl", "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.embeddings is not None:
            np.save(output_dir / "article_embeddings.npy", self.embeddings)

        if self.index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.index, str(output_dir / "faiss.index"))

        self.manifest = ArtifactManifest(
            model_name="content_based_faiss",
            model_version="2.0.0",
            artifact_type="content_based",
            dataset_source=dataset_source,
            dataset_fingerprint=dataset_fingerprint,
            hyperparameters={
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "num_articles": len(self.article_ids),
                "faiss_available": FAISS_AVAILABLE,
            },
        )
        self.manifest.save(output_dir / "content_manifest.json")

    @classmethod
    def load(
        cls,
        input_dir: Path,
        expected_source: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
        fail_closed: bool = True,
    ) -> "ContentBasedRecommender":
        """Load content model artifacts and validate fail-closed provenance compatibility."""
        from pulserank.data.provenance import ArtifactManifest, enforce_artifact_compatibility, validate_artifact_compatibility
        with open(input_dir / "content_meta.pkl", "rb") as f:
            meta = pickle.load(f)

        recommender = cls(
            model_name=meta["model_name"],
            embedding_dim=meta["embedding_dim"],
        )
        recommender.article_ids = meta["article_ids"]
        recommender.article_id_to_idx = meta["article_id_to_idx"]

        emb_path = input_dir / "article_embeddings.npy"
        if emb_path.exists():
            recommender.embeddings = np.load(emb_path)

        index_path = input_dir / "faiss.index"
        if index_path.exists() and FAISS_AVAILABLE:
            recommender.index = faiss.read_index(str(index_path))

        manifest_path = input_dir / "content_manifest.json"
        if manifest_path.exists():
            recommender.manifest = ArtifactManifest.load(manifest_path)
        else:
            recommender.manifest = None

        if expected_source is not None:
            if fail_closed:
                enforce_artifact_compatibility(
                    recommender.manifest,
                    expected_dataset_source=expected_source,
                    expected_fingerprint=expected_fingerprint,
                    artifact_name="ContentBasedRecommender",
                )
            else:
                compat, reason = validate_artifact_compatibility(
                    recommender.manifest, expected_source, expected_fingerprint
                )
                if not compat:
                    logger.warning(f"Incompatible content artifact: {reason}")

        return recommender

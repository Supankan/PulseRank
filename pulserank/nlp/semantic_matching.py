"""Contrastive / Triplet Bi-Encoder Fine-Tuning for News Recommendation."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

from pulserank.data.provenance import ArtifactManifest

logger = logging.getLogger(__name__)


@dataclass
class NewsTriplet:
    anchor: str    # e.g., user reading history summary or last read title
    positive: str  # clicked article title/abstract
    negative: str  # unclicked impression candidate


class TripletDataset(Dataset):
    def __init__(self, triplets: List[NewsTriplet]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        t = self.triplets[idx]
        return t.anchor, t.positive, t.negative


class ContrastiveBiEncoder:
    """
    Contrastive/Triplet bi-encoder fine-tuner for semantic relevance.
    Optimizes TripletMarginLoss over (user_history, clicked, unclicked) triples,
    sharpening embedding separation compared to out-of-the-box sentence transformers.
    """

    def __init__(
        self,
        base_model_name: str = "all-MiniLM-L6-v2",
        margin: float = 0.3,
        device: Optional[str] = None,
    ):
        self.base_model_name = base_model_name
        self.margin = margin
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SentenceTransformer(base_model_name, device=self.device)
        self.is_fine_tuned = False

    def evaluate_triplets(self, triplets: List[NewsTriplet]) -> Dict[str, float]:
        """
        Benchmark positive vs negative cosine similarities and triplet separation accuracy.
        """
        if not triplets:
            return {"accuracy": 0.0, "mean_pos_sim": 0.0, "mean_neg_sim": 0.0, "margin_gap": 0.0}

        anchors = [t.anchor for t in triplets]
        positives = [t.positive for t in triplets]
        negatives = [t.negative for t in triplets]

        a_emb = self.model.encode(anchors, normalize_embeddings=True, show_progress_bar=False)
        p_emb = self.model.encode(positives, normalize_embeddings=True, show_progress_bar=False)
        n_emb = self.model.encode(negatives, normalize_embeddings=True, show_progress_bar=False)

        pos_sims = np.sum(a_emb * p_emb, axis=1)
        neg_sims = np.sum(a_emb * n_emb, axis=1)

        correct = np.sum(pos_sims > neg_sims)
        acc = float(correct / len(triplets))
        mean_pos = float(np.mean(pos_sims))
        mean_neg = float(np.mean(neg_sims))
        gap = float(mean_pos - mean_neg)

        return {
            "triplet_accuracy": round(acc, 4),
            "mean_positive_similarity": round(mean_pos, 4),
            "mean_negative_similarity": round(mean_neg, 4),
            "separation_margin_gap": round(gap, 4),
            "num_triplets": len(triplets),
        }

    def fine_tune(
        self,
        triplets: List[NewsTriplet],
        epochs: int = 1,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        mode: str = "quick",
    ) -> Dict[str, Any]:
        """
        Fine-tune bi-encoder embeddings with PyTorch TripletMarginLoss.
        """
        if mode == "quick" and len(triplets) > 40:
            triplets = triplets[:40]
            epochs = 1
            batch_size = 8

        eval_before = self.evaluate_triplets(triplets)

        torch_model = self.model._first_module().auto_model
        tokenizer = self.model.tokenizer

        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=learning_rate)
        triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

        torch_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(triplets), batch_size):
                batch_triplets = triplets[i : i + batch_size]
                optimizer.zero_grad()

                def encode_texts(texts):
                    tokens = tokenizer(
                        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
                    ).to(self.device)
                    out = torch_model(**tokens)
                    # Mean pooling
                    mask = tokens["attention_mask"].unsqueeze(-1)
                    vecs = (out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                    return nn.functional.normalize(vecs, p=2, dim=1)

                a_vecs = encode_texts([t.anchor for t in batch_triplets])
                p_vecs = encode_texts([t.positive for t in batch_triplets])
                n_vecs = encode_texts([t.negative for t in batch_triplets])

                loss = triplet_loss(a_vecs, p_vecs, n_vecs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"Bi-Encoder Epoch {epoch + 1}/{epochs} Loss: {total_loss / max(1, len(triplets)//batch_size):.4f}")

        self.is_fine_tuned = True
        eval_after = self.evaluate_triplets(triplets)

        return {
            "before": eval_before,
            "after": eval_after,
            "num_triplets_trained": len(triplets),
            "mode": mode,
        }

    def save(self, output_dir: Path, manifest_path: Optional[Path] = None, dataset_fingerprint: str = "") -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))

        if manifest_path is not None:
            manifest = ArtifactManifest(
                artifact_type="contrastive_bi_encoder",
                dataset_fingerprint=dataset_fingerprint,
                parameters={
                    "base_model_name": self.base_model_name,
                    "margin": self.margin,
                    "is_fine_tuned": self.is_fine_tuned,
                },
            )
            manifest.save(Path(manifest_path))

    @classmethod
    def load(cls, model_dir: Path) -> "ContrastiveBiEncoder":
        instance = cls(base_model_name=str(model_dir))
        instance.is_fine_tuned = True
        return instance

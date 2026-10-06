"""Transformer-based Topic Classifier using DistilBERT."""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from pulserank.data.provenance import (
    ArtifactManifest,
    enforce_artifact_compatibility,
    validate_artifact_compatibility,
)

logger = logging.getLogger(__name__)


class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer: Any, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DistilBertTopicClassifier:
    """
    DistilBERT fine-tuning classifier for news topic categorization.
    Supports quick mode (subset/1 epoch) for fast local verification and full mode for production fine-tuning.
    Authentically tracks validation loss and selects best model checkpoint.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: Any = None
        self.model: Any = None
        self.classes_: List[str] = []
        self.label_to_idx: Dict[str, int] = {}
        self.is_fitted = False
        self.manifest: Optional[ArtifactManifest] = None

    def fit(
        self,
        texts: List[str],
        labels: List[str],
        epochs: int = 2,
        mode: str = "quick",
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[str]] = None,
    ) -> "DistilBertTopicClassifier":
        """
        Fine-tune DistilBERT on text classification.
        mode='quick' limits training to a lightweight subset (80 samples, 1 epoch) for rapid CPU execution.
        mode='full' fine-tunes on the complete training set.
        Genuinely evaluates on val_texts / val_labels and tracks best checkpoint.
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.classes_ = sorted(list(set(labels)))
        self.label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        num_labels = len(self.classes_)

        train_texts = list(texts)
        train_labels = list(labels)

        if mode == "quick" and len(train_texts) > 80:
            rng = np.random.RandomState(42)
            sample_indices = rng.choice(len(train_texts), size=80, replace=False)
            train_texts = [train_texts[i] for i in sample_indices]
            train_labels = [train_labels[i] for i in sample_indices]
            epochs = 1
            logger.info(f"[quick mode] Subsampled to {len(train_texts)} samples, 1 epoch for rapid execution.")

        logger.info(f"Loading DistilBERT tokenizer and model ({self.model_name}) with {num_labels} classes...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

        int_labels = [self.label_to_idx[l] for l in train_labels]
        dataset = TextClassificationDataset(train_texts, int_labels, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build validation dataloader if provided
        val_dataloader = None
        if val_texts and val_labels:
            val_int_labels = [self.label_to_idx.get(l, 0) for l in val_labels]
            val_dataset = TextClassificationDataset(val_texts, val_int_labels, self.tokenizer, self.max_length)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, batch_labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / max(len(dataloader), 1)

            # Evaluate on validation split if present
            if val_dataloader is not None:
                self.model.eval()
                total_val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        batch_labels = batch["label"].to(self.device)

                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        loss = criterion(outputs.logits, batch_labels)
                        total_val_loss += loss.item()

                        preds = torch.argmax(outputs.logits, dim=-1)
                        val_correct += (preds == batch_labels).sum().item()
                        val_total += len(batch_labels)

                avg_val_loss = total_val_loss / max(len(val_dataloader), 1)
                val_acc = val_correct / max(val_total, 1)
                logger.info(
                    f"DistilBERT Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
            else:
                logger.info(f"DistilBERT Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")

        # Restore best weights if validation set was used
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("Restored best checkpoint based on validation loss.")

        self.is_fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        if not texts:
            return np.empty((0, len(self.classes_)))

        self.model.eval()
        dataset = TextClassificationDataset(texts, None, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: List[str]) -> List[str]:
        probs = self.predict_proba(texts)
        if len(probs) == 0:
            return []
        pred_indices = np.argmax(probs, axis=1)
        return [self.classes_[i] for i in pred_indices]

    def predict_top_k(self, texts: List[str], k: int = 3) -> List[List[Tuple[str, float]]]:
        probs = self.predict_proba(texts)
        results = []
        for p in probs:
            top_indices = np.argsort(-p)[:k]
            results.append([(self.classes_[idx], float(round(p[idx], 4))) for idx in top_indices])
        return results

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        preds = self.predict(texts)
        acc = float(accuracy_score(labels, preds))
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
        cm = confusion_matrix(labels, preds, labels=self.classes_).tolist()
        report = classification_report(labels, preds, labels=self.classes_, output_dict=True, zero_division=0)

        return {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "classes": self.classes_,
            "confusion_matrix": cm,
            "classification_report": report,
            "num_test_samples": len(texts),
        }

    def save(
        self,
        save_dir: Path,
        manifest_path: Optional[Path] = None,
        dataset_source: str = "mind-small",
        dataset_fingerprint: str = "",
    ) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        metadata = {
            "classes": self.classes_,
            "label_to_idx": self.label_to_idx,
            "model_name": self.model_name,
            "max_length": self.max_length,
        }
        with open(save_dir / "classifier_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        actual_manifest_path = manifest_path or (save_dir / "distilbert_manifest.json")
        self.manifest = ArtifactManifest(
            model_name="distilbert_topic_classifier",
            model_version="2.0.0",
            artifact_type="transformer_topic_classifier",
            dataset_source=dataset_source,
            dataset_fingerprint=dataset_fingerprint,
            hyperparameters={
                "model_name": self.model_name,
                "max_length": self.max_length,
                "classes": self.classes_,
                "num_classes": len(self.classes_),
            },
        )
        self.manifest.save(actual_manifest_path)

    @classmethod
    def load(
        cls,
        save_dir: Path,
        expected_source: Optional[str] = None,
        expected_fingerprint: Optional[str] = None,
        fail_closed: bool = True,
    ) -> "DistilBertTopicClassifier":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        save_dir = Path(save_dir)
        meta_file = save_dir / "classifier_meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"DistilBERT metadata not found at {meta_file}")

        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(model_name=meta["model_name"], max_length=meta["max_length"])
        instance.classes_ = meta["classes"]
        instance.label_to_idx = meta["label_to_idx"]
        instance.tokenizer = AutoTokenizer.from_pretrained(save_dir)
        instance.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        instance.model.to(instance.device)
        instance.is_fitted = True

        manifest_path = save_dir / "distilbert_manifest.json"
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
                    artifact_name="DistilBertTopicClassifier",
                )
            else:
                compat, reason = validate_artifact_compatibility(
                    instance.manifest, expected_source, expected_fingerprint
                )
                if not compat:
                    logger.warning(f"Incompatible DistilBERT artifact: {reason}")

        return instance

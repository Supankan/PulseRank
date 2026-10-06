"""Active Learning Annotation Queue, Pre-Labeling Providers, and Inter-Annotator Agreement."""

import abc
import math
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


@dataclass
class AnnotationItem:
    article_id: str
    title: str
    abstract: str
    predicted_label: str
    prediction_confidence: float
    entropy: float
    margin: float
    uncertainty_score: float  # blended ranking key
    pre_label: Optional[str] = None
    pre_label_rationale: Optional[str] = None
    human_label: Optional[str] = None
    annotator_id: Optional[str] = None


class ActiveLearningQueue:
    """
    Prioritizes unlabeled or ambiguous articles for active learning triage.
    Supports prediction entropy and margin sampling strategies.
    """

    @staticmethod
    def calculate_entropy(probs: np.ndarray) -> float:
        """Shannon entropy: H(p) = -sum(p_i * log2(p_i))."""
        probs = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def calculate_margin(probs: np.ndarray) -> float:
        """Smallest margin = most uncertain: margin = p_top1 - p_top2."""
        if len(probs) < 2:
            return 1.0
        sorted_p = np.sort(probs)[::-1]
        return float(sorted_p[0] - sorted_p[1])

    def rank_for_annotation(
        self,
        articles: List[Dict[str, Any]],
        prediction_probabilities: np.ndarray,
        classes: List[str],
        top_n: int = 50,
        strategy: str = "entropy",  # 'entropy', 'margin', or 'hybrid'
    ) -> List[AnnotationItem]:
        """
        Score articles by model uncertainty and return top-N items for the labeling queue.
        """
        items: List[AnnotationItem] = []

        for idx, art in enumerate(articles):
            probs = prediction_probabilities[idx]
            entropy = self.calculate_entropy(probs)
            margin = self.calculate_margin(probs)
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            pred_class = classes[top_idx]

            # Invert margin so higher score = higher priority
            margin_uncertainty = 1.0 - margin

            if strategy == "entropy":
                uncertainty = entropy
            elif strategy == "margin":
                uncertainty = margin_uncertainty
            else:
                # Normalized hybrid
                uncertainty = 0.5 * entropy + 0.5 * margin_uncertainty

            items.append(AnnotationItem(
                article_id=art.get("article_id", f"art_{idx}"),
                title=art.get("title", ""),
                abstract=art.get("abstract", ""),
                predicted_label=pred_class,
                prediction_confidence=round(top_prob, 4),
                entropy=round(entropy, 4),
                margin=round(margin, 4),
                uncertainty_score=round(uncertainty, 4),
            ))

        items.sort(key=lambda x: x.uncertainty_score, reverse=True)
        return items[:top_n]


class PreLabelProvider(abc.ABC):
    """Abstract base class for active learning pre-label providers."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def is_available(self) -> bool:
        pass

    @abc.abstractmethod
    def pre_label(
        self,
        title: str,
        abstract: str,
        available_classes: List[str],
        return_metadata: bool = False,
    ) -> Any:
        """
        Returns (predicted_category, reasoning_rationale, confidence) or
        (predicted_category, reasoning_rationale, confidence, metadata) if return_metadata=True.
        """
        pass


class MockLLMPreLabeler(PreLabelProvider):
    """
    MOCK — Keyword-based test provider.
    Explicitly disclosed as a simulated rule engine for offline development and testing.
    Never mislabeled as an external LLM.
    """

    KEYWORD_MAPPINGS = {
        "news": ["politic", "government", "president", "senate", "election", "court", "state", "police", "bill"],
        "sports": ["game", "nfl", "nba", "team", "coach", "season", "championship", "score", "football", "basketball"],
        "finance": ["market", "stock", "dollar", "inflation", "bank", "invest", "economy", "ceo", "fund", "crypto", "shares"],
        "entertainment": ["movie", "film", "actor", "actress", "star", "singer", "music", "hollywood", "tv", "series", "show"],
        "technology": ["ai", "tech", "google", "apple", "software", "computer", "phone", "robot", "data", "cyber", "app"],
        "lifestyle": ["food", "recipe", "health", "travel", "diet", "fitness", "style", "home", "garden", "living"],
    }

    @property
    def provider_name(self) -> str:
        return "MOCK — keyword-based test provider"

    @property
    def is_available(self) -> bool:
        return True

    def pre_label(
        self,
        title: str,
        abstract: str,
        available_classes: List[str],
        return_metadata: bool = False,
    ) -> Any:
        t0 = time.perf_counter()
        text = f"{title} {abstract}".lower()

        scores: Dict[str, int] = {c: 0 for c in available_classes}
        for cat, kws in self.KEYWORD_MAPPINGS.items():
            for c in available_classes:
                if cat in c.lower() or c.lower() in cat:
                    for kw in kws:
                        if kw in text:
                            scores[c] += 1

        best_class = max(scores, key=scores.get)
        best_count = scores[best_class]

        if best_count > 0:
            rationale = f"[MOCK] Keyword rules matched {best_count} topical keywords corresponding to '{best_class}'."
            confidence = min(0.60 + 0.10 * best_count, 0.95)
        else:
            best_class = available_classes[0] if available_classes else "news"
            rationale = "[MOCK] Default classification assigned; no distinctive keywords detected in title/abstract."
            confidence = 0.50

        latency_ms = (time.perf_counter() - t0) * 1000.0
        meta = {
            "provider": self.provider_name,
            "latency_ms": round(latency_ms, 2),
            "token_usage": 0,
            "is_mock": True,
        }
        if return_metadata:
            return best_class, rationale, round(confidence, 2), meta
        return best_class, rationale, round(confidence, 2)


class ExternalLLMPreLabelProvider(PreLabelProvider):
    """
    External LLM pre-labeler (e.g. OpenAI/Anthropic/Gemini).
    Disabled when API keys are not provided.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model

    @property
    def provider_name(self) -> str:
        return f"ExternalLLM ({self.model})"

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def pre_label(
        self,
        title: str,
        abstract: str,
        available_classes: List[str],
        return_metadata: bool = False,
    ) -> Any:
        if not self.is_available:
            raise RuntimeError("External LLM provider is not configured with valid API credentials.")
        # Future endpoint integration
        raise NotImplementedError("External LLM API call not executed without active network credentials.")


def compute_inter_annotator_agreement(
    rater1_labels: List[str],
    rater2_labels: List[str],
    labels_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute Cohen's Kappa metric between two independent annotation tracks.
    kappa = (p_o - p_e) / (1 - p_e)
    """
    if len(rater1_labels) != len(rater2_labels) or not rater1_labels:
        return {"cohen_kappa": 0.0, "observed_agreement": 0.0, "interpretation": "insufficient_data"}

    kappa = float(cohen_kappa_score(rater1_labels, rater2_labels, labels=labels_order))
    agreements = sum(1 for a, b in zip(rater1_labels, rater2_labels) if a == b)
    p_o = float(agreements / len(rater1_labels))

    # Standard Landis & Koch (1977) interpretation
    if kappa < 0:
        interp = "Poor (Less than chance)"
    elif kappa <= 0.20:
        interp = "Slight"
    elif kappa <= 0.40:
        interp = "Fair"
    elif kappa <= 0.60:
        interp = "Moderate"
    elif kappa <= 0.80:
        interp = "Substantial"
    else:
        interp = "Almost Perfect"

    return {
        "cohen_kappa": round(kappa, 4),
        "observed_agreement": round(p_o, 4),
        "total_samples": len(rater1_labels),
        "interpretation": interp,
    }

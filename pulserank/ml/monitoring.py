"""Inference Monitoring, Latency Tracking, and Population Stability Index (PSI) Drift Analysis."""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ServingStats:
    total_requests: int
    mean_latency_ms: float
    p95_latency_ms: float
    channel_contribution_shares: Dict[str, float]
    mean_prediction_entropy: float


@dataclass
class DriftReport:
    psi_score: float
    status: str  # 'stable', 'moderate_drift', 'significant_drift'
    baseline_distribution: Dict[str, float]
    serving_distribution: Dict[str, float]
    per_category_psi: Dict[str, float]
    alert_triggered: bool
    recommended_action: str


class InferenceMonitor:
    """
    Monitors online serving health:
    - Measures request latencies and per-channel candidate sourcing shares.
    - Computes Population Stability Index (PSI) on category exposure distributions.
    - Triggers automated alerts when drift crosses standard statistical bounds.
    """

    def __init__(self, baseline_category_dist: Optional[Dict[str, float]] = None):
        self.baseline_category_dist = baseline_category_dist or {}
        self.request_logs: List[Dict[str, Any]] = []

    def log_inference_event(
        self,
        endpoint: str,
        latency_ms: float,
        user_id: Optional[str] = None,
        channels: Optional[Dict[str, int]] = None,
        categories_served: Optional[List[str]] = None,
        prediction_scores: Optional[List[float]] = None,
    ) -> None:
        """Record an inference serving event."""
        entropy = 0.0
        if prediction_scores and len(prediction_scores) > 1:
            # Softmax normalize scores for entropy
            exp_s = np.exp(prediction_scores - np.max(prediction_scores))
            probs = exp_s / max(np.sum(exp_s), 1e-9)
            entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-9, 1.0))))

        self.request_logs.append({
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "user_id": user_id,
            "channels": channels or {},
            "categories_served": categories_served or [],
            "prediction_entropy": round(entropy, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_serving_statistics(self) -> ServingStats:
        """Aggregate recent serving performance."""
        if not self.request_logs:
            return ServingStats(0, 0.0, 0.0, {}, 0.0)

        latencies = [log["latency_ms"] for log in self.request_logs]
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))

        # Aggregate channel contributions
        channel_counts: Dict[str, int] = {}
        for log in self.request_logs:
            for ch, count in log.get("channels", {}).items():
                channel_counts[ch] = channel_counts.get(ch, 0) + count

        total_ch = max(sum(channel_counts.values()), 1)
        channel_shares = {ch: round(c / total_ch, 4) for ch, c in channel_counts.items()}

        entropies = [log["prediction_entropy"] for log in self.request_logs if log["prediction_entropy"] > 0]
        mean_ent = float(np.mean(entropies)) if entropies else 0.0

        return ServingStats(
            total_requests=len(self.request_logs),
            mean_latency_ms=round(mean_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            channel_contribution_shares=channel_shares,
            mean_prediction_entropy=round(mean_ent, 4),
        )

    def compute_psi_drift(
        self,
        serving_category_dist: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-4,
    ) -> DriftReport:
        """
        Compute Population Stability Index (PSI):
        PSI = sum((Serving_i - Baseline_i) * ln(Serving_i / Baseline_i))
        """
        # Extract serving distribution from recent logs if not provided
        if serving_category_dist is None:
            served_cats: Dict[str, int] = {}
            for log in self.request_logs:
                for cat in log.get("categories_served", []):
                    served_cats[cat] = served_cats.get(cat, 0) + 1
            total_served = max(sum(served_cats.values()), 1)
            serving_dist = {k: v / total_served for k, v in served_cats.items()}
        else:
            serving_dist = dict(serving_category_dist)

        baseline_dist = self.baseline_category_dist or serving_dist
        all_categories = sorted(list(set(baseline_dist.keys()) | set(serving_dist.keys())))

        total_psi = 0.0
        per_cat_psi: Dict[str, float] = {}

        for cat in all_categories:
            b_prob = max(baseline_dist.get(cat, 0.0), epsilon)
            s_prob = max(serving_dist.get(cat, 0.0), epsilon)

            # PSI contribution
            psi_i = (s_prob - b_prob) * math.log(s_prob / b_prob)
            per_cat_psi[cat] = round(psi_i, 5)
            total_psi += psi_i

        total_psi = round(total_psi, 4)

        if total_psi < 0.10:
            status = "stable"
            alert = False
            action = "Distribution aligns with baseline training corpus. No action required."
        elif total_psi < 0.25:
            status = "moderate_drift"
            alert = True
            action = "Moderate category distribution shift observed. Monitor recommendation entropy."
        else:
            status = "significant_drift"
            alert = True
            action = "Severe distribution shift detected (PSI >= 0.25). Model retraining recommended."

        return DriftReport(
            psi_score=total_psi,
            status=status,
            baseline_distribution={k: round(v, 4) for k, v in baseline_dist.items()},
            serving_distribution={k: round(v, 4) for k, v in serving_dist.items()},
            per_category_psi=per_cat_psi,
            alert_triggered=alert,
            recommended_action=action,
        )

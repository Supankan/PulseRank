"""Experimentation Engine: Paired Bootstrap Statistical Testing, A/B Sizing, and Online Routing."""

import math
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats

from pulserank.ml.evaluation import Evaluator, compute_impression_auc, compute_impression_mrr, compute_impression_ndcg

logger = logging.getLogger(__name__)


@dataclass
class BootstrapMetricDelta:
    metric_name: str
    model_a_mean: float
    model_b_mean: float
    delta_mean: float
    ci_lower_95: float
    ci_upper_95: float
    p_value: float
    is_statistically_significant: bool


@dataclass
class SampleSizeResult:
    baseline_ctr: float
    mde_relative: float
    mde_absolute: float
    alpha: float
    power: float
    required_sample_size_per_variant: int
    total_sample_size: int


class ExperimentationEngine:
    """
    A/B testing, statistical hypothesis testing, and sample-size sizing engine.
    """

    @staticmethod
    def calculate_ab_sample_size(
        baseline_rate: float = 0.04,  # e.g. 4% baseline CTR
        mde_relative: float = 0.05,   # e.g. 5% relative lift -> target 4.2%
        alpha: float = 0.05,          # 5% significance (two-sided)
        power: float = 0.80,          # 80% statistical power
    ) -> SampleSizeResult:
        """
        Calculate required sample size per variant for two-tailed two-proportion z-test.
        N = [Z_{alpha/2} * sqrt(2*p_bar*(1-p_bar)) + Z_{beta} * sqrt(p1*(1-p1) + p2*(1-p2))]^2 / (p2 - p1)^2
        """
        p1 = baseline_rate
        delta_abs = p1 * mde_relative
        p2 = p1 + delta_abs
        p_bar = (p1 + p2) / 2.0

        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(power)

        term1 = z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        term2 = z_beta * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
        n_per_variant = int(math.ceil(((term1 + term2) ** 2) / (delta_abs ** 2)))

        return SampleSizeResult(
            baseline_ctr=round(p1, 4),
            mde_relative=round(mde_relative, 4),
            mde_absolute=round(delta_abs, 6),
            alpha=alpha,
            power=power,
            required_sample_size_per_variant=n_per_variant,
            total_sample_size=n_per_variant * 2,
        )

    def run_paired_bootstrap(
        self,
        behaviors: List[Any],
        score_func_a: Any,
        score_func_b: Any,
        n_bootstraps: int = 500,
        random_seed: int = 42,
    ) -> Dict[str, BootstrapMetricDelta]:
        """
        Paired bootstrap test across impression-level dev behaviors.
        Calculates 95% Confidence Intervals on delta and empirical two-sided p-values.
        """
        rng = np.random.RandomState(random_seed)

        # Precompute per-impression metrics for both models
        auc_a, auc_b = [], []
        mrr_a, mrr_b = [], []
        ndcg5_a, ndcg5_b = [], []

        for b in behaviors:
            if not b.impressions:
                continue
            cand_ids = [item.news_id for item in b.impressions]
            labels = [item.clicked for item in b.impressions]
            num_pos = sum(labels)
            if num_pos == 0 or num_pos == len(labels):
                continue

            try:
                s_a = score_func_a(b.user_id, cand_ids, user_history=b.history)
                s_b = score_func_b(b.user_id, cand_ids, user_history=b.history)
            except Exception:
                continue

            # Model A
            ranked_a = [lbl for _, lbl in sorted(zip(s_a, labels), reverse=True)]
            auc_a.append(compute_impression_auc(labels, s_a))
            mrr_a.append(compute_impression_mrr(ranked_a))
            ndcg5_a.append(compute_impression_ndcg(ranked_a, num_pos, 5))

            # Model B
            ranked_b = [lbl for _, lbl in sorted(zip(s_b, labels), reverse=True)]
            auc_b.append(compute_impression_auc(labels, s_b))
            mrr_b.append(compute_impression_mrr(ranked_b))
            ndcg5_b.append(compute_impression_ndcg(ranked_b, num_pos, 5))

        n_samples = len(auc_a)
        if n_samples == 0:
            logger.warning("No evaluatable impressions for bootstrap testing.")
            return {}

        arr_auc_a, arr_auc_b = np.array(auc_a), np.array(auc_b)
        arr_mrr_a, arr_mrr_b = np.array(mrr_a), np.array(mrr_b)
        arr_ndcg5_a, arr_ndcg5_b = np.array(ndcg5_a), np.array(ndcg5_b)

        boot_delta_auc = []
        boot_delta_mrr = []
        boot_delta_ndcg5 = []

        for _ in range(n_bootstraps):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_delta_auc.append(float(np.mean(arr_auc_b[indices]) - np.mean(arr_auc_a[indices])))
            boot_delta_mrr.append(float(np.mean(arr_mrr_b[indices]) - np.mean(arr_mrr_a[indices])))
            boot_delta_ndcg5.append(float(np.mean(arr_ndcg5_b[indices]) - np.mean(arr_ndcg5_a[indices])))

        def summarize_metric(name, vals_a, vals_b, deltas):
            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            delta_m = mean_b - mean_a
            ci_low = float(np.percentile(deltas, 2.5))
            ci_high = float(np.percentile(deltas, 97.5))

            # Two-sided empirical p-value
            p_val = 2.0 * min(np.mean(np.array(deltas) <= 0.0), np.mean(np.array(deltas) >= 0.0))
            p_val = min(max(float(p_val), 1e-4), 1.0)
            sig = bool(ci_low > 0 or ci_high < 0)

            return BootstrapMetricDelta(
                metric_name=name,
                model_a_mean=round(mean_a, 4),
                model_b_mean=round(mean_b, 4),
                delta_mean=round(delta_m, 4),
                ci_lower_95=round(ci_low, 4),
                ci_upper_95=round(ci_high, 4),
                p_value=round(p_val, 4),
                is_statistically_significant=sig,
            )

        return {
            "auc": summarize_metric("AUC", arr_auc_a, arr_auc_b, boot_delta_auc),
            "mrr": summarize_metric("MRR", arr_mrr_a, arr_mrr_b, boot_delta_mrr),
            "ndcg_at_5": summarize_metric("nDCG@5", arr_ndcg5_a, arr_ndcg5_b, boot_delta_ndcg5),
        }

    @staticmethod
    def assign_variant(user_id: str, experiment_salt: str = "exp_ranker_v2", traffic_split: float = 0.5) -> str:
        """
        Deterministic user traffic assignment using SHA-256 hash modulo.
        """
        token = f"{experiment_salt}_{user_id}"
        hash_val = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        normalized_val = (hash_val % 10000) / 10000.0
        return "variant_a" if normalized_val < traffic_split else "variant_b"

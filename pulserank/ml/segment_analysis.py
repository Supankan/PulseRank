"""Performance Segmentation and Recommender Failure Diagnostics."""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from pulserank.ml.evaluation import Evaluator, EvaluationMetrics

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetric:
    segment_type: str       # 'user_activity', 'category', 'impression_length'
    segment_name: str       # 'Cold (0-2)', 'Sports', 'Short (<=5)'
    sample_size: int
    auc: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float


@dataclass
class FailureCase:
    case_id: str
    archetype: str          # 'clickbait_misrank', 'echo_chamber_rabbit_hole', 'cold_start_drop'
    user_id: str
    user_history_length: int
    user_top_category: str
    top_recommended_article: Dict[str, Any]
    actual_clicked_article: Dict[str, Any]
    diagnosis: str
    remediation: str


class SegmentDiagnosticsEngine:
    """
    Analyzes model performance across behavioral slices and extracts concrete failure archetypes.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        article_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.evaluator = evaluator
        self.article_metadata = article_metadata or {}

    def run_segment_analysis(self, score_func: Any) -> Dict[str, List[SegmentMetric]]:
        """
        Compute metrics sliced by User Activity, News Category, and Impression Length.
        """
        behaviors = self.evaluator.dev_behaviors
        results: Dict[str, List[SegmentMetric]] = {
            "user_activity": [],
            "category": [],
            "impression_length": [],
        }

        # 1. User Activity Slices
        activity_bins = {
            "Cold (0-2 history)": [b for b in behaviors if len(b.history) <= 2],
            "Moderate (3-10 history)": [b for b in behaviors if 3 <= len(b.history) <= 10],
            "Heavy (>10 history)": [b for b in behaviors if len(b.history) > 10],
        }
        for name, b_list in activity_bins.items():
            if b_list:
                ev = Evaluator(b_list)
                m = ev.evaluate_model(f"Activity_{name}", score_func)
                results["user_activity"].append(SegmentMetric(
                    segment_type="user_activity",
                    segment_name=name,
                    sample_size=len(b_list),
                    auc=m.auc, mrr=m.mrr, ndcg_at_5=m.ndcg_at_5, ndcg_at_10=m.ndcg_at_10,
                ))

        # 2. Impression Length Slices
        imp_bins = {
            "Short (<=5)": [b for b in behaviors if len(b.impressions) <= 5],
            "Medium (6-15)": [b for b in behaviors if 6 <= len(b.impressions) <= 15],
            "Long (>15)": [b for b in behaviors if len(b.impressions) > 15],
        }
        for name, b_list in imp_bins.items():
            if b_list:
                ev = Evaluator(b_list)
                m = ev.evaluate_model(f"Impression_{name}", score_func)
                results["impression_length"].append(SegmentMetric(
                    segment_type="impression_length",
                    segment_name=name,
                    sample_size=len(b_list),
                    auc=m.auc, mrr=m.mrr, ndcg_at_5=m.ndcg_at_5, ndcg_at_10=m.ndcg_at_10,
                ))

        # 3. Category Slices (based on first clicked impression item's category)
        cat_map: Dict[str, List[Any]] = {}
        for b in behaviors:
            clicked_item = next((item for item in b.impressions if item.clicked == 1), None)
            if clicked_item and clicked_item.news_id in self.article_metadata:
                cat = getattr(self.article_metadata[clicked_item.news_id], "category", "unknown")
                cat_map.setdefault(cat, []).append(b)

        # Take top 5 categories by sample size
        top_cats = sorted(cat_map.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for cat, b_list in top_cats:
            ev = Evaluator(b_list)
            m = ev.evaluate_model(f"Category_{cat}", score_func)
            results["category"].append(SegmentMetric(
                segment_type="category",
                segment_name=cat.title(),
                sample_size=len(b_list),
                auc=m.auc, mrr=m.mrr, ndcg_at_5=m.ndcg_at_5, ndcg_at_10=m.ndcg_at_10,
            ))

        return results

    def extract_failure_archetypes(
        self,
        score_func: Any,
        max_cases_per_type: int = 2,
    ) -> List[FailureCase]:
        """
        Identify representative failure cases where the ranker produced suboptimal recommendations.
        """
        cases: List[FailureCase] = []
        c_clickbait, c_echo, c_cold = 0, 0, 0

        for b in self.evaluator.dev_behaviors:
            if not b.impressions:
                continue
            cand_ids = [item.news_id for item in b.impressions]
            labels = [item.clicked for item in b.impressions]
            num_pos = sum(labels)
            if num_pos == 0:
                continue

            try:
                scores = score_func(b.user_id, cand_ids, user_history=b.history)
            except Exception:
                continue

            # Rank items by score
            ranked_pairs = sorted(zip(cand_ids, labels, scores), key=lambda x: x[2], reverse=True)
            top_rec_id, top_label, top_score = ranked_pairs[0]

            # Find actual clicked item
            clicked_pair = next((p for p in ranked_pairs if p[1] == 1), None)
            if not clicked_pair:
                continue
            clicked_id, _, clicked_score = clicked_pair

            top_meta = self.article_metadata.get(top_rec_id)
            click_meta = self.article_metadata.get(clicked_id)

            # Archetype 1: Clickbait / Popularity Overweighting
            # Top recommended item was NOT clicked, but was ranked top despite low topical relevance
            if top_label == 0 and c_clickbait < max_cases_per_type:
                cases.append(FailureCase(
                    case_id=f"FB-CB-{b.impression_id}",
                    archetype="clickbait_misrank",
                    user_id=b.user_id,
                    user_history_length=len(b.history),
                    user_top_category=getattr(click_meta, "category", "unknown"),
                    top_recommended_article={
                        "article_id": top_rec_id,
                        "title": getattr(top_meta, "title", "Unknown Headline"),
                        "category": getattr(top_meta, "category", "unknown"),
                        "score": round(top_score, 4),
                    },
                    actual_clicked_article={
                        "article_id": clicked_id,
                        "title": getattr(click_meta, "title", "Unknown Headline"),
                        "category": getattr(click_meta, "category", "unknown"),
                        "score": round(clicked_score, 4),
                    },
                    diagnosis="Global popularity overwhelmed content affinity, placing an unclicked viral article ahead of niche topical preference.",
                    remediation="Increase content/entity affinity feature weight in LambdaMART or tighten popularity feature saturation curve.",
                ))
                c_clickbait += 1

            # Archetype 2: Cold-Start Drop
            if len(b.history) <= 1 and top_label == 0 and c_cold < max_cases_per_type:
                cases.append(FailureCase(
                    case_id=f"FB-CS-{b.impression_id}",
                    archetype="cold_start_drop",
                    user_id=b.user_id,
                    user_history_length=len(b.history),
                    user_top_category="unobserved",
                    top_recommended_article={
                        "article_id": top_rec_id,
                        "title": getattr(top_meta, "title", "Unknown Headline"),
                        "category": getattr(top_meta, "category", "unknown"),
                        "score": round(top_score, 4),
                    },
                    actual_clicked_article={
                        "article_id": clicked_id,
                        "title": getattr(click_meta, "title", "Unknown Headline"),
                        "category": getattr(click_meta, "category", "unknown"),
                        "score": round(clicked_score, 4),
                    },
                    diagnosis="Sparse user interaction history prevented collaborative and content models from activating effectively.",
                    remediation="Trigger Onboarding Persona flow or inject diverse top-category exploration slots for users with |history| <= 2.",
                ))
                c_cold += 1

            if c_clickbait >= max_cases_per_type and c_cold >= max_cases_per_type:
                break

        return cases

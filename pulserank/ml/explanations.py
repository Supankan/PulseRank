"""Transparent, engineering-grade recommendation explanations."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from pulserank.ml.hybrid import CandidateItem


@dataclass
class RecommendationExplanation:
    article_id: str
    rank: int
    final_score: float
    sources: List[str]
    component_scores: Dict[str, float]
    feature_breakdown: Dict[str, float]
    primary_reason: str
    summary_text: str


class ExplanationGenerator:
    """Generates verifiable explanations based directly on calculated model features."""

    @staticmethod
    def explain(
        article_id: str,
        rank: int,
        score: float,
        candidate_item: Optional[CandidateItem],
        feature_values: Optional[Dict[str, float]] = None,
        article_category: Optional[str] = None,
    ) -> RecommendationExplanation:
        """
        Build structured explanation from actual pipeline values.
        """
        sources = candidate_item.sources if candidate_item else ["unknown"]
        content_score = candidate_item.content_score if candidate_item else 0.0
        collab_score = candidate_item.collaborative_score if candidate_item else 0.0
        pop_score = candidate_item.popularity_score if candidate_item else 0.0

        component_scores = {
            "content": round(content_score, 4),
            "collaborative": round(collab_score, 4),
            "popularity": round(pop_score, 4),
        }

        feat_dict = feature_values or {}
        cat_aff = feat_dict.get("category_affinity", 0.0)
        entity_overlap = int(feat_dict.get("entity_overlap_count", 0))

        # Determine primary driving factor
        reasons: List[str] = []
        if len(sources) > 1:
            reasons.append(f"Multi-channel discovery via {', '.join(sources)}")
        elif "content" in sources:
            reasons.append(f"High semantic alignment with recent reading history (similarity: {content_score:.2f})")
        elif "collaborative" in sources:
            reasons.append("Frequently engaged by readers with overlapping interests")
        elif "popularity" in sources:
            reasons.append("Trending content with high recency-adjusted engagement")

        if cat_aff > 0.3:
            cat_pct = int(cat_aff * 100)
            reasons.append(f"{cat_pct}% of past interactions match category '{article_category or 'this topic'}'")

        if entity_overlap > 0:
            reasons.append(f"Shares {entity_overlap} key entity/subject reference(s) from past reading")

        primary_reason = reasons[0] if reasons else "Selected by multi-stage retrieval."
        summary_text = " • ".join(reasons)

        return RecommendationExplanation(
            article_id=article_id,
            rank=rank,
            final_score=round(score, 4),
            sources=sources,
            component_scores=component_scores,
            feature_breakdown={k: round(v, 4) for k, v in feat_dict.items()},
            primary_reason=primary_reason,
            summary_text=summary_text,
        )

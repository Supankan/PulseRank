"""Failure and Behavioral Segment Diagnostics Routes."""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException

from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.ml.evaluation import Evaluator
from pulserank.ml.segment_analysis import SegmentDiagnosticsEngine

router = APIRouter(prefix="/diagnostics", tags=["Diagnostics"])


@router.get("/segments")
def get_segment_diagnostics(services: ServiceContainer = Depends(get_service_container)):
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    evaluator = Evaluator(services.dataset.dev_behaviors[:120])
    diag = SegmentDiagnosticsEngine(evaluator=evaluator, article_metadata=services.dataset.articles)

    def score_ltr(uid, cids, hist=None):
        if services.ranker_model.model is None:
            return services.hybrid_generator.score_candidates_pointwise(uid, cids, user_history=hist)
        feats = [services.feature_extractor.extract_features(uid, a, user_history=hist) for a in cids]
        return [float(p) for p in services.ranker_model.model.predict(feats)]

    slices = diag.run_segment_analysis(score_ltr)

    serialized = {
        s_type: [
            {
                "segment_type": m.segment_type,
                "segment_name": m.segment_name,
                "sample_size": m.sample_size,
                "auc": m.auc,
                "mrr": m.mrr,
                "ndcg_at_5": m.ndcg_at_5,
                "ndcg_at_10": m.ndcg_at_10,
            }
            for m in m_list
        ]
        for s_type, m_list in slices.items()
    }

    return {"segments": serialized}


@router.get("/failures")
def get_failure_archetypes(services: ServiceContainer = Depends(get_service_container)):
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    evaluator = Evaluator(services.dataset.dev_behaviors[:150])
    diag = SegmentDiagnosticsEngine(evaluator=evaluator, article_metadata=services.dataset.articles)

    def score_ltr(uid, cids, hist=None):
        if services.ranker_model.model is None:
            return services.hybrid_generator.score_candidates_pointwise(uid, cids, user_history=hist)
        feats = [services.feature_extractor.extract_features(uid, a, user_history=hist) for a in cids]
        return [float(p) for p in services.ranker_model.model.predict(feats)]

    cases = diag.extract_failure_archetypes(score_ltr)

    serialized = [
        {
            "case_id": c.case_id,
            "archetype": c.archetype,
            "user_id": c.user_id,
            "user_history_length": c.user_history_length,
            "user_top_category": c.user_top_category,
            "top_recommended_article": c.top_recommended_article,
            "actual_clicked_article": c.actual_clicked_article,
            "diagnosis": c.diagnosis,
            "remediation": c.remediation,
        }
        for c in cases
    ]

    return {"failure_cases": serialized, "count": len(serialized)}

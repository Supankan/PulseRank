import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from pulserank.db.session import get_db
from pulserank.db.models import Interaction
from pulserank.api.schemas import ExplanationResponse, InteractionRequest, InteractionResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Recommendations & Interactions"])


@router.get("/recommendations/{recommendation_id}/explanation", response_model=ExplanationResponse)
def get_recommendation_explanation(
    recommendation_id: str,
    services: ServiceContainer = Depends(get_service_container),
):
    """
    Retrieve transparent engineering-grade explanation and feature breakdown
    for a specific generated recommendation.
    """
    cached = services.explanation_cache.get(recommendation_id)
    if not cached:
        raise HTTPException(
            status_code=404,
            detail=f"Recommendation '{recommendation_id}' not found in active session cache.",
        )

    expl = cached["explanation"]
    user_id = cached["user_id"]

    return ExplanationResponse(
        recommendation_id=recommendation_id,
        article_id=expl.article_id,
        user_id=user_id,
        rank=expl.rank,
        final_score=expl.final_score,
        sources=expl.sources,
        component_scores=expl.component_scores,
        feature_breakdown=expl.feature_breakdown,
        primary_reason=expl.primary_reason,
        summary_text=expl.summary_text,
    )


@router.post("/interactions", response_model=InteractionResponse)
def record_interaction(
    req: InteractionRequest,
    db: Session = Depends(get_db),
    services: ServiceContainer = Depends(get_service_container),
):
    """
    Record user feedback action (e.g., 'like', 'save', 'open', 'dismiss')
    without modifying official MIND evaluation ground truth.
    """
    now = datetime.now(timezone.utc)
    try:
        interaction = Interaction(
            user_id=req.user_id,
            article_id=req.article_id,
            interaction_type=req.interaction_type,
            timestamp=now,
        )
        db.add(interaction)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to commit user interaction to database: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database commit failed while recording interaction: {str(e)}",
        )

    return InteractionResponse(
        status="recorded",
        user_id=req.user_id,
        article_id=req.article_id,
        interaction_type=req.interaction_type,
        timestamp=now.isoformat(),
    )

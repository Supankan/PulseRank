"""Health check route."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from pulserank.config import settings
from pulserank.db.session import get_db
from pulserank.api.schemas import HealthResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    db: Session = Depends(get_db),
    services: ServiceContainer = Depends(get_service_container),
):
    """Verify application health, database connectivity, and ML model readiness."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    models_ready = {
        "popularity": services.popularity_model is not None,
        "content_based": services.content_model is not None,
        "collaborative": services.collaborative_model is not None,
        "hybrid": services.hybrid_generator is not None,
        "ranker": services.ranker_model is not None and services.ranker_model.model is not None,
    }

    dataset_loaded = services.dataset is not None
    article_count = len(services.dataset.articles) if services.dataset else 0
    user_count = len(services.dataset.user_histories) if services.dataset else 0

    return HealthResponse(
        status="healthy" if db_connected and dataset_loaded else "degraded",
        app_name=settings.app_name,
        env=settings.env,
        database_connected=db_connected,
        models_ready=models_ready,
        dataset_loaded=dataset_loaded,
        article_count=article_count,
        user_count=user_count,
    )

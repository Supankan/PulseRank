"""Model registry and operational status routes."""

from typing import List
from fastapi import APIRouter, Depends

from pulserank.config import settings
from pulserank.api.schemas import ModelStatusResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=List[ModelStatusResponse])
def list_models(
    services: ServiceContainer = Depends(get_service_container),
):
    """List operational status and descriptions of all recommendation models."""
    pop_art = (settings.artifacts_dir / "popularity.pkl").exists()
    content_art = (settings.artifacts_dir / "content" / "content_meta.pkl").exists()
    als_art = (settings.artifacts_dir / "als.pkl").exists()
    ranker_art = (settings.artifacts_dir / "ranker.pkl").exists()

    models = [
        ModelStatusResponse(
            name="Popularity Baseline",
            model_type="popularity",
            status="ready" if services.popularity_model else "uninitialized",
            artifact_available=pop_art,
            description="Recency-decayed interaction popularity score serving as baseline and cold-start fallback.",
        ),
        ModelStatusResponse(
            name="Sentence-Transformers Content Recommender",
            model_type="content_based",
            status="ready" if services.content_model else "uninitialized",
            artifact_available=content_art,
            description="Dense semantic retrieval using all-MiniLM-L6-v2 embeddings indexed in FAISS.",
        ),
        ModelStatusResponse(
            name="Implicit ALS Collaborative Filtering",
            model_type="collaborative",
            status="ready" if services.collaborative_model else "uninitialized",
            artifact_available=als_art,
            description="Implicit feedback matrix factorization modeling latent user and article vectors.",
        ),
        ModelStatusResponse(
            name="Hybrid Candidate Fusion",
            model_type="hybrid",
            status="ready" if services.hybrid_generator else "uninitialized",
            artifact_available=pop_art and content_art and als_art,
            description="Multi-channel candidate generator fusing content, collaborative, and popularity candidates.",
        ),
        ModelStatusResponse(
            name="LightGBM LambdaMART Ranker",
            model_type="ranker",
            status="ready" if (services.ranker_model and services.ranker_model.model) else "uninitialized",
            artifact_available=ranker_art,
            description="Supervised learning-to-rank model trained on impression-grouped pairwise preferences.",
        ),
    ]
    return models

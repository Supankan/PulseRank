"""Article metadata exploration routes."""

from fastapi import APIRouter, Depends, HTTPException

from pulserank.api.schemas import ArticleResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container

router = APIRouter(prefix="/articles", tags=["Articles"])


@router.get("/{article_id}", response_model=ArticleResponse)
def get_article(
    article_id: str,
    services: ServiceContainer = Depends(get_service_container),
):
    """Retrieve detailed metadata for a specific article."""
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    art = services.dataset.articles.get(article_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found.")

    clicks = services.dataset.article_click_counts.get(article_id, 0)

    return ArticleResponse(
        article_id=art.news_id,
        category=art.category,
        subcategory=art.subcategory,
        title=art.title,
        abstract=art.abstract,
        url=art.url,
        title_entities=art.title_entities,
        abstract_entities=art.abstract_entities,
        click_count=clicks,
    )

"""User management and profile exploration routes."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from pulserank.api.schemas import UserSummaryResponse, UserProfileResponse, EntityPreferenceResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("", response_model=List[UserSummaryResponse])
def list_users(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    services: ServiceContainer = Depends(get_service_container),
):
    """List available MIND users with history summary."""
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    user_ids = sorted(list(services.dataset.user_histories.keys()))
    paginated_ids = user_ids[offset : offset + limit]

    summaries: List[UserSummaryResponse] = []
    for uid in paginated_ids:
        hist = services.dataset.user_histories.get(uid, [])
        cat_counts = {}
        for aid in hist:
            art = services.dataset.articles.get(aid)
            if art:
                cat_counts[art.category] = cat_counts.get(art.category, 0) + 1
        top_cats = [c[0] for c in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]]

        summaries.append(
            UserSummaryResponse(
                user_id=uid,
                history_count=len(hist),
                top_categories=top_cats,
            )
        )
    return summaries


@router.get("/{user_id}", response_model=UserSummaryResponse)
def get_user(
    user_id: str,
    services: ServiceContainer = Depends(get_service_container),
):
    """Retrieve specific user details."""
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    hist = services.dataset.user_histories.get(user_id, [])
    cat_counts = {}
    for aid in hist:
        art = services.dataset.articles.get(aid)
        if art:
            cat_counts[art.category] = cat_counts.get(art.category, 0) + 1
    top_cats = [c[0] for c in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:3]]

    return UserSummaryResponse(
        user_id=user_id,
        history_count=len(hist),
        top_categories=top_cats,
    )


@router.get("/{user_id}/profile", response_model=UserProfileResponse)
def get_user_profile(
    user_id: str,
    services: ServiceContainer = Depends(get_service_container),
):
    """Retrieve interpretable user interest profile."""
    if not services.dataset or not services.profile_builder:
        raise HTTPException(status_code=503, detail="Profile service not ready.")

    hist = services.dataset.user_histories.get(user_id, [])
    profile = services.profile_builder.build_profile(user_id, hist)

    return UserProfileResponse(
        user_id=profile.user_id,
        history_size=profile.history_size,
        category_distribution=profile.category_distribution,
        subcategory_distribution=profile.subcategory_distribution,
        top_entities=[
            EntityPreferenceResponse(
                name=e.name,
                wikidata_id=e.wikidata_id,
                count=e.count,
                category=e.category,
            )
            for e in profile.top_entities
        ],
        top_categories=profile.top_categories,
        recent_articles=profile.recent_articles,
        semantic_summary=profile.semantic_summary,
    )

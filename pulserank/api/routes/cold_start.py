"""Cold-Start Onboarding Persona and Sliced Segment routes."""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.ml.cold_start import OnboardingPersona

router = APIRouter(prefix="/cold-start", tags=["Cold-Start"])


class OnboardingRequest(BaseModel):
    selected_categories: List[str]
    selected_subcategories: Optional[List[str]] = None
    seed_article_ids: Optional[List[str]] = None
    limit: int = 10


@router.post("/onboard")
def create_onboarding_persona(
    request: OnboardingRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    persona = OnboardingPersona(
        selected_categories=request.selected_categories,
        selected_subcategories=request.selected_subcategories,
        seed_article_ids=request.seed_article_ids,
    )
    vec, cat_priors, subcat_priors = services.cold_start_engine.create_onboarding_profile(
        persona, return_subcategories=True
    )

    # Score candidates across the eligible catalog using principled composite ranking
    recommended = []
    if services.dataset:
        recommended = services.cold_start_engine.score_onboarding_candidates(
            synthetic_vec=vec,
            cat_priors=cat_priors,
            subcat_priors=subcat_priors,
            articles=services.dataset.articles,
            content_model=services.content_model,
            popularity_model=services.popularity_model,
            top_k=request.limit,
        )

    return {
        "category_priors": cat_priors,
        "subcategory_priors": subcat_priors,
        "initial_slate": recommended,
        "status": "onboarding_profile_ready",
    }


@router.get("/segments")
def get_cold_start_segments(services: ServiceContainer = Depends(get_service_container)):
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    # Item interaction counts
    item_counts: Dict[str, int] = {}
    for b in services.dataset.train_behaviors:
        for nid in b.history:
            item_counts[nid] = item_counts.get(nid, 0) + 1
        for item in b.impressions:
            if item.clicked == 1:
                item_counts[item.news_id] = item_counts.get(item.news_id, 0) + 1

    sliced = services.cold_start_engine.slice_evaluation_segments(
        dev_behaviors=services.dataset.dev_behaviors,
        item_interaction_counts=item_counts,
    )

    return {
        "zero_history_count": len(sliced["zero_history"]),
        "history_1_to_2_count": len(sliced["history_1_to_2"]),
        "history_3_to_10_count": len(sliced["history_3_to_10"]),
        "history_10_plus_count": len(sliced["history_10_plus"]),
        "cold_users_count": len(sliced["cold_users"]),
        "warm_users_count": len(sliced["warm_users"]),
        "cold_item_impressions_count": len(sliced["cold_item_impressions"]),
        "total_impressions": len(sliced["all"]),
    }

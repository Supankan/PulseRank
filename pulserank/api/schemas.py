"""Pydantic schemas for API request and response validation."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# --- Health & Info ---
class HealthResponse(BaseModel):
    status: str
    app_name: str
    env: str
    database_connected: bool
    models_ready: Dict[str, bool]
    dataset_loaded: bool
    article_count: int
    user_count: int


# --- Articles ---
class ArticleResponse(BaseModel):
    article_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: Optional[str] = None
    title_entities: Optional[List[Dict[str, Any]]] = None
    abstract_entities: Optional[List[Dict[str, Any]]] = None
    click_count: int = 0


# --- Users & Profiles ---
class UserSummaryResponse(BaseModel):
    user_id: str
    history_count: int
    top_categories: List[str] = []


class EntityPreferenceResponse(BaseModel):
    name: str
    wikidata_id: Optional[str] = None
    count: int
    category: Optional[str] = None


class UserProfileResponse(BaseModel):
    user_id: str
    history_size: int
    category_distribution: Dict[str, float]
    subcategory_distribution: Dict[str, float]
    top_entities: List[EntityPreferenceResponse]
    top_categories: List[str]
    recent_articles: List[Dict[str, Any]]
    semantic_summary: str


# --- Feed & Recommendations ---
class FeedItemResponse(BaseModel):
    recommendation_id: str
    article_id: str
    title: str
    abstract: str
    category: str
    subcategory: str
    url: Optional[str] = None
    rank: int
    score: float
    sources: List[str]
    primary_reason: str


class FeedResponse(BaseModel):
    user_id: str
    model_used: str
    total_returned: int
    items: List[FeedItemResponse]


# --- Explanation ---
class ExplanationResponse(BaseModel):
    recommendation_id: str
    article_id: str
    user_id: str
    rank: int
    final_score: float
    sources: List[str]
    component_scores: Dict[str, float]
    feature_breakdown: Dict[str, float]
    primary_reason: str
    summary_text: str


# --- User Interaction ---
class InteractionRequest(BaseModel):
    user_id: str
    article_id: str
    interaction_type: str = Field(..., description="'like', 'save', 'open', or 'dismiss'")


class InteractionResponse(BaseModel):
    status: str
    user_id: str
    article_id: str
    interaction_type: str
    timestamp: str


# --- Models & Evaluations ---
class ModelStatusResponse(BaseModel):
    name: str
    model_type: str
    status: str
    artifact_available: bool
    description: str


class EvaluationResultResponse(BaseModel):
    model_name: str
    auc: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    recall_at_k: float
    coverage: float
    num_impressions: int
    evaluated_at: str
    dataset_source: Optional[str] = "mind-small"
    evaluation_mode: Optional[str] = "impression"
    impression_coverage: Optional[float] = None
    catalog_coverage: Optional[float] = None
    hit_rate_at_k: Optional[float] = None

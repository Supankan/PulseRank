"""Active Learning Annotation Queue, Pre-labeling, and Agreement Routes."""

from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.db.session import get_db
from pulserank.db.models import AnnotationRecord
from pulserank.nlp.annotation import (
    MockLLMPreLabeler,
    compute_inter_annotator_agreement,
)

router = APIRouter(prefix="/annotations", tags=["Annotation Queue"])


class PreLabelRequest(BaseModel):
    title: str
    abstract: str = ""


class SubmitAnnotationRequest(BaseModel):
    article_id: str
    task_type: str = "topic_classification"
    true_label: str
    predicted_label: Optional[str] = None
    annotator_id: str = "annotator_1"
    uncertainty_score: float = 0.0
    pre_label_provider: Optional[str] = None
    meta_info: Optional[Dict[str, Any]] = None


class AgreementRequest(BaseModel):
    rater1_labels: List[str]
    rater2_labels: List[str]


@router.get("/queue")
def get_annotation_queue(
    top_n: int = Query(20, ge=1, le=100),
    strategy: str = Query("entropy", description="'entropy', 'margin', or 'hybrid'"),
    services: ServiceContainer = Depends(get_service_container),
):
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    clf = services.topic_classifier_baseline
    if not clf or not clf.is_fitted:
        raise HTTPException(status_code=503, detail="Classifier not ready.")

    articles = list(services.dataset.articles.values())[:100]
    art_dicts = [
        {"article_id": getattr(a, "news_id", getattr(a, "article_id", str(i))), "title": a.title, "abstract": a.abstract}
        for i, a in enumerate(articles)
    ]

    texts = [f"{a.title} {a.abstract}" for a in articles]
    probs = clf.predict_proba(texts)

    queue = services.active_learning_queue.rank_for_annotation(
        articles=art_dicts,
        prediction_probabilities=probs,
        classes=clf.classes_,
        top_n=top_n,
        strategy=strategy,
    )

    serialized = [
        {
            "article_id": item.article_id,
            "title": item.title,
            "abstract": item.abstract,
            "predicted_label": item.predicted_label,
            "prediction_confidence": item.prediction_confidence,
            "entropy": item.entropy,
            "margin": item.margin,
            "uncertainty_score": item.uncertainty_score,
        }
        for item in queue
    ]

    return {"queue": serialized, "strategy": strategy, "count": len(serialized)}


@router.post("/pre-label")
def get_mock_llm_pre_label(
    request: PreLabelRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    pre_labeler = MockLLMPreLabeler()
    classes = services.topic_classifier_baseline.classes_ if services.topic_classifier_baseline else ["news", "sports", "finance", "entertainment"]
    cat, rationale, conf, meta = pre_labeler.pre_label(
        request.title, request.abstract, classes, return_metadata=True
    )

    return {
        "pre_label": cat,
        "rationale": rationale,
        "confidence": conf,
        "provider": pre_labeler.provider_name,
        "metadata": meta,
    }


@router.post("/submit")
def submit_annotation(
    request: SubmitAnnotationRequest,
    db: Session = Depends(get_db),
):
    annotation_id = f"annot_{uuid4().hex[:12]}"
    meta = request.meta_info or {}
    meta["review_state"] = "reviewed"
    meta["submitted_at"] = datetime.now(timezone.utc).isoformat()
    if request.pre_label_provider:
        meta["pre_label_provider"] = request.pre_label_provider

    record = AnnotationRecord(
        id=annotation_id,
        article_id=request.article_id,
        task_type=request.task_type,
        predicted_label=request.predicted_label,
        true_label=request.true_label,
        annotator_id=request.annotator_id,
        uncertainty_score=request.uncertainty_score,
        meta_info=meta,
        created_at=datetime.now(timezone.utc),
    )

    try:
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist annotation to database: {e}",
        )

    return {
        "status": "recorded",
        "annotation_id": annotation_id,
        "article_id": request.article_id,
        "true_label": request.true_label,
        "annotator_id": request.annotator_id,
    }


@router.get("/history")
def list_annotations(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieve persisted human-in-the-loop annotations."""
    records = db.query(AnnotationRecord).order_by(AnnotationRecord.created_at.desc()).limit(limit).all()
    items = [
        {
            "id": r.id,
            "annotation_id": r.id,
            "article_id": r.article_id,
            "task_type": r.task_type,
            "predicted_label": r.predicted_label,
            "true_label": r.true_label,
            "annotator_id": r.annotator_id,
            "uncertainty_score": r.uncertainty_score,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "meta_info": r.meta_info,
        }
        for r in records
    ]
    return {
        "annotations": items,
        "history": items,
        "count": len(items),
    }


@router.post("/agreement")
def compute_agreement(request: AgreementRequest):
    res = compute_inter_annotator_agreement(request.rater1_labels, request.rater2_labels)
    return res

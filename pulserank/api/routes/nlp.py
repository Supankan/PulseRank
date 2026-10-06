"""NLP Intelligence API routes: Topic Classification, NER, and Semantic Matching."""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.nlp.ner import ExtractedEntity

router = APIRouter(prefix="/nlp", tags=["NLP Intelligence"])


class TextClassificationRequest(BaseModel):
    title: str
    abstract: str = ""
    model_type: str = "baseline"  # 'baseline' / 'tfidf' or 'transformer' / 'distilbert'


class ClassificationResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_k_probabilities: List[Dict[str, Any]]
    model_type: str
    is_calibrated: bool = False


class EntityExtractionRequest(BaseModel):
    text: str
    article_id: Optional[str] = None


class EntityExtractionResponse(BaseModel):
    entities: List[Dict[str, Any]]
    total_extracted: int
    categories_found: List[str]
    pipeline_type: str
    mind_wikidata_entities: Optional[List[Dict[str, Any]]] = None


class CrossEncoderRerankRequest(BaseModel):
    user_context: str
    candidate_ids: List[str]
    top_k: int = 10


@router.post("/classify", response_model=ClassificationResponse)
def classify_text(
    request: TextClassificationRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    text = f"{request.title} {request.abstract}".strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text provided for classification.")

    model_choice = request.model_type.lower().strip()

    if model_choice in ("baseline", "tfidf"):
        clf = services.topic_classifier_baseline
        if not clf or not getattr(clf, "is_fitted", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Baseline TF-IDF topic classifier is not ready/trained.",
            )
        resolved_type = "tfidf"
        is_calibrated = getattr(clf, "is_calibrated", False)
    elif model_choice in ("transformer", "distilbert", "transformer_distilbert"):
        clf = getattr(services, "topic_classifier_transformer", None)
        if not clf or not getattr(clf, "is_fitted", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="DistilBERT topic classifier is not fitted/trained. Status: not_trained. Run training workflow first.",
            )
        resolved_type = "distilbert"
        is_calibrated = False
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model_type '{request.model_type}'. Supported choices: 'baseline' / 'tfidf', 'transformer' / 'distilbert'.",
        )

    top_k = clf.predict_top_k([text], k=3)[0]
    pred_cat, top_conf = top_k[0]

    return ClassificationResponse(
        predicted_category=pred_cat,
        confidence=round(float(top_conf), 4),
        top_k_probabilities=[{"category": c, "probability": p} for c, p in top_k],
        model_type=resolved_type,
        is_calibrated=is_calibrated,
    )


@router.post("/entities", response_model=EntityExtractionResponse)
@router.post("/ner/extract", response_model=EntityExtractionResponse)
def extract_entities(
    request: EntityExtractionRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text provided for NER extraction.")

    pipeline = services.named_entity_pipeline
    entities: List[ExtractedEntity] = pipeline.extract_entities(text)

    pipeline_type = "transformer_bert_base_ner" if pipeline.use_transformer and pipeline._hf_pipeline else "pattern_regex"

    mind_wikidata = None
    # If article_id provided and in dataset, keep MIND Wikidata entities explicitly distinct
    if request.article_id and services.dataset and request.article_id in services.dataset.articles:
        art = services.dataset.articles[request.article_id]
        mind_raw = []
        for ent_list in [getattr(art, "title_entities", []), getattr(art, "abstract_entities", [])]:
            if isinstance(ent_list, list):
                mind_raw.extend(ent_list)
        mind_wikidata = mind_raw
        # Also perform Wikidata resolution matching on predicted entities
        entities = pipeline.resolve_mind_entities(entities, mind_raw)

    serialized = [
        {
            "text": e.text,
            "label": e.label,
            "start": e.start,
            "end": e.end,
            "confidence": e.confidence,
            "wikidata_id": e.wikidata_id,
        }
        for e in entities
    ]
    labels_found = sorted(list({e.label for e in entities}))

    return EntityExtractionResponse(
        entities=serialized,
        total_extracted=len(serialized),
        categories_found=labels_found,
        pipeline_type=pipeline_type,
        mind_wikidata_entities=mind_wikidata,
    )


@router.post("/cross-encoder/rerank")
def rerank_cross_encoder(
    request: CrossEncoderRerankRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    candidates = []
    for aid in request.candidate_ids:
        art = services.dataset.articles.get(aid)
        if art:
            candidates.append({
                "article_id": aid,
                "title": art.title,
                "abstract": art.abstract,
                "category": art.category,
                "score": 0.5,
            })

    reranked, lat_ms = services.pairwise_cross_encoder.rerank_candidates(
        user_context=request.user_context,
        candidates=candidates,
        top_k=request.top_k,
    )

    return {
        "reranked_candidates": reranked,
        "latency_ms": lat_ms,
        "count": len(reranked),
        "model_status": "optional_pretrained_reranker",
    }

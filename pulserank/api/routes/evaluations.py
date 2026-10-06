"""Offline evaluation results and benchmark comparison routes."""

import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from pulserank.config import settings
from pulserank.api.schemas import EvaluationResultResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.ml.evaluation import OfflineEvaluator, save_evaluation_results

router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


@router.get("", response_model=List[EvaluationResultResponse])
def get_evaluation_results(
    mode: Optional[str] = Query(None, description="Filter by 'impression' or 'catalog_retrieval'"),
    services: ServiceContainer = Depends(get_service_container),
):
    """
    Retrieve offline evaluation benchmarks across all 5 models:
    - Popularity Baseline
    - Content-Based (Sentence-Transformers)
    - Collaborative Filtering (Implicit ALS)
    - Hybrid Candidate Fusion
    - Hybrid + LightGBM LambdaMART Ranker
    Metrics include AUC, MRR, nDCG@5, nDCG@10, Recall@K, and explicit Impression / Catalog Coverage.
    Includes dataset_source provenance ('synthetic_fixture' vs 'mind-small').
    """
    eval_file = settings.artifacts_dir / "evaluation_results.json"

    # If results exist on disk, return them directly
    if eval_file.exists():
        try:
            with open(eval_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            responses = [EvaluationResultResponse(**item) for item in data]
            if mode:
                responses = [r for r in responses if r.evaluation_mode == mode]
            return responses
        except Exception:
            pass

    # If not evaluated yet, run evaluation if validation behaviors exist
    if (
        services.dataset
        and services.dataset.dev_behaviors
        and services.popularity_model
        and services.content_model
        and services.collaborative_model
        and services.hybrid_generator
        and services.ranker_model
        and services.feature_extractor
    ):
        evaluator = OfflineEvaluator(services.dataset.dev_behaviors)
        dataset_src = getattr(services.dataset, "dataset_source", "mind-small")
        metrics_list = evaluator.evaluate_all(
            popularity_model=services.popularity_model,
            content_model=services.content_model,
            collaborative_model=services.collaborative_model,
            hybrid_generator=services.hybrid_generator,
            ranker_model=services.ranker_model,
            feature_extractor=services.feature_extractor,
            dataset_source=dataset_src,
        )
        save_evaluation_results(metrics_list, eval_file)
        responses = [
            EvaluationResultResponse(
                model_name=m.model_name,
                auc=m.auc,
                mrr=m.mrr,
                ndcg_at_5=m.ndcg_at_5,
                ndcg_at_10=m.ndcg_at_10,
                recall_at_k=m.recall_at_k,
                coverage=m.coverage,
                num_impressions=m.num_impressions,
                evaluated_at=m.evaluated_at,
                dataset_source=m.dataset_source,
                evaluation_mode=m.evaluation_mode,
                impression_coverage=m.impression_coverage,
                catalog_coverage=m.catalog_coverage,
                hit_rate_at_k=m.hit_rate_at_k,
            )
            for m in metrics_list
        ]
        if mode:
            responses = [r for r in responses if r.evaluation_mode == mode]
        return responses

    # If no evaluation is available and cannot run, return empty list (UI handles uninitialized state cleanly)
    return []

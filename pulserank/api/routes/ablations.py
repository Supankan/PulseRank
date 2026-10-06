"""9-Configuration Systematic Ablation Study Route."""

import json
import logging
from dataclasses import asdict
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, Query

from pulserank.config import settings
from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.data.provenance import compute_dataset_fingerprint
from pulserank.ml.evaluation import Evaluator
from pulserank.ml.ablation import AblationFramework

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ablations", tags=["Ablations"])


@router.get("")
def get_ablation_matrix(
    max_impressions: int = Query(80, ge=10, le=500),
    rerun: bool = Query(False, description="Force re-running the 9-stage benchmark instead of serving cached run"),
    services: ServiceContainer = Depends(get_service_container),
):
    """Run and return the 9-stage systematic ablation matrix."""
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    active_fingerprint = compute_dataset_fingerprint(services.dataset)
    active_source = getattr(services.dataset, "dataset_source", "mind-small")
    cached_path = settings.artifacts_dir / "ablation_results.json"

    # Serve cached measured results if compatible and rerun not requested
    if not rerun and cached_path.exists():
        try:
            with open(cached_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_fp = cached_data.get("dataset_fingerprint")
            cached_src = cached_data.get("dataset_source")
            if cached_fp == active_fingerprint and cached_src == active_source:
                cached_data["cached"] = True
                cached_data["provenance_status"] = "measured"
                return cached_data
            else:
                logger.info("Cached ablation results fingerprint mismatch with active dataset; rerunning...")
        except Exception as e:
            logger.warning(f"Failed to read cached ablation results ({e}); rerunning...")

    evaluator = Evaluator(services.dataset.dev_behaviors)
    framework = AblationFramework(
        evaluator=evaluator,
        diversity_engine=services.diversity_engine,
        popularity_model=services.popularity_model,
        content_model=services.content_model,
        collaborative_model=services.collaborative_model,
        hybrid_generator=services.hybrid_generator,
        ranker_model=services.ranker_model,
        feature_extractor=services.feature_extractor,
        dynamic_profile_manager=getattr(services, "dynamic_profile_manager", None),
        exploration_engine=getattr(services, "exploration_engine", None),
        catalog_article_ids=list(services.dataset.articles.keys()),
        dataset_source=active_source,
        dataset_fingerprint=active_fingerprint,
        train_behaviors=getattr(services.dataset, "train_behaviors", None),
    )

    results = framework.run_all_ablations(
        max_impressions=max_impressions,
        artifacts_dir=settings.artifacts_dir,
    )

    serialized = [asdict(r) for r in results]

    return {
        "ablations": serialized,
        "num_configurations": len(serialized),
        "cached": False,
        "provenance_status": "measured",
        "dataset_source": active_source,
        "dataset_fingerprint": active_fingerprint,
    }


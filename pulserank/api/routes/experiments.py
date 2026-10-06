import hashlib
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
import numpy as np

from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.ml.experimentation import ExperimentationEngine, SampleSizeResult

router = APIRouter(prefix="/experiments", tags=["Experimentation"])

SUPPORTED_POLICIES = [
    "popularity",
    "content",
    "collaborative",
    "pointwise_hybrid",
    "rrf",
    "ranker",
    "ranker_mmr",
]


class SampleSizeRequest(BaseModel):
    baseline_ctr: float = 0.04
    mde_relative: float = 0.05
    alpha: float = 0.05
    power: float = 0.80


class BootstrapTestRequest(BaseModel):
    model_a: str = "pointwise_hybrid"
    model_b: str = "ranker"
    num_bootstraps: int = 200


def resolve_scoring_policy(policy_name: str, services: ServiceContainer):
    """
    Resolve model string identifier into an executable scoring policy function:
    (user_id, candidate_ids, user_history) -> List[float].
    """
    p = policy_name.lower().strip()

    if p in ("popularity", "pop"):
        if not services.popularity_model:
            raise HTTPException(status_code=503, detail="Popularity model not ready.")
        return lambda uid, cids, hist=None: [services.popularity_model.get_score(a) for a in cids]

    elif p in ("content", "content_based"):
        if not services.content_model:
            raise HTTPException(status_code=503, detail="Content model not ready.")
        def score_content(uid, cids, hist=None):
            h = hist if hist is not None else services.feature_extractor.user_histories.get(uid, [])
            u_vec = services.content_model.build_user_vector(h) if h else None
            return [services.content_model.compute_similarity(u_vec, a) if u_vec is not None else 0.0 for a in cids]
        return score_content

    elif p in ("collaborative", "collab", "als"):
        if not services.collaborative_model:
            raise HTTPException(status_code=503, detail="Collaborative model not ready.")
        return lambda uid, cids, hist=None: [services.collaborative_model.get_score(uid, a) for a in cids]

    elif p in ("pointwise_hybrid", "hybrid"):
        return lambda uid, cids, hist=None: services.hybrid_generator.score_candidates_pointwise(uid, cids, user_history=hist)

    elif p in ("rrf", "rrf_hybrid"):
        def score_rrf(uid, cids, hist=None):
            if not cids:
                return []
            h = hist if hist is not None else services.feature_extractor.user_histories.get(uid, [])
            u_vec = services.content_model.build_user_vector(h) if (services.content_model and h) else None
            c_scores = [services.content_model.compute_similarity(u_vec, a) if (services.content_model and u_vec is not None) else 0.0 for a in cids]
            als_scores = [services.collaborative_model.get_score(uid, a) if services.collaborative_model else 0.0 for a in cids]
            pop_scores = [services.popularity_model.get_score(a) if services.popularity_model else 0.0 for a in cids]

            c_ranks = np.argsort(np.argsort(-np.array(c_scores))) + 1
            als_ranks = np.argsort(np.argsort(-np.array(als_scores))) + 1
            pop_ranks = np.argsort(np.argsort(-np.array(pop_scores))) + 1

            return [
                float(1.0 / (60 + c_ranks[i]) + 1.0 / (60 + als_ranks[i]) + 1.0 / (60 + pop_ranks[i]))
                for i in range(len(cids))
            ]
        return score_rrf

    elif p in ("ranker", "lambdamart"):
        if not services.ranker_model or not services.ranker_model.model:
            raise HTTPException(status_code=503, detail="LambdaMART ranker model not ready.")
        def score_ranker(uid, cids, hist=None):
            feats = [services.feature_extractor.extract_features(uid, a, user_history=hist) for a in cids]
            return [float(pred) for pred in services.ranker_model.model.predict(np.vstack(feats))]
        return score_ranker

    elif p in ("ranker_mmr", "mmr"):
        if not services.ranker_model or not services.ranker_model.model:
            raise HTTPException(status_code=503, detail="LambdaMART ranker model not ready.")
        def score_mmr(uid, cids, hist=None):
            feats = [services.feature_extractor.extract_features(uid, a, user_history=hist) for a in cids]
            base_scores = [float(pred) for pred in services.ranker_model.model.predict(np.vstack(feats))]
            cands = [
                {
                    "article_id": aid,
                    "score": s,
                    "category": getattr(services.dataset.articles.get(aid), "category", "unknown") if services.dataset else "unknown",
                }
                for aid, s in zip(cids, base_scores)
            ]
            reranked = services.diversity_engine.rerank_mmr(cands, lambda_param=0.7, top_k=len(cids))
            score_map = {item["article_id"]: item.get("mmr_score", item.get("score", 0.0)) for item in reranked}
            return [score_map.get(a, 0.0) for a in cids]
        return score_mmr

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown policy '{policy_name}'. Supported policies: {', '.join(SUPPORTED_POLICIES)}",
        )


@router.post("/sample-size", response_model=Dict[str, Any])
def compute_sample_size(request: SampleSizeRequest):
    res: SampleSizeResult = ExperimentationEngine.calculate_ab_sample_size(
        baseline_rate=request.baseline_ctr,
        mde_relative=request.mde_relative,
        alpha=request.alpha,
        power=request.power,
    )
    return {
        "baseline_ctr": res.baseline_ctr,
        "mde_relative": res.mde_relative,
        "mde_absolute": res.mde_absolute,
        "alpha": res.alpha,
        "power": res.power,
        "required_sample_size_per_variant": res.required_sample_size_per_variant,
        "total_sample_size": res.total_sample_size,
    }


@router.post("/bootstrap-test")
def run_bootstrap_test(
    request: BootstrapTestRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    score_func_a = resolve_scoring_policy(request.model_a, services)
    score_func_b = resolve_scoring_policy(request.model_b, services)

    engine = ExperimentationEngine()
    behaviors = services.dataset.dev_behaviors[:150]

    deltas = engine.run_paired_bootstrap(
        behaviors=behaviors,
        score_func_a=score_func_a,
        score_func_b=score_func_b,
        n_bootstraps=request.num_bootstraps,
    )

    serialized = {
        k: {
            "metric_name": v.metric_name,
            "model_a_mean": v.model_a_mean,
            "model_b_mean": v.model_b_mean,
            "delta_mean": v.delta_mean,
            "ci_lower_95": v.ci_lower_95,
            "ci_upper_95": v.ci_upper_95,
            "p_value": v.p_value,
            "is_statistically_significant": v.is_statistically_significant,
        }
        for k, v in deltas.items()
    }

    return {
        "model_a": request.model_a,
        "model_b": request.model_b,
        "num_bootstraps": request.num_bootstraps,
        "num_impressions_evaluated": len(behaviors),
        "metrics": serialized,
    }


@router.get("/runs")
def list_experiment_runs(services: ServiceContainer = Depends(get_service_container)):
    runs = services.mlflow_tracker.list_runs()
    return {"runs": runs, "count": len(runs)}


class EvaluateCandidatesRequest(BaseModel):
    experiment_id: str
    user_id: str
    candidate_article_ids: List[str]
    model_a: str = "popularity"
    model_b: str = "ranker"
    limit: int = 10


@router.post("/evaluate-candidates")
def evaluate_candidates(
    request: EvaluateCandidatesRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    """Dynamically evaluate and rank candidates under an A/B test policy variant."""
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    score_func_a = resolve_scoring_policy(request.model_a, services)
    score_func_b = resolve_scoring_policy(request.model_b, services)

    # Deterministic variant hash based on user_id and experiment_id
    h = int(hashlib.md5(f"{request.user_id}:{request.experiment_id}".encode()).hexdigest(), 16)
    assigned_variant = "A" if (h % 100) < 50 else "B"
    active_policy = request.model_a if assigned_variant == "A" else request.model_b
    active_func = score_func_a if assigned_variant == "A" else score_func_b

    scores = active_func(request.user_id, request.candidate_article_ids)
    ranked = sorted(zip(request.candidate_article_ids, scores), key=lambda x: x[1], reverse=True)
    slate = [{"article_id": aid, "score": round(float(s), 4)} for aid, s in ranked[:request.limit]]

    return {
        "experiment_id": request.experiment_id,
        "user_id": request.user_id,
        "assigned_variant": assigned_variant,
        "active_policy": active_policy,
        "slate": slate,
    }


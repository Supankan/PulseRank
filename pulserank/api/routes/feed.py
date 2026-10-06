"""Personalized feed generation route supporting multiple recommendation models."""

from typing import List, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Query

from pulserank.api.schemas import FeedResponse, FeedItemResponse
from pulserank.api.dependencies import ServiceContainer, get_service_container
from pulserank.ml.hybrid import CandidateItem
from pulserank.ml.explanations import ExplanationGenerator
from pulserank.ml.exploration import build_exploration_pool

router = APIRouter(prefix="/feed", tags=["Feed"])


@router.get("/{user_id}", response_model=FeedResponse)
def get_personalized_feed(
    user_id: str,
    model: str = Query("ranker", description="Model: 'ranker', 'hybrid', 'content', 'collaborative', or 'popularity'"),
    limit: int = Query(20, ge=1, le=100),
    diversity_lambda: Optional[float] = Query(None, ge=0.0, le=1.0, description="MMR lambda trade-off (1.0 = relevance only, 0.0 = diversity only)"),
    explore_epsilon: Optional[float] = Query(None, ge=0.0, le=1.0, description="Epsilon-greedy exploration probability"),
    services: ServiceContainer = Depends(get_service_container),
):
    import time
    t_start = time.perf_counter()
    """
    Generate personalized feed for user_id using the selected recommendation engine:
    - ranker: Hybrid candidate fusion + LightGBM LambdaMART reranking
    - hybrid: Multi-source candidate fusion (content + collaborative + popularity)
    - content: Sentence-Transformers semantic retrieval
    - collaborative: Implicit ALS matrix factorization
    - popularity: Recency-decayed engagement baseline
    """
    if not services.dataset:
        raise HTTPException(status_code=503, detail="Dataset not ready.")

    history = services.dataset.user_histories.get(user_id, [])
    exclude = set(history)
    articles = services.dataset.articles

    feed_items: List[FeedItemResponse] = []

    if model == "popularity":
        if not services.popularity_model:
            raise HTTPException(status_code=503, detail="Popularity model not ready.")
        recs = services.popularity_model.recommend(k=limit, exclude_article_ids=exclude)
        for rank_idx, (aid, score) in enumerate(recs, 1):
            art = articles.get(aid)
            if not art:
                continue
            rec_id = f"rec_{uuid4().hex[:12]}"
            cand = CandidateItem(
                article_id=aid,
                user_id=user_id,
                sources=["popularity"],
                source_scores={"popularity": score},
                popularity_score=score,
            )
            expl = ExplanationGenerator.explain(
                article_id=aid,
                rank=rank_idx,
                score=score,
                candidate_item=cand,
                feature_values={"popularity_score": score},
                article_category=art.category,
            )
            services.explanation_cache[rec_id] = {
                "recommendation_id": rec_id,
                "user_id": user_id,
                "explanation": expl,
            }
            feed_items.append(
                FeedItemResponse(
                    recommendation_id=rec_id,
                    article_id=aid,
                    title=art.title,
                    abstract=art.abstract,
                    category=art.category,
                    subcategory=art.subcategory,
                    url=art.url,
                    rank=rank_idx,
                    score=round(score, 4),
                    sources=["popularity"],
                    primary_reason=expl.primary_reason,
                )
            )

    elif model == "content":
        if not services.content_model:
            raise HTTPException(status_code=503, detail="Content model not ready.")
        recs = services.content_model.recommend(
            history_article_ids=history,
            k=limit,
            exclude_article_ids=exclude,
        )
        for rank_idx, (aid, score) in enumerate(recs, 1):
            art = articles.get(aid)
            if not art:
                continue
            rec_id = f"rec_{uuid4().hex[:12]}"
            cand = CandidateItem(
                article_id=aid,
                user_id=user_id,
                sources=["content"],
                source_scores={"content": score},
                content_score=score,
            )
            expl = ExplanationGenerator.explain(
                article_id=aid,
                rank=rank_idx,
                score=score,
                candidate_item=cand,
                feature_values={"content_similarity": score},
                article_category=art.category,
            )
            services.explanation_cache[rec_id] = {
                "recommendation_id": rec_id,
                "user_id": user_id,
                "explanation": expl,
            }
            feed_items.append(
                FeedItemResponse(
                    recommendation_id=rec_id,
                    article_id=aid,
                    title=art.title,
                    abstract=art.abstract,
                    category=art.category,
                    subcategory=art.subcategory,
                    url=art.url,
                    rank=rank_idx,
                    score=round(score, 4),
                    sources=["content"],
                    primary_reason=expl.primary_reason,
                )
            )

    elif model == "collaborative":
        if not services.collaborative_model:
            raise HTTPException(status_code=503, detail="Collaborative model not ready.")
        recs = services.collaborative_model.recommend(
            user_id=user_id,
            k=limit,
            exclude_article_ids=exclude,
        )
        # Cold start fallback if collaborative has no candidates
        if not recs and services.popularity_model:
            recs = services.popularity_model.recommend(k=limit, exclude_article_ids=exclude)

        for rank_idx, (aid, score) in enumerate(recs, 1):
            art = articles.get(aid)
            if not art:
                continue
            rec_id = f"rec_{uuid4().hex[:12]}"
            cand = CandidateItem(
                article_id=aid,
                user_id=user_id,
                sources=["collaborative"],
                source_scores={"collaborative": score},
                collaborative_score=score,
            )
            expl = ExplanationGenerator.explain(
                article_id=aid,
                rank=rank_idx,
                score=score,
                candidate_item=cand,
                feature_values={"collaborative_score": score},
                article_category=art.category,
            )
            services.explanation_cache[rec_id] = {
                "recommendation_id": rec_id,
                "user_id": user_id,
                "explanation": expl,
            }
            feed_items.append(
                FeedItemResponse(
                    recommendation_id=rec_id,
                    article_id=aid,
                    title=art.title,
                    abstract=art.abstract,
                    category=art.category,
                    subcategory=art.subcategory,
                    url=art.url,
                    rank=rank_idx,
                    score=round(score, 4),
                    sources=["collaborative"],
                    primary_reason=expl.primary_reason,
                )
            )

    elif model == "hybrid":
        if not services.hybrid_generator:
            raise HTTPException(status_code=503, detail="Hybrid generator not ready.")
        cands = services.hybrid_generator.generate_candidates(
            user_id=user_id,
            user_history=history,
            exclude_article_ids=exclude,
            article_metadata=articles,
        )[:limit]

        for rank_idx, cand in enumerate(cands, 1):
            art = articles.get(cand.article_id)
            if not art:
                continue
            rec_id = f"rec_{uuid4().hex[:12]}"
            feat_values = {
                "content_similarity": cand.content_score,
                "collaborative_score": cand.collaborative_score,
                "popularity_score": cand.popularity_score,
            }
            expl = ExplanationGenerator.explain(
                article_id=cand.article_id,
                rank=rank_idx,
                score=cand.hybrid_score,
                candidate_item=cand,
                feature_values=feat_values,
                article_category=art.category,
            )
            services.explanation_cache[rec_id] = {
                "recommendation_id": rec_id,
                "user_id": user_id,
                "explanation": expl,
            }
            feed_items.append(
                FeedItemResponse(
                    recommendation_id=rec_id,
                    article_id=cand.article_id,
                    title=art.title,
                    abstract=art.abstract,
                    category=art.category,
                    subcategory=art.subcategory,
                    url=art.url,
                    rank=rank_idx,
                    score=round(cand.hybrid_score, 4),
                    sources=cand.sources,
                    primary_reason=expl.primary_reason,
                )
            )

    else:  # Default: "ranker" (Hybrid + LightGBM LambdaMART)
        if not services.hybrid_generator or not services.feature_extractor:
            raise HTTPException(status_code=503, detail="Ranking service not ready.")

        candidates = services.hybrid_generator.generate_candidates(
            user_id=user_id,
            user_history=history,
            exclude_article_ids=exclude,
            article_metadata=articles,
        )

        if services.ranker_model:
            ranked_pairs = services.ranker_model.rerank_candidates(
                candidates=candidates,
                feature_extractor=services.feature_extractor,
                user_history=history,
            )[:limit]
        else:
            ranked_pairs = [(c, c.hybrid_score) for c in candidates[:limit]]

        for rank_idx, (cand, final_score) in enumerate(ranked_pairs, 1):
            art = articles.get(cand.article_id)
            if not art:
                continue

            raw_feats = services.feature_extractor.extract_features(
                user_id=user_id,
                article_id=cand.article_id,
                candidate_item=cand,
                user_history=history,
            )
            from pulserank.ml.ranker import FEATURE_NAMES
            feat_values = {name: float(val) for name, val in zip(FEATURE_NAMES, raw_feats)}

            rec_id = f"rec_{uuid4().hex[:12]}"
            expl = ExplanationGenerator.explain(
                article_id=cand.article_id,
                rank=rank_idx,
                score=final_score,
                candidate_item=cand,
                feature_values=feat_values,
                article_category=art.category,
            )
            services.explanation_cache[rec_id] = {
                "recommendation_id": rec_id,
                "user_id": user_id,
                "explanation": expl,
            }
            feed_items.append(
                FeedItemResponse(
                    recommendation_id=rec_id,
                    article_id=cand.article_id,
                    title=art.title,
                    abstract=art.abstract,
                    category=art.category,
                    subcategory=art.subcategory,
                    url=art.url,
                    rank=rank_idx,
                    score=round(final_score, 4),
                    sources=cand.sources,
                    primary_reason=expl.primary_reason,
                )
            )

    # Apply optional MMR diversity reranking
    if diversity_lambda is not None and len(feed_items) > 2 and services.diversity_engine:
        dict_items = [
            {
                "article_id": it.article_id,
                "score": it.score,
                "final_score": it.score,
                "category": it.category,
                "original_item": it,
            }
            for it in feed_items
        ]
        reranked_dict = services.diversity_engine.rerank_mmr(
            candidates=dict_items,
            lambda_param=diversity_lambda,
            top_k=limit,
        )
        feed_items = [d["original_item"] for d in reranked_dict]
        for new_rank, item in enumerate(feed_items, 1):
            item.rank = new_rank

    # Apply optional exploration injection
    if explore_epsilon is not None and explore_epsilon > 0.0 and services.exploration_engine:
        dict_items = [
            {
                "article_id": it.article_id,
                "score": it.score,
                "final_score": it.score,
                "category": it.category,
                "original_item": it,
            }
            for it in feed_items
        ]
        # Build candidate pool for exploration from eligible articles not in history or feed
        already_served = {it.article_id for it in feed_items}
        pool_candidates = build_exploration_pool(
            catalog_article_ids=list(articles.keys()),
            articles=articles,
            user_history=history,
            content_model=services.content_model,
            popularity_model=services.popularity_model,
            exposure_counts=services.exploration_engine.item_exposure_counts,
            exclude_ids=set(exclude) | already_served,
        )

        explore_pool = []
        for cand in pool_candidates:
            aid = cand["article_id"]
            art = articles[aid]
            rec_id = f"rec_{uuid4().hex[:12]}"
            c_copy = dict(cand)
            c_copy["original_item"] = FeedItemResponse(
                recommendation_id=rec_id,
                article_id=aid,
                title=art.title,
                abstract=art.abstract,
                category=art.category,
                subcategory=art.subcategory,
                url=art.url,
                rank=999,
                score=c_copy["score"],
                sources=["exploration"],
                primary_reason="Surfaced for topic exploration and serendipity",
            )
            explore_pool.append(c_copy)

        explored_dict = services.exploration_engine.explore_slate(
            ranked_candidates=dict_items,
            exploration_pool=explore_pool,
            top_k=limit,
            epsilon=explore_epsilon,
        )
        feed_items = [d["original_item"] for d in explored_dict]
        for new_rank, item in enumerate(feed_items, 1):
            item.rank = new_rank

    # Record served item exposures to track true catalog exposure
    if services.exploration_engine and feed_items:
        services.exploration_engine.record_exposures([it.article_id for it in feed_items])

    # Measure actual elapsed latency
    real_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    # Log inference event for monitoring with measured latency
    if services.inference_monitor:
        cat_served = [it.category for it in feed_items]
        scores_served = [it.score for it in feed_items]
        ch_counts = {}
        for it in feed_items:
            for s in it.sources:
                ch_counts[s] = ch_counts.get(s, 0) + 1
        services.inference_monitor.log_inference_event(
            endpoint=f"/feed/{user_id}",
            latency_ms=real_latency_ms,
            user_id=user_id,
            channels=ch_counts,
            categories_served=cat_served,
            prediction_scores=scores_served,
        )

    return FeedResponse(
        user_id=user_id,
        model_used=model,
        total_returned=len(feed_items),
        items=feed_items,
    )


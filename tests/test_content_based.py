"""Tests for content-based semantic recommender."""

import numpy as np
from pulserank.ml.content_based import ContentBasedRecommender


def test_content_embedding_generation(processed_dataset):
    model = ContentBasedRecommender(embedding_dim=64)
    model.build_index(processed_dataset.articles)

    assert model.embeddings is not None
    assert model.embeddings.shape[0] == len(processed_dataset.articles)
    # Check L2 normalization: norm should be close to 1.0
    norms = np.linalg.norm(model.embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_content_recommend_and_exclude(processed_dataset):
    model = ContentBasedRecommender(embedding_dim=64)
    model.build_index(processed_dataset.articles)

    user_hist = ["N10001", "N10002"]
    recs = model.recommend(history_article_ids=user_hist, k=5)

    rec_ids = [r[0] for r in recs]
    # Consumed articles must be excluded
    for aid in user_hist:
        assert aid not in rec_ids

    # Scores should be in [0.0, 1.0]
    for _, score in recs:
        assert 0.0 <= score <= 1.0


def test_content_save_load(processed_dataset, temp_dir):
    model = ContentBasedRecommender(embedding_dim=64)
    model.build_index(processed_dataset.articles)

    out_dir = temp_dir / "content_test"
    model.save(out_dir)

    loaded = ContentBasedRecommender.load(out_dir)
    assert len(loaded.article_ids) == len(model.article_ids)
    assert loaded.embeddings.shape == model.embeddings.shape

    # Verify reload preserves exact FAISS retrieval alignment
    recs_orig = model.recommend(history_article_ids=["N10001"], k=3)
    recs_loaded = loaded.recommend(history_article_ids=["N10001"], k=3)
    assert [r[0] for r in recs_orig] == [r[0] for r in recs_loaded]
    np.testing.assert_allclose([r[1] for r in recs_orig], [r[1] for r in recs_loaded], atol=1e-5)

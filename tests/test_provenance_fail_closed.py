"""Tests verifying fail-closed artifact provenance validation."""

import pytest
from pathlib import Path
from pulserank.data.provenance import (
    ArtifactManifest,
    IncompatibleArtifactError,
    validate_artifact_compatibility,
    enforce_artifact_compatibility,
    compute_dataset_fingerprint,
)
from pulserank.ml.popularity import PopularityRecommender


def test_manifest_validation_rejects_dataset_mismatch():
    """Verify that an artifact trained on synthetic_fixture is rejected when mind-small is expected."""
    manifest = ArtifactManifest(
        model_name="popularity",
        dataset_source="synthetic_fixture",
        dataset_fingerprint="abc12345",
    )

    compat, reason = validate_artifact_compatibility(
        manifest,
        expected_dataset_source="mind-small",
        expected_fingerprint="abc12345",
    )
    assert compat is False
    assert "Dataset source mismatch" in reason

    with pytest.raises(IncompatibleArtifactError) as exc_info:
        enforce_artifact_compatibility(
            manifest,
            expected_dataset_source="mind-small",
            expected_fingerprint="abc12345",
            artifact_name="PopularityModel",
        )
    assert "Dataset source mismatch" in str(exc_info.value)


def test_manifest_validation_rejects_fingerprint_mismatch():
    """Verify that an artifact with stale fingerprint is rejected."""
    manifest = ArtifactManifest(
        model_name="ranker",
        dataset_source="mind-small",
        dataset_fingerprint="old_fingerprint_123",
    )

    with pytest.raises(IncompatibleArtifactError) as exc_info:
        enforce_artifact_compatibility(
            manifest,
            expected_dataset_source="mind-small",
            expected_fingerprint="new_fingerprint_456",
            strict_fingerprint=True,
        )
    assert "Dataset fingerprint mismatch" in str(exc_info.value)


def test_manifest_validation_rejects_missing_manifest():
    """Verify that a missing manifest fails closed."""
    compat, reason = validate_artifact_compatibility(
        manifest=None,
        expected_dataset_source="mind-small",
    )
    assert compat is False
    assert "missing or corrupt" in reason

    with pytest.raises(IncompatibleArtifactError):
        enforce_artifact_compatibility(
            manifest=None,
            expected_dataset_source="mind-small",
        )


def test_popularity_recommender_load_fails_closed(tmp_path: Path):
    """Verify PopularityRecommender.load raises IncompatibleArtifactError on source mismatch."""
    pop = PopularityRecommender()
    pop.article_scores = {"A1": 1.0}
    pop.ranked_articles = [("A1", 1.0)]

    out_file = tmp_path / "popularity.pkl"
    pop.save(out_file, dataset_source="synthetic_fixture", dataset_fingerprint="synth_fp_123")

    # Loading with expected_source="mind-small" must fail closed
    with pytest.raises(IncompatibleArtifactError):
        PopularityRecommender.load(
            out_file,
            expected_source="mind-small",
            expected_fingerprint="synth_fp_123",
            fail_closed=True,
        )

    # Loading with exact expected source and fingerprint succeeds
    loaded = PopularityRecommender.load(
        out_file,
        expected_source="synthetic_fixture",
        expected_fingerprint="synth_fp_123",
        fail_closed=True,
    )
    assert loaded.get_score("A1") == 1.0


def test_dataset_fingerprint_changes_on_single_click_label_change():
    """Verify that a single altered click label in train_behaviors produces a distinct SHA-256 fingerprint."""
    from pulserank.data.mind_parser import BehaviorRecord, ImpressionItem, NewsArticleRecord
    from pulserank.data.preprocessor import ProcessedDataset

    articles = {
        "N1": NewsArticleRecord(news_id="N1", category="news", subcategory="world", title="Title 1", abstract="Abstract 1"),
        "N2": NewsArticleRecord(news_id="N2", category="sports", subcategory="football", title="Title 2", abstract="Abstract 2"),
    }

    beh1 = [
        BehaviorRecord(
            impression_id="1",
            user_id="U1",
            time=None,
            history=["N1"],
            impressions=[ImpressionItem(news_id="N2", clicked=0)],
        )
    ]
    ds1 = ProcessedDataset(
        articles=articles,
        train_behaviors=beh1,
        dev_behaviors=[],
        user_histories={"U1": ["N1"]},
        dataset_source="synthetic_fixture",
    )
    fp1 = compute_dataset_fingerprint(ds1)

    # Identical dataset except clicked: 0 -> 1
    beh2 = [
        BehaviorRecord(
            impression_id="1",
            user_id="U1",
            time=None,
            history=["N1"],
            impressions=[ImpressionItem(news_id="N2", clicked=1)],
        )
    ]
    ds2 = ProcessedDataset(
        articles=articles,
        train_behaviors=beh2,
        dev_behaviors=[],
        user_histories={"U1": ["N1"]},
        dataset_source="synthetic_fixture",
    )
    fp2 = compute_dataset_fingerprint(ds2)

    assert fp1 != fp2, "Fingerprint failed to differentiate datasets with different click labels!"


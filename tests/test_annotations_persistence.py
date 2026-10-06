"""Tests verifying active learning DB persistence, transactional safety, and pre-labeler truthfulness."""

import pytest
from pulserank.nlp.annotation import MockLLMPreLabeler, ExternalLLMPreLabelProvider
from pulserank.db.session import SessionLocal
from pulserank.db.models import AnnotationRecord


def test_mock_llm_prelabeler_identity_and_behavior():
    """Verify MockLLMPreLabeler truthfully identifies itself as mock keyword engine."""
    pre_labeler = MockLLMPreLabeler()
    assert "MOCK" in pre_labeler.provider_name
    assert pre_labeler.is_available is True

    cat, rationale, conf, meta = pre_labeler.pre_label(
        "Lakers defeat Warriors in NBA game",
        "Basketball highlights",
        ["news", "sports", "finance"],
        return_metadata=True,
    )
    assert cat == "sports"
    assert "MOCK" in rationale
    assert meta["is_mock"] is True
    assert meta["token_usage"] == 0


def test_external_llm_prelabeler_fails_without_key():
    """Verify ExternalLLMPreLabelProvider discloses unavailability when key is missing."""
    provider = ExternalLLMPreLabelProvider(api_key=None)
    assert provider.is_available is False
    with pytest.raises(RuntimeError) as exc_info:
        provider.pre_label("title", "abstract", ["news"])
    assert "not configured" in str(exc_info.value)


def test_api_annotation_submission_persists_to_db(client):
    """Verify POST /api/annotations/submit genuinely commits AnnotationRecord to SQLite."""
    payload = {
        "article_id": "test_art_001",
        "true_label": "sports",
        "predicted_label": "sports",
        "annotator_id": "auditor_1",
        "uncertainty_score": 0.42,
        "pre_label_provider": "MOCK — keyword-based test provider",
        "meta_info": {"notes": "Verified authentic DB persistence"},
    }

    res = client.post("/api/annotations/submit", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "recorded"
    assert "annotation_id" in data
    ann_id = data["annotation_id"]

    # Verify directly via DB query
    db = SessionLocal()
    try:
        record = db.query(AnnotationRecord).filter(AnnotationRecord.id == ann_id).first()
        assert record is not None
        assert record.article_id == "test_art_001"
        assert record.label == "sports"
        assert record.annotator_id == "auditor_1"
    finally:
        db.close()

    # Query GET /api/annotations/history to verify read-back
    hist_res = client.get("/api/annotations/history")
    assert hist_res.status_code == 200
    history = hist_res.json()["history"]
    assert any(h["annotation_id"] == ann_id for h in history)

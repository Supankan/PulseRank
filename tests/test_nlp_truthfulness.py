"""Tests verifying NLP classifier calibration, truthfulness, and failure states."""

import pytest
import numpy as np
from pulserank.nlp.classifier_baseline import TFIDFTopicClassifier


def test_tfidf_classifier_calibration_and_metrics():
    """Verify that TFIDFTopicClassifier uses genuine Platt calibration and reports real log-loss and Brier scores."""
    texts = [
        "Lakers defeat Warriors in thrilling NBA overtime basketball thriller",
        "LeBron James scores 35 points in playoffs victory",
        "Apple announces new iPhone with high performance M3 chip technology",
        "Google deep learning AI model breaks performance benchmarks in science",
        "Federal reserve raises interest rates to combat inflation in stock markets",
        "Wall Street stocks drop as banking sector faces credit uncertainty",
        "Hollywood movie star wins best actor award at international film festival",
        "New cinema trailer reveals superhero blockbuster coming this summer",
    ] * 5

    labels = ["sports", "sports", "tech", "tech", "finance", "finance", "entertainment", "entertainment"] * 5

    clf = TFIDFTopicClassifier(max_features=200, calibrate=True)
    clf.fit(texts, labels)

    assert clf.is_fitted is True
    assert clf.is_calibrated is True
    assert len(clf.classes_) == 4

    # Test predict_proba sums to 1.0
    test_texts = ["NBA basketball star player", "Silicon Valley tech software startup"]
    probs = clf.predict_proba(test_texts)
    assert probs.shape == (2, 4)
    for p in probs:
        assert abs(float(np.sum(p)) - 1.0) < 1e-4

    # Test real evaluation metrics
    metrics = clf.evaluate(test_texts, ["sports", "tech"])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "log_loss" in metrics
    assert "brier_score" in metrics
    assert isinstance(metrics["log_loss"], float)
    assert isinstance(metrics["brier_score"], float)
    assert metrics["brier_score"] >= 0.0


def test_api_distilbert_unfitted_returns_503(client):
    """Verify that requesting DistilBERT classification when model is unfitted returns HTTP 503 truthfully."""
    res = client.post(
        "/api/nlp/classify",
        json={
            "title": "Stock market surges after inflation report",
            "abstract": "Investors optimistic about economic growth and rate cuts.",
            "model_type": "transformer_distilbert",
        },
    )
    # Since transformer DistilBERT is not fitted in test environment, it must fail truthfully with 503
    assert res.status_code == 503
    assert "DistilBERT topic classifier is not" in res.json()["detail"]


def test_api_ner_distinguishes_regex_from_transformer(client):
    """Verify that NER extraction clearly labels pattern_regex vs transformer extraction."""
    res = client.post(
        "/api/nlp/entities",
        json={
            "text": "Microsoft CEO Satya Nadella announced new investments in OpenAI in Seattle.",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "pipeline_type" in data
    assert data["pipeline_type"] in ["pattern_regex", "transformer_bert_base_ner"]
    assert isinstance(data["entities"], list)


def test_deterministic_train_test_split_articles():
    """Verify that train_test_split_articles produces deterministic 80/20 splits without leakage."""
    from pulserank.nlp.classifier_baseline import train_test_split_articles

    class SimpleArticle:
        def __init__(self, aid):
            self.news_id = aid

    arts = [SimpleArticle(f"N{i}") for i in range(100)]
    train_1, test_1, mode_1 = train_test_split_articles(arts, test_size=0.2, random_state=42)
    train_2, test_2, mode_2 = train_test_split_articles(arts, test_size=0.2, random_state=42)

    assert mode_1 == "deterministic_held_out_split"
    assert mode_2 == "deterministic_held_out_split"
    assert len(train_1) == 80
    assert len(test_1) == 20
    assert [a.news_id for a in train_1] == [a.news_id for a in train_2]
    assert [a.news_id for a in test_1] == [a.news_id for a in test_2]

    # Zero leakage between train and test
    train_ids = {a.news_id for a in train_1}
    test_ids = {a.news_id for a in test_1}
    assert len(train_ids.intersection(test_ids)) == 0

    # Small fixture semantics (< 30 articles)
    small_arts = arts[:15]
    train_s, test_s, mode_s = train_test_split_articles(small_arts)
    assert mode_s == "demo_resubstitution_not_held_out"
    assert len(train_s) == 15
    assert len(test_s) == 15
    assert [a.news_id for a in train_s] == [a.news_id for a in test_s]


def test_tfidf_classifier_evaluation_metadata_persistence():
    """Verify that TFIDFTopicClassifier records evaluation_mode and sample counts in evaluation and manifest."""
    texts = [
        "Lakers defeat Warriors in thrilling basketball game",
        "Apple launches new iPhone with fast processor",
    ] * 10
    labels = ["sports", "tech"] * 10
    clf = TFIDFTopicClassifier(max_features=100, calibrate=False)
    clf.fit(texts, labels)

    eval_out = clf.evaluate(
        texts[:4],
        labels[:4],
        evaluation_mode="demo_resubstitution_not_held_out",
        train_sample_count=len(texts),
        test_sample_count=4,
        split_seed=42,
    )
    assert eval_out["evaluation_mode"] == "demo_resubstitution_not_held_out"
    assert eval_out["train_sample_count"] == 20
    assert eval_out["test_sample_count"] == 4
    assert eval_out["split_seed"] == 42
    assert clf.last_evaluation == eval_out



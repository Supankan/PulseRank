"""Unit and integration tests for NLP Intelligence components."""

import pytest
import numpy as np
from pulserank.nlp.classifier_baseline import TFIDFTopicClassifier
from pulserank.nlp.ner import NamedEntityPipeline, ExtractedEntity
from pulserank.nlp.semantic_matching import ContrastiveBiEncoder, NewsTriplet
from pulserank.nlp.cross_encoder import PairwiseCrossEncoder
from pulserank.nlp.annotation import ActiveLearningQueue, MockLLMPreLabeler, compute_inter_annotator_agreement


def test_tfidf_topic_classifier():
    clf = TFIDFTopicClassifier(max_features=200)
    texts = [
        "Government passes new budget and inflation legislation in senate",
        "President signs federal bill into law following congressional vote",
        "Lakers win NBA championship game with spectacular fourth quarter",
        "Quarterback throws game winning touchdown pass in NFL playoffs",
        "Tech giant announces new artificial intelligence chip architecture",
        "Software startup raises Series A venture capital funding for neural networks",
    ]
    labels = ["news", "news", "sports", "sports", "technology", "technology"]

    clf.fit(texts, labels)
    assert clf.is_fitted
    assert len(clf.classes_) == 3

    preds = clf.predict(["Senate votes on tax reform bill"])
    assert preds[0] in clf.classes_

    probs = clf.predict_proba(["Quarterback scores touchdown"])
    assert probs.shape == (1, 3)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-3)

    metrics = clf.evaluate(texts, labels)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] >= 0.80


def test_ner_pipeline_extraction_and_resolution():
    pipeline = NamedEntityPipeline(use_transformer=False)
    sample_text = "Apple CEO Tim Cook announced new iPhone manufacturing in Texas and India."

    entities = pipeline.extract_entities(sample_text)
    assert len(entities) > 0
    labels = {e.label for e in entities}
    assert any(l in ("ORG", "LOC", "PER", "MISC") for l in labels)

    # Test MIND Wikidata resolution
    mind_raw = [
        {"Label": "Apple", "WikidataId": "Q312"},
        {"Label": "Texas", "WikidataId": "Q1439"},
    ]
    resolved = pipeline.resolve_mind_entities(entities, mind_raw)
    apple_ent = next((e for e in resolved if "apple" in e.text.lower()), None)
    if apple_ent:
        assert apple_ent.wikidata_id == "Q312"


def test_contrastive_bi_encoder_triplets():
    encoder = ContrastiveBiEncoder(base_model_name="all-MiniLM-L6-v2")
    triplets = [
        NewsTriplet(
            anchor="President signs economy relief bill into law",
            positive="Federal congress passes stimulus package for small businesses",
            negative="Star basketball player scores 40 points in playoff victory",
        ),
        NewsTriplet(
            anchor="Coach leads soccer team to champions league final",
            positive="Striker nets hat trick in European tournament win",
            negative="Tech stock drops following quarterly earnings report",
        ),
    ]

    metrics = encoder.evaluate_triplets(triplets)
    assert "triplet_accuracy" in metrics
    assert "mean_positive_similarity" in metrics
    assert "mean_negative_similarity" in metrics
    # Cosine similarity to positive should be higher than negative
    assert metrics["separation_margin_gap"] > 0.0


def test_cross_encoder_reranker():
    reranker = PairwiseCrossEncoder()
    candidates = [
        {"article_id": "A1", "title": "Soccer World Cup Final Announced", "abstract": "Teams prepare for tournament", "score": 0.4},
        {"article_id": "A2", "title": "Artificial Intelligence Breakthrough", "abstract": "Researchers unveil LLM", "score": 0.9},
        {"article_id": "A3", "title": "Champions League Highlights", "abstract": "Goals scored in semifinal", "score": 0.5},
    ]

    reranked, latency = reranker.rerank_candidates("World cup soccer tournament news", candidates, top_k=2)
    assert len(reranked) == 2
    assert latency >= 0.0
    assert "cross_encoder_score" in reranked[0]
    assert "final_score" in reranked[0]


def test_active_learning_queue_and_cohens_kappa():
    queue_mgr = ActiveLearningQueue()
    probs = np.array([
        [0.34, 0.33, 0.33],  # Highly uncertain (entropy high, margin small)
        [0.90, 0.05, 0.05],  # Highly certain
        [0.51, 0.49, 0.00],  # Small margin
    ])
    articles = [
        {"article_id": "1", "title": "Ambiguous headline", "abstract": "text"},
        {"article_id": "2", "title": "Clear Sports headline", "abstract": "text"},
        {"article_id": "3", "title": "Borderline Politics vs Economy", "abstract": "text"},
    ]
    classes = ["news", "sports", "tech"]

    items = queue_mgr.rank_for_annotation(articles, probs, classes, top_n=2, strategy="entropy")
    assert len(items) == 2
    # The most uncertain item (article 1) should be ranked highest for annotation
    assert items[0].article_id == "1"

    # Pre-labeler
    pre_labeler = MockLLMPreLabeler()
    cat, rationale, conf = pre_labeler.pre_label("Lakers defeat Celtics in overtime", "NBA match report", classes)
    assert cat == "sports"
    assert "keyword" in rationale.lower()

    # Inter-annotator Cohen's kappa
    r1 = ["news", "sports", "tech", "news", "sports"]
    r2 = ["news", "sports", "tech", "news", "news"]
    agreement = compute_inter_annotator_agreement(r1, r2)
    assert "cohen_kappa" in agreement
    assert agreement["observed_agreement"] == 0.8

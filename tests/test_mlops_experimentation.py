"""Unit and integration tests for MLOps, Experimentation, and API routes."""

import pytest
from pulserank.ml.experimentation import ExperimentationEngine
from pulserank.ml.monitoring import InferenceMonitor
from pulserank.ml.registry import ModelRegistry
from pulserank.ml.tracking import MLflowTracker
from pulserank.data.provenance import ArtifactManifest


def test_ab_sample_size_calculator():
    res = ExperimentationEngine.calculate_ab_sample_size(
        baseline_rate=0.04,
        mde_relative=0.05,
        alpha=0.05,
        power=0.80,
    )
    assert res.baseline_ctr == 0.04
    assert res.mde_relative == 0.05
    assert res.required_sample_size_per_variant > 10000
    assert res.total_sample_size == res.required_sample_size_per_variant * 2


def test_psi_drift_detection():
    baseline = {"tech": 0.40, "sports": 0.30, "news": 0.20, "finance": 0.10}
    monitor = InferenceMonitor(baseline_category_dist=baseline)

    # 1. Stable distribution
    stable_serving = {"tech": 0.39, "sports": 0.31, "news": 0.20, "finance": 0.10}
    report_stable = monitor.compute_psi_drift(stable_serving)
    assert report_stable.psi_score < 0.10
    assert report_stable.status == "stable"
    assert report_stable.alert_triggered is False

    # 2. Significant drift distribution
    drift_serving = {"tech": 0.05, "sports": 0.10, "news": 0.70, "finance": 0.15}
    report_drift = monitor.compute_psi_drift(drift_serving)
    assert report_drift.psi_score > 0.25
    assert report_drift.status == "significant_drift"
    assert report_drift.alert_triggered is True


def test_model_registry_lifecycle(tmp_path):
    registry_file = tmp_path / "registry.json"
    registry = ModelRegistry(registry_file=registry_file)

    manifest = ArtifactManifest(
        artifact_type="ranker",
        dataset_fingerprint="abc12345",
        parameters={"lr": 0.05},
    )

    # Register v1
    v1 = registry.register_version(
        model_name="ranker",
        version="v1.0.0",
        artifact_path="/models/ranker_v1.pkl",
        metrics={"auc": 0.720, "ndcg": 0.650},
        manifest=manifest,
        stage="production",
    )
    assert v1.stage == "production"
    assert registry.get_production_model("ranker").version == "v1.0.0"

    # Register v2 with lower AUC (should fail promotion gate without force)
    v2 = registry.register_version(
        model_name="ranker",
        version="v2.0.0",
        artifact_path="/models/ranker_v2.pkl",
        metrics={"auc": 0.710, "ndcg": 0.640},
        stage="staging",
    )
    success, msg = registry.promote_to_production("ranker", "v2.0.0", force=False)
    assert not success
    assert "rejected" in msg.lower()

    # Register v3 with higher AUC (should pass promotion gate)
    v3 = registry.register_version(
        model_name="ranker",
        version="v3.0.0",
        artifact_path="/models/ranker_v3.pkl",
        metrics={"auc": 0.745, "ndcg": 0.680},
        stage="staging",
    )
    success, msg = registry.promote_to_production("ranker", "v3.0.0", force=False)
    assert success
    assert registry.get_production_model("ranker").version == "v3.0.0"


def test_pass2_api_routes(client):
    # 1. NLP Classification
    res_clf = client.post("/api/nlp/classify", json={"title": "Lakers defeat Celtics", "abstract": "NBA game"})
    assert res_clf.status_code == 200
    assert "predicted_category" in res_clf.json()

    # 2. NER Extraction
    res_ner = client.post("/api/nlp/ner/extract", json={"text": "Google announced partnership in London with BBC."})
    assert res_ner.status_code == 200
    assert res_ner.json()["total_extracted"] > 0

    # 3. Experiment Sample Size
    res_size = client.post("/api/experiments/sample-size", json={"baseline_ctr": 0.03, "mde_relative": 0.1})
    assert res_size.status_code == 200
    assert "required_sample_size_per_variant" in res_size.json()

    # 4. Cold-Start Onboarding
    res_onboard = client.post("/api/cold-start/onboard", json={"selected_categories": ["sports", "news"]})
    assert res_onboard.status_code == 200
    assert "category_priors" in res_onboard.json()

    # 5. Monitoring Stats & Drift
    res_mon = client.get("/api/monitoring/stats")
    assert res_mon.status_code == 200
    res_drift = client.get("/api/monitoring/drift")
    assert res_drift.status_code == 200
    assert "psi_score" in res_drift.json()

    # 6. Annotation Queue
    res_annot = client.get("/api/annotations/queue?top_n=5")
    assert res_annot.status_code == 200
    assert "queue" in res_annot.json()

    # 7. Model Registry
    res_reg = client.get("/api/registry/models")
    assert res_reg.status_code == 200
    assert "models" in res_reg.json()

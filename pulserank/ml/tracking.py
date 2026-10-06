"""Local MLflow Experiment Tracking for PulseRank Pipelines."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from pulserank.data.provenance import ArtifactManifest

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Manages experiment runs, metrics logging, artifact tracking, and parameter logging
    backed by a local MLflow file store.
    """

    def __init__(self, tracking_dir: Optional[Path] = None):
        self.tracking_dir = tracking_dir or Path("mlruns")
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(f"file:{self.tracking_dir.resolve()}")

    @contextmanager
    def start_run(
        self,
        experiment_name: str = "pulserank_recsys",
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Context manager for tracking a model training or evaluation run."""
        if not MLFLOW_AVAILABLE:
            logger.info(f"[Local Tracking Fallback] Run started: {run_name} (MLflow unavailable)")
            yield self
            return

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            self.active_run_id = active_run.info.run_id
            try:
                yield self
            finally:
                self.active_run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log training hyperparameters or configuration settings."""
        if not MLFLOW_AVAILABLE:
            return
        flattened = {}
        for k, v in params.items():
            if isinstance(v, (dict, list)):
                flattened[k] = str(v)
            else:
                flattened[k] = v
        mlflow.log_params(flattened)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log evaluation metrics (AUC, MRR, nDCG, ILD, etc.)."""
        if not MLFLOW_AVAILABLE:
            return
        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, artifact_path: Path) -> None:
        """Log a model or evaluation artifact file."""
        if not MLFLOW_AVAILABLE:
            return
        if Path(artifact_path).exists():
            mlflow.log_artifact(str(artifact_path))

    def log_manifest(self, manifest: ArtifactManifest) -> None:
        """Log an artifact manifest and attach dataset provenance tags."""
        if not MLFLOW_AVAILABLE:
            return
        temp_path = self.tracking_dir / "temp_manifest.json"
        manifest.save(temp_path)
        mlflow.log_artifact(str(temp_path), artifact_path="provenance")
        if temp_path.exists():
            temp_path.unlink()
        mlflow.set_tags({
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "artifact_type": manifest.artifact_type,
            "schema_version": manifest.schema_version,
        })

    def list_runs(self, experiment_name: str = "pulserank_recsys") -> List[Dict[str, Any]]:
        """Query logged runs and their metrics from local storage."""
        if not MLFLOW_AVAILABLE:
            return []
        try:
            exp = mlflow.get_experiment_by_name(experiment_name)
            if not exp:
                return []
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            records = []
            for _, row in runs.iterrows():
                records.append({
                    "run_id": row.get("run_id", ""),
                    "run_name": row.get("tags.mlflow.runName", ""),
                    "status": row.get("status", ""),
                    "start_time": str(row.get("start_time", "")),
                    "metrics": {col.replace("metrics.", ""): float(row[col]) for col in runs.columns if col.startswith("metrics.") and not row[col] != row[col]},
                })
            return records
        except Exception as e:
            logger.warning(f"Failed to search MLflow runs: {e}")
            return []

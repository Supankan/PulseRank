"""Model Registry, Lifecycle State Management, and Promotion Gates."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


from pulserank.data.provenance import ArtifactManifest

logger = logging.getLogger(__name__)


@dataclass
class RegisteredModelVersion:
    model_name: str
    version: str
    stage: str  # 'staging', 'production', 'archived'
    artifact_path: str
    metrics: Dict[str, float]
    manifest: Optional[Dict[str, Any]]
    created_at: str
    promoted_at: Optional[str] = None
    description: str = ""


class ModelRegistry:
    """
    Manages versioned ML artifacts across lifecycle environments:
    - Tracks versions, metrics, artifact locations, and dataset provenance manifests.
    - Implements automated promotion criteria against the current champion model.
    """

    def __init__(self, registry_file: Optional[Path] = None):
        self.registry_file = registry_file or Path("artifacts/registry.json")
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, List[RegisteredModelVersion]] = {}
        self._load()

    def _load(self) -> None:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for m_name, versions in data.items():
                    self.models[m_name] = [RegisteredModelVersion(**v) for v in versions]
            except Exception as e:
                logger.warning(f"Failed to load model registry from {self.registry_file}: {e}")
                self.models = {}

    def _save(self) -> None:
        data = {m_name: [asdict(v) for v in versions] for m_name, versions in self.models.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_version(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        metrics: Dict[str, float],
        manifest: Optional[ArtifactManifest] = None,
        stage: str = "staging",
        description: str = "",
    ) -> RegisteredModelVersion:
        """Register a new model checkpoint or pipeline artifact."""
        entry = RegisteredModelVersion(
            model_name=model_name,
            version=version,
            stage=stage,
            artifact_path=str(artifact_path),
            metrics=metrics,
            manifest=asdict(manifest) if manifest else None,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
        )

        self.models.setdefault(model_name, []).append(entry)
        self._save()
        logger.info(f"Registered model '{model_name}' version '{version}' in stage '{stage}'.")
        return entry

    def get_production_model(self, model_name: str) -> Optional[RegisteredModelVersion]:
        """Retrieve the active production champion version."""
        for v in self.models.get(model_name, []):
            if v.stage == "production":
                return v
        return None

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        min_metric_gate: str = "auc",
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Promote a candidate model to production:
        - Verifies that candidate meets or exceeds the current champion\'s primary metric.
        - Archives the previously active production model.
        """
        versions = self.models.get(model_name, [])
        candidate = next((v for v in versions if v.version == version), None)
        if not candidate:
            return False, f"Version '{version}' for model '{model_name}' not found."

        current_prod = self.get_production_model(model_name)

        if current_prod and not force:
            cand_score = candidate.metrics.get(min_metric_gate, 0.0)
            prod_score = current_prod.metrics.get(min_metric_gate, 0.0)
            if cand_score < prod_score:
                return False, (
                    f"Promotion rejected: candidate {min_metric_gate} ({cand_score}) < "
                    f"current production {min_metric_gate} ({prod_score}). Use force=True to override."
                )

        # Archive current production
        if current_prod:
            current_prod.stage = "archived"

        candidate.stage = "production"
        candidate.promoted_at = datetime.now(timezone.utc).isoformat()
        self._save()

        return True, f"Successfully promoted {model_name}:{version} to production."

    def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return full catalog of registered models and versions."""
        return {m_name: [asdict(v) for v in versions] for m_name, versions in self.models.items()}

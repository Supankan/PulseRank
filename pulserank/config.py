"""Application and ML configuration using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    app_name: str = "PulseRank"
    env: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Project Root
    project_root: Path = Path(__file__).resolve().parent.parent

    # Database
    database_url: str = "postgresql+psycopg2://pulserank:pulserank@localhost:5432/pulserank_db"
    test_database_url: str = "sqlite:///./data/pulserank_test.db"

    # Paths
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    artifacts_dir: Path = project_root / "artifacts"

    # Retrieval & Pipeline Hyperparameters
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    popularity_decay_hours: float = 72.0
    
    # Collaborative Filtering (ALS)
    als_factors: int = 64
    als_regularization: float = 0.05
    als_iterations: int = 15
    als_alpha: float = 40.0

    # LightGBM LambdaMART Ranker
    ranker_max_depth: int = 6
    ranker_num_leaves: int = 31
    ranker_learning_rate: float = 0.05
    ranker_n_estimators: int = 150
    ranker_random_state: int = 42

    # Top-K candidate pool sizes
    content_top_k: int = 50
    collab_top_k: int = 50
    popularity_top_k: int = 50
    final_feed_limit: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()

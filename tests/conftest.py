"""Test fixtures and shared configuration."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from pulserank.config import settings
from pulserank.data.fixtures import write_mind_tsv_fixtures
from pulserank.data.preprocessor import build_processed_dataset, ProcessedDataset
from pulserank.db.base import Base
from pulserank.db.session import engine, SessionLocal, get_db
from pulserank.api.app import app
from pulserank.api.dependencies import container


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory cleaned up after session."""
    d = Path(tempfile.mkdtemp(prefix="pulserank_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_mind_data(temp_dir: Path):
    """Generate authentic MIND fixtures in temporary directory."""
    raw_dir = temp_dir / "raw"
    train_news, train_behaviors, dev_news, dev_behaviors = write_mind_tsv_fixtures(raw_dir)
    return {
        "train_news": train_news,
        "train_behaviors": train_behaviors,
        "dev_news": dev_news,
        "dev_behaviors": dev_behaviors,
    }


@pytest.fixture(scope="session")
def processed_dataset(sample_mind_data) -> ProcessedDataset:
    """Build processed dataset from sample fixture data."""
    return build_processed_dataset(
        train_news_path=sample_mind_data["train_news"],
        train_behaviors_path=sample_mind_data["train_behaviors"],
        dev_news_path=sample_mind_data["dev_news"],
        dev_behaviors_path=sample_mind_data["dev_behaviors"],
    )


@pytest.fixture(scope="session")
def test_db():
    """Create in-memory SQLite tables for testing."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
def client(processed_dataset, test_db) -> Generator[TestClient, None, None]:
    """TestClient wired with initialized service container."""
    container.dataset = processed_dataset
    container.initialize()
    with TestClient(app) as c:
        yield c

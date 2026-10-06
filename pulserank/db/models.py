"""Database schema models for PulseRank metadata and recommendations."""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Integer, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pulserank.db.base import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    history_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")


class Article(Base):
    __tablename__ = "articles"

    article_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    subcategory: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    title_entities: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    abstract_entities: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    click_count: Mapped[int] = mapped_column(Integer, default=0)


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.user_id"), index=True)
    article_id: Mapped[str] = mapped_column(String(64), ForeignKey("articles.article_id"), index=True)
    interaction_type: Mapped[str] = mapped_column(String(32), index=True)  # history, click, impression, like, save
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    user = relationship("User", back_populates="interactions")
    article = relationship("Article")


class RecommendationRun(Base):
    __tablename__ = "recommendation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # UUID
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.user_id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    outputs = relationship("RecommendationOutput", back_populates="run", cascade="all, delete-orphan")


class RecommendationOutput(Base):
    __tablename__ = "recommendation_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_run_id: Mapped[str] = mapped_column(String(64), ForeignKey("recommendation_runs.id"), index=True)
    article_id: Mapped[str] = mapped_column(String(64), ForeignKey("articles.article_id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    sources: Mapped[Any] = mapped_column(JSON)
    explanation: Mapped[Any] = mapped_column(JSON)

    run = relationship("RecommendationRun", back_populates="outputs")
    article = relationship("Article")


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    model_type: Mapped[str] = mapped_column(String(64))  # popularity, content_based, collaborative, hybrid, ranker
    status: Mapped[str] = mapped_column(String(32), default="ready")
    metrics: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    auc: Mapped[float] = mapped_column(Float)
    mrr: Mapped[float] = mapped_column(Float)
    ndcg_at_5: Mapped[float] = mapped_column(Float)
    ndcg_at_10: Mapped[float] = mapped_column(Float)
    recall_at_k: Mapped[float] = mapped_column(Float)
    coverage: Mapped[float] = mapped_column(Float)
    num_impressions: Mapped[int] = mapped_column(Integer)
    num_users: Mapped[int] = mapped_column(Integer)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    details: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    dataset_source: Mapped[str] = mapped_column(String(64), default="mind-small")
    evaluation_mode: Mapped[str] = mapped_column(String(32), default="impression")
    impression_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    catalog_coverage: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hit_rate_at_k: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_scored_impressions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class AnnotationRecord(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    article_id: Mapped[str] = mapped_column(String(64), ForeignKey("articles.article_id"), index=True)
    task_type: Mapped[str] = mapped_column(String(32), index=True)  # topic_classification, ner
    predicted_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    true_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    annotator_id: Mapped[str] = mapped_column(String(64), default="human")
    uncertainty_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    meta_info: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)

    @property
    def label(self) -> Optional[str]:
        """Convenience alias for true_label."""
        return self.true_label


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_name: Mapped[str] = mapped_column(String(64), index=True)
    variant_a: Mapped[str] = mapped_column(String(64))
    variant_b: Mapped[str] = mapped_column(String(64))
    traffic_split: Mapped[float] = mapped_column(Float, default=0.5)
    metrics_summary: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    p_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")  # running, completed, archived
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str] = mapped_column(String(32), default="staging")  # staging, production, archived
    artifact_path: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    manifest_data: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    metrics: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class MonitoringLog(Base):
    __tablename__ = "monitoring_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    endpoint: Mapped[str] = mapped_column(String(64), index=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    user_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    channel_distribution: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    prediction_entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    drift_psi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


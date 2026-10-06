"""Inference Health Monitoring and Population Stability Index (PSI) Drift Routes."""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends
from pulserank.api.dependencies import ServiceContainer, get_service_container

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/stats")
def get_serving_statistics(services: ServiceContainer = Depends(get_service_container)):
    stats = services.inference_monitor.get_serving_statistics()
    return {
        "total_requests": stats.total_requests,
        "mean_latency_ms": stats.mean_latency_ms,
        "p95_latency_ms": stats.p95_latency_ms,
        "channel_contribution_shares": stats.channel_contribution_shares,
        "mean_prediction_entropy": stats.mean_prediction_entropy,
    }


@router.get("/drift")
def get_drift_report(services: ServiceContainer = Depends(get_service_container)):
    report = services.inference_monitor.compute_psi_drift()
    return {
        "psi_score": report.psi_score,
        "status": report.status,
        "alert_triggered": report.alert_triggered,
        "recommended_action": report.recommended_action,
        "baseline_distribution": report.baseline_distribution,
        "serving_distribution": report.serving_distribution,
        "per_category_psi": report.per_category_psi,
    }

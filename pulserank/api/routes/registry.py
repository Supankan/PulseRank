"""Model Registry Lifecycle Management Routes."""

from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from pulserank.api.dependencies import ServiceContainer, get_service_container

router = APIRouter(prefix="/registry", tags=["Model Registry"])


class PromoteRequest(BaseModel):
    model_name: str
    version: str
    min_metric_gate: str = "auc"
    force: bool = False


@router.get("/models")
def list_registry_models(services: ServiceContainer = Depends(get_service_container)):
    all_models = services.model_registry.list_all_models()
    return {"models": all_models}


@router.post("/promote")
def promote_model(
    request: PromoteRequest,
    services: ServiceContainer = Depends(get_service_container),
):
    success, msg = services.model_registry.promote_to_production(
        model_name=request.model_name,
        version=request.version,
        min_metric_gate=request.min_metric_gate,
        force=request.force,
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"success": True, "message": msg}

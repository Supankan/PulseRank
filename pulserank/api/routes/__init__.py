"""API routes package."""

from pulserank.api.routes.health import router as health_router
from pulserank.api.routes.users import router as users_router
from pulserank.api.routes.feed import router as feed_router
from pulserank.api.routes.articles import router as articles_router
from pulserank.api.routes.recommendations import router as recommendations_router
from pulserank.api.routes.models import router as models_router
from pulserank.api.routes.evaluations import router as evaluations_router
from pulserank.api.routes.nlp import router as nlp_router
from pulserank.api.routes.experiments import router as experiments_router
from pulserank.api.routes.ablations import router as ablations_router
from pulserank.api.routes.cold_start import router as cold_start_router
from pulserank.api.routes.monitoring import router as monitoring_router
from pulserank.api.routes.annotations import router as annotations_router
from pulserank.api.routes.diagnostics import router as diagnostics_router
from pulserank.api.routes.registry import router as registry_router

__all__ = [
    "health_router",
    "users_router",
    "feed_router",
    "articles_router",
    "recommendations_router",
    "models_router",
    "evaluations_router",
    "nlp_router",
    "experiments_router",
    "ablations_router",
    "cold_start_router",
    "monitoring_router",
    "annotations_router",
    "diagnostics_router",
    "registry_router",
]


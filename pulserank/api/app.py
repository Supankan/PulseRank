"""FastAPI application factory and middleware configuration."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pulserank.config import settings
from pulserank.db.session import init_db
from pulserank.api.dependencies import container
from pulserank.api.routes import (
    health_router,
    users_router,
    feed_router,
    articles_router,
    recommendations_router,
    models_router,
    evaluations_router,
    nlp_router,
    experiments_router,
    ablations_router,
    cold_start_router,
    monitoring_router,
    annotations_router,
    diagnostics_router,
    registry_router,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} in {settings.env} mode...")
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    try:
        container.initialize()
        logger.info("PulseRank ML pipelines ready.")
    except Exception as e:
        logger.error(f"Error during ML container initialization: {e}")

    yield
    logger.info("PulseRank application shutting down.")


def create_app() -> FastAPI:
    """Factory creating configured FastAPI app."""
    app = FastAPI(
        title=settings.app_name,
        description="Personalized Content Intelligence & Multi-Stage Recommendation Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Structured error handling (protects internal stack traces)
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected server error occurred while processing the recommendation request.",
                "path": request.url.path,
            },
        )

    # Register routes
    api_prefix = "/api"
    app.include_router(health_router, prefix=api_prefix)
    app.include_router(users_router, prefix=api_prefix)
    app.include_router(feed_router, prefix=api_prefix)
    app.include_router(articles_router, prefix=api_prefix)
    app.include_router(recommendations_router, prefix=api_prefix)
    app.include_router(models_router, prefix=api_prefix)
    app.include_router(evaluations_router, prefix=api_prefix)
    app.include_router(nlp_router, prefix=api_prefix)
    app.include_router(experiments_router, prefix=api_prefix)
    app.include_router(ablations_router, prefix=api_prefix)
    app.include_router(cold_start_router, prefix=api_prefix)
    app.include_router(monitoring_router, prefix=api_prefix)
    app.include_router(annotations_router, prefix=api_prefix)
    app.include_router(diagnostics_router, prefix=api_prefix)
    app.include_router(registry_router, prefix=api_prefix)


    # Mount static frontend production build if available
    dist_dir = settings.project_root / "web" / "dist"
    if dist_dir.exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static_web")

    return app


app = create_app()

"""Main FastAPI application for OCR service."""
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime


from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog
import time

from app.core import settings, optimize_for_inference, get_performance_info
from app.core.image_utils import validate_file_upload, load_image_from_bytes
from app.models import get_ocr_pipeline, validate_image_for_ocr
from app.models.schemas import (
    OCRRequest, OCRResponse, BatchOCRRequest, BatchJobResponse,
    BatchJobStatus, BatchJobResult, HealthCheckResponse, 
    ModelsResponse, ErrorResponse, ValidationError, JobStatus
)
from app.api import inference_router, batch_router, health_router, models_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Global variables for cleanup
app_state = {
    "start_time": None,
    "shutdown_event": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting OCR FastAPI service", 
                environment=settings.environment,
                debug=settings.debug)
    
    # Optimize for inference
    optimize_for_inference()
    
    # Set startup time
    app_state["start_time"] = datetime.now()
    app_state["shutdown_event"] = asyncio.Event()
    
    # Validate model paths
    try:
        detection_path = settings.get_detection_model_path()
        recognition_path = settings.get_recognition_model_path()
        
        if not detection_path.exists():
            raise FileNotFoundError(f"Detection model not found: {detection_path}")
        if not recognition_path.exists():
            raise FileNotFoundError(f"Recognition model not found: {recognition_path}")
            
        logger.info("Model paths validated successfully")
        
    except Exception as e:
        logger.error("Failed to validate model paths", error=str(e))
        raise
    
    logger.info("OCR FastAPI service started successfully")
    
    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down OCR FastAPI service")
        app_state["shutdown_event"].set()


# Create FastAPI application
application = FastAPI(
    title="OCR FastAPI Service",
    description="Production-ready OCR service for Devanagari text detection and recognition",
    version="1.0.0",
    docs_url="/docs" if settings.is_development() else None,
    redoc_url="/redoc" if settings.is_development() else None,
    lifespan=lifespan
)

# Add middleware
application.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

application.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Add rate limiting
application.state.limiter = limiter
application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include API routes
application.include_router(inference_router, prefix="/api/v1/ocr", tags=["inference"])
application.include_router(batch_router, prefix="/api/v1/ocr", tags=["batch"])
application.include_router(health_router, prefix="/api/v1", tags=["health"])
application.include_router(models_router, prefix="/api/v1", tags=["models"])


@application.get("/", response_model=dict)
async def root():
    """Root endpoint with service information."""
    return {
        "service": "OCR FastAPI Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs" if settings.is_development() else "disabled in production",
        "environment": settings.environment
    }


@application.get("/api/v1/status", response_model=dict)
async def service_status():
    """Service status endpoint."""
    uptime = None
    if app_state["start_time"]:
        uptime = (datetime.now() - app_state["start_time"]).total_seconds()
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "start_time": app_state["start_time"].isoformat() if app_state["start_time"] else None,
        "environment": settings.environment,
        "debug": settings.debug,
        "models_loaded": len(get_ocr_pipeline().get_pipeline_stats().get("models", {}))
    }


@application.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning("HTTP exception occurred", 
                   status_code=exc.status_code,
                   detail=exc.detail,
                   path=request.url.path)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "type": "http_exception"
            },
            "timestamp": datetime.now().isoformat()
        }
    )


@application.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error("Unhandled exception occurred",
                 error=str(exc),
                 error_type=type(exc).__name__,
                 path=request.url.path)
    
    # Don't leak internal error details in production
    if settings.is_production():
        message = "Internal server error"
    else:
        message = f"Internal server error: {str(exc)}"
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": message,
                "type": "internal_error"
            },
            "timestamp": datetime.now().isoformat()
        }
    )


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        if app_state["shutdown_event"]:
            app_state["shutdown_event"].set()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


# Setup signal handlers on import
setup_signal_handlers()


if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn
    uvicorn_config = {
        "host": settings.host,
        "port": settings.port,
        "log_level": settings.log_level.lower(),
        "access_log": True,
    }
    
    if settings.reload and settings.is_development():
        uvicorn_config["reload"] = True
        uvicorn_config["reload_dirs"] = ["app"]
    
    logger.info("Starting uvicorn server", config=uvicorn_config)
    uvicorn.run("app.main:application", **uvicorn_config)
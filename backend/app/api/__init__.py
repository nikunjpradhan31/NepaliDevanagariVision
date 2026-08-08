"""API package for OCR FastAPI service."""

from .health import router as health_router
from .inference import router as inference_router
from .models import router as models_router

# For backwards compatibility with main.py
inference = inference_router
health = health_router
models = models_router

__all__ = [
    "health_router",
    "inference_router",
    "models_router",
    "inference",
    "health",
    "models"
]

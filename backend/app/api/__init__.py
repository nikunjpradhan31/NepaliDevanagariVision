"""API package for OCR FastAPI service."""

# Import routers from each module
#from .batch import router as batch_router
from .health import router as health_router
from .inference import router as inference_router
from .models import router as models_router

# For backwards compatibility with main.py
inference = inference_router
#batch = batch_router
health = health_router
models = models_router

__all__ = [
    "batch_router",
    "health_router", 
    "inference_router",
    "models_router",
    "inference",
    #"batch", 
    "health",
    "models"
]
"""Models package for OCR FastAPI service."""

# Import from detection
from .detection import LineDetectionModel, get_detection_model

# Import from model_manager
from .model_manager import (
    ModelInfo as ModelManagerInfo,
    ModelManager,
    model_manager,
    load_detection_model,
    load_recognition_model,
    get_detection_model,
    get_recognition_model,
    check_models_health,
    get_models_stats
)

# Import from ocr_pipeline
from .ocr_pipeline import OCRPipeline, get_ocr_pipeline, validate_image_for_ocr

# Import from recognition
from .recognition import CTCLabelConverter, TextRecognitionModel, get_recognition_model

# Import from schemas (avoiding name conflicts with ModelManagerInfo)
from .schemas import (
    JobStatus,
    OCRRequest,
    OCRResponse,
    BatchOCRRequest,
    BatchJobResponse,
    BatchJobStatus,
    BatchJobResult,
    HealthCheckResponse,
    ModelInfo as SchemaModelInfo,
    ModelsResponse,
    ErrorResponse,
    ValidationError
)

__all__ = [
    # detection
    "LineDetectionModel",
    "get_detection_model",
    # model_manager
    "ModelManagerInfo",
    "ModelManager",
    "model_manager",
    "load_detection_model",
    "load_recognition_model", 
    "get_detection_model",
    "get_recognition_model",
    "check_models_health",
    "get_models_stats",
    # ocr_pipeline
    "OCRPipeline",
    "get_ocr_pipeline",
    "validate_image_for_ocr",
    # recognition
    "CTCLabelConverter",
    "TextRecognitionModel",
    "get_recognition_model",
    # schemas
    "JobStatus",
    "OCRRequest",
    "OCRResponse",
    "BatchOCRRequest",
    "BatchJobResponse",
    "BatchJobStatus",
    "BatchJobResult",
    "HealthCheckResponse",
    "SchemaModelInfo",
    "ModelsResponse",
    "ErrorResponse",
    "ValidationError"
]
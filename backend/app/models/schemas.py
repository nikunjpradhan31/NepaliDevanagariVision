"""Pydantic models for OCR FastAPI service request/response validation."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OCRRequest(BaseModel):
    """Request model for OCR inference."""
    
    include_masks: bool = Field(default=False, description="Whether to include mask data in response")
    confidence_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Override default confidence threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "include_masks": True,
                "confidence_threshold": 0.5
            }
        }


class OCRResponse(BaseModel):
    """Response model for OCR inference."""
    
    image_width: int = Field(description="Width of the input image")
    image_height: int = Field(description="Height of the input image")
    detections: List[Dict[str, Any]] = Field(default_factory=list, description="List of detected text lines")
    total_lines: int = Field(description="Total number of detected text lines")
    processing_time: float = Field(description="Time taken for processing in seconds")
    models_used: Dict[str, str] = Field(description="Models used for processing")
    parsed_text: str = Field(description="Concatenated parsed text from all detections")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_width": 1024,
                "image_height": 768,
                "detections": [
                    {
                        "line_id": 0,
                        "box": [50, 100, 500, 130],
                        "crop_box": [0, 85, 600, 145],
                        "confidence": 0.95,
                        "text": "नमस्ते विश्व",
                        "class": 0
                    }
                ],
                "total_lines": 1,
                "processing_time": 1.234,
                "models_used": {
                    "detection": "LineDetectionv4",
                    "recognition": "ResNetBiLSTMCTCv1"
                }
            }
        }

class MultiOCRResponse(BaseModel):
    """Response model for OCR inference."""
    
    OCR: List[OCRResponse] = Field(description="List of OCR responses for each image in batch")
    
    class Config:
        json_schema_extra = {
            "example": {
                "OCR": [
                {
                "image_width": 1024,
                "image_height": 768,
                "detections": [
                    {
                        "line_id": 0,
                        "box": [50, 100, 500, 130],
                        "crop_box": [0, 85, 600, 145],
                        "confidence": 0.95,
                        "text": "नमस्ते विश्व",
                        "class": 0
                    }
                ],
                "total_lines": 1,
                "processing_time": 1.234,
                "models_used": {
                    "detection": "LineDetectionv4",
                    "recognition": "ResNetBiLSTMCTCv1"
                }
            }
                ]
            }


        }

# class BatchOCRRequest(BaseModel):
#     """Request model for batch OCR processing."""
    
#     include_masks: bool = Field(default=True, description="Whether to include mask data in response")
#     confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override default confidence threshold")
#     priority: int = Field(default=0, ge=0, le=10, description="Job priority (0=normal, 10=highest)")
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "include_masks": True,
#                 "confidence_threshold": 0.5,
#                 "priority": 5
#             }
#         }


# class BatchJobResponse(BaseModel):
#     """Response model for batch job submission."""
    
#     job_id: str = Field(description="Unique job identifier")
#     status: JobStatus = Field(description="Current job status")
#     total_images: int = Field(description="Total number of images in batch")
#     submitted_at: datetime = Field(description="When the job was submitted")
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "job_id": "batch_abc123def456",
#                 "status": "pending",
#                 "total_images": 5,
#                 "submitted_at": "2025-11-30T12:00:00Z"
#             }
#         }


# class BatchJobStatus(BaseModel):
#     """Response model for batch job status."""
    
#     job_id: str = Field(description="Unique job identifier")
#     status: JobStatus = Field(description="Current job status")
#     total_images: int = Field(description="Total number of images in batch")
#     processed_images: int = Field(default=0, description="Number of images processed")
#     failed_images: int = Field(default=0, description="Number of images that failed")
#     progress_percentage: float = Field(default=0.0, description="Progress percentage (0-100)")
#     created_at: datetime = Field(description="When the job was created")
#     updated_at: datetime = Field(description="When the job was last updated")
#     estimated_completion: Optional[datetime] = Field(description="Estimated completion time")
#     processing_time: Optional[float] = Field(description="Current processing time in seconds")
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "job_id": "batch_abc123def456",
#                 "status": "processing",
#                 "total_images": 5,
#                 "processed_images": 3,
#                 "failed_images": 0,
#                 "progress_percentage": 60.0,
#                 "created_at": "2025-11-30T12:00:00Z",
#                 "updated_at": "2025-11-30T12:01:30Z",
#                 "estimated_completion": "2025-11-30T12:02:00Z",
#                 "processing_time": 90.5
#             }
#         }


# class BatchJobResult(BaseModel):
#     """Response model for batch job results."""
    
#     job_id: str = Field(description="Unique job identifier")
#     status: JobStatus = Field(description="Final job status")
#     results: List[Dict[str, Any]] = Field(default_factory=list, description="OCR results for each image")
#     failed_images: List[Dict[str, Any]] = Field(default_factory=list, description="Failed image information")
#     total_processing_time: float = Field(description="Total processing time in seconds")
#     completed_at: datetime = Field(description="When the job was completed")
    
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "job_id": "batch_abc123def456",
#                 "status": "completed",
#                 "results": [
#                     {
#                         "image_name": "document1.jpg",
#                         "ocr_result": {
#                             "image_width": 1024,
#                             "image_height": 768,
#                             "detections": [],
#                             "total_lines": 10,
#                             "processing_time": 1.5
#                         }
#                     }
#                 ],
#                 "failed_images": [],
#                 "total_processing_time": 7.5,
#                 "completed_at": "2025-11-30T12:02:00Z"
#             }
#         }


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(description="Overall service status")
    timestamp: datetime = Field(description="When the health check was performed")
    models: Dict[str, bool] = Field(description="Model health status")
    system: Dict[str, Any] = Field(description="System information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-11-30T12:00:00Z",
                "models": {
                    "detection": True,
                    "recognition": True
                },
                "system": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "device": "cpu"
                }
            }
        }


class ModelInfo(BaseModel):
    """Response model for model information."""
    
    name: str = Field(description="Model name")
    path: str = Field(description="Model file path")
    healthy: bool = Field(description="Model health status")
    loaded_at: str = Field(description="When the model was loaded")
    last_used: str = Field(description="When the model was last used")
    inference_count: int = Field(description="Number of inferences performed")
    avg_inference_time: float = Field(description="Average inference time in seconds")
    inputs: List[Dict[str, Any]] = Field(default_factory=list, description="Model input information")
    outputs: List[Dict[str, Any]] = Field(default_factory=list, description="Model output information")
    providers: List[str] = Field(default_factory=list, description="ONNX execution providers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "detection",
                "path": "models/LineDetectionv4.onnx",
                "healthy": True,
                "loaded_at": "2025-11-30T12:00:00Z",
                "last_used": "2025-11-30T12:01:00Z",
                "inference_count": 150,
                "avg_inference_time": 0.234,
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1, 3, 1024, 1024],
                        "type": "float32"
                    }
                ],
                "outputs": [
                    {
                        "name": "output",
                        "shape": [1, 300, 38],
                        "type": "float32"
                    }
                ],
                "providers": ["CPUExecutionProvider"]
            }
        }


class ModelsResponse(BaseModel):
    """Response model for models endpoint."""
    
    models: Dict[str, ModelInfo] = Field(description="Model information by name")
    character_set_info: Dict[str, Any] = Field(description="Character set information for recognition model")
    
    class Config:
        json_schema_extra = {
            "example": {
                "models": {
                    "detection": {},
                    "recognition": {}
                },
                "character_set_info": {
                    "character_set": "नमस्ते विश्व...",
                    "num_characters": 128,
                    "blank_token": "[blank]",
                    "dictionary_size": 128
                }
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: Dict[str, Any] = Field(description="Error information")
    timestamp: datetime = Field(description="When the error occurred")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": {
                    "code": "INVALID_FILE_FORMAT",
                    "message": "Unsupported file format. Allowed: jpg, jpeg, png, tiff, pdf",
                    "details": {
                        "filename": "document.bmp",
                        "received_format": "bmp"
                    }
                },
                "timestamp": "2025-11-30T12:00:00Z"
            }
        }


class ValidationError(BaseModel):
    """Response model for validation errors."""
    
    detail: List[Dict[str, Any]] = Field(description="Validation error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": [
                    {
                        "loc": ["body", "confidence_threshold"],
                        "msg": "ensure this value is less than or equal to 1",
                        "type": "value_error.number.not_le"
                    }
                ]
            }
        }
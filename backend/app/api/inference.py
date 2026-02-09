"""Real-time inference endpoints for OCR service."""
import asyncio
import time
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.image_utils import validate_file_upload, load_image_from_bytes, validate_image_quality
from app.models.ocr_pipeline import get_ocr_pipeline
from app.models.schemas import OCRRequest, OCRResponse, ValidationError, MultiOCRResponse
from app.core.device import cleanup_memory

logger = structlog.get_logger()

router = APIRouter()


async def process_image_file(file: UploadFile, request: OCRRequest) -> MultiOCRResponse:
    """Process uploaded image file through OCR pipeline."""
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file
        is_valid, error_msg, extension = validate_file_upload(file_content, file.filename)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_FILE",
                    "message": error_msg,
                    "filename": file.filename
                }
            )
        
        # Load and validate image
        try:
            images = load_image_from_bytes(file_content, extension)
        except Exception as e:
            logger.error("Failed to load image", error=str(e), filename=file.filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "IMAGE_LOAD_FAILED",
                    "message": f"Could not load image: {str(e)}",
                    "filename": file.filename
                }
            )
        
        # Process first image (for real-time endpoint)
        #image = images[0]
        Response = []
        for image in images:
        # Additional validation for OCR suitability
            is_valid_image, warnings = validate_image_quality(image)
            if not is_valid_image and settings.is_production():
                # In production, just log warnings but continue processing
                logger.warning("Image quality warnings", 
                            filename=file.filename, 
                            warnings=warnings)
            
            # Override confidence threshold if specified in request
            if request.confidence_threshold is not None:
                original_threshold = settings.detection_confidence_threshold
                settings.detection_confidence_threshold = request.confidence_threshold
            
            try:
                # Get OCR pipeline and process image
                ocr_pipeline = get_ocr_pipeline()
                
                # Validate image for OCR
                is_valid_for_ocr, validation_error = ocr_pipeline.validate_image(image)
                if not is_valid_for_ocr:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "code": "INVALID_IMAGE_FOR_OCR",
                            "message": validation_error,
                            "filename": file.filename
                        }
                    )
                
                # Process through OCR pipeline
                result = ocr_pipeline.process_image(image, include_masks=request.include_masks)
                
                processing_time = time.time() - start_time
                
                # Ensure processing time is recorded
                result["processing_time"] = processing_time
                
                logger.info("OCR inference completed", 
                        filename=file.filename,
                        processing_time=processing_time,
                        lines_detected=result["total_lines"])
                Response.append(OCRResponse(**result))
                
            finally:
                # Restore original confidence threshold if it was overridden
                if request.confidence_threshold is not None:
                    settings.detection_confidence_threshold = original_threshold
                
                # Cleanup memory
                cleanup_memory()
        return MultiOCRResponse(OCR=Response)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Unexpected error during OCR inference", 
                    error=str(e), 
                    error_type=type(e).__name__,
                    filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "INFERENCE_FAILED",
                "message": "Failed to process image through OCR pipeline",
                "filename": file.filename
            }
        )


@router.post(
    "/inference",
    response_model=MultiOCRResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-time OCR inference",
    description="Upload an image for real-time OCR processing. Supports JPEG, PNG, TIFF, and PDF formats."
)
async def ocr_inference(
    request: Request,
    file: UploadFile = File(..., description="Image file to process"),
    ocr_request: OCRRequest = Depends()
):
    """
    Perform real-time OCR on an uploaded image.
    
    - **file**: Image file (JPEG, PNG, TIFF, PDF)
    - **include_masks**: Whether to include segmentation masks in response
    - **confidence_threshold**: Override default confidence threshold (0.0-1.0)
    
    Returns OCR results with detected text lines and bounding boxes.
    """
    return await process_image_file(file, ocr_request)


@router.post(
    "/inference/detect-only",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Line detection only",
    description="Upload an image for line detection only (no text recognition)."
)
async def detect_lines_only(
    request: Request,
    file: UploadFile = File(..., description="Image file to process")
):
    """
    Perform line detection only (no text recognition).
    
    This endpoint is faster and useful when you only need to find text regions
    without recognizing the actual text.
    """
    start_time = time.time()
    
    try:
        # Read and validate file
        file_content = await file.read()
        is_valid, error_msg, extension = validate_file_upload(file_content, file.filename)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_FILE",
                    "message": error_msg,
                    "filename": file.filename
                }
            )
        
        # Load image
        images = load_image_from_bytes(file_content, extension)
        image = images[0]
        
        # Get OCR pipeline and detect lines only
        ocr_pipeline = get_ocr_pipeline()
        detections = ocr_pipeline.detect_only(image)
        
        processing_time = time.time() - start_time
        
        result = {
            "image_width": image.width,
            "image_height": image.height,
            "detections": detections,
            "total_lines": len(detections),
            "processing_time": processing_time,
            "mode": "detection_only"
        }
        
        logger.info("Line detection completed", 
                   filename=file.filename,
                   processing_time=processing_time,
                   lines_detected=len(detections))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during line detection", 
                    error=str(e), 
                    filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "DETECTION_FAILED",
                "message": "Failed to detect lines in image",
                "filename": file.filename
            }
        )


@router.post(
    "/inference/recognize-single",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Text recognition only",
    description="Upload a cropped text line for recognition (no detection)."
)
async def recognize_single_line(
    request: Request,
    file: UploadFile = File(..., description="Cropped text line image")
):
    """
    Recognize text from a single cropped text line.
    
    This endpoint assumes the image contains only one text line and skips
    the detection step, making it faster for already-cropped text regions.
    """
    start_time = time.time()
    
    try:
        # Read and validate file
        file_content = await file.read()
        is_valid, error_msg, extension = validate_file_upload(file_content, file.filename)
        
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "INVALID_FILE",
                    "message": error_msg,
                    "filename": file.filename
                }
            )
        
        # Load and validate image
        images = load_image_from_bytes(file_content, extension)
        image = images[0]
        
        # Get OCR pipeline and recognize text
        ocr_pipeline = get_ocr_pipeline()
        text = ocr_pipeline.process_single_line(image)
        
        processing_time = time.time() - start_time
        
        result = {
            "text": text,
            "processing_time": processing_time,
            "image_width": image.width,
            "image_height": image.height,
            "mode": "recognition_only"
        }
        
        logger.info("Text recognition completed", 
                   filename=file.filename,
                   processing_time=processing_time,
                   recognized_text=text)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error during text recognition", 
                    error=str(e), 
                    filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "RECOGNITION_FAILED",
                "message": "Failed to recognize text in image",
                "filename": file.filename
            }
        )
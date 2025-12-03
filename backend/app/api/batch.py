"""Batch processing endpoints for OCR service."""
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, status
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.image_utils import validate_file_upload, load_image_from_bytes
from app.models.ocr_pipeline import get_ocr_pipeline
from app.models.schemas import (
    BatchOCRRequest, BatchJobResponse, BatchJobStatus, BatchJobResult, 
    JobStatus, ValidationError
)
from app.core.device import cleanup_memory

logger = structlog.get_logger()

router = APIRouter()

# In-memory job storage (replace with Redis in production)
job_storage = {}
processing_jobs = set()


class BatchJob:
    """Simple batch job representation."""
    
    def __init__(self, job_id: str, files: List[UploadFile], request: BatchOCRRequest):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.files = files
        self.request = request
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.total_images = len(files)
        self.processed_images = 0
        self.failed_images = 0
        self.results = []
        self.failed_files = []
        self.processing_start_time = None
        self.processing_end_time = None
        self.error_message = None
    
    def to_status_response(self) -> BatchJobStatus:
        """Convert job to status response."""
        progress_percentage = 0.0
        if self.total_images > 0:
            progress_percentage = ((self.processed_images + self.failed_images) / self.total_images) * 100
        
        estimated_completion = None
        if self.status == JobStatus.PROCESSING and self.processing_start_time:
            processed = self.processed_images + self.failed_images
            if processed > 0:
                elapsed = (datetime.now() - self.processing_start_time).total_seconds()
                avg_time_per_image = elapsed / processed
                remaining_images = self.total_images - processed
                estimated_seconds = remaining_images * avg_time_per_image
                estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return BatchJobStatus(
            job_id=self.job_id,
            status=self.status,
            total_images=self.total_images,
            processed_images=self.processed_images,
            failed_images=self.failed_images,
            progress_percentage=progress_percentage,
            created_at=self.created_at,
            updated_at=self.updated_at,
            estimated_completion=estimated_completion,
            processing_time=(datetime.now() - self.created_at).total_seconds()
            if self.processing_start_time else None
        )
    
    def to_result_response(self) -> BatchJobResult:
        """Convert job to result response."""
        return BatchJobResult(
            job_id=self.job_id,
            status=self.status,
            results=self.results,
            failed_images=self.failed_files,
            total_processing_time=(
                (self.processing_end_time - self.processing_start_time).total_seconds()
                if self.processing_start_time and self.processing_end_time else 0
            ),
            completed_at=datetime.now()
        )


async def process_batch_job(job: BatchJob):
    """Process a batch job in the background."""
    try:
        job.status = JobStatus.PROCESSING
        job.processing_start_time = datetime.now()
        job.updated_at = datetime.now()
        
        logger.info("Starting batch job processing", job_id=job.job_id, total_images=job.total_images)
        
        ocr_pipeline = get_ocr_pipeline()
        
        # Process each file
        for i, file in enumerate(job.files):
            try:
                # Read file content
                file_content = await file.read()
                
                # Validate file
                is_valid, error_msg, extension = validate_file_upload(file_content, file.filename)
                if not is_valid:
                    raise ValueError(f"File validation failed: {error_msg}")
                
                # Load image(s)
                images = load_image_from_bytes(file_content, extension)
                
                # Process first image (assuming single page for batch)
                image = images[0]
                
                # Override confidence threshold if specified
                original_threshold = settings.detection_confidence_threshold
                if job.request.confidence_threshold is not None:
                    settings.detection_confidence_threshold = job.request.confidence_threshold
                
                try:
                    # Process through OCR pipeline
                    result = ocr_pipeline.process_image(image, include_masks=job.request.include_masks)
                    
                    # Add file information to result
                    result["image_name"] = file.filename
                    result["image_index"] = i
                    
                    job.results.append({
                        "image_name": file.filename,
                        "image_index": i,
                        "ocr_result": result
                    })
                    
                    job.processed_images += 1
                    
                finally:
                    # Restore original threshold
                    if job.request.confidence_threshold is not None:
                        settings.detection_confidence_threshold = original_threshold
                
                # Cleanup memory
                cleanup_memory()
                
            except Exception as e:
                logger.error("Failed to process file in batch", 
                           job_id=job.job_id, 
                           filename=file.filename, 
                           error=str(e))
                
                job.failed_images += 1
                job.failed_files.append({
                    "image_name": file.filename,
                    "image_index": i,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
            
            # Update job progress
            job.updated_at = datetime.now()
        
        job.status = JobStatus.COMPLETED
        job.processing_end_time = datetime.now()
        
        logger.info("Batch job completed", 
                   job_id=job.job_id, 
                   processed=job.processed_images,
                   failed=job.failed_images)
        
    except Exception as e:
        logger.error("Batch job failed", job_id=job.job_id, error=str(e))
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.processing_end_time = datetime.now()
        job.updated_at = datetime.now()


@router.post(
    "/batch",
    response_model=BatchJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit batch OCR job",
    description="Submit multiple images for batch OCR processing."
)
async def submit_batch_job(
    request: Request,
    files: List[UploadFile] = File(..., description="Image files to process"),
    batch_request: BatchOCRRequest = Depends()
):
    """
    Submit multiple images for batch OCR processing.
    
    - **files**: List of image files (max 20 files)
    - **include_masks**: Whether to include segmentation masks in results
    - **confidence_threshold**: Override default confidence threshold
    - **priority**: Job priority (0-10, higher = more priority)
    
    Returns job ID for tracking progress.
    """
    # Validate file count
    if len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "NO_FILES",
                "message": "At least one file must be provided"
            }
        )
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TOO_MANY_FILES",
                "message": f"Maximum {settings.max_batch_size} files allowed, got {len(files)}"
            }
        )
    
    # Generate unique job ID
    job_id = f"batch_{uuid.uuid4().hex[:12]}"
    
    # Create batch job
    job = BatchJob(job_id, files, batch_request)
    job_storage[job_id] = job
    
    # Start processing in background (simplified - in production use Arq)
    import asyncio
    asyncio.create_task(process_batch_job(job))
    
    logger.info("Batch job submitted", job_id=job_id, file_count=len(files))
    
    return BatchJobResponse(
        job_id=job_id,
        status=job.status,
        total_images=job.total_images,
        submitted_at=job.created_at
    )


@router.get(
    "/batch/{job_id}",
    response_model=BatchJobStatus,
    status_code=status.HTTP_200_OK,
    summary="Get batch job status",
    description="Get the status and progress of a batch job."
)
async def get_batch_job_status(job_id: str):
    """
    Get the status and progress of a batch job.
    
    Args:
        job_id: Job identifier returned from batch submission
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' not found"
            }
        )
    
    job = job_storage[job_id]
    return job.to_status_response()


@router.get(
    "/batch/{job_id}/result",
    response_model=BatchJobResult,
    status_code=status.HTTP_200_OK,
    summary="Get batch job results",
    description="Get the results of a completed batch job."
)
async def get_batch_job_result(job_id: str):
    """
    Get the results of a completed batch job.
    
    Args:
        job_id: Job identifier returned from batch submission
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' not found"
            }
        )
    
    job = job_storage[job_id]
    
    if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "JOB_NOT_COMPLETED",
                "message": f"Job is not completed yet. Current status: {job.status.value}"
            }
        )
    
    return job.to_result_response()


@router.post(
    "/batch/{job_id}/cancel",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Cancel batch job",
    description="Cancel a pending or processing batch job."
)
async def cancel_batch_job(job_id: str):
    """
    Cancel a batch job.
    
    Args:
        job_id: Job identifier returned from batch submission
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' not found"
            }
        )
    
    job = job_storage[job_id]
    
    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "JOB_CANNOT_BE_CANCELLED",
                "message": f"Job cannot be cancelled. Current status: {job.status.value}"
            }
        )
    
    job.status = JobStatus.CANCELLED
    job.updated_at = datetime.now()
    
    logger.info("Batch job cancelled", job_id=job_id)
    
    return {
        "job_id": job_id,
        "status": job.status.value,
        "message": "Job cancelled successfully"
    }


@router.get(
    "/batch",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="List batch jobs",
    description="List all batch jobs with their current status."
)
async def list_batch_jobs():
    """List all batch jobs with their current status."""
    jobs_summary = []
    
    for job_id, job in job_storage.items():
        jobs_summary.append({
            "job_id": job_id,
            "status": job.status.value,
            "total_images": job.total_images,
            "processed_images": job.processed_images,
            "failed_images": job.failed_images,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat()
        })
    
    # Sort by creation time (newest first)
    jobs_summary.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "jobs": jobs_summary,
        "total_jobs": len(jobs_summary)
    }


@router.delete(
    "/batch/{job_id}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Delete batch job",
    description="Delete a completed or failed batch job and its results."
)
async def delete_batch_job(job_id: str):
    """
    Delete a batch job and its results.
    
    Args:
        job_id: Job identifier returned from batch submission
    """
    if job_id not in job_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job '{job_id}' not found"
            }
        )
    
    job = job_storage[job_id]
    
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "JOB_CANNOT_BE_DELETED",
                "message": "Cannot delete a job that is currently processing"
            }
        )
    
    del job_storage[job_id]
    
    logger.info("Batch job deleted", job_id=job_id)
    
    return {
        "job_id": job_id,
        "message": "Job deleted successfully"
    }
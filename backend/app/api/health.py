"""Health check endpoints for OCR service."""
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
import structlog
import psutil

from app.core.config import settings
from app.models.model_manager import check_models_health, get_models_stats
from app.core.device import get_performance_info, get_memory_info
from app.models.schemas import HealthCheckResponse

logger = structlog.get_logger()

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    description="Comprehensive health check including models, and system status."
)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns:
        HealthCheckResponse: Complete service health status
    """
    start_time = time.time()
    
    try:
        # Check model health
        models_health = check_models_health()
        

        
        # Get system information
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_info = {
                "cpu_percent": cpu_percent,
                "memory_total": memory.total,
                "memory_used": memory.used,
                "memory_percent": memory.percent,
                "disk_total": disk.total,
                "disk_used": disk.used,
                "disk_percent": (disk.used / disk.total) * 100,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            logger.warning("Failed to get system info", error=str(e))
            system_info = {"error": str(e)}
        
        # Get device and performance info
        try:
            performance_info = get_performance_info()
            memory_info = get_memory_info()
            device_info = {
                **performance_info,
                "memory": memory_info
            }
        except Exception as e:
            logger.warning("Failed to get device info", error=str(e))
            device_info = {"error": str(e)}
        
        # Determine overall status
        models_ok = all(models_health.values())
        
        overall_status = "healthy"
        if not models_ok:
            overall_status = "degraded"
        
        health_response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.now(),
            models=models_health,
            system={
                **system_info,
                "device": device_info,
                "check_duration": time.time() - start_time
            }
        )
        
        # Log health check result
        log_level = logger.info if overall_status == "healthy" else logger.warning
        log_level("Health check completed", 
                 status=overall_status,
                 models_healthy=models_ok,
                 )
        
        return health_response
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/health/live",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes readiness/liveness probes."
)
async def liveness_check():
    """Simple liveness check."""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time()  # Simplified for k8s probes
    }


@router.get(
    "/health/ready",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Check if service is ready to accept traffic."
)
async def readiness_check():
    """Readiness check - verify critical components are ready."""
    try:
        # Check models are loaded
        models_health = check_models_health()
        if not models_health:
            raise Exception("No models loaded")
        
       
        
        if all(models_health.values()):
            return {
                "status": "ready",
                "timestamp": datetime.now().isoformat(),
                "models_healthy": True,
            }
        else:
            raise Exception("Models not healthy")
            
    except Exception as e:
        logger.warning("Readiness check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@router.get(
    "/metrics",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Service metrics",
    description="Get service metrics for monitoring."
)
async def get_metrics():
    """Get service metrics."""
    try:
        # Get model statistics
        models_stats = get_models_stats()
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Calculate totals
        total_inferences = sum(
            stats.get("inference_count", 0) 
            for stats in models_stats.values()
        )
        
        total_avg_time = 0
        if models_stats:
            total_avg_time = sum(
                stats.get("avg_inference_time", 0) 
                for stats in models_stats.values()
            ) / len(models_stats)
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024)
            },
            "models": {
                "total_inferences": total_inferences,
                "average_inference_time": total_avg_time,
                "models_loaded": len(models_stats),
                "models": {
                    name: {
                        "inferences": stats.get("inference_count", 0),
                        "avg_time": stats.get("avg_inference_time", 0),
                        "healthy": stats.get("is_healthy", False)
                    }
                    for name, stats in models_stats.items()
                }
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
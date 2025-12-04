"""Model information endpoints for OCR service."""
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
import structlog

from app.models.model_manager import get_models_stats
from app.models.detection import get_detection_model
from app.models.recognition import get_recognition_model
from app.models.schemas import ModelsResponse, ModelInfo
from app.models import get_ocr_pipeline
logger = structlog.get_logger()

router = APIRouter()


@router.get(
    "/models",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get model information",
    description="Get detailed information about loaded OCR models."
)
async def get_models_info():
    """
    Get comprehensive information about all loaded models.
    
    Returns:
        ModelsResponse: Model information and statistics
    """
    try:
        # Get model statistics
        models_stats = get_models_stats()
        
        # Convert to ModelInfo format
        models_info = {}
        for model_name, stats in models_stats.items():
            models_info[model_name] = stats
        
        # Get character set info from recognition model
        try:
            recognition_model = get_recognition_model()
            character_set_info = recognition_model.get_character_info()
            models_info["recognition"]["character_set"] = character_set_info
        except Exception as e:
            logger.warning("Failed to get character set info", error=str(e))
            character_set_info = {}
        
        # response = ModelsResponse(
        #     models=models_info,
        #     character_set_info=character_set_info
        # )
        
        return models_info
        
    except Exception as e:
        logger.error("Failed to get models info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "message": "Failed to retrieve model information"
            }
        )


@router.get(
    "/models/{model_name}",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get specific model information",
    description="Get detailed information about a specific model."
)
async def get_specific_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model (detection or recognition)
    """
    try:
        if model_name == "detection":
            model = get_detection_model()
            info = model.get_model_info()
        elif model_name == "recognition":
            model = get_recognition_model()
            info = model.get_model_info()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "MODEL_NOT_FOUND",
                    "message": f"Model '{model_name}' not found. Available: detection, recognition"
                }
            )
        
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "MODEL_INFO_NOT_AVAILABLE",
                    "message": f"Information not available for model '{model_name}'"
                }
            )
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get model info", 
                    model_name=model_name, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "message": f"Failed to retrieve information for model '{model_name}'"
            }
        )


@router.get(
    "/models/{model_name}/health",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Check model health",
    description="Check the health status of a specific model."
)
async def check_model_health(model_name: str):
    """
    Check the health status of a specific model.
    
    Args:
        model_name: Name of the model (detection or recognition)
    """
    try:
        if model_name == "detection":
            model = get_detection_model()
            is_healthy = model.is_healthy()
        elif model_name == "recognition":
            model = get_recognition_model()
            is_healthy = model.is_healthy()
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "MODEL_NOT_FOUND",
                    "message": f"Model '{model_name}' not found. Available: detection, recognition"
                }
            )
        
        return {
            "model_name": model_name,
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "unhealthy"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to check model health", 
                    model_name=model_name, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "message": f"Failed to check health for model '{model_name}'"
            }
        )


@router.get(
    "/pipeline/stats",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get pipeline statistics",
    description="Get comprehensive statistics about the OCR pipeline."
)
async def get_pipeline_stats():
    """
    Get comprehensive statistics about the OCR pipeline.
    """
    try:
        ocr_pipeline = get_ocr_pipeline()
        stats = ocr_pipeline.get_pipeline_stats()
        
        return {
            "pipeline_stats": stats,
            "timestamp": "now"  # Could add actual timestamp
        }
        
    except Exception as e:
        logger.error("Failed to get pipeline stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "message": "Failed to retrieve pipeline statistics"
            }
        )


@router.get(
    "/pipeline/character-set",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Get character set information",
    description="Get information about the character set used for text recognition."
)
async def get_character_set_info():
    """
    Get information about the character set used for text recognition.
    """
    try:
        recognition_model = get_recognition_model()
        character_info = recognition_model.get_character_info()
        
        return {
            "character_set_info": character_info,
            "supported_scripts": ["Devanagari", "Latin numerals", "Punctuation"],
            "languages": ["Nepali", "Hindi", "Sanskrit"]
        }
        
    except Exception as e:
        logger.error("Failed to get character set info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "message": "Failed to retrieve character set information"
            }
        )
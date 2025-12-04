"""Model loading and caching for OCR models."""
import os
import time
import onnxruntime as ort
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging
from threading import Lock
from dataclasses import dataclass
from datetime import datetime

# DEBUG: Add detailed logging to diagnose import issues
import sys
# print(f"DEBUG: model_manager.py - Current working directory: {os.getcwd()}")
# print(f"DEBUG: model_manager.py - Python path: {sys.path}")
# print(f"DEBUG: model_manager.py - __name__: {__name__}")
# print(f"DEBUG: model_manager.py - __package__: {__package__}")

from datetime import datetime

from app.core.config import settings
from app.core.device import get_device_for_model


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    path: Path
    session: ort.InferenceSession
    input_name: str
    output_names: list
    loaded_at: datetime
    last_used: datetime
    inference_count: int = 0
    total_inference_time: float = 0.0
    is_healthy: bool = True


class ModelManager:
    """Manages loading, caching, and health monitoring of ONNX models."""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._locks: Dict[str, Lock] = {}
        self._health_check_interval = settings.health_check_interval
        self._last_health_check = time.time()
    
    def load_model(self, model_name: str, model_path: str, providers: Optional[list] = None) -> ModelInfo:
        """Load a model and cache it."""
        if model_name in self._models:
            model_info = self._models[model_name]
            model_info.last_used = datetime.now()
            logger.info(f"Model {model_name} already loaded, returning cached version")
            return model_info
        
        # Create lock for this model
        if model_name not in self._locks:
            self._locks[model_name] = Lock()
        
        # Double-check locking pattern for thread safety
        with self._locks[model_name]:
            if model_name in self._models:
                model_info = self._models[model_name]
                model_info.last_used = datetime.now()
                return model_info
            
            logger.info(f"Loading model {model_name} from {model_path}")
            
            # Validate model path
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            if not path.is_file():
                raise ValueError(f"Path is not a file: {model_path}")
            
            # Get optimal providers for the device
            if providers is None:
                available_providers = ort.get_available_providers()
                device_info = get_device_for_model(model_name)
                
                if device_info.device_type == "cuda" and "CUDAExecutionProvider" in available_providers:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
            
            try:
                # Load ONNX session with optimized options
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                session_options.enable_mem_pattern = True
                session_options.enable_cpu_mem_arena = True
                session_options.log_severity_level = 2  # Error level only
                
                session = ort.InferenceSession(
                    str(path),
                    providers=providers,
                    sess_options=session_options
                )
                
                # Get input and output information
                input_name = session.get_inputs()[0].name
                output_names = [output.name for output in session.get_outputs()]
                
                # Create model info
                model_info = ModelInfo(
                    name=model_name,
                    path=path,
                    session=session,
                    input_name=input_name,
                    output_names=output_names,
                    loaded_at=datetime.now(),
                    last_used=datetime.now()
                )
                
                self._models[model_name] = model_info
                
                logger.info(f"Successfully loaded model {model_name}")
                logger.info(f"Input: {input_name}")
                logger.info(f"Outputs: {output_names}")
                logger.info(f"Providers: {providers}")
                
                return model_info
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get a loaded model if available."""
        if model_name in self._models:
            model_info = self._models[model_name]
            model_info.last_used = datetime.now()
            return model_info
        return None
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self._models:
            try:
                del self._models[model_name]
                if model_name in self._locks:
                    del self._locks[model_name]
                
                logger.info(f"Unloaded model {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to unload model {model_name}: {str(e)}")
                return False
        return False
    
    def unload_all_models(self) -> Dict[str, bool]:
        """Unload all models."""
        results = {}
        for model_name in list(self._models.keys()):
            results[model_name] = self.unload_model(model_name)
        return results
    
    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all loaded models."""
        stats = {}
        for model_name, model_info in self._models.items():
            avg_inference_time = 0.0
            if model_info.inference_count > 0:
                avg_inference_time = model_info.total_inference_time / model_info.inference_count
            
            stats[model_name] = {
                "name": model_info.name,
                "path": str(model_info.path),
                "loaded_at": model_info.loaded_at.isoformat(),
                "last_used": model_info.last_used.isoformat(),
                "inference_count": model_info.inference_count,
                "avg_inference_time": avg_inference_time,
                "is_healthy": model_info.is_healthy,
                "input_name": model_info.input_name,
                "output_names": model_info.output_names
            }
        return stats
    
    def update_inference_stats(self, model_name: str, inference_time: float) -> None:
        """Update inference statistics for a model."""
        if model_name in self._models:
            model_info = self._models[model_name]
            model_info.inference_count += 1
            model_info.total_inference_time += inference_time
    
    def check_model_health(self, model_name: str) -> bool:
        """Check health of a specific model."""
        if model_name not in self._models:
            return False
        
        try:
            model_info = self._models[model_name]
            
            # Check if session is still valid
            if model_info.session is None:
                return False
            
            # Try to get input/output info
            _ = model_info.session.get_inputs()
            _ = model_info.session.get_outputs()
            
            model_info.is_healthy = True
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for model {model_name}: {str(e)}")
            if model_name in self._models:
                self._models[model_name].is_healthy = False
            return False
    
    def check_all_models_health(self) -> Dict[str, bool]:
        """Check health of all loaded models."""
        current_time = time.time()
        
        # Only check periodically to avoid overhead
        if current_time - self._last_health_check < self._health_check_interval:
            return {name: info.is_healthy for name, info in self._models.items()}
        
        self._last_health_check = current_time
        
        results = {}
        for model_name in self._models:
            results[model_name] = self.check_model_health(model_name)
        
        return results
    
    def get_loaded_models(self) -> list:
        """Get list of loaded model names."""
        return list(self._models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model."""
        if model_name not in self._models:
            return None
        
        model_info = self._models[model_name]
        
        # Get input/output shape information
        try:
            inputs = model_info.session.get_inputs()
            outputs = model_info.session.get_outputs()
            
            input_info = []
            for inp in inputs:
                input_info.append({
                    "name": inp.name,
                    "shape": inp.shape,
                    "type": inp.type
                })
            
            output_info = []
            for out in outputs:
                output_info.append({
                    "name": out.name,
                    "shape": out.shape,
                    "type": out.type
                })
            
            return {
                "name": model_info.name,
                "path": str(model_info.path),
                "loaded_at": model_info.loaded_at.isoformat(),
                "last_used": model_info.last_used.isoformat(),
                "inference_count": model_info.inference_count,
                "avg_inference_time": model_info.total_inference_time / max(model_info.inference_count, 1),
                "is_healthy": model_info.is_healthy,
                "inputs": input_info,
                "outputs": output_info,
                "providers": model_info.session.get_providers()
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {str(e)}")
            return None


# Global model manager instance
model_manager = ModelManager()


# Convenience functions
def load_detection_model() -> ModelInfo:
    """Load the detection model."""
    return model_manager.load_model(
        "detection",
        str(settings.get_detection_model_path())
    )


def load_recognition_model() -> ModelInfo:
    """Load the recognition model."""
    return model_manager.load_model(
        "recognition",
        str(settings.get_recognition_model_path())
    )


def get_detection_model() -> Optional[ModelInfo]:
    """Get the loaded detection model."""
    return model_manager.get_model("detection")


def get_recognition_model() -> Optional[ModelInfo]:
    """Get the loaded recognition model."""
    return model_manager.get_model("recognition")


def check_models_health() -> Dict[str, bool]:
    """Check health of all models."""
    return model_manager.check_all_models_health()


def get_models_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all models."""
    return model_manager.get_model_stats()
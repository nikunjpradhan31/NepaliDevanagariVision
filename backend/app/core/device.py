"""Device management utilities for OCR models."""
import platform
import psutil
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from .config import settings


@dataclass
class DeviceInfo:
    """Information about the compute device."""
    device_type: str  # 'cpu', 'cuda'
    device_name: str
    memory_total: int = 0  # in bytes
    memory_available: int = 0  # in bytes
    compute_capability: str = "N/A"


class DeviceManager:
    """Manages device selection and optimization for OCR models."""
    
    def __init__(self):
        self._cpu_info = self._get_cpu_info()
        self._cuda_info = None  # CUDA support removed
        
    def get_optimal_device(self) -> DeviceInfo:
        """Get the optimal device for model inference."""
        # Always return CPU device since CUDA support is removed
        return self._cpu_info
    
    def get_device_for_model(self, model_name: str) -> DeviceInfo:
        """Get device optimized for specific model."""
        # For now, use the same device for both models
        # In the future, this could be optimized per model
        return self.get_optimal_device()
    
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return False  # CUDA support removed
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the current device."""
        # Always return CPU memory info since CUDA support is removed
        return {
            "device": "cpu",
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "memory_used": psutil.virtual_memory().used,
            "memory_percent": psutil.virtual_memory().percent
        }
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information for the current device."""
        device = self.get_optimal_device()
        
        return {
            "device_type": device.device_type,
            "device_name": device.device_name,
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total": self.get_memory_info()["memory_total"],
        }
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information."""
        return DeviceInfo(
            device_type="cpu",
            device_name=f"{platform.processor()} ({psutil.cpu_count()} cores)",
            memory_total=psutil.virtual_memory().total,
            memory_available=psutil.virtual_memory().available
        )
    
    def optimize_for_inference(self) -> None:
        """Optimize settings for inference."""
        # No CUDA optimizations available since CUDA support is removed
        pass
    
    def cleanup_memory(self) -> None:
        """Cleanup memory for the current device."""
        # Force garbage collection (no CUDA cleanup needed)
        import gc
        gc.collect()


# Global device manager instance
device_manager = DeviceManager()


def get_device() -> DeviceInfo:
    """Get the optimal device for model inference."""
    return device_manager.get_optimal_device()


def get_device_for_model(model_name: str) -> DeviceInfo:
    """Get device optimized for specific model."""
    return device_manager.get_device_for_model(model_name)


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    return device_manager.is_cuda_available()


def get_memory_info() -> Dict[str, Any]:
    """Get memory information for the current device."""
    return device_manager.get_memory_info()


def get_performance_info() -> Dict[str, Any]:
    """Get performance information for the current device."""
    return device_manager.get_performance_info()


def optimize_for_inference() -> None:
    """Optimize settings for inference."""
    device_manager.optimize_for_inference()


def cleanup_memory() -> None:
    """Cleanup memory for the current device."""
    device_manager.cleanup_memory()
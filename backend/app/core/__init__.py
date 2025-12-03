"""Core package for OCR FastAPI service."""

# Import from config
from .config import Settings, settings

# Import from device
from .device import (
    DeviceInfo,
    DeviceManager,
    get_device,
    get_device_for_model,
    is_cuda_available,
    get_memory_info,
    get_performance_info,
    optimize_for_inference,
    cleanup_memory
)

# Import from image_utils
from .image_utils import (
    validate_file_upload,
    get_file_extension,
    load_image_from_bytes,
    convert_pdf_to_images,
    create_temp_image_file,
    cleanup_temp_file,
    cleanup_temp_files,
    resize_image_for_display,
    image_to_base64,
    base64_to_image,
    get_image_info,
    validate_image_quality
)

__all__ = [
    # config
    "Settings",
    "settings",
    # device
    "DeviceInfo",
    "DeviceManager", 
    "get_device",
    "get_device_for_model",
    "is_cuda_available",
    "get_memory_info",
    "get_performance_info",
    "optimize_for_inference",
    "cleanup_memory",
    # image_utils
    "validate_file_upload",
    "get_file_extension",
    "load_image_from_bytes",
    "convert_pdf_to_images",
    "create_temp_image_file",
    "cleanup_temp_file",
    "cleanup_temp_files",
    "resize_image_for_display",
    "image_to_base64",
    "base64_to_image",
    "get_image_info",
    "validate_image_quality"
]
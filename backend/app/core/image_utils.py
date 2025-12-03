"""Image processing utilities for OCR service."""
import os
import tempfile
import magic
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from PIL import Image
from pdf2image import convert_from_path
import numpy as np

from .config import settings

logger = logging.getLogger(__name__)


def validate_file_upload(file_content: bytes, filename: str) -> Tuple[bool, str, str]:
    """
    Validate uploaded file for OCR processing.
    
    Args:
        file_content: Raw file content
        filename: Original filename
        
    Returns:
        Tuple of (is_valid, error_message, detected_extension)
    """
    try:
        # Check file size
        if len(file_content) > settings.max_file_size:
            return False, f"File too large: {len(file_content)} bytes (max: {settings.max_file_size})", ""
        
        # Check magic bytes to detect actual file type
        try:
            mime_type = magic.from_buffer(file_content, mime=True)
        except Exception:
            # Fallback to file extension if magic detection fails
            mime_type = ""
        
        # Determine file extension from content or filename
        extension = get_file_extension(file_content, filename, mime_type)
        
        # Check if extension is allowed
        allowed_extensions = settings.get_allowed_extensions()
        if extension.lower() not in allowed_extensions:
            return False, f"Unsupported file format: .{extension}. Allowed: {', '.join(allowed_extensions)}", extension
        
        return True, "", extension
        
    except Exception as e:
        logger.error(f"Error validating file {filename}: {str(e)}")
        return False, f"Error validating file: {str(e)}", ""


def get_file_extension(file_content: bytes, filename: str, mime_type: str = "") -> str:
    """Get file extension from content, filename, or MIME type."""
    # First try to get from filename
    if filename:
        path = Path(filename)
        if path.suffix:
            return path.suffix.lower().lstrip(".")
    
    # Try to detect from MIME type
    mime_to_ext = {
        "image/jpeg": "jpg",
        "image/png": "png", 
        "image/tiff": "tiff",
        "image/bmp": "bmp",
        "application/pdf": "pdf"
    }
    
    if mime_type in mime_to_ext:
        return mime_to_ext[mime_type]
    
    # Try to detect from magic bytes
    if file_content.startswith(b'\xff\xd8\xff'):
        return "jpg"  # JPEG
    elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):
        return "png"  # PNG
    elif file_content.startswith(b'II*\x00') or file_content.startswith(b'MM\x00*'):
        return "tiff"  # TIFF
    elif file_content.startswith(b'%PDF'):
        return "pdf"  # PDF
    else:
        return "unknown"


def load_image_from_bytes(file_content: bytes, extension: str) -> List[Image.Image]:
    """
    Load image(s) from file content.
    
    Args:
        file_content: Raw file content
        extension: File extension
        
    Returns:
        List of PIL Images (multiple for multi-page documents)
    """
    try:
        if extension.lower() == "pdf":
            return convert_pdf_to_images(file_content)
        else:
            image = Image.open(io.BytesIO(file_content))
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return [image]
            
    except Exception as e:
        logger.error(f"Error loading image: {str(e)}")
        raise ValueError(f"Could not load image: {str(e)}")


def convert_pdf_to_images(file_content: bytes) -> List[Image.Image]:
    """
    Convert PDF content to images.
    
    Args:
        file_content: Raw PDF content
        
    Returns:
        List of PIL Images (one per page)
    """
    try:
        # Create temporary file for PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_content)
            temp_pdf_path = temp_pdf.name
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                temp_pdf_path,
                dpi=300,  # High DPI for better OCR quality
                first_page=1,
                last_page=None  # All pages
            )
            
            # Convert to RGB if necessary
            rgb_images = []
            for image in images:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                rgb_images.append(image)
            
            logger.info(f"Converted PDF to {len(rgb_images)} images")
            return rgb_images
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary PDF file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error converting PDF to images: {str(e)}")
        raise ValueError(f"Could not convert PDF: {str(e)}")


def create_temp_image_file(image: Image.Image) -> str:
    """
    Create a temporary image file.
    
    Args:
        image: PIL Image to save
        
    Returns:
        Path to temporary image file
    """
    temp_dir = settings.temp_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg",
        dir=temp_dir
    )
    
    image.save(temp_file.name, "JPEG", quality=95)
    return temp_file.name


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file.
    
    Args:
        file_path: Path to file to delete
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not delete temporary file {file_path}: {str(e)}")


def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    Clean up multiple temporary files.
    
    Args:
        file_paths: List of file paths to delete
    """
    for file_path in file_paths:
        cleanup_temp_file(file_path)


def resize_image_for_display(image: Image.Image, max_size: Tuple[int, int] = (800, 600)) -> Image.Image:
    """
    Resize image for display purposes while maintaining aspect ratio.
    
    Args:
        image: PIL Image to resize
        max_size: Maximum size (width, height)
        
    Returns:
        Resized PIL Image
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 95) -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image to convert
        format: Output format (JPEG, PNG)
        quality: JPEG quality (1-100)
        
    Returns:
        Base64 encoded image string
    """
    import io
    import base64
    
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=quality)
    image_data = buffer.getvalue()
    
    return base64.b64encode(image_data).decode()


def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image
    """
    import io
    import base64
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def get_image_info(image: Image.Image) -> dict:
    """
    Get information about an image.
    
    Args:
        image: PIL Image to analyze
        
    Returns:
        Dictionary containing image information
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
        "size_bytes": image.width * image.height * (len(image.getbands())),
        "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
    }


def validate_image_quality(image: Image.Image) -> Tuple[bool, List[str]]:
    """
    Validate image quality for OCR processing.
    
    Args:
        image: PIL Image to validate
        
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    # Check resolution
    if image.width < 300 or image.height < 300:
        warnings.append("Low resolution image, OCR quality may be poor")
    
    # Check aspect ratio
    aspect_ratio = image.width / image.height if image.height > 0 else 0
    if aspect_ratio > 10 or aspect_ratio < 0.1:
        warnings.append("Unusual aspect ratio, image may be rotated or skewed")
    
    # Check for very dark or very bright images
    if image.mode == 'RGB':
        grayscale = image.convert('L')
        np_image = np.array(grayscale)
        mean_brightness = np.mean(np_image)
        
        if mean_brightness < 50:
            warnings.append("Image appears to be too dark")
        elif mean_brightness > 200:
            warnings.append("Image appears to be too bright")
    
    # Check for very small text (rough estimate)
    if image.width * image.height < 50000:  # Very small image
        warnings.append("Image is very small, text may be too small to read")
    
    return len(warnings) == 0, warnings


# Import io module for the functions that need it
import io
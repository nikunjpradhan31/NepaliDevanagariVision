"""Unified OCR pipeline combining detection and recognition models."""
import time
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import logging

from .detection import get_detection_model
from .recognition import get_recognition_model
from ..core.config import settings

logger = logging.getLogger(__name__)


class OCRPipeline:
    """Complete OCR pipeline combining line detection and text recognition."""
    
    def __init__(self):
        self.detection_model = get_detection_model()
        self.recognition_model = get_recognition_model()
    
    def process_image(self, image: Image.Image, include_masks: bool = True) -> Dict[str, Any]:
        """
        Process an image through the complete OCR pipeline.
        
        Args:
            image: PIL Image to process
            include_masks: Whether to include mask data in response
            
        Returns:
            Dictionary containing detections and recognized text
        """
        start_time = time.time()
        
        try:
            # Step 1: Line detection
            logger.info("Starting line detection")
            detections = self.detection_model.detect_lines(image)
            
            if not detections:
                logger.warning("No text lines detected in image")
                return {
                    "image_width": image.width,
                    "image_height": image.height,
                    "detections": [],
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: Get crops for each detection
            logger.info(f"Extracting {len(detections)} text line crops")
            crops = self.detection_model.get_detection_crops(image, detections)
            
            # Step 3: Recognize text from each crop
            logger.info("Starting text recognition")
            crop_images = [crop[0] for crop in crops]
            recognized_texts = self.recognition_model.recognize_batch(crop_images)
            
            # Step 4: Combine results
            final_detections = []
            for i, ((crop_image, detection_info), text) in enumerate(zip(crops, recognized_texts)):
                result = {
                    "line_id": i,
                    "box": detection_info["box"],
                    "crop_box": detection_info["crop_box"],
                    "confidence": detection_info["score"],
                    "text": text,
                    "class": detection_info["class"]
                }
                
                # Include mask if requested
                if include_masks:
                    mask = detection_info["mask"]
                    # Convert boolean mask to base64 encoded image
                    mask_img = Image.fromarray((mask * 255).astype('uint8'), mode='L')
                    buffer = BytesIO()
                    mask_img.save(buffer, format='PNG')
                    result["mask_base64"] = base64.b64encode(buffer.getvalue()).decode()
                
                final_detections.append(result)
            
            # Step 5: Compile final response
            processing_time = time.time() - start_time
            
            response = {
                "image_width": image.width,
                "image_height": image.height,
                "detections": final_detections,
                "total_lines": len(final_detections),
                "processing_time": processing_time,
                "models_used": {
                    "detection": "LineDetectionv4",
                    "recognition": "ResNetBiLSTMCTCv1"
                }
            }
            
            logger.info(f"OCR pipeline completed in {processing_time:.3f}s, "
                       f"detected {len(final_detections)} lines")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in OCR pipeline: {str(e)}")
            raise
    
    def process_single_line(self, image: Image.Image) -> str:
        """
        Process a single cropped text line image directly.
        
        Args:
            image: PIL Image containing a single text line
            
        Returns:
            Recognized text string
        """
        try:
            text = self.recognition_model.recognize_text(image)
            logger.debug(f"Single line recognition: '{text}'")
            return text
        except Exception as e:
            logger.error(f"Error in single line recognition: {str(e)}")
            raise
    
    def detect_only(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Perform only line detection without recognition.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of detection results without text recognition
        """
        try:
            detections = self.detection_model.detect_lines(image)
            crops = self.detection_model.get_detection_crops(image, detections)
            
            results = []
            for i, (crop_image, detection_info) in enumerate(crops):
                result = {
                    "line_id": i,
                    "box": detection_info["box"],
                    "crop_box": detection_info["crop_box"],
                    "confidence": detection_info["score"],
                    "class": detection_info["class"]
                }
                results.append(result)
            
            logger.info(f"Detection only completed, found {len(results)} lines")
            return results
            
        except Exception as e:
            logger.error(f"Error in detection-only pipeline: {str(e)}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the OCR pipeline."""
        try:
            detection_stats = self.detection_model.get_model_info() or {}
            recognition_stats = self.recognition_model.get_model_info() or {}
            
            return {
                "detection_model": {
                    "name": "LineDetectionv4",
                    "healthy": self.detection_model.is_healthy(),
                    "stats": detection_stats
                },
                "recognition_model": {
                    "name": "ResNetBiLSTMCTCv1", 
                    "healthy": self.recognition_model.is_healthy(),
                    "stats": recognition_stats
                },
                "configuration": {
                    "confidence_threshold": settings.detection_confidence_threshold,
                    "crop_padding_x": settings.crop_padding_x,
                    "crop_padding_y": settings.crop_padding_y,
                    "input_sizes": {
                        "detection": settings.detection_input_size,
                        "recognition": settings.recognition_input_size
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {str(e)}")
            return {}
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, str]:
        """
        Validate if an image is suitable for OCR processing.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if image is valid
            if image is None:
                return False, "Image is None"
            
            # Check image dimensions
            if image.width == 0 or image.height == 0:
                return False, f"Invalid image dimensions: {image.width}x{image.height}"
            
            # Check minimum size
            if image.width < 50 or image.height < 50:
                return False, f"Image too small: {image.width}x{image.height} (minimum 50x50)"
            
            # Check maximum size (prevent memory issues)
            max_pixels = 10000 * 10000  # 100MP
            if image.width * image.height > max_pixels:
                return False, f"Image too large: {image.width}x{image.height}"
            
            # Check image mode
            if image.mode not in ['RGB', 'L', 'RGBA']:
                return False, f"Unsupported image mode: {image.mode}"
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating image: {str(e)}"
    
    def get_character_set_info(self) -> Dict[str, Any]:
        """Get information about the character set used for recognition."""
        return self.recognition_model.get_character_info()


# Global pipeline instance
_ocr_pipeline = None


def get_ocr_pipeline() -> OCRPipeline:
    """Get or create the global OCR pipeline instance."""
    global _ocr_pipeline
    if _ocr_pipeline is None:
        _ocr_pipeline = OCRPipeline()
    return _ocr_pipeline


def validate_image_for_ocr(image: Image.Image) -> Tuple[bool, str]:
    """Convenience function to validate image for OCR."""
    pipeline = get_ocr_pipeline()
    return pipeline.validate_image(image)
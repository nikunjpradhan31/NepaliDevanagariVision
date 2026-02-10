"""Line detection model wrapper for OCR pipeline."""
import time
import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
import logging

from .model_manager import get_detection_model, model_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class LineDetectionModel:
    """Wrapper for YOLOv8-based line detection model."""
    
    def __init__(self):
        self.confidence_threshold = settings.detection_confidence_threshold
        self.input_size = settings.detection_input_size
        self.model_name = settings.detection_model_data["model_name"]
        self.model_type = "detection"
        # Load the model
        self.model_info = model_manager.load_model(
            self.model_name,
            str(settings.get_detection_model_path()),self.model_type
        )
    
    def reload(self) -> None:
        """Reload the detection model."""
        try:
            logger.info("Reloading detection model")
            self.model_name = settings.detection_model_data["model_name"]
            self.model_info = model_manager.load_model(
                self.model_name,
                str(settings.get_detection_model_path()), self.model_type,force_reload=True
            )
            logger.info("Detection model reloaded successfully")
        except Exception as e:
            logger.error(f"Error reloading detection model: {str(e)}")
            raise

    def letterbox(self, img: Image.Image, new_shape: Tuple[int, int] = None) -> Tuple[Image.Image, int, int, int, int, float]:
        """
        Resize image while maintaining aspect ratio using letterboxing.
        
        Args:
            img: PIL Image to resize
            new_shape: Target size (width, height)
            
        Returns:
            Tuple of (resized_image, new_width, new_height, pad_x, pad_y, scale)
        """
        if new_shape is None:
            new_shape = self.input_size
        
        orig_w, orig_h = img.size
        r = min(new_shape[1] / orig_h, new_shape[0] / orig_w)
        new_unpad = int(orig_w * r), int(orig_h * r)
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]
        dw /= 2
        dh /= 2

        # resize
        img_resized = img.resize(new_unpad, Image.BILINEAR)
        # create padded image
        new_img = Image.new("RGB", new_shape, (114, 114, 114))
        new_img.paste(img_resized, (int(dw), int(dh)))
        return new_img, new_unpad[0], new_unpad[1], int(dw), int(dh), r
    
    def post_process_segmentation(self, pred: np.ndarray, proto: np.ndarray,
                                mask_threshold: float, pad_x: int, pad_y: int,
                                orig_w: int, orig_h: int, new_w: int, new_h: int) -> List[Dict[str, Any]]:
        """
        Convert ONNX outputs into masks + boxes in ORIGINAL IMAGE coordinates.
        
        Args:
            pred: Model predictions (300, 38)
            proto: Prototype masks (32, 256, 256)
            mask_threshold: Threshold for mask generation
            pad_x, pad_y: Padding applied during preprocessing
            orig_w, orig_h: Original image dimensions
            new_w, new_h: Preprocessed image dimensions
            
        Returns:
            List of detection dictionaries
        """
        pred = pred[0]  # (300, 38)
        proto = proto[0]  # (32, 256, 256)

        # remove zero rows (confidence = 0)
        pred = pred[pred[:, 4] > 0]

        if len(pred) == 0:
            return []

        boxes = pred[:, 0:4]  # x1,y1,x2,y2 in letterboxed coordinates
        scores = pred[:, 4]
        class_ids = pred[:, 5].astype(int)
        coeffs = pred[:, 6:]  # (N, 32)

        # compute prototype masks (256×256 → 1024×1024 → unpad → orig)
        masks = []
        for c in coeffs:
            m = np.tensordot(c, proto.reshape(32, -1), axes=1)
            m = m.reshape(256, 256)
            m = 1 / (1 + np.exp(-m))  # sigmoid
            m = cv2.resize(m, (new_w, new_h))

            # unpad
            m = m[pad_y:pad_y+new_h, pad_x:pad_x+new_w]

            # resize to original
            m = cv2.resize(m, (orig_w, orig_h))
            masks.append(m > mask_threshold)

        # convert boxes to original-image coordinates
        final_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b

            # remove padding
            x1 -= pad_x
            x2 -= pad_x
            y1 -= pad_y
            y2 -= pad_y

            # scale to original image
            x1 = x1 * (orig_w / new_w)
            x2 = x2 * (orig_w / new_w)
            y1 = y1 * (orig_h / new_h)
            y2 = y2 * (orig_h / new_h)

            final_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        detections = []
        for box, score, cls, mask in zip(final_boxes, scores, class_ids, masks):
            detections.append({
                "box": box,  # [x1,y1,x2,y2] in ORIGINAL coordinates
                "score": float(score),
                "class": int(cls),
                "mask": mask  # binary mask in ORIGINAL resolution (orig_h×orig_w)
            })

        return detections
    
    def detect_lines(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect text lines in an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List of detection dictionaries with box, score, class, and mask
        """
        start_time = time.time()
        
        try:
            # Store original dimensions
            orig_w, orig_h = image.size
            
            # Preprocess image
            padded, new_w, new_h, pad_x, pad_y, scale = self.letterbox(image, self.input_size)
            
            # Convert to numpy and normalize
            image_array = np.array(padded).astype(np.float32) / 255.0
            image_array = image_array.transpose(2, 0, 1)[None, ...]  # Add batch dimension
            
            # Run inference
            outputs = self.model_info.session.run(
                self.model_info.output_names,
                {self.model_info.input_name: image_array}
            )
            
            pred, proto = outputs[0], outputs[1]
            
            # Post-process results
            detections = self.post_process_segmentation(
                pred, proto, self.confidence_threshold,
                pad_x, pad_y, orig_w, orig_h, new_w, new_h
            )
            
            # Update inference statistics
            inference_time = time.time() - start_time
            model_manager.update_inference_stats(self.model_name, inference_time)
            
            logger.debug(f"Line detection completed in {inference_time:.3f}s, found {len(detections)} lines")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during line detection: {str(e)}")
            raise
    
    def get_detection_crops(self, image: Image.Image, detections: List[Dict[str, Any]]) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Extract cropped regions for detected text lines.
        
        Args:
            image: Original PIL Image
            detections: List of detection results from detect_lines
            
        Returns:
            List of tuples (cropped_image, detection_info)
        """
        crops = []
        orig_w, orig_h = image.size
        
        for i, detection in enumerate(detections):
            box = detection["box"]
            x1, y1, x2, y2 = box
            
            # Apply padding as per configuration
            x1 = max(0, x1 - settings.crop_padding_x)
            y1 = max(0, y1 - settings.crop_padding_y)
            x2 = min(orig_w, x2 + settings.crop_padding_x)
            y2 = min(orig_h, y2 + settings.crop_padding_y)
            
            # Crop the image
            crop = image.crop((x1, y1, x2, y2))
            
            # Add crop info to detection
            crop_info = detection.copy()
            crop_info["crop_box"] = [x1, y1, x2, y2]
            crop_info["line_id"] = i
            
            crops.append((crop, crop_info))
        
        return crops
    
    def is_healthy(self) -> bool:
        """Check if the detection model is healthy."""
        return model_manager.check_model_health(self.model_name)
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed model information."""
        return model_manager.get_model_info(self.model_name)


# Global instance
_detection_model = None


def get_detection_model() -> LineDetectionModel:
    """Get or create the global detection model instance."""
    global _detection_model
    if _detection_model is None:
        _detection_model = LineDetectionModel()
    return _detection_model
"""Text recognition model wrapper for OCR pipeline."""
import time
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any
import logging

from .model_manager import get_recognition_model, model_manager
from app.core.config import settings

logger = logging.getLogger(__name__)


class CTCLabelConverter:
    """Convert between text-label and text-index for CTC decoder."""
    
    def __init__(self, character: str):
        self.character = character
        dict_character = list(character)
        self.dict = {}
        
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1
        
        self.character_list = ['[blank]'] + dict_character
        self.ignore_idx = [0]  # Ignore blank token
    
    def encode(self, text: str, batch_max_length: int = 25) -> tuple:
        """convert text-label into text-index."""
        length = [len(text)]
        text = ''.join(text)
        text = [self.dict[char] for char in text]
        return np.array(text, dtype=np.int32), np.array(length, dtype=np.int32)
    
    def decode_greedy(self, text_index: np.ndarray, length: np.ndarray) -> List[str]:
        """convert text-index into text-label using greedy decoding."""
        texts = []
        index = 0
        
        for l in length:
            t = text_index[index:index + l]
            char_list = []
            
            for i in range(l):
                if t[i] not in self.ignore_idx and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.character_list[t[i]])
            
            text = ''.join(char_list)
            texts.append(text)
            index += l
        
        return texts

class AttnLabelConverter:
    """ Convert between text-label and text-index """

    def __init__(self, character: str):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character
        self.character_list = self.character
        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """
        Convert text-label into text-index using NumPy only.

        input:
            text: list of strings, length = batch_size
            batch_max_length: max length of text label in the batch (default 25)

        output:
            batch_text: np.ndarray of shape [batch_size, max_length + 2]
                        index 0 is [GO], padded with [GO] after [s]
            length: np.ndarray of shape [batch_size], includes [s]
        """
        length = np.array([len(s) + 1 for s in text], dtype=np.int32)  # +1 for [s]

        batch_max_length += 1  # +1 for [s]
        batch_size = len(text)

        # Initialize with [GO] token index (assumed to be 0, same as original)
        batch_text = np.zeros((batch_size, batch_max_length + 1), dtype=np.int64)

        for i, t in enumerate(text):
            chars = list(t)
            chars.append('[s]')
            indices = [self.dict[char] for char in chars]
            batch_text[i, 1:1 + len(indices)] = np.array(indices, dtype=np.int64)

        return batch_text, length

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character_list[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
    
class TextRecognitionModel:
    """Wrapper for CRNN-based text recognition model."""
    
    def __init__(self):
        self.input_size = settings.recognition_input_size
        self.character_set = settings.recognition_character_set
        self.model_name = settings.recognition_model_data["model_name"]
        self.batch_max_length = 200
        self.model_type = "recognition"
        # Initialize label converter
        self.label_converter = CTCLabelConverter(self.character_set) if settings.recognition_model_data["decoder"] == "CTC" else AttnLabelConverter(self.character_set)
        
        # Load the model
        self.model_info = model_manager.load_model(
            self.model_name,
            str(settings.get_recognition_model_path()),
            self.model_type
        )
    
    def reload(self) -> None:
        logger.info("Reloading recognition model")
        self.model_name = settings.recognition_model_data["model_name"]
        self.model_info = model_manager.load_model(
            self.model_name,
            str(settings.get_recognition_model_path()),
            self.model_type,
            force_reload=True
        )

        self.label_converter = (
            CTCLabelConverter(self.character_set)
            if settings.recognition_model_data["decoder"] == "CTC"
            else AttnLabelConverter(self.character_set)
        )

        logger.info("Recognition model reloaded successfully")

    def center_and_resize_image(self, img: Image.Image, target_size: tuple = None) -> Image.Image:
        """
        Resize the image to fit inside target_size while maintaining aspect ratio.
        If the image is smaller, center it on a black background.
        
        Args:
            img: PIL Image to resize
            target_size: Target size (width, height)
            
        Returns:
            Resized and centered PIL Image
        """
        if target_size is None:
            target_size = self.input_size
        
        target_w, target_h = target_size
        
        if img.width > target_w or img.height > target_h:
            img.thumbnail((target_w, target_h), Image.LANCZOS)
        
        new_img = Image.new("RGB", (target_w, target_h), color="black")
        paste_x = (target_w - img.width) // 2
        paste_y = (target_h - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img
    
    def preprocess_image(self, image: Image.Image, return_tensor: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline for the text recognition model.
        
        Args:
            image: PIL Image to preprocess
            return_tensor: Whether to return numpy array or PIL Image
            
        Returns:
            Preprocessed array ready for inference
        """
        # Step 1: Center and resize
        processed_pil = self.center_and_resize_image(image, self.input_size)
        
        if not return_tensor:
            return np.array(processed_pil)
        
        # Step 2: Convert to numpy and normalize
        image_np = np.array(processed_pil)
        image_np = image_np.astype(np.float32) / 255.0
        
        # Step 3: Apply ImageNet normalization
        # Mean and std for ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        
        # Step 4: Convert to tensor-like format (HWC to CHW) and add batch dimension
        image_np = np.transpose(image_np, (2, 0, 1))  # HWC to CHW
        image_np = np.expand_dims(image_np, axis=0)   # Add batch dimension
        
        return image_np
    
    def recognize_text(self, image: Image.Image) -> str:
        """
        Recognize text from a cropped text line image.
        
        Args:
            image: PIL Image containing a single text line
            
        Returns:
            Recognized text string
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess image
            image_np = self.preprocess_image(image)
            if image_np.dtype != np.float32:
                image_np = image_np.astype(np.float32)            
            
            outputs = self.model_info.session.run(
                self.model_info.output_names,
                {self.model_info.input_name: image_np}
            )
            
            preds = outputs[0]  # [batch, seq_len, num_classes]
            
            # Apply CTC decoding for single image
            batch_size = image_np.shape[0]
            preds_size = np.array([preds.shape[1]] * batch_size, dtype=np.int32)
            preds_index = preds.argmax(axis=2).flatten()
            
            # Decode using CTC greedy decoder
            preds_str = self.label_converter.decode_greedy(preds_index, preds_size)
            result = preds_str[0] if preds_str else ""


            inference_time = time.time() - start_time
            model_manager.update_inference_stats(self.model_name, inference_time)
            logger.debug(f"Text recognition completed in {inference_time:.3f}s: '{result}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during text recognition: {str(e)}")
            raise
    
    def recognize_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Recognize text from multiple images.
        
        Args:
            images: List of PIL Images containing text lines
            
        Returns:
            List of recognized text strings
        """
        start_time = time.time()
        
        try:
            if not images:
                return []
            
            # Preprocess all images
            processed_arrays = []
            for image in images:
                array = self.preprocess_image(image)
                processed_arrays.append(array)
            
            # Since ONNX model can only handle single images, process each image individually
            results = []
            for i, (image, image_np) in enumerate(zip(images, processed_arrays)):
                try:
                    # Ensure data type is float32 for ONNX compatibility
                    if image_np.dtype != np.float32:
                        image_np = image_np.astype(np.float32)
                    
                    logger.debug(f"Processing image {i+1}/{len(images)} - shape: {image_np.shape}, dtype: {image_np.dtype}")
                    
                    # Run ONNX inference on single image
                    outputs = self.model_info.session.run(
                        self.model_info.output_names,
                        {self.model_info.input_name: image_np}
                    )
                    
                    preds = outputs[0]  # [batch, seq_len, num_classes]
                    
                    # Apply CTC decoding for single image
                    batch_size = image_np.shape[0]
                    preds_size = np.array([preds.shape[1]] * batch_size, dtype=np.int32)
                    preds_index = preds.argmax(axis=2).flatten()
                    
                    # Decode using CTC greedy decoder
                    preds_str = self.label_converter.decode_greedy(preds_index, preds_size)
                    result = preds_str[0] if preds_str else ""
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing image {i+1}/{len(images)}: {str(e)}")
                    results.append("")  # Append empty string for failed recognition
            
            # Update inference statistics
            inference_time = time.time() - start_time
            model_manager.update_inference_stats(self.model_name, inference_time)
            
            logger.info(f"Sequential text recognition completed in {inference_time:.3f}s for {len(images)} images")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during batch text recognition: {str(e)}")
            raise
    
    def get_character_info(self) -> Dict[str, Any]:
        """Get information about the character set used by the model."""
        return {
            "character_set": self.character_set,
            "num_characters": len(self.character_set),
            "blank_token": "[blank]",
            "dictionary_size": len(self.label_converter.dict),
            "character_list_length": len(self.label_converter.character_list)
        }
    
    def is_healthy(self) -> bool:
        """Check if the recognition model is healthy."""
        return model_manager.check_model_health(self.model_name)
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get detailed model information."""
        info = model_manager.get_model_info(self.model_name)
        if info:
            info.update({
                "input_size": self.input_size,
                "character_set_size": len(self.character_set),
                "batch_max_length": self.batch_max_length
            })
        return info


# Global instance
_recognition_model = None


def get_recognition_model() -> TextRecognitionModel:
    """Get or create the global recognition model instance."""
    global _recognition_model
    if _recognition_model is None:
        _recognition_model = TextRecognitionModel()
    return _recognition_model
"""Configuration management for OCR FastAPI service."""
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import yaml
from pydantic import Field, validator
from pydantic_settings import SettingsConfigDict, BaseSettings

DetectionModelData = {"model_name": "LineDetectionv4", "type": "detection", "version": "4.0", "model_file": "models/LineDetectionv4.onnx"}
RecognitionModelData = {"model_name": "ResNetBiLSTMCTCv1", "type": "recognition", "version": "1.0", "model_file": "models/ResNetBiLSTMCTCv1.onnx","decoder": "CTC"}
class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow"
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Environment type")
    debug: bool = Field(default=False, description="Debug mode")
    reload: bool = Field(default=False, description="Auto-reload in development")

    # Model Paths
    models_dir: str = Field(default="models", description="Directory containing model YAML files")
    detection_model_data: dict = Field(default=DetectionModelData, description="Metadata for the current detection model")
    recognition_model_data: dict = Field(default=RecognitionModelData, description="Metadata for the current recognition model")
    
    # Available Models

    available_detection_models: List[dict] = Field(default=[DetectionModelData], description="List of available detection models")
    available_recognition_models: List[dict] = Field(default=[RecognitionModelData], description="List of available recognition models")

    # Processing Configuration
    detection_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
    crop_padding_x: int = Field(default=100, ge=0, description="Horizontal padding for crops")
    crop_padding_y: int = Field(default=15, ge=0, description="Vertical padding for crops")
    max_file_size: int = Field(default=10485760, ge=1024, description="Maximum file size in bytes (10MB)")
    max_batch_size: int = Field(default=20, ge=1, le=100, description="Maximum batch size")

    # Request Timeouts
    realtime_request_timeout: int = Field(default=5, ge=1, le=300, description="Real-time request timeout in seconds")
    batch_request_timeout: int = Field(default=300, ge=60, le=1800, description="Batch request timeout in seconds")


    # Arq Queue Configuration
    arq_queue_name: str = Field(default="ocr_queue", description="Arq queue name")
    arq_max_jobs: int = Field(default=100, ge=1, description="Maximum jobs in queue")

    # Security and Rate Limiting
    allowed_origins: List[str] = Field(default=["http://localhost:3000"], description="Allowed CORS origins")
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, description="Rate limit per minute")

    # Image Processing
    allowed_image_extensions: List[str] = Field(
        default=["jpg", "jpeg", "png", "tiff", "tif", "pdf"],
        description="Allowed image file extensions"
    )
    temp_dir: str = Field(default="/tmp/ocr_temp", description="Temporary file directory")
    cleanup_temp_files: bool = Field(default=True, description="Clean up temporary files")

    # Device Configuration
    preferred_device: str = Field(default="cpu", description="Preferred compute device (cpu/cuda)")
    force_cpu: bool = Field(default=False, description="Force CPU usage even if GPU available")

    # Monitoring and Health Checks
    health_check_timeout: int = Field(default=10, ge=1, description="Health check timeout")
    health_check_interval: int = Field(default=60, ge=10, description="Model health check interval")

    # Model-specific constants
    detection_input_size: tuple = Field(default=(1024, 1024), description="Detection model input size")
    recognition_input_size: tuple = Field(default=(1220, 80), description="Recognition model input size")
    
    # Character set for recognition model
    recognition_character_set: str = Field(
        default=r"""०१२३४५६७८९0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{}~।॥—‘’“”… अआइईउऊऋएऐओऔअंअःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञािीुूृेैोौंःँॅॉ""",
        description="Character set for recognition model"
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @validator("preferred_device")
    def validate_device(cls, v):
        """Validate preferred device."""
        valid_devices = ["cpu", "cuda"]
        if v.lower() not in valid_devices:
            raise ValueError(f"Invalid device: {v}. Must be one of {valid_devices}")
        return v.lower()

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "production", "test"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

    @validator("temp_dir")
    def create_temp_dir(cls, v):
        """Create temporary directory if it doesn't exist."""
        temp_path = Path(v)
        temp_path.mkdir(parents=True, exist_ok=True)
        return str(temp_path)

    def get_detection_model_path(self) -> Path:
        """Get absolute path to detection model."""
        return Path(self.detection_model_data["model_file"]).resolve()

    def get_recognition_model_path(self) -> Path:
        """Get absolute path to recognition model."""
        return Path(self.recognition_model_data["model_file"]).resolve()
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    def is_test(self) -> bool:
        """Check if running in test mode."""
        return self.environment == "test"

    def get_cors_origins(self) -> List[str]:
        """Get CORS origins as list."""
        return [origin.strip() for origin in self.allowed_origins]

    def get_allowed_extensions(self) -> List[str]:
        """Get allowed image extensions as lowercase list."""
        return [ext.lower().lstrip(".") for ext in self.allowed_image_extensions]

    def get_available_models_file(self, dir: Optional[str] = None) -> None:
        """Get list of available detection models."""
        if not os.path.exists(dir):
            raise NotADirectoryError(f"Invalid directory: {dir}")
        yaml_files = [f for f in os.listdir(dir) if f.endswith(".yaml")]
        if not yaml_files:
            raise FileNotFoundError(f"No YAML files found in directory: {dir}")
        
        available_detection_models = []
        available_recognition_models = []

        for yaml_file in yaml_files:
            with open(os.path.join(dir, yaml_file), "r") as f:
                try:
                    model_config_file = yaml.safe_load(f)
                    if model_config_file.get("type") == "detection":
                        available_detection_models.append(model_config_file)
                    elif model_config_file.get("type") == "recognition":
                        available_recognition_models.append(model_config_file)
                except yaml.YAMLError as e:
                    raise ValueError(f"Error parsing YAML file {yaml_file}: {e}")
        self.available_detection_models = available_detection_models
        self.available_recognition_models = available_recognition_models
        

# Global settings instance
settings = Settings()
# OCR FastAPI Service

A production-ready FastAPI service for Devanagari text detection and recognition using ONNX models.

## 🚀 Features

- **Line Detection**: YOLOv8-based segmentation to detect text lines in document images
- **Text Recognition**: CRNN (ResNet + BiLSTM + CTC) for Devanagari script recognition
- **Real-time Processing**: Single image OCR with 5-second response time target
- **Batch Processing**: Multiple image processing with job queue management
- **Multi-format Support**: JPEG, PNG, TIFF, and PDF file formats
- **PDF Processing**: Server-side PDF-to-image conversion
- **Health Monitoring**: Comprehensive health checks and metrics
- **Production Ready**: Docker, CORS, rate limiting, and error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│  OCR Pipeline    │───▶│   ONNX Models   │
│                 │    │                  │    │                 │
│ • Real-time API │    │ • Line Detection │    │ • LineDetection │
│ • Batch Jobs    │    │ • Text Recognition│   │ • ResNetBiLSTM  │
│ • Health Checks │    │ • Image Processing│   │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Redis Queue   │    │  File Processing │    │  Monitoring &   │
│                 │    │                  │    │  Logging        │
│ • Job Storage   │    │ • PDF Conversion │    │                 │
│ • Status Tracking│   │ • Image Validation│   │ • Metrics       │
│ • Queue Management│  │ • Temp File Mgmt │    │ • Health Checks │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- ONNX models: `LineDetectionv4.onnx` and `ResNetBiLSTMCTCv1.onnx`

## 🛠️ Installation

### Local Development

1. **Clone and setup environment**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Setup Redis** (required for batch processing):
   ```bash
   # Install Redis locally or use Docker
   docker run -d -p 6379:6379 redis:7-alpine
   ```

4. **Run the service**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Access the service**:
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Redis Commander: http://localhost:8081

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Model Paths
DETECTION_MODEL_PATH=models/LineDetectionv4.onnx
RECOGNITION_MODEL_PATH=models/ResNetBiLSTMCTCv1.onnx

# Processing Configuration
DETECTION_CONFIDENCE_THRESHOLD=0.5
MAX_BATCH_SIZE=20
MAX_FILE_SIZE=10485760  # 10MB

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Request Timeouts
REALTIME_REQUEST_TIMEOUT=5
BATCH_REQUEST_TIMEOUT=300

# CORS Settings
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

See `.env.example` for complete configuration options.

## 📡 API Endpoints

### Real-time OCR

**Single Image Processing**:
```http
POST /api/v1/ocr/inference
Content-Type: multipart/form-data

file: <image_file>
include_masks: true
confidence_threshold: 0.5
```

**Line Detection Only**:
```http
POST /api/v1/ocr/inference/detect-only
Content-Type: multipart/form-data

file: <image_file>
```

**Text Recognition Only**:
```http
POST /api/v1/ocr/inference/recognize-single
Content-Type: multipart/form-data

file: <cropped_text_line_image>
```

### Batch Processing

**Submit Batch Job**:
```http
POST /api/v1/ocr/batch
Content-Type: multipart/form-data

files: <multiple_image_files>
include_masks: true
priority: 5
```

**Check Job Status**:
```http
GET /api/v1/ocr/batch/{job_id}
```

**Get Job Results**:
```http
GET /api/v1/ocr/batch/{job_id}/result
```

**Cancel Job**:
```http
POST /api/v1/ocr/batch/{job_id}/cancel
```

### Health & Monitoring

**Health Check**:
```http
GET /api/v1/health
```

**Liveness Probe**:
```http
GET /api/v1/health/live
```

**Readiness Probe**:
```http
GET /api/v1/health/ready
```

**Metrics**:
```http
GET /api/v1/metrics
```

### Model Information

**Get Models Info**:
```http
GET /api/v1/models
```

**Specific Model Info**:
```http
GET /api/v1/models/detection
GET /api/v1/models/recognition
```

## 📝 API Response Examples

### OCR Inference Response

```json
{
  "image_width": 1024,
  "image_height": 768,
  "detections": [
    {
      "line_id": 0,
      "box": [50, 100, 500, 130],
      "crop_box": [0, 85, 600, 145],
      "confidence": 0.95,
      "text": "नमस्ते विश्व",
      "class": 0,
      "mask_base64": "data:image/png;base64,..."
    }
  ],
  "total_lines": 1,
  "processing_time": 1.234,
  "models_used": {
    "detection": "LineDetectionv4",
    "recognition": "ResNetBiLSTMCTCv1"
  }
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2025-11-30T12:00:00Z",
  "models": {
    "detection": true,
    "recognition": true
  },
  "redis": true,
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 67.8,
    "device": "cpu"
  }
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest app/tests/ -v
```

### Integration Tests
```bash
pytest app/tests/integration/ -v
```

### Load Testing
```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📊 Performance

- **Target Response Time**: 5 seconds for single images
- **Batch Processing**: Up to 20 images per batch
- **Supported Formats**: JPEG, PNG, TIFF, PDF
- **Maximum File Size**: 10MB per image
- **Memory Usage**: ~2-4GB depending on image size

## 🔐 Security

- File upload validation with magic byte checking
- Rate limiting (60 requests/minute per IP)
- CORS configuration for allowed origins
- Input sanitization and validation
- Non-root container execution

## 🐳 Production Deployment

### Docker Build
```bash
docker build -t ocr-fastapi:latest .
```

### Kubernetes Deployment
See `k8s/` directory for Kubernetes manifests including:
- Deployment configuration
- Service definition
- Ingress setup
- ConfigMaps and Secrets

### Environment-specific Configurations

**Production**:
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
MAX_WORKERS=4
```

**Development**:
```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true
```

## 🛠️ Development

### Project Structure
```
backend/
├── app/
│   ├── api/endpoints/     # API route handlers
│   ├── core/             # Core utilities and config
│   ├── models/           # Model wrappers and pipeline
│   ├── workers/          # Background task workers
│   └── storage/          # File and data storage
├── models/               # ONNX model files
├── tests/                # Test suite
├── docker/              # Docker configurations
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Features

1. **New Endpoints**: Add to `app/api/endpoints/`
2. **Model Updates**: Update wrapper classes in `app/models/`
3. **Background Tasks**: Add to `app/workers/`
4. **Storage**: Modify `app/storage/`

## 🐛 Troubleshooting

### Common Issues

**Model Loading Failures**:
- Check ONNX model file paths in configuration
- Verify model file integrity
- Check available memory

**Redis Connection Issues**:
- Verify Redis is running: `redis-cli ping`
- Check REDIS_URL configuration
- Check network connectivity

**Performance Issues**:
- Monitor memory usage
- Check batch size limits
- Verify image sizes and formats

**Health Check Failures**:
- Check model health: `GET /api/v1/models/{name}/health`
- Monitor system resources
- Check logs for detailed errors

### Logging

Logs are structured and include:
- Request/response logging
- Model inference timing
- Error tracking with context
- Performance metrics

Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

## 📈 Monitoring

### Metrics Available
- Model inference times
- Request counts and response times
- Memory and CPU usage
- Job queue statistics
- Error rates and types

### Health Indicators
- Model loading status
- Redis connectivity
- System resource usage
- Service availability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 for object detection
- CRNN architecture for text recognition
- ONNX Runtime for model inference
- FastAPI for the web framework
- Arq for background task processing
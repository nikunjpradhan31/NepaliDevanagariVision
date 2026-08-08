# OCR FastAPI Inference Service

A lightweight FastAPI service for Devanagari text detection and recognition using ONNX models. Models are downloaded into the container on startup from URLs configured via `.env`.

## Features

- **Line Detection**: YOLOv8-based segmentation to detect text lines in document images
- **Text Recognition**: CRNN (ResNet + BiLSTM) with CTC or Attention decoding
- **Real-time Processing**: Single image OCR (`/inference`), detection-only, and recognition-only endpoints
- **Multi-format Support**: JPEG, PNG, TIFF, and PDF file formats
- **Health Monitoring**: Liveness, readiness, and comprehensive health checks
- **Lightweight Docker**: Models downloaded at startup, non-root user, single uvicorn worker

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│  OCR Pipeline    │───▶│   ONNX Models   │
│                 │    │                  │    │                 │
│ • Inference API │    │ • Line Detection │    │ • LineDetection │
│ • Health Checks │    │ • Text Recognition│   │ • ResNetBiLSTM  │
│ • Model Info    │    │ • Image Processing│   │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- ONNX models (downloaded automatically if URLs are configured)

## Local Development

1. **Create and activate a virtual environment**:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Optionally set model download URLs (see below)
   ```

3. **Run the service**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 2083
   ```

## Docker

```bash
# Build
docker build -t ocr-backend ./backend

# Run (models download into the container on startup)
docker run -p 10007:10007 --env-file ./backend/.env ocr-backend
```

### Model Download on Startup

The container downloads the ONNX models on application startup from the URLs in `.env`:

```bash
DETECTION_MODEL_URL=https://github.com/nikunjpradhan31/NepaliDevanagariVision/releases/download/v1/LineDetectionv4.onnx
RECOGNITION_MODEL_URL=https://github.com/nikunjpradhan31/NepaliDevanagariVision/releases/download/v1/ResNetBiLSTMCTCv1.onnx
RECOGNITION_ATTN_MODEL_URL=https://github.com/nikunjpradhan31/NepaliDevanagariVision/releases/download/v1/ResNetBiLSTMAttnv1.onnx
MODEL_DOWNLOAD_TIMEOUT=300
```

- Files are downloaded into `MODELS_DIR` (default `models/`) at the path given by each model's `model_file` metadata.
- If a file already exists and is non-empty, it is not re-downloaded.
- Leave the URL empty to skip downloading (e.g. when models are mounted externally).
- Model metadata (name, type, decoder, file path) is defined in the `model/__.yaml` files and/or the `DETECTION_MODEL_DATA` / `RECOGNITION_MODEL_DATA` env vars.

## API Endpoints

### Inference

**Full OCR** (`POST /api/v1/ocr/inference`, multipart form `file`):
```bash
curl -X POST http://localhost:10007/api/v1/ocr/inference -F file=@document.jpg -F include_masks=false
```

**Detection only** (`POST /api/v1/ocr/inference/detect-only`):
```bash
curl -X POST http://localhost:10007/api/v1/ocr/inference/detect-only -F file=@document.jpg
```

**Recognition only** (`POST /api/v1/ocr/inference/recognize-single`):
```bash
curl -X POST http://localhost:10007/api/v1/ocr/inference/recognize-single -F file=@crop.jpg
```

### Health

- `GET /api/v1/health` — comprehensive health (models + system)
- `GET /api/v1/health/live` — liveness probe
- `GET /api/v1/health/ready` — readiness probe

### Model Information

- `GET /api/v1/models` — all loaded models
- `GET /api/v1/models/{name}` — specific model info (`detection`, `recognition`)
- `GET /api/v1/models/{name}/health` — model health
- `GET /api/v1/pipeline/stats`, `/pipeline/available-models`, `/pipeline/character-set`, `/pipeline/select-models`

### Service

- `GET /` — service info
- `GET /api/v1/status` — status and uptime

## Configuration

Key options in `.env` (see `.env.example` for the full list):

```bash
HOST=0.0.0.0
PORT=10007
ENVIRONMENT=production
MODELS_DIR=models

# Model download URLs (optional)
DETECTION_MODEL_URL=...
RECOGNITION_MODEL_URL=...
RECOGNITION_ATTN_MODEL_URL=...

DETECTION_CONFIDENCE_THRESHOLD=0.5
CROP_PADDING_X=100
CROP_PADDING_Y=15
MAX_FILE_SIZE=10485760

ALLOWED_ORIGINS=["http://localhost:3000"]
```

## Testing

```bash
pytest app/tests/ -v
```

## Performance

- **Supported Formats**: JPEG, PNG, TIFF, PDF
- **Maximum File Size**: 10MB per image

## Troubleshooting

**Model Loading Failures**:
- Verify the download URLs in `.env` are reachable
- Check `MODELS_DIR` and the `model_file` paths match
- Check available memory

**Health Check Failures**:
- Check model health: `GET /api/v1/models/{name}/health`
- Monitor system resources

## License

MIT — see the LICENSE file in the repository root.

## Acknowledgments

- YOLOv8 for object detection
- CRNN architecture for text recognition
- ONNX Runtime for model inference
- FastAPI for the web framework

# OCR FastAPI Service - Lightweight Inference Dockerfile
#
# Build:
#   docker build -t ocr-backend ./backend
#
# Run (models download into the container on startup via env vars):
#   docker run -p 10007:10007 --env-file ./backend/.env ocr-backend

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies (poppler-utils for PDF support)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app/app
COPY models /app/models
COPY run_script.sh /app/
COPY .env.example /app/.env.example

# NOTE: no .env is baked in. Configuration (model download URLs, etc.) is
# supplied at runtime via --env-file, which pydantic-settings reads as
# environment variables (see run command in the header comments).

# Create necessary directories. The ONNX model files are downloaded here on
# startup from the URLs configured in .env, so they are NOT baked into the image.
RUN mkdir -p /app/temp_crops /app/logs /app/models

# Create non-root user
RUN useradd --create-home --shell /bin/bash ocruser && \
    chown -R ocruser:ocruser /app
USER ocruser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:2083/api/v1/health/live || exit 1

# Expose port
EXPOSE 2083

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "2083", "--workers", "1"]

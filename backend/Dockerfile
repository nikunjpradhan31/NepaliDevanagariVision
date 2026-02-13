# OCR FastAPI Service Dockerfile
#
# Usage:
#   Build:
#     docker build -t ocr-backend ./backend
#
#   Run with external models:
#     docker run -p 10007:10007 \
#       -v /path/to/models:/app/models \
#       -v /path/to/model_configs:/app/backend/models \
#       ocr-backend
#
#   Or using docker-compose, mount the models directory at /app/models
#
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY environment.yml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r environment.yml

# Copy application code (excluding models directory to keep image small)
# The models will be mounted externally at runtime
COPY app /app/app
COPY run_script.sh /app/

# Copy environment file for configuration
COPY .env /app/.env

# Create necessary directories
# /app/models - External mount point for both ONNX model files AND YAML configs
# The application expects:
#   - /app/models/*.yaml (model configuration files)
#   - /app/models/*.onnx (model weight files)
# 
# When running the container, mount your models directory:
#   docker run -v /path/to/your/models:/app/models ...
RUN mkdir -p /app/temp_crops /app/logs /app/models

# Create non-root user
RUN useradd --create-home --shell /bin/bash ocruser && \
    chown -R ocruser:ocruser /app
USER ocruser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10007/api/v1/health/live || exit 1

# Expose port
EXPOSE 10007

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10007", "--workers", "1"]
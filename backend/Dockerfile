# OCR FastAPI Service - Lightweight Inference Dockerfile
#
# Build:
#   docker build -t ocr-backend ./backend
#
# Run (models download into the container on startup via env vars):
#   docker run -p 10007:10007 --env-file ./backend/.env ocr-backend

FROM continuumio/miniconda3:24.11.1-0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libmagic1 \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -afy

COPY main.py /app/
COPY app /app/app

RUN mkdir -p /app/temp_crops /app/logs /app/models

RUN useradd --create-home --shell /bin/bash ocruser && \
    chown -R ocruser:ocruser /app
USER ocruser

ARG PORT=10007

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/v1/health/live || exit 1

EXPOSE ${PORT}

CMD ["/opt/conda/envs/OCRService/bin/python", "main.py"]

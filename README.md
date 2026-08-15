# Nepali Devanagari OCR System

A comprehensive **Optical Character Recognition (OCR) system for Nepali Devanagari script** that combines deep learning models with a inferencing service.

## Project Overview

This project provides a complete end-to-end solution for recognizing Nepali text from document images. The system consists of four main components:

### Core Components

1. **Production FastAPI Service** (`backend/`) - A API service with ONNX model integration
2. **Line Detection System** (`CNN_Detection/`) - YOLOv11-based text line segmentation and detection
3. **Text Recognition Training** (`CRNN/`) - Complete CRNN training infrastructure with PyTorch
4. **Inference Pipeline** (`inference/`) - Optimized ONNX-based inference implementation

### Model Architecture- **Detection Model**: YOLOv8 segmentation variant exporting 300 predictions with prototype masks
- **Recognition Model**: ResNet backbone + BiLSTM + CTC decoder supporting 70+ Devanagari character classes
- **Character Set**: Complete Devanagari script including numerals (०१२३४५६७८९), consonants, vowels, matras, punctuation

## Technical Architecture

### Detection Pipeline
```
Input Image (any resolution) → Letterbox Resize (1024×1024) → YOLOv8 Segmentation 
→ Prototype Mask Processing → Original Coordinate Transformation → Text Line Crops
```

### Recognition Pipeline  
```
Text Line Crop → Center & Resize (1220×80) → ImageNet Normalization 
→ ResNet+BiLSTM Feature Extraction → CTC/Attention Decoding → Unicode Text Output
```

### API Service Architecture
```
FastAPI Application → Model Manager → ONNX Runtime Sessions
     ↓
Real-time Processing 
     ↓                    
Response Formatting
```

## API Capabilities

### Real-time OCR
- **Single image processing** with synchronous response
- **Line detection + text recognition** in one request
- **Configurable confidence thresholds** and processing parameters
- **Mask visualization** support for detection debugging

### Health & Monitoring
- **Comprehensive health checks** for models, and system resources
- **Performance metrics** tracking and logging
- **Structured logging** with JSON output for production monitoring
- **Rate limiting** and security middleware

## Technical Specifications

### Model Performance
- **Detection Input**: 1024×1024 pixels (letterboxed)
- **Recognition Input**: 1220×80 pixels (aspect-ratio maintained)
- **Supported Formats**: JPEG, PNG, TIFF, PDF
- **Maximum File Size**: 10MB per image
- **Processing Speed**: ~500ms detection + ~50ms per line recognition
- **Memory Usage**: 2-4GB depending on image size

### Character Support
- **Devanagari Numerals**: ०१२३४५६७८९ (0-9 in Devanagari)
- **Latin Numerals**: 0-9
- **Devanagari Script**: Complete set including consonants, vowels, dependent vowel signs (matras)
- **Punctuation**: Special characters and symbols
- **Unicode Compliance**: Full Unicode Devanagari block support

## Research & Development Features

### Training Infrastructure
- **PyTorch Lightning** integration for scalable training
- **Mixed precision training** with AMP support
- **Data augmentation** pipeline with configurable parameters
- **Multiple decoder support** (CTC, Attention, Beam Search)
- **Validation metrics**: Character Error Rate (CER), Word Error Rate (WER), normalized edit distance

## Performance Benchmarks

### Current Capabilities
- **Line Detection Accuracy**: YOLOv8-based segmentation with high precision
- **Text Recognition**: Supports complex Devanagari sequences with matras
- **Processing Throughput**: ~25 pages/minute (single worker)
- **Batch Processing**: Up to 16 images per batch
- **Response Time**: <5 seconds for typical documents

### Quality Metrics
- **Character Recognition**: Multi-class classification for 132+ character types
- **Sequence Modeling**: BiLSTM for temporal pattern recognition
- **CTC Decoding**: Optimal path finding for sequence alignment
- **Confidence Scoring**: Per-character and per-sequence confidence metrics -->
<!-- 
## Getting Started

### Local Development
```bash
# Backend API
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py

# Training (CRNN)
cd CRNN/trainer_CRNN
python train.py --config config_files/ne_config.yaml

# Detection Training
cd CNN_Detection
python YOLOv8_Detection_Train.ipynb
```

### Usage Examples
```bash
# Single image OCR
curl -X POST "http://localhost:8000/api/v1/ocr/inference" \
     -F "file=@document.jpg"

# Batch processing
curl -X POST "http://localhost:8000/api/v1/ocr/batch" \
     -F "files=@page1.jpg" -F "files=@page2.jpg"

# Health check
curl http://localhost:8000/api/v1/health
``` -->

<!-- ## Use Cases

### Document Digitization
- **Printed books and newspapers** in Nepali Devanagari script
- **Historical documents** and manuscripts
- **Administrative documents** and forms
- **Academic papers** and research materials

### Digital Archives
- **Searchable text archives** for Nepali content
- **Library digitization** projects
- **Cultural heritage preservation**
- **Government document** processing -->

## Future Enhancements

### Planned Features
- **GPU Acceleration** with CUDA/TensorRT optimization
- **Advanced Post-processing** with spell checking and grammar validation
- **Web Interface** for interactive OCR processing
- **Mobile Application** for on-device OCR

### Research Directions
- **Transformer-based Models** for improved accuracy
- **Domain Adaptation** for specialized document types
- **Active Learning** for continuous model improvement

## Project Structure

```
NepaliDevanagariVision/
├── backend/                    # Production FastAPI service
│   ├── app/                   # Application code
│   │   ├── api/              # API endpoints
│   │   ├── core/             # Core utilities
│   │   └── models/           # Model wrappers
│   ├── models/               # ONNX model files
├── CNN_Detection/            # Line detection training
│   ├── config.yaml           # YOLOv8 configuration
│   └── YOLOv8_Detection_Train.ipynb
├── CRNN/                     # Text recognition training
│   └── trainer_CRNN/         # Complete training pipeline
│       ├── config_files/     # Training configurations
│       ├── modules/          # Model components
│       └── train.py          # Training script
├── inference/                # ONNX inference pipeline
│   ├── inference_onnx.ipynb  # Complete inference demo
│   └── modules/              # Inference utilities
└── README.md                 # This documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built by Nikunj Pradhan** - A comprehensive solution making Nepali Devanagari text accessible, searchable, and usable in digital formats through OCR technology.

# Nepali Devanagari OCR + NLP

## Overview

This project aims to build a **state-of-the-art OCR (Optical Character Recognition) system for Nepali Devanagari script** and a complementary NLP pipeline to process text into both **Devanagari Unicode** and **romanized Nepali**. The system is designed to handle printed and handwritten text, reconstruct valid Nepali sequences, and support further applications like search, translation, and text analysis.

## Key Objectives

- Develop a **robust OCR pipeline** capable of detecting and recognizing Nepali characters, vowels, matras, digits, and punctuation.
- Build a **flexible annotation and preprocessing framework** to efficiently train models with both real and synthetic datasets.
- Implement **post-processing logic** to recombine base consonants and vowel signs into linguistically valid sequences.
- Integrate an **NLP pipeline** for tokenization, sentence segmentation, spell correction, and romanization.
- Deploy models for **production use** with a FastAPI service and a simple React web interface.
- Enable **active learning** by collecting user corrections and retraining models for continuous improvement.

## Features

- Detection and recognition of 70+ Nepali character classes, including consonants, vowels, dependent vowel signs (matras), digits, and punctuation.
- Flexible dataset creation, including synthetic data generation with multiple fonts and augmentations.
- Post-processing to ensure Unicode-correct Nepali text output.
- Romanization module for transliteration to Latin script.
- Modular architecture for easy extension to other Indic scripts.
- Metrics tracking: Character Error Rate (CER), Word Error Rate (WER), and detection F1 scores.

## Tech Stack

- **Deep Learning:** PyTorch, PyTorch Lightning, CRNN, Transformer-based models
- **OCR Tools:** DBNet, CRAFT, MMOCR, PaddleOCR
- **NLP:** Hugging Face Transformers, KenLM, IndicNLP library
- **Deployment:** FastAPI, ONNX, React, Uvicorn
- **Data Annotation:** Label Studio, CVAT
- **Augmentation & Preprocessing:** Albumentations, custom Python scripts

## Roadmap

1. **Data Collection & Annotation** – Gather diverse Nepali text sources and annotate line/word images.
2. **Synthetic Data Generation** – Produce additional training data with font variations and augmentations.
3. **Model Training** – Train OCR models (detection + recognition or end-to-end) on annotated datasets.
4. **Postprocessing & NLP** – Recombine OCR outputs, tokenize, spell-correct, and romanize.
5. **Deployment** – Serve the models via FastAPI with a React UI and batch processing CLI.
6. **Monitoring & Active Learning** – Track performance metrics, collect user feedback, and retrain models periodically.

## Use Cases

- Digitization of Nepali printed books, newspapers, and historical documents.
- Handwriting recognition for educational or administrative purposes.
- Searchable Nepali text archives.
- Romanized Nepali text for cross-lingual applications, translation, and NLP tasks.

## Contribution

This is a research-driven engineering project. Contributions are welcome for dataset collection, model optimization, deployment improvements, and NLP enhancements.

---

This project combines **cutting-edge deep learning** with practical NLP pipelines to create a complete solution for Nepali text recognition and processing, making Nepali content more **accessible, searchable, and usable in digital formats**.


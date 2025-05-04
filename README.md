# Face Mask Detection

**Real-time Face Mask Detection Using MobileNetV2**

This project implements a complete deep learning pipeline to detect whether a person is wearing a face mask or not using a convolutional neural network (CNN) based on the MobileNetV2 architecture. It includes model training, evaluation, and real-time video stream inference using OpenCV and TensorFlow.

---

## 🧠 Overview

The goal is to develop a lightweight and accurate neural network model that classifies face images as either **with mask** or **without mask**. The model is trained on a dataset of face images and tested on real-time webcam footage.

---

## 🚀 Features

### ✅ Dataset Preparation
- Images categorized into `with_mask` and `without_mask` folders
- Image resizing and preprocessing for model compatibility

### 🧪 Data Augmentation
- Performed using `ImageDataGenerator` with:
  - Rotation, zoom, brightness shift, width/height shift, and flipping
- Helps improve model generalization

### 🏗️ Model Architecture
- **MobileNetV2** as the base model with pre-trained ImageNet weights
- Custom head with GlobalAveragePooling, Dense, and Dropout layers
- Output layer with `softmax` activation for binary classification

### ⚙️ Training
- Loss Function: `binary_crossentropy`
- Optimizer: Adam
- Training for 20 epochs with a batch size of 32
- Metrics: Accuracy and Loss

### 📈 Evaluation
- Final training and validation accuracy and loss are plotted
- Model saved as `mask_detector.model` for inference

---

## 🎥 Real-time Inference

### 📦 Video Stream Detection
- Implemented using `detectMaskVideo.py`
- Captures frames from webcam and performs face detection using Haar cascades
- Classifies face region as **mask** or **no mask**
- Displays results with bounding boxes and labels in real time

---

## 🧰 Requirements

Install the required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt

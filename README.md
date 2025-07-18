# Computer Vision Practice Project

This repository is a personal learning project focused on building foundational skills in computer vision using Python and OpenCV.

## Features Implemented

### 1. Color Detection from Webcam
- Detects a specific color range in real-time using webcam input.
- Utilizes HSV color space for better color segmentation.
- Support dynamic color range adjustment using trackbars.

### 2. Face Anonymizer
- Detects and anonymizes human faces in various media sources:
    - Static images
    - Video files
    - Live webcam feed
- Uses `MediaPipe` for face detection.
- Supports blurring method for anonymization.

### 3. Emotion Detection from Facial Landmarks
- Detects facial landmarks using MediaPipe FaceMesh (468 keypoints per face).
- Extracts and preprocesses 3D facial landmark coordinates (x, y, z) for emotion classification.
- Trained a Random Forest Classifier on a custom dataset labeled with 5 basic emotions:
    `Anger`, `Happy`, `Neutral`, `Sad`, `Surprise`
- Real-time webcam demo that:
    - Detects face
    - Predicts emotion
    - Displays label and bounding box
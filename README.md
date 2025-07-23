# Computer Vision Practice Project

This repository is a personal learning project focused on building foundational skills in computer vision using Python and OpenCV. Shout out to my instructor! [(ComputerVisionEngineer)](https://www.youtube.com/@ComputerVisionEngineer)

## Features Implemented

### 1. Emotion Detection from Facial Landmarks
- Detects facial landmarks using `MediaPipe FaceMesh` (468 keypoints per face).
- Extracts and preprocesses 3D facial landmark coordinates (x, y, z) for emotion classification.
- Trained a Random Forest Classifier on a custom dataset labeled with 5 basic emotions:
    `Anger`, `Happy`, `Neutral`, `Sad`, `Surprise`
- Real-time webcam demo that:
    - Detects face
    - Predicts emotion
    - Displays label and bounding box

### 2. Hand Sign Detection (mini-heart detection)
- Detects hand gestures in real-time using webcam.
- Uses `MediaPipe Hands` to extract 21 key landmarks per hand.
- Trained a custom classifier on a dataset I collected using my own hands.
- Supports recognition of 3 hand signs:
    - `Mini-Heart`
    - `Peace`
    - `I Love You`
- Displays bounding box and label for each detected hand gesture.

### 3. Color Detection from Webcam
- Detects a specific color range in real-time using webcam input.
- Utilizes `HSV` color space for better color segmentation.
- Support dynamic color range adjustment using trackbars.

### 4. Face Anonymizer
- Detects and anonymizes human faces in various media sources:
    - Static images
    - Video files
    - Live webcam feed
- Uses `MediaPipe` for face detection.
- Supports blurring method for anonymization.


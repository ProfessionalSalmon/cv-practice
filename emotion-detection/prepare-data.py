import os
import cv2
import numpy as np
import mediapipe as mp
from util import get_face_landmarks

# Root directory of the dataset
data_root = "emotion-detection/emotion-data"

# Which subsets to process
subsets = ["train", "valid"]

output = []

# Initialize MediaPipe FaceMesh once
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,  # static images mode for dataset processing
    max_num_faces=1,
    min_detection_confidence=0.5
)

for subset in subsets:
    image_dir = os.path.join(data_root, subset, "images")
    label_dir = os.path.join(data_root, subset, "labels")

    for image_file in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image_file)

        # Assume label file has same name with .txt extension and contains the emotion index
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue  # skip unreadable image

        # Get face landmarks
        face_landmarks = get_face_landmarks(image, face_mesh, draw=False)

        # Facial Landmarks = specific key points on a person's face that represent anatomical structures. 
        # MediaPipe FaceMesh predicts 468 landmarks per detected face, and each landmark has x, y, z coordinates.
        # Only use samples with 468 landmarks (each has x, y, z = 468*3 = 1404 values)
        if len(face_landmarks) == 1404 and os.path.exists(label_path):
            # Read label (emotion index)
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                parts = line.split()
                if len(parts) == 0:
                    print(f"[SKIP] Empty label file: {label_path}")
                    continue
                try:
                    emotion_index = int(parts[0])  # <-- Take only the first number as label
                except ValueError:
                    print(f"[SKIP] Invalid label content in: {label_path}")
                    continue

            face_landmarks.append(emotion_index)
            output.append(face_landmarks)

face_mesh.close()

if output:
    np.savetxt("emotion-detection/data.txt", np.asarray(output))
    print("Data saved!")
else:
    print("No valid samples to save.")
import pickle
import cv2
import mediapipe as mp
import numpy as np
from util import get_face_landmarks

emotion_labels = ['Anger', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load trained model
with open('emotion-detection/model', 'rb') as f:
    model = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(img_rgb)

    if result.detections:
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Ensure box is within frame
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x1 + w, W)
            y2 = min(y1 + h, H)

            face_roi = frame[y1:y2, x1:x2]
            landmarks = get_face_landmarks(face_roi, draw=False, face_mesh=face_mesh)

            if len(landmarks) == 1404:
                pred = model.predict([landmarks])[0]
                emotion = emotion_labels[int(pred)]

                # Draw green rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            emotion,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)
            else:
                print("Face landmarks not found")
    else:
        print("No face detected")

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
face_detection.close()

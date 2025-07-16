import cv2
import mediapipe as mp
import os
import argparse


def face_anonymizer(img, face_detection, H, W):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    print(out.detections)  # check confidence score

    if out.detections is None:
        print("Face not detected")
    else:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # clamp coordinates to be inside the image
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x1 + w, W)
            y2 = min(y1 + h, H)

            # blur faces
            img[y1:y2, x1:x2, :] = cv2.blur(img[y1:y2, x1:x2, :], (30, 30))
            # blur entire photo --> img = cv2.blur(img, (10, 10))
    
    return img

# ──────────────────────────
# Parse CLI arguments
args = argparse.ArgumentParser()
args.add_argument('--mode', default='webcam')
args.add_argument('--filePath', default=None)
args = args.parse_args()

# ──────────────────────────
# Face detection
mp_face_detection = mp.solutions.face_detection

# model_selection = 0 --> short-range model that works best for faces within 2 meters from the camera
# model_selection = 1 --> full-range model best for faces within 5 meters
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode in ['image']:
        img = cv2.imread(args.filePath)
        if img is None:
            raise FileNotFoundError(f"Image not found: {args.filePath}")
        H, W, _ = img.shape
        img = face_anonymizer(img, face_detection, H, W)

        # save image
        input_dir = os.path.dirname(args.filePath)
        filename_wo_ext = os.path.splitext(os.path.basename(args.filePath))[0]
        output_filename = f"{filename_wo_ext}-blurred.png"
        output_path = os.path.join(input_dir, output_filename)
        cv2.imwrite(output_path, img)
        print(f"Saved blurred image to: {output_path}")
    
    elif args.mode in ['video']:
        cap = cv2.VideoCapture(args.filePath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found: {args.filePath}")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # create output video path similarly to image output
        input_dir = os.path.dirname(args.filePath)
        filename_wo_ext = os.path.splitext(os.path.basename(args.filePath))[0]
        output_filename = f"{filename_wo_ext}-blurred.mp4"
        output_path = os.path.join(input_dir, output_filename)

        out_vid = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            H, W, _ = frame.shape
            frame = face_anonymizer(frame, face_detection, H, W)
            out_vid.write(frame)

            # Show frame
            cv2.imshow("Face Anonymizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Video interrupted by user.")
                break

        cap.release()
        out_vid.release()
        print(f"Saved blurred video to: {output_path}")

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            H, W, _ = frame.shape
            frame = face_anonymizer(frame, face_detection, H, W)

            cv2.imshow('webcam', frame)
            # if use waitKey = 0, The frame will be frozen until you press a key.
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Webcam interrupted by user.")
                break

        cap.release()
        cv2.destroyAllWindows()

print("🧙 Successfully run")

# How to use this script
# in terminal
# python face-anonumizer/function.py --mode image --filePath .\face-anonumber\img\face1.jpg
# python face-anonymizer/function.py --mode video --filePath .\face-anonymizer\vid\vid1.mp4 

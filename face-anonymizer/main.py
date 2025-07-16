import cv2
import mediapipe as mp
import os


img_path = "face-anonymizer/img/face1.jpg"
input_dir = os.path.dirname(img_path)
filename_wo_ext = os.path.splitext(os.path.basename(img_path))[0]
output_filename = f"{filename_wo_ext}-blurred.png"
output_path = os.path.join(input_dir, output_filename)

# read image
img = cv2.imread(img_path)

H, W, _ = img.shape

# detect faces 
mp_face_detection = mp.solutions.face_detection

# model_selection = 0 --> short-range model that works best for faces within 2 meters from the camera
# model_selection = 1 --> full-range model best for faces within 5 meters
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
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

            # blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (30, 30))
            # blur entire photo --> img = cv2.blur(img, (10, 10))
    
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# save image
cv2.imwrite(output_path, img)
print(f"Saved blurred image to: {output_path}")
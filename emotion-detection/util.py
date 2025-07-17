import cv2
import mediapipe as mp


def get_face_landmarks(image, face_mesh, draw=False):
    """
    Detects face landmarks from a given image using MediaPipe FaceMesh.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        face_mesh (mp.solutions.face_mesh.FaceMesh): Pre-initialized FaceMesh object.
        draw (bool): If True, draws the detected landmarks on the image.

    Returns:
        list: A flat list of normalized and shifted x, y, z coordinates of the landmarks for a single face.
    """
    
    # Facial Landmarks = specific key points on a person's face that represent anatomical structures. 
    # MediaPipe FaceMesh predicts 468 landmarks per detected face, and each landmark has x, y, z coordinates.

    # Convert BGR to RGB
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get landmarks
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = [idx.x for idx in ls_single_face]
        ys_ = [idx.y for idx in ls_single_face]
        zs_ = [idx.z for idx in ls_single_face]

        min_x, min_y, min_z = min(xs_), min(ys_), min(zs_)

        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min_x)
            image_landmarks.append(ys_[j] - min_y)
            image_landmarks.append(zs_[j] - min_z)

     # Returns a flat list: [x0', y0', z0', x1', y1', z1', ..., x467', y467', z467']
    return image_landmarks  # Length = 468 landmarks × 3 = 1404, if successful

   
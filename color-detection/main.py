import cv2
from util import get_limits
from PIL import Image


yellow = [0, 255, 255]  # yellow in BGR colorspace
red = [0, 0, 255]
green = [0, 255, 0]
purple = [128, 0, 128]
pink = [203, 192, 255]

cap = cv2.VideoCapture(0)  # Open camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # convert original image from BGR to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerLimit, upperLimit = get_limits(color=pink)

    # the location of all the pixels containing the information we wamt
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # convert image from being numpy array (opencv representation) to pillow
    mask_ = Image.fromarray(mask)

    # bounding box
    bbox = mask_.getbbox()
    # print(bbox)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)  # color: green, thickness = 5

    cv2.imshow('Camera', frame)

    # Wait for key press for 1 ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Turn camera off by releasing the capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

import cv2
from util import get_limits

# Define colors in BGR
colors = {
    'r': ([0, 0, 255], 'Red'),
    'g': ([0, 255, 0], 'Green'),
    'b': ([255, 0, 0], 'Blue'),
    'y': ([0, 255, 255], 'Yellow'),
    'c': ([255, 255, 0], 'Cyan'),
    'm': ([255, 0, 255], 'Magenta'),
    'o': ([0, 165, 255], 'Orange'),
    'p': ([203, 192, 255], 'Pink'),
    'k': ([128, 0, 128], 'Purple'),
    'w': ([255, 255, 255], 'White'),
    'd': ([50, 50, 50], 'Dark Gray'),
    'l': ([0, 255, 128], 'Lime'),
    'n': ([128, 0, 0], 'Navy'),
    'e': ([19, 69, 139], 'Brown')
}

# Set initial color (default to pink)
color_key = 'y'
target_color, color_name = colors[color_key]

# Open the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert original image from BGR to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV limits for the currently selected color
    lowerLimit, upperLimit = get_limits(color=target_color)

    # Create a binary mask for the selected color
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Find contours of detected regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h > 500:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    # Display current target color on frame
    cv2.putText(frame, f"Tracking: {color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show the result
    cv2.imshow('Camera', frame)

    # Handle key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key) in colors:
        target_color, color_name = colors[chr(key)]

# Release camera and close all windows
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2

hsv_ranges = {
    'red1': (np.array([0, 70, 50]), np.array([15, 255, 255])),
    'red2': (np.array([165, 70, 50]), np.array([180, 255, 255])),
    'green': (np.array([36,  50,  70]), np.array([89, 255, 255])),
    'blue':  (np.array([90,  60,  0]),  np.array([128, 255, 255])),
    'yellow':(np.array([28, 200, 200]),  np.array([35, 255, 255])),
    'orange':(np.array([10, 100, 20]),  np.array([25, 255, 255])),
    'pink':  (np.array([145, 50, 70]),  np.array([170, 255, 255])),
    'purple': (np.array([130, 50, 50]), np.array([160, 255, 255])),
}


def get_limits(color):
    """
    Accepts either:
    - a key string like 'red1', 'green', etc. → uses fixed HSV ranges
    - a BGR list like [0, 255, 255]           → converts to HSV with loose bounds
    """
    if isinstance(color, str) and color in hsv_ranges:
        return hsv_ranges[color]
    
    # Otherwise, assume it's a BGR triplet (list or np.array)
    c = np.uint8([[color]])  # BGR → HSV
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = hsvC
    lowerLimit = np.array([max(h - 15, 0), 50, 50])
    upperLimit = np.array([min(h + 15, 179), 255, 255])

    return lowerLimit, upperLimit

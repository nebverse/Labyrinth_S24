# red and green

import cv2
import numpy as np

def detect_red_and_blue(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for red color
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Define HSV range for blue color
    lower_blue = np.array([35, 100, 100])
    upper_blue = np.array([85, 255, 255])

    # Create masks for red and blue colors
    mask_red= cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_red, mask_blue)

    # Apply the combined mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    return result, combined_mask

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    # Detect red and blue colors in the frame
    result, combined_mask = detect_red_and_blue(frame)
    
    # Display the original frame, combined mask, and result
    cv2.imshow('Original', frame)
    cv2.imshow('Combined Mask', combined_mask)
    cv2.imshow('Red and Green Detected', result)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


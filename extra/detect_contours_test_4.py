import cv2
import numpy as np
from rembg import remove

# Define HSV color range for dark pink
#dark_pink_lower = np.array([145, 50, 75])
#dark_pink_upper = np.array([165, 255, 255])

sky_blue_lower = np.array([90, 50, 50])
sky_blue_upper = np.array([110, 255, 255])

def calculate_contour_distance(contour1, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1 / 2
    c_y1 = y1 + h1 / 2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2 / 2
    c_y2 = y2 + h2 / 2

    return max(abs(c_x1 - c_x2) - (w1 + w2) / 2, abs(c_y1 - c_y2) - (h1 + h2) / 2)

def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)

def agglomerative_cluster(contours, threshold_distance=40.0):
    current_contours = list(contours)  # Convert tuple to list
    while len(current_contours) > 1:
        min_distance = None
        min_coordinate = None

        for x in range(len(current_contours) - 1):
            for y in range(x + 1, len(current_contours)):
                distance = calculate_contour_distance(current_contours[x], current_contours[y])
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                    min_coordinate = (x, y)

        if min_distance is not None and min_distance < threshold_distance:
            index1, index2 = min_coordinate
            current_contours[index1] = merge_contours(current_contours[index1], current_contours[index2])
            del current_contours[index2]
        else:
            break

    return current_contours

# Function to detect objects based on color range and draw contours around them
def detect_objects(frame, lower_bound, upper_bound):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Perform agglomerative clustering on the contours
    clustered_contours = agglomerative_cluster(contours, threshold_distance=40.0)
    
    # Draw clustered contours on the original frame
    cv2.drawContours(frame, clustered_contours, -1, (0, 255, 0), 2)
    
    return frame, mask, clustered_contours

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Remove the background
    frame_with_bg_removed = remove(frame)
    
    # Convert the frame to a writable array
    frame_with_bg_removed = np.array(frame_with_bg_removed)
    
    # Detect objects and draw contours on the frame
    frame_with_contours, mask, contours = detect_objects(frame_with_bg_removed, sky_blue_lower, sky_blue_upper)
    
    print(len(contours))
    
    # Display the original frame with contours
    cv2.imshow('Sky Blue Objects', frame_with_contours)
    
    # Display the mask for debugging
    cv2.imshow('Mask', mask)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


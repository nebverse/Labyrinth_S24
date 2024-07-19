import cv2
import numpy as np
# from rembg import remove 

# Define HSV color ranges for different shades of blue, including more dark blue shades

# Define HSV range for green screen


blue_ranges = [
    #("Midnight Blue", np.array([110, 100, 40]), np.array([130, 255, 100]))
    #("White", np.array([0, 0, 200]), np.array([180, 55, 255])),
    #("RED1", np.array([0, 100, 100]), np.array([10, 255, 255])),
    #("RED2", np.array([160, 100, 100]), np.array([180, 255, 255])),
    #("LG", np.array([35, 100, 100]), np.array([85, 255, 255])),
    #("Yellow", np.array([20, 100, 100]), np.array([30, 255, 255])),
    #("DP", np.array([160, 50, 50]), np.array([170, 255, 255]))
    #("Navy Blue", np.array([110, 150, 30]), np.array([130, 255, 90]))
    #("Sky Blue", np.array([90, 50, 50]), np.array([110, 255, 255])),
    #("Silver", np.array([0, 0, 160]), np.array([180, 50, 255]))
    #("Black", np.array([0, 0, 0]), np.array([180, 255, 50]))
    ("Green", np.array([35, 100, 100]), np.array([85, 255, 255]))
    #("Blue", np.array([100, 100, 100]), np.array([130, 255, 255]))
]

# Function to detect blue markers and draw approximated contours with exactly 4 vertices
def detect_blue_markers(frame, lower_bound, upper_bound):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the specified blue range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw approximated contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    return frame, mask, contours

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    #frame = remove(frame) 
    
    if not ret:
        break
    
    # Apply each blue range to detect markers
    for name, lower_bound, upper_bound in blue_ranges:
        frame_with_contours, mask, contours = detect_blue_markers(frame.copy(), lower_bound, upper_bound)
        
        print(len(contours))
        
        # Display the original frame with contours
        cv2.imshow(f'Blue Markers - {name}', frame_with_contours)
        
        # Display the mask for debugging
        cv2.imshow(f'Mask - {name}', mask)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


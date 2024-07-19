import cv2
import numpy as np
#import reward_path

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
def detect_objects(frame, lr, ur, lg, ug):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create masks for red and green colors
    mask_red= cv2.inRange(hsv, lr, ur)
    mask_green = cv2.inRange(hsv, lg, ug)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_red, mask_green)

    # Apply the combined mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Perform agglomerative clustering on the contours
    clustered_contours = agglomerative_cluster(contours, threshold_distance=40.0)
    
    # Draw clustered contours on the original frame
    cv2.drawContours(frame, clustered_contours, -1, (0, 255, 0), 2)
    
    return frame, combined_mask, clustered_contours
    
# Function to detect dark pink objects and draw contours around them
def detect_green_objects(frame, lg, ug):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the dark pink color
    mask = cv2.inRange(hsv, lg, ug)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Perform agglomerative clustering on the contours
    clustered_contours = agglomerative_cluster(contours, threshold_distance=40.0)
    
    # Draw clustered contours on the original frame
    cv2.drawContours(frame, clustered_contours, -1, (0, 255, 0), 2)
    
    return frame, mask, clustered_contours

# Function to calculate the reward based on the progress along the path
def calculate_reward(ball_position, previous_ball_position, waypoints):
    # Find the closest point on the path to the current ball position
    current_progress = min([np.linalg.norm(np.array(ball_position) - np.array(wp)) for wp in waypoints])
    
    # Find the closest point on the path to the previous ball position
    previous_progress = min([np.linalg.norm(np.array(previous_ball_position) - np.array(wp)) for wp in waypoints])
    
    # Calculate the reward as the progress made
    reward = current_progress - previous_progress
    
    return reward
    
def extract_reward(frame, previous_ball_position, lower_red, upper_red, lower_green, upper_green):
    frame_with_ball, ball_mask, ball_contours = detect_green_objects(frame, lower_green, upper_green)
    # Assuming ball_contours[0] is the ball contour
    print(len(ball_contours))
    if len(ball_contours) > 0:
        ball_contour = ball_contours[0]
        ball_moments = cv2.moments(ball_contour)
        
        if ball_moments['m00'] != 0:
            ball_position = (int(ball_moments['m10'] / ball_moments['m00']), int(ball_moments['m01'] / ball_moments['m00']))
            
            # Load waypoints
            edge_points = np.loadtxt('path_edge_points.txt')
            waypoints = [(x, y) for x, y in edge_points]
            
            # Calculate reward
            reward = calculate_reward(ball_position, previous_ball_position, waypoints)
            
            # Update previous ball position
            previous_ball_position = ball_position
            
            ball_hole = False
        else:
            reward = 0
            ball_position = previous_ball_position  # Maintain previous position if m00 is zero
            waypoints = 0
            ball_hole = True
    else:
        reward = 0
        ball_position = previous_ball_position  # Maintain previous position if no contours found
        waypoints = 0
        ball_hole = True

    return reward, previous_ball_position, waypoints, ball_hole

    
previous_ball_position = (0, 0)
# Define HSV color ranges for detecting the ball and markers
lower_red = np.array([160, 100, 100])
upper_red = np.array([180, 255, 255])
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])
# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    # Detect red and blue colors in the frame
    reward, previous_ball_position, waypoints, ball_hole = extract_reward(
            frame, previous_ball_position,
            lower_red, upper_red,
            lower_green, upper_green
        )
    
    # Display the original frame, combined mask, and result
    print(f'reward: {reward}, previous_ball_position: {previous_ball_position}, ball_hole: {ball_hole}')
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

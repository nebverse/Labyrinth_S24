import cv2
import numpy as np

# Define HSV color ranges for the blue ball and blue markers
ball_hsv_lower = np.array([35, 100, 100])
ball_hsv_upper = np.array([85, 255, 255])
marker_hsv_lower = np.array([90, 50, 50])
marker_hsv_upper = np.array([110, 255, 255])

labyrinth_width = 280
labyrinth_height = 230
  

# Load camera calibration parameters
# Assume camera_matrix and dist_coeffs are obtained from a previous calibration process

camera_matrix = np.array([[709.2319106, 0, 334.18540366],
                          [0, 707.85698237, 261.58897357],
                          [0, 0, 1]])
dist_coeffs = np.array([1.73242018e+00, -7.41757783e+01, -5.69601941e-03, 2.04623423e-02, 1.08123868e+03])

#camera_matrix = np.array([[472.76170318, 0, 337.85499143],
#                          [0, 472.27089845, 258.2768827],
#                          [0, 0, 1]])
#dist_coeffs = np.array([5.37429522e-01, -5.79711380e+00, 3.95646440e-03, 7.43915585e-03, 6.54852446e+00])

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

# Function to detect the ball and markers
def detect_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    ball_mask = cv2.inRange(hsv, ball_hsv_lower, ball_hsv_upper)
    marker_mask = cv2.inRange(hsv, marker_hsv_lower, marker_hsv_upper)

    # Find contours for the ball and markers
    ball_contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ball_contours = agglomerative_cluster(ball_contours, threshold_distance=40.0)
    marker_contours = agglomerative_cluster(marker_contours, threshold_distance=40.0)

    ball_position = None
    marker_positions = []

    if ball_contours:
        ball_contour = max(ball_contours, key=cv2.contourArea)
        M = cv2.moments(ball_contour)
        if M["m00"] != 0:
            ball_position = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

    for marker_contour in marker_contours:
        M = cv2.moments(marker_contour)
        if M["m00"] != 0:
            marker_positions.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    return ball_position, marker_positions

# Function to estimate inclination angles
def estimate_inclination_angles(marker_positions, camera_matrix, dist_coeffs):
    # Assuming marker positions are known in the labyrinth coordinate system
    object_points = np.array([[0, 0, 0], [labyrinth_width, 0, 0], [0, labyrinth_height, 0], [labyrinth_width, labyrinth_height, 0]], dtype=np.float32)
    
    print("dc", len(marker_positions))
    if len(marker_positions) == 4:
        image_points = np.array(marker_positions, dtype=np.float32)
        success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
        print(success)
        if success:
            R, _ = cv2.Rodrigues(rvec)
            angles = -cv2.decomposeProjectionMatrix(np.hstack((R, tvec)))[6]
            alpha = angles[0]
            beta = angles[1]
            return alpha, beta
    return None, None

# Function to extract rectified image patch
def extract_image_patch(frame, ball_position, alpha, beta, window_size=64):
    if ball_position is not None:
        x, y = ball_position
        half_size = window_size // 2

        # Apply transformations to extract rectified patch
        M = cv2.getRotationMatrix2D((x, y), -alpha, 1)
        rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        M = cv2.getRotationMatrix2D((x, y), -beta, 1)
        rotated_frame = cv2.warpAffine(rotated_frame, M, (frame.shape[1], frame.shape[0]))

        patch = rotated_frame[y-half_size:y+half_size, x-half_size:x+half_size]
        patch = cv2.resize(patch, (64, 64))

        return patch
    return None

# Example usage
def main():
    cap = cv2.VideoCapture(0)  # Adjust the index based on your camera setup

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ball_position, marker_positions = detect_objects(frame)
        #alpha, beta = estimate_inclination_angles(marker_positions, camera_matrix, dist_coeffs)
        #image_patch = extract_image_patch(frame, ball_position, alpha, beta)

        #if image_patch is not None:
            #cv2.imshow('Rectified Image Patch', image_patch)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

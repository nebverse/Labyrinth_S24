import cv2
import numpy as np

# Define HSV color ranges for the blue ball and blue markers
ball_hsv_lower = np.array([100, 150, 0])
ball_hsv_upper = np.array([140, 255, 255])
marker_hsv_lower = np.array([110, 100, 40])
marker_hsv_upper = np.array([130, 255, 100])

labyrinth_width = 280
labyrinth_height = 230

# Load camera calibration parameters
# Assume camera_matrix and dist_coeffs are obtained from a previous calibration process
camera_matrix = np.array([[481.12735695, 0, 316.70088386],
                          [0, 481.46581572, 248.35756658],
                          [0, 0, 1]])
dist_coeffs = np.array([0.06873575, -0.02451401, 0.00021207, -0.00031585, -0.20998459])

# Function to detect the ball and markers
def detect_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    ball_mask = cv2.inRange(hsv, ball_hsv_lower, ball_hsv_upper)
    marker_mask = cv2.inRange(hsv, marker_hsv_lower, marker_hsv_upper)

    # Find contours for the ball and markers
    ball_contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker_contours, _ = cv2.findContours(marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.imshow('Imagetest',marker_contours)

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
        alpha, beta = estimate_inclination_angles(marker_positions, camera_matrix, dist_coeffs)
        image_patch = extract_image_patch(frame, ball_position, alpha, beta)

        if image_patch is not None:
            cv2.imshow('Rectified Image Patch', image_patch)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

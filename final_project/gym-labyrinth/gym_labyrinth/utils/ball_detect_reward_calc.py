import cv2
import numpy as np
import pkg_resources

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

def detect_objects(frame, lr, ur, lg, ug):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv, lr, ur)
    mask_green = cv2.inRange(hsv, lg, ug)
    combined_mask = cv2.bitwise_or(mask_red, mask_green)
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clustered_contours = agglomerative_cluster(contours, threshold_distance=40.0)
    frame_writable = np.copy(frame)  # Ensure the frame is writable
    cv2.drawContours(frame_writable, clustered_contours, -1, (0, 255, 0), 2)
    return frame_writable, combined_mask, clustered_contours

def detect_green_objects(frame, lg, ug):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lg, ug)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clustered_contours = agglomerative_cluster(contours, threshold_distance=40.0)
    frame_writable = np.copy(frame)  # Ensure the frame is writable
    cv2.drawContours(frame_writable, clustered_contours, -1, (0, 255, 0), 2)
    return frame_writable, mask, clustered_contours

def calculate_reward(ball_position, previous_ball_position, waypoints):
    current_progress = min([np.linalg.norm(np.array(ball_position) - np.array(wp)) for wp in waypoints])
    previous_progress = min([np.linalg.norm(np.array(previous_ball_position) - np.array(wp)) for wp in waypoints])
    reward = current_progress - previous_progress
    return reward

def extract_reward(frame, previous_ball_position, lower_red, upper_red, lower_green, upper_green):
    frame_with_ball, ball_mask, ball_contours = detect_green_objects(frame, lower_green, upper_green)
    if len(ball_contours) > 0:
        ball_contour = ball_contours[0]
        ball_moments = cv2.moments(ball_contour)
        if ball_moments['m00'] != 0:
            ball_position = (int(ball_moments['m10'] / ball_moments['m00']), int(ball_moments['m01'] / ball_moments['m00']))
            filename = pkg_resources.resource_filename(__name__, 'resources/path_edge_points.txt')
            edge_points = np.loadtxt(filename)
            waypoints = [(x, y) for x, y in edge_points]
            reward = calculate_reward(ball_position, previous_ball_position, waypoints)
            previous_ball_position = ball_position
            ball_hole = False
        else:
            reward = 0
            ball_position = previous_ball_position
            waypoints = 0
            ball_hole = True
    else:
        reward = 0
        ball_position = previous_ball_position
        waypoints = 0
        ball_hole = True

    return reward, previous_ball_position, waypoints, ball_hole

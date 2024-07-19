import cv2
import numpy as np

def get_lines(lines_in, lines):
    if cv2.__version__ < '3.0':
        return lines_in[0]
    return [l[0] for l in lines]

def remove_circular_contours(canny_image, min_radius=1, max_radius=100):
    # Find contours in the Canny edge-detected image
    contours, _ = cv2.findContours(canny_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask to remove circular contours
    mask = np.ones(canny_image.shape, dtype="uint8") * 255
    
    for contour in contours:
        # Fit a minimum enclosing circle to the contour
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        # Check if the radius is within the range for circles
        if min_radius < radius < max_radius:
            # Draw the circular contour on the mask (white on black)
            cv2.drawContours(mask, [contour], -1, 0, -1)
    
    # Remove circular contours from the Canny image using the mask
    canny_filtered = cv2.bitwise_and(canny_image, canny_image, mask=mask)
    
    return canny_filtered

def mask_outside_region(frame, x_start, y_start, x_end, y_end):
    mask = np.zeros_like(frame)
    mask[y_start:y_end, x_start:x_end] = frame[y_start:y_end, x_start:x_end]
    return mask

def normalize_lighting(img_gray):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)
    return img_clahe

def extract_path(img, x_start, y_start, x_end, y_end):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = img_gray[y_start:y_end, x_start:x_end]
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    cannied_roi = cv2.Canny(roi, threshold1=200, threshold2=250, apertureSize=3, L2gradient=True)
    cannied = np.zeros_like(img_gray)
    cannied[y_start:y_end, x_start:x_end] = cannied_roi
    cannied = remove_circular_contours(cannied)
    
    # Find external contours and draw them
    contours, _ = cv2.findContours(cannied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thinned = np.zeros_like(cannied)
    cv2.drawContours(thinned, contours, 0, (255), 1)
    
    # Extract the edge points from the thinned image
    edge_points = np.column_stack(np.where(thinned > 0))
    
    return img, thinned, edge_points
    
def nearest_neighbor_sort_with_endpoints(edge_points, start_point, end_point):
    sorted_points = [start_point]
    remaining_points = list(edge_points)
    
    current_point = start_point
    while remaining_points:
        distances = np.linalg.norm(remaining_points - current_point, axis=1)
        nearest_index = np.argmin(distances)
        next_point = remaining_points.pop(nearest_index)
        sorted_points.append(next_point)
        current_point = next_point
    
        # Stop if we reach the end point
        if np.allclose(current_point, end_point):
            break
    
    return np.array(sorted_points)
    
def angle_sort_with_endpoints(edge_points, start_point, end_point):
    sorted_points = [start_point]
    remaining_points = list(edge_points)
    
    reference_vector = end_point - start_point
    reference_angle = np.arctan2(reference_vector[1], reference_vector[0])
    
    def angle_to_reference(point):
        vector = point - start_point
        angle = np.arctan2(vector[1], vector[0])
        return np.abs(reference_angle - angle)
    
    remaining_points.sort(key=angle_to_reference)
    sorted_points.extend(remaining_points)
    
    return np.array(sorted_points)

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Change the index if you have multiple cameras

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Define the region of interest
    x_start, y_start, x_end, y_end = 215, 120, 450, 415

    # Mask the region outside the specified range
    frame = mask_outside_region(frame, x_start, y_start, x_end, y_end)
    
    if not ret:
        break
    
    # Extract the path using Canny Edge Detection and thinning
    img, thinned, edge_points = extract_path(frame, x_start, y_start, x_end, y_end)
    
    start_point = np.array([316, 431])
    end_point = np.array([412, 330])
    sorted_edge_points = nearest_neighbor_sort_with_endpoints(edge_points, start_point, end_point)
    #sorted_edge_points = angle_sort_with_endpoints(edge_points, start_point, end_point)
    
    # Display the original frame with the path
    cv2.imshow('lines', img)
    cv2.imshow('thinned', thinned)
    
    # Print the edge points
    print("Edge Points (x, y):", sorted_edge_points)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save sorted edge points to a file
np.savetxt('path_edge_points.txt', sorted_edge_points)

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()


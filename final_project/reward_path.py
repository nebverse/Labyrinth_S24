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

def extract_path(img, x_start=215, y_start=120, x_end=450, y_end=415):
    img = mask_outside_region(img, x_start, y_start, x_end, y_end)
    img_gray = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_gray = normalize_lighting(img_gray)
    cannied = np.zeros_like(img_gray)
    roi = img_gray[y_start:y_end, x_start:x_end]
    #roi = cv2.blur(roi, ksize = (5,5))
    #roi = cv2.medianBlur(roi, ksize = 3)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    cannied_roi = cv2.Canny(roi, threshold1=200, threshold2=250, apertureSize=3, L2gradient=True)
    cannied[y_start:y_end, x_start:x_end] = cannied_roi
    cannied = remove_circular_contours(cannied)
    edge_points = np.column_stack(np.where(cannied > 0))
    start_point = np.array([316, 431])
    end_point = np.array([412, 330])
    sorted_edge_points = nearest_neighbor_sort_with_endpoints(edge_points, start_point, end_point)
    
    return img, cannied, sorted_edge_points



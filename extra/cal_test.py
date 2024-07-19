import cv2
import numpy as np

# Function to undistort an image
def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted_img

# Example usage: Load an image and undistort it
img = cv2.imread('/home/utilisateur/cyberrunner/my_code/testimage.jpg')  # Change to your image path

camera_matrix = np.array([[481.12735695, 0, 316.70088386],
                          [0, 481.46581572, 248.35756658],
                          [0, 0, 1]])
dist_coeffs = np.array([0.06873575, -0.02451401, 0.00021207, -0.00031585, -0.20998459])


undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)

cv2.imshow('Original Image', img)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

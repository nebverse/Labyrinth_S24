import cv2
import numpy as np

edge_points = np.loadtxt('path_edge_points.txt')
print(edge_points.shape)
print(len([(x, y) for x, y in edge_points]))

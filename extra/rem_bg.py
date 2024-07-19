import cv2
from rembg import remove 

#camera_index is the video device number of the camera 
camera_index = 0
cam = cv2.VideoCapture(camera_index)

while True:
 ret, image = cam.read()
 image = remove(image) 
 cv2.imshow('Imagetest',image)
 k = cv2.waitKey(1)
 if k != -1:
  break
cv2.imwrite('/home/utilisateur/cyberrunner/my_code/testimage.jpg', image)
cam.release()
cv2.destroyAllWindows()

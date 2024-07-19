import cv2

#camera_index is the video device number of the camera 
camera_index = 0
cam = cv2.VideoCapture(camera_index)
i = 0
while True:
 ret, image = cam.read()
 filename = f'/home/utilisateur/cyberrunner/my_code/ball_dataset/testimage_{i}.jpg'
 cv2.imshow('image', image)
 cv2.imwrite(filename, image)
 i += 1
 key = cv2.waitKey(2000)
 if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

# import cv2
# import ray

# ray.init()

# @ray.remote
# def capture_webcam():
#     # Open a connection to the webcam (0 is usually the built-in webcam)
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         # Display the frame
#         cv2.imshow('Webcam', frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the capture and close any OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the capture_webcam function remotely
# capture_webcam.remote()


# import cv2

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     cv2.imshow('Webcam', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import ray

ray.init()

@ray.remote
def capture_webcam():
    # Open a connection to the webcam (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the capture_webcam function remotely
future = capture_webcam.remote()

# Wait for the task to complete
ray.get(future)

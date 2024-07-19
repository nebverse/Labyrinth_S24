import inference
import cv2

# Load a pre-trained yolov8n model
model = inference.get_model("ball-detection-nsov0/1")

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    if not success:
        break
    
    # Run inference on the captured image
    results = model.infer(img)[0]
       
    # Extract predictions
    predictions = results.predictions

    for prediction in predictions:
        # Extract bounding box coordinates
        x_center = prediction.x
        y_center = prediction.y
        width = prediction.width
        height = prediction.height
        confidence = prediction.confidence
        class_name = prediction.class_name

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put the label and confidence score on the image
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the results
    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


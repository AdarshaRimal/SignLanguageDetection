import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("../models/best.pt")

# Open webcam
cap = cv2.VideoCapture(1)  # Use 0 for the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame not captured

    # Run YOLO inference on the frame
    results = model(frame, conf=0.5, iou=0.5)  # Adjust confidence threshold if needed

    # Loop through results and draw detections
    for result in results:
        frame_with_boxes = result.plot()  # Draw bounding boxes, class names, and confidence

    # Show the frame
    cv2.imshow("Sign Language Detection", frame_with_boxes)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

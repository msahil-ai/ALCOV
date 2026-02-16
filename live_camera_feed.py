import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)  # Use the default camera (0)
model = YOLO('yolov8n.pt')  # Load the YOLOv

while True:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        break  # If the frame is not captured successfully, exit the loop

    results = model(frame)  # Run the YOLO model on the captured frame
    annotated_frame = results[0].plot()  # Get the annotated frame with detections

    cv2.imshow('Live Camera Feed', annotated_frame)  # Display the annotated frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break
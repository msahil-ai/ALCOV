import cv2
from ultralytics import YOLO

cap=cv2.VideoCapture('new_12fps.mp4')

model=YOLO('yolov8m.pt')

while True:
    ret,frame =cap.read()
    results=model(frame)

    annotated_frame=results[0].plot()

    cv2.imshow('Detection Video',annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
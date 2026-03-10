import cv2 
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

img=cv2.imread(r'C:\Users\user\Desktop\person_detection\jr.jpg')

results= model(img)

annotated_image=results[0].plot()
cv2.imshow('YOLOv8 Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
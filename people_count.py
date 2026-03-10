import cv2
from ultralytics import YOLO
import numpy as np

# ----------------------------
# Load model and video
# ----------------------------
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('video2.mp4')

# ----------------------------
# VideoWriter setup
# ----------------------------
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'video2_people_count.mp4',
    fourcc,
    fps,
    (width, height)
)

# ----------------------------
# Tracking state
# ----------------------------
unique_id = set()

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        classes=[0],
        persist=True,
        conf=0.6,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Count unique people
    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        for oid in ids:
            unique_id.add(int(oid))

    # Draw count
    cv2.putText(
        annotated_frame,
        f'Count: {len(unique_id)}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show frame
    cv2.imshow('Object Tracking', annotated_frame)

    # ✅ SAVE FRAME
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

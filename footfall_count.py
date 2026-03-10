
import cv2
from ultralytics import YOLO
import numpy as np

# ----------------------------
# Load model and video
# ----------------------------
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('video2.mp4')

# ----------------------------
# VideoWriter setup (original size)
# ----------------------------
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'new_tracked.mp4',
    fourcc,
    fps,
    (width, height)
)

# ----------------------------
# Tracking state
# ----------------------------
unique_id = set()
zone_ids = set()   

# ----------------------------
# ZONE COORDINATES
# ----------------------------
x1, y1 = 916, 1255   # bottom left
x2, y2 = 1640, 1255   # bottom right
x3, y3 = 916, 400   # top left
x4, y4 = 1640, 400   # top right

# ----------------------------
# Display resize size (SMALL WINDOW)
# ----------------------------
display_width = 800   # Change this as needed
display_height = 500

# ----------------------------
# Mouse callback function
# ----------------------------
def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:

        # Convert resized coordinates to original frame coordinates
        orig_x = int(x * width / display_width)
        orig_y = int(y * height / display_height)

        print(f"Mouse Position -> X: {orig_x}, Y: {orig_y}", end="\r")


# Create window and attach mouse callback
cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Tracking", display_width, display_height)
cv2.setMouseCallback("Object Tracking", get_coordinates)

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


    # Draw Zone Rectangle
    cv2.rectangle(annotated_frame, (x3, y3), (x2, y1), (0, 0, 255), 2)

    if results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, oid in zip(boxes, ids):
            unique_id.add(int(oid))

            x_min, y_min, x_max, y_max = box
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            # Draw center point
            cv2.circle(annotated_frame, (center_x, center_y), 4, (255, 0, 0), -1)

            # Check if inside zone
            if (x3 <= center_x <= x2) and (y3 <= center_y <= y1):
                zone_ids.add(int(oid))


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
    cv2.putText(annotated_frame,
                f'Zone Footfall: {len(zone_ids)}',
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    # SAVE FRAME (original size)
    out.write(annotated_frame)

    # Resize ONLY for display
    display_frame = cv2.resize(annotated_frame, (display_width, display_height))

    # Show smaller frame
    cv2.imshow('Object Tracking', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
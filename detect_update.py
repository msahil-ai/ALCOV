import cv2
from ultralytics import YOLO
from collections import defaultdict

# -----------------------------
# Load trained YOLO model
# -----------------------------
model = YOLO("models/best.pt")

# -----------------------------
# Initialize Voting Dictionary
# -----------------------------
# This will store the count of gender predictions for each unique person (track ID)
# Format: {track_id: {class_index_0: count, class_index_1: count}}
gender_votes = defaultdict(lambda: defaultdict(int))

# -----------------------------
# Video input
# -----------------------------
cap = cv2.VideoCapture("input/test_video.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output/output_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -----------------------------
# Processing
# -----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Tracker enabled
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get instantaneous prediction
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Fallback label if tracking fails for a frame
            stable_label = model.names[cls]

            # If the tracker successfully assigned an ID to this box
            if box.id is not None:
                track_id = int(box.id[0])
                
                # 1. Cast a vote for the current frame's prediction
                gender_votes[track_id][cls] += 1
                
                # 2. Find which class has the most votes historically for this ID
                best_cls = max(gender_votes[track_id], key=gender_votes[track_id].get)
                
                # 3. Assign the stable label
                stable_label = model.names[best_cls]

            text = f"{stable_label} {conf:.2f}"

            # Draw Box and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    out.write(frame)
    cv2.imshow("YOLO Gender Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# -----------------------------
# Final Count Tally (Executes after video finishes)
# -----------------------------
male_count = 0
female_count = 0

for track_id, votes in gender_votes.items():
    # Find the class that got the most votes for this specific track_id
    best_cls = max(votes, key=votes.get)
    label = model.names[best_cls].lower()
    
    if label == "male":
        male_count += 1
    elif label == "female":
        female_count += 1

print("\n" + "="*40)
print("FINAL GENDER COUNT REPORT")
print("="*40)
print(f"Total Unique Males Detected:   {male_count}")
print(f"Total Unique Females Detected: {female_count}")
print("="*40 + "\n")
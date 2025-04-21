import cv2
import json
import numpy as np
import time

# Load pose data from JSON
with open("pose_data.json", "r") as f:
    frame_data = json.load(f)

# Define colors
keypoint_color = (0, 255, 0)  # Green
edge_color = (255, 0, 0)      # Blue

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Playback loop
prev_timestamp = 0.0
start_time = time.time()
frame_index = 0
total_frames = len(frame_data)

while frame_index < total_frames:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Resize frame to desired canvas size if needed
    canvas = frame.copy()

    # Get current pose data
    frame_info = frame_data[frame_index]
    current_timestamp = frame_info.get("timestamp_sec", 0.0)

    # Calculate delay based on timestamp difference
    delay = current_timestamp - prev_timestamp
    prev_timestamp = current_timestamp

    # Draw edges
    for edge in frame_info.get("edges", []):
        start_xy = edge.get("start_xy")
        end_xy = edge.get("end_xy")
        if start_xy and end_xy:
            start_point = tuple(map(int, start_xy))
            end_point = tuple(map(int, end_xy))
            cv2.line(canvas, start_point, end_point, edge_color, 2)

    # Draw keypoints
    for kp in frame_info.get("keypoints", []):
        x = int(kp.get("x", 0))
        y = int(kp.get("y", 0))
        cv2.circle(canvas, (x, y), 3, keypoint_color, -1)

    # Display the frame
    cv2.imshow("Pose Estimation Playback with Webcam", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Wait for the appropriate delay
    time.sleep(delay)
    frame_index += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()


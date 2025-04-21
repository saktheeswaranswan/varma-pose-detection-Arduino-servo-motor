import cv2
import time
import json
import mediapipe as mp

# Initialize MediaPipe Pose (or any other pipeline you prefer)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture('varma.mp4')
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Prepare data storage
frame_data = []
start_time = time.time()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process with MediaPipe
    results = pose.process(image_rgb)

    # Annotate frame (optional)
    annotated_frame = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS)

    # Calculate timestamp
    timestamp = time.time() - start_time

    # Record metadata
    frame_info = {
        "timestamp_sec": round(timestamp, 3),
        # Optionally record number of landmarks detected
        "landmarks_count": len(results.pose_landmarks.landmark) if results.pose_landmarks else 0
    }
    frame_data.append(frame_info)

    # Display
    cv2.imshow('MediaPipe Live Pose', annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save JSON metadata
output_filename = 'video_frame_data.json'
with open(output_filename, 'w') as f:
    json.dump(frame_data, f, indent=2)

print(f"Metadata saved to {output_filename}")


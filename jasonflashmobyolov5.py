import cv2
import time
import json
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
connections = list(mp_pose.POSE_CONNECTIONS)

# Open video file
video_path = "kannadu.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize data storage
frame_data = []

print("Processing video. Press 'q' to quit.")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate timestamp
    timestamp = round(frame_idx / fps, 3)

    # Convert the BGR image to RGB before processing.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    keypoints = []
    edges = []

    if results.pose_landmarks:
        h, w = frame.shape[:2]
        # Extract keypoints
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keypoints.append({
                "id": idx,
                "x": round(lm.x * w, 2),
                "y": round(lm.y * h, 2),
                "z": round(lm.z, 4),
                "visibility": round(lm.visibility, 3)
            })

        # Extract edges
        for a, b in connections:
            kp_a = keypoints[a]
            kp_b = keypoints[b]
            edges.append({
                "start_id": a,
                "end_id": b,
                "start_xy": [kp_a["x"], kp_a["y"]],
                "end_xy": [kp_b["x"], kp_b["y"]]
            })

        # Draw landmarks and connections on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Append data for the current frame
    frame_data.append({
        "timestamp_sec": timestamp,
        "keypoints": keypoints,
        "edges": edges
    })

    # Display the frame
    cv2.imshow("Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()

# Save data to JSON file
with open("pose_data.json", "w") as f:
    json.dump(frame_data, f, indent=2)

print("Saved pose data to pose_data.json")


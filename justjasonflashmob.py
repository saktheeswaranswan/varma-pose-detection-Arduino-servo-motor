import cv2
import time
import json
import torch
import mediapipe as mp

# 1. Load YOLOv5 (person class only)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5   # detection confidence threshold
model.iou = 0.45   # NMS IoU threshold

# 2. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
connections = list(mp_pose.POSE_CONNECTIONS)

# 3. Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

frame_data = []
start_time = time.time()

print("Recording... Press Ctrl+C to stop and save metadata.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(time.time() - start_time, 3)
        results = model(frame)
        detections = results.xyxy[0]

        persons = []
        for idx, det in enumerate(detections.tolist()):
            x1, y1, x2, y2, conf, cls = det
            if int(cls) != 0:
                continue

            # Crop and run MediaPipe
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_res = pose.process(crop_rgb)

            # Collect keypoints and edges
            keypoints = []
            edges = []
            if mp_res.pose_landmarks:
                h, w = crop.shape[:2]
                for i, lm in enumerate(mp_res.pose_landmarks.landmark):
                    px = x1 + lm.x * w
                    py = y1 + lm.y * h
                    keypoints.append({
                        "id": i,
                        "x": round(px, 2),
                        "y": round(py, 2),
                        "z": round(lm.z, 4),
                        "visibility": round(lm.visibility, 3)
                    })
                for (a, b) in connections:
                    kp_a = keypoints[a]
                    kp_b = keypoints[b]
                    edges.append({
                        "start_id": a,
                        "end_id": b,
                        "start_xy": [kp_a["x"], kp_a["y"]],
                        "end_xy": [kp_b["x"], kp_b["y"]]
                    })

            persons.append({
                "person_id": idx,
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "detection_conf": round(conf, 3),
                "keypoints": keypoints,
                "edges": edges
            })

        frame_data.append({
            "timestamp_sec": timestamp,
            "persons": persons
        })

except KeyboardInterrupt:
    print("Stopping recording...")

# Cleanup
cap.release()
pose.close()

# Export metadata
with open("multi_pose_data.json", "w") as f:
    json.dump(frame_data, f, indent=2)

print(f"Saved metadata for {len(frame_data)} frames to multi_pose_data.json")


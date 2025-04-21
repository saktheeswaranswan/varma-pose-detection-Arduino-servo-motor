import cv2
import time
import csv
import math
import json
import mediapipe as mp
from collections import defaultdict

# Setup MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture('kannadu.mp4')
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

fps = 30  # Assuming 30 FPS webcam feed

# CSV & JSON Setup
pose_data_json = []
csv_filename = "pose_joint_data_webcam.csv"
csv_fields = ["timestamp_sec", "joint", "x", "y", "angle_deg"]
csv_file = open(csv_filename, mode="w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
csv_writer.writeheader()

# For difference recording
keypoint_log_by_time = defaultdict(list)

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba == 0 or mag_bc == 0:
        return 0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    angle = math.acos(cos_angle)
    return round(math.degrees(angle), 2)

# Joint sets
joint_sets = {
    "left_elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    "right_elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    "left_shoulder": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    "right_shoulder": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    "left_knee": (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    "right_knee": (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
}

landmark_names = {lm.value: lm.name for lm in mp_pose.PoseLandmark}

frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = round(time.time() - start_time, 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        h, w = frame.shape[:2]
        lm_dict = {}
        keypoints_frame = []

        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y, z = int(lm.x * w), int(lm.y * h), round(lm.z, 4)
            lm_dict[idx] = (x, y)
            keypoints_frame.append({
                "id": idx,
                "x": x,
                "y": y,
                "z": z,
                "visibility": round(lm.visibility, 3)
            })
            csv_writer.writerow({
                "timestamp_sec": timestamp,
                "joint": landmark_names.get(idx, f"id_{idx}"),
                "x": x,
                "y": y,
                "angle_deg": ""
            })

        for name, (a_idx, b_idx, c_idx) in joint_sets.items():
            if a_idx.value in lm_dict and b_idx.value in lm_dict and c_idx.value in lm_dict:
                angle = calculate_angle(lm_dict[a_idx.value], lm_dict[b_idx.value], lm_dict[c_idx.value])
                bx, by = lm_dict[b_idx.value]
                csv_writer.writerow({
                    "timestamp_sec": timestamp,
                    "joint": name,
                    "x": bx,
                    "y": by,
                    "angle_deg": angle
                })
                cv2.putText(frame, f"{angle}°", (bx + 10, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.ellipse(frame, (bx, by), (20, 20), 0, 0, angle, (255, 0, 255), 2)

        pose_data_json.append({
            "timestamp_sec": timestamp,
            "keypoints": keypoints_frame
        })

        keypoint_log_by_time[round(timestamp)][0:0] = keypoints_frame

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Live Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Save raw pose JSON
with open("pose_data_webcam.json", "w") as f:
    json.dump(pose_data_json, f, indent=2)

# Save displacement every 5 seconds
timestamps = sorted(keypoint_log_by_time.keys())
json_diff = []
csv_diff_rows = []

for i in range(0, len(timestamps)-1):
    t1 = timestamps[i]
    t2 = round(t1 + 5.0, 3)
    if t2 not in keypoint_log_by_time:
        continue

    kps1 = {kp["id"]: kp for kp in keypoint_log_by_time[t1]}
    kps2 = {kp["id"]: kp for kp in keypoint_log_by_time[t2]}

    for idx in kps1:
        if idx in kps2:
            start = kps1[idx]
            end = kps2[idx]
            dx = round(end["x"] - start["x"], 2)
            dy = round(end["y"] - start["y"], 2)
            dz = round(end["z"] - start["z"], 4)
            csv_diff_rows.append({
                "timestamp_start": t1,
                "timestamp_end": t2,
                "keypoint_id": idx,
                "x_start": start["x"],
                "y_start": start["y"],
                "z_start": start["z"],
                "x_end": end["x"],
                "y_end": end["y"],
                "z_end": end["z"],
                "dx": dx,
                "dy": dy,
                "dz": dz
            })
            json_diff.append({
                "keypoint_id": idx,
                "from_timestamp": t1,
                "to_timestamp": t2,
                "start_pos": [start["x"], start["y"], start["z"]],
                "end_pos": [end["x"], end["y"], end["z"]],
                "diff": [dx, dy, dz]
            })

# Save CSV and JSON
with open("pose_diff_5s_webcam.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_diff_rows[0].keys())
    writer.writeheader()
    writer.writerows(csv_diff_rows)

with open("pose_diff_5s_webcam.json", "w") as f:
    json.dump(json_diff, f, indent=2)

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()
csv_file.close()

print(f"Saved live pose data to:\n- {csv_filename}\n- pose_data_webcam.json\n- pose_diff_5s_webcam.csv\n- pose_diff_5s_webcam.json")


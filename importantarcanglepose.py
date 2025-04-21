import cv2
import time
import csv
import math
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

# Open video file
video_path = "kannadu.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Could not open video file.")

fps = cap.get(cv2.CAP_PROP_FPS)

# CSV Setup
csv_filename = "pose_joint_data.csv"
csv_fields = ["timestamp_sec", "joint", "x", "y", "angle_deg"]
csv_file = open(csv_filename, mode="w", newline="")
csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
csv_writer.writeheader()

# Angle calculation (safe acos with clamping)
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = round(frame_idx / fps, 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        h, w = frame.shape[:2]
        lm_dict = {}

        # Landmark coords
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            lm_dict[idx] = (x, y)
            csv_writer.writerow({
                "timestamp_sec": timestamp,
                "joint": landmark_names.get(idx, f"id_{idx}"),
                "x": x,
                "y": y,
                "angle_deg": ""
            })

        # Angles with arcs and text
        for name, (a_idx, b_idx, c_idx) in joint_sets.items():
            if a_idx.value in lm_dict and b_idx.value in lm_dict and c_idx.value in lm_dict:
                a, b, c = lm_dict[a_idx.value], lm_dict[b_idx.value], lm_dict[c_idx.value]
                angle = calculate_angle(a, b, c)
                bx, by = b

                # Save to CSV
                csv_writer.writerow({
                    "timestamp_sec": timestamp,
                    "joint": name,
                    "x": bx,
                    "y": by,
                    "angle_deg": angle
                })

                # Draw angle text
                cv2.putText(frame, f"{angle}°", (bx + 10, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw arc (ellipse)
                cv2.ellipse(frame, (bx, by), (20, 20), 0, 0, angle, (255, 0, 255), 2)

        # Draw skeleton
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose Estimation with Joint Angles", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
pose.close()
csv_file.close()
print(f"Pose and angle data saved to {csv_filename}")


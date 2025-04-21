import csv
import json
from collections import defaultdict

# === FILES ===
csv_filename = "pose_joint_data.csv"
json_filename = "pose_data.json"
output_filename = "joint_angles_every_5s.csv"

# === LOAD CSV DATA ===
csv_data = defaultdict(dict)  # {timestamp: {joint: angle}}
timestamps = set()

with open(csv_filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ts = float(row['timestamp_sec'])
        joint = row['joint']
        angle_str = row['angle_deg'].strip()
        if angle_str:  # only store if angle is valid
            angle = float(angle_str)
            csv_data[ts][joint] = angle
            timestamps.add(ts)

timestamps = sorted(timestamps)

# === PROCESS DIFFERENCES EVERY 5 SECONDS ===
output_rows = []

for i in range(len(timestamps)):
    t_current = timestamps[i]
    t_prev = t_current - 5.0

    if t_prev in csv_data:
        for joint, angle_now in csv_data[t_current].items():
            angle_prev = csv_data[t_prev].get(joint)
            if angle_prev is not None:
                angle_diff = round(angle_now - angle_prev, 2)
                output_rows.append({
                    "timestamp_sec": t_current,
                    "joint": joint,
                    "angle_diff_deg": angle_diff
                })

# === SAVE OUTPUT CSV ===
with open(output_filename, "w", newline="") as f:
    fieldnames = ["timestamp_sec", "joint", "angle_diff_deg"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"Joint angle differences saved every 5 seconds to {output_filename}")


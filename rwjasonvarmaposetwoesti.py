import json
import csv

# Load pose data
with open("pose_data.json", "r") as f:
    pose_data = json.load(f)

# Index frames by timestamp
pose_by_time = {frame["timestamp_sec"]: frame for frame in pose_data}
timestamps = sorted(pose_by_time.keys())

# Prepare CSV and JSON data
csv_rows = []
json_output = []

for t_start in timestamps:
    t_end = round(t_start + 5.0, 3)
    if t_end in pose_by_time:
        kp_start = {kp["id"]: kp for kp in pose_by_time[t_start]["keypoints"]}
        kp_end = {kp["id"]: kp for kp in pose_by_time[t_end]["keypoints"]}

        for kp_id in kp_start:
            if kp_id in kp_end:
                start = kp_start[kp_id]
                end = kp_end[kp_id]
                dx = round(end["x"] - start["x"], 2)
                dy = round(end["y"] - start["y"], 2)
                dz = round(end["z"] - start["z"], 4)

                # Add to CSV
                csv_rows.append({
                    "timestamp_start": t_start,
                    "timestamp_end": t_end,
                    "keypoint_id": kp_id,
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

                # Add to JSON
                json_output.append({
                    "keypoint_id": kp_id,
                    "from_timestamp": t_start,
                    "to_timestamp": t_end,
                    "start_pos": [start["x"], start["y"], start["z"]],
                    "end_pos": [end["x"], end["y"], end["z"]],
                    "diff": [dx, dy, dz]
                })

# Save to CSV
csv_fieldnames = ["timestamp_start", "timestamp_end", "keypoint_id",
                  "x_start", "y_start", "z_start", "x_end", "y_end", "z_end",
                  "dx", "dy", "dz"]

with open("pose_diff_5s.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

# Save to JSON
with open("pose_diff_5s.json", "w") as f:
    json.dump(json_output, f, indent=2)

print("Exported pose changes every 5 seconds to 'pose_diff_5s.csv' and 'pose_diff_5s.json'")


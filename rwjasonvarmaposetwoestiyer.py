import json
import csv

# Input and output file paths
json_input_path = "pose_data.json"
csv_output_path = "pose_diff_5s.csv"
json_output_path = "pose_diff_5s.json"

# Load JSON data
with open(json_input_path, "r") as f:
    pose_data = json.load(f)

# Build timestamp-indexed map for quick access
pose_by_time = {frame["timestamp_sec"]: frame for frame in pose_data}
timestamps = sorted(pose_by_time.keys())

# Prepare CSV and JSON outputs
csv_rows = []
json_output = []

# Process every 5s pair of frames
for t_start in timestamps:
    t_end = round(t_start + 5.0, 3)
    if t_end in pose_by_time:
        frame_start = pose_by_time[t_start]
        frame_end = pose_by_time[t_end]

        # Build keypoint maps for both frames
        kp_start = {kp["id"]: kp for kp in frame_start["keypoints"]}
        kp_end = {kp["id"]: kp for kp in frame_end["keypoints"]}

        for kp_id in kp_start:
            if kp_id in kp_end:
                start = kp_start[kp_id]
                end = kp_end[kp_id]
                dx = round(end["x"] - start["x"], 2)
                dy = round(end["y"] - start["y"], 2)
                dz = round(end["z"] - start["z"], 4)

                row = {
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
                }

                csv_rows.append(row)
                json_output.append({
                    "keypoint_id": kp_id,
                    "from_timestamp": t_start,
                    "to_timestamp": t_end,
                    "start_pos": [start["x"], start["y"], start["z"]],
                    "end_pos": [end["x"], end["y"], end["z"]],
                    "diff": [dx, dy, dz]
                })

# === Save to CSV ===
csv_fieldnames = ["timestamp_start", "timestamp_end", "keypoint_id",
                  "x_start", "y_start", "z_start",
                  "x_end", "y_end", "z_end",
                  "dx", "dy", "dz"]

with open(csv_output_path, "w", newline="") as f_csv:
    writer = csv.DictWriter(f_csv, fieldnames=csv_fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

# === Save to JSON ===
with open(json_output_path, "w") as f_json:
    json.dump(json_output, f_json, indent=2)

print(f"Exported pose changes every 5 seconds to '{csv_output_path}' and '{json_output_path}'")


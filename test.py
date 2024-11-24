import os
import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(point_a, point_b, point_c):
    ba = [point_a[0] - point_b[0], point_a[1] - point_b[1], point_a[2] - point_b[2]]
    bc = [point_c[0] - point_b[0], point_c[1] - point_b[1], point_c[2] - point_b[2]]
    dot_product = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
    cos_angle = dot_product / (mag_ba * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Path to your video files
video_paths = ["dumbbelloverheadshoulderpress_S3_BL.mp4", "dumbbelloverheadshoulderpress_S3_BR.mp4", "dumbbelloverheadshoulderpress_S3_FL.mp4", "dumbbelloverheadshoulderpress_S3_FR.mp4",
               "dumbbelloverheadshoulderpress_S4_BL.mp4", "dumbbelloverheadshoulderpress_S4_BR.mp4", "dumbbelloverheadshoulderpress_S4_FL.mp4", "dumbbelloverheadshoulderpress_S4_FR.mp4",
               "dumbbelloverheadshoulderpress_S5_BL.mp4", "dumbbelloverheadshoulderpress_S5_BR.mp4", "dumbbelloverheadshoulderpress_S5_FL.mp4", "dumbbelloverheadshoulderpress_S5_FR.mp4",
               "dumbbelloverheadshoulderpress_S7_BL.mp4", "dumbbelloverheadshoulderpress_S7_BR.mp4", "dumbbelloverheadshoulderpress_S7_FL.mp4", "dumbbelloverheadshoulderpress_S7_FR.mp4",
               "dumbbelloverheadshoulderpress_S8_BL.mp4", "dumbbelloverheadshoulderpress_S8_BR.mp4", "dumbbelloverheadshoulderpress_S8_FL.mp4", "dumbbelloverheadshoulderpress_S8_FR.mp4",
               "dumbbelloverheadshoulderpress_S9_BL.mp4", "dumbbelloverheadshoulderpress_S9_BR.mp4", "dumbbelloverheadshoulderpress_S9_FL.mp4", "dumbbelloverheadshoulderpress_S9_FR.mp4",
               "dumbbelloverheadshoulderpress_S10_BL.mp4", "dumbbelloverheadshoulderpress_S10_BR.mp4", "dumbbelloverheadshoulderpress_S10_FL.mp4", "dumbbelloverheadshoulderpress_S10_FR.mp4",
               "dumbbelloverheadshoulderpress_S11_BL.mp4", "dumbbelloverheadshoulderpress_S11_BR.mp4", "dumbbelloverheadshoulderpress_S11_FL.mp4", "dumbbelloverheadshoulderpress_S11_FR.mp4",
               "dumbbelloverheadshoulderpress_T2_BL.mp4",
               "dumbbelloverheadshoulderpress_T12_BL.mp4",
               "dumbbelloverheadshoulderpress_T13_BL.mp4",
               ]  # Replace with your actual video paths

# Loop through each video
all_features = []

for video_path in video_paths:
    # Extract metadata from the filename
    filename = os.path.basename(video_path)
    parts = filename.split("_")
    
    exercise_name = parts[0]
    subject_id = parts[1]
    camera_angle = parts[2].replace(".mp4", "")

    # Open the video
    cap = cv2.VideoCapture(video_path)
    keypoints_data = []  # Store keypoints for each frame
    elbow_angles = []  # Store elbow angles for each frame
    wrist_y_positions = []  # Store y-coordinates of wrists for each frame

    # Process video frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB as Mediapipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with Mediapipe Pose
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract keypoints of interest
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            # Calculate elbow angles and track wrist movement
            left_elbow_angle = calculate_angle(
                (left_shoulder.x, left_shoulder.y, left_shoulder.z),
                (left_elbow.x, left_elbow.y, left_elbow.z),
                (left_wrist.x, left_wrist.y, left_wrist.z)
            )
            right_elbow_angle = calculate_angle(
                (right_shoulder.x, right_shoulder.y, right_shoulder.z),
                (right_elbow.x, right_elbow.y, right_elbow.z),
                (right_wrist.x, right_wrist.y, right_wrist.z)
            )

            # Track vertical wrist movement (y-coordinate)
            left_wrist_y = left_wrist.y
            right_wrist_y = right_wrist.y

            # Store keypoints with metadata
            keypoints_data.append({
                "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                "subject_id": subject_id,
                "camera_angle": camera_angle,
                "left_elbow_angle": left_elbow_angle,
                "right_elbow_angle": right_elbow_angle,
                "left_wrist_y": left_wrist_y,
                "right_wrist_y": right_wrist_y
            })

    cap.release()

    # Convert keypoints to DataFrame and add to all_features
    keypoints_df = pd.DataFrame(keypoints_data)
    all_features.append(keypoints_df)

# Combine all features into a single DataFrame
combined_features_df = pd.concat(all_features, ignore_index=True)

# Save to CSV for further analysis
combined_features_df.to_csv("combined_keypoints_with_metadata.csv", index=False)
print("Keypoints with metadata saved to combined_keypoints_with_metadata.csv")
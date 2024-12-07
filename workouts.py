import cv2
import mediapipe as mp
import math
from collections import deque
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.signal import argrelextrema

# Initialize mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Function to smooth angles using a moving average
def smooth_angle(angle, window):
    window.append(angle)
    return np.mean(window)

# Function to track shoulder presses for both arms
def track_shoulder_presses(video_source=0, output_csv="keypoints_data.csv"):
    # Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["Frame", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Elbow_X", "Left_Elbow_Y", 
                         "Left_Hip_X", "Left_Hip_Y", "Right_Shoulder_X", "Right_Shoulder_Y", 
                         "Right_Elbow_X", "Right_Elbow_Y", "Right_Hip_X", "Right_Hip_Y", 
                         "Left_Angle", "Right_Angle", "Time_Elapsed"])

        # Video feed
        cap = cv2.VideoCapture(video_source)

        # Mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Rep counters and states
            left_rep_count, right_rep_count = 0, 0
            left_state, right_state = "EXTENDED", "EXTENDED"
            left_angle_window, right_angle_window = deque(maxlen=5), deque(maxlen=5)

            frame_count = 0

            #maintains time for each frame
            start = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break

                # Recolor image
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Extract landmarks
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates for left arm
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1],
                                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]

                    # Get coordinates for right arm
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * frame.shape[1],
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * frame.shape[0]]

                    # Calculate and smooth angles
                    left_angle = smooth_angle(calculate_angle(left_shoulder, left_elbow, left_hip), left_angle_window)
                    right_angle = smooth_angle(calculate_angle(right_shoulder, right_elbow, right_hip), right_angle_window)

                    # Rep counting logic for left arm
                    if left_state == "EXTENDED" and left_angle < 100:
                        left_state = "FLEXED"
                    elif left_state == "FLEXED" and left_angle > 160:
                        left_state = "EXTENDED"
                        left_rep_count += 1

                    # Rep counting logic for right arm
                    if right_state == "EXTENDED" and right_angle < 100:
                        right_state = "FLEXED"
                    elif right_state == "FLEXED" and right_angle > 160:
                        right_state = "EXTENDED"
                        right_rep_count += 1

                    # Save the keypoints data to CSV
                    writer.writerow([frame_count, left_shoulder[0], left_shoulder[1], left_elbow[0], left_elbow[1], 
                                     left_hip[0], left_hip[1], right_shoulder[0], right_shoulder[1], 
                                     right_elbow[0], right_elbow[1], right_hip[0], right_hip[1], 
                                     left_angle, right_angle, time.time() - start])

                    frame_count += 1

                # Render detections
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Display frame
                cv2.imshow('Shoulder Press Tracker', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'Q' to exit
                    break
            cap.release()
            cv2.destroyAllWindows()


# Initialize mediapipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = math.acos(cos_angle) * 180.0 / math.pi
    return angle

# Function to smooth angles using a moving average
def smooth_angle(angle, window):
    window.append(angle)
    return np.mean(window)

# Function to get frames where repitions begin
# n dictates how sensitive the data is to noise; higher values make it less sensitive
def get_shoulder_press_reps(filename_csv="keypoints_data.csv", n=20):
    data = pd.read_csv(filename_csv)
    rep_frames = data.iloc[argrelextrema(data["Left_Elbow_Y"].values, np.less_equal, order=n)[0]]
    return rep_frames

# Function to get the average time between reps, as well as the number of reps
def get_shoulder_press_rep_info(rep_frames):
    average_time_between_reps = round(rep_frames["Time_Elapsed"].diff().mean(), 2)
    num_of_reps = len(rep_frames)

    return {"num_of_reps": num_of_reps, "average_time_between_reps": average_time_between_reps}
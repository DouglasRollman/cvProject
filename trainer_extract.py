import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

trainer_video_paths = ["trainer_video1.mp4", "trainer_video2.mp4"]  # Paths to trainer videos
trainer_pose_data = []

for path in trainer_video_paths:
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Extract pose keypoints
            keypoints = [(lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark]
            trainer_pose_data.append({"keypoints": keypoints, "label": "Good Form"})

    cap.release()
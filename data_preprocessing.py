import os
import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Pose module setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_from_video(video_path):
    """
    Extract pose landmarks from a video file.

    Args:
        video_path (str): Path to the video file.
    Returns:
        List of frames with pose landmarks and associated metadata.
    """
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # Extract pose landmarks if detected
        if results.pose_landmarks:
            landmarks = [
                [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
            ]
            landmarks_list.append(landmarks)
        else:
            # Append None for frames with no detection
            landmarks_list.append(None)

    cap.release()
    return landmarks_list

def smooth_landmarks(landmarks, window_size=5):
    """
    Apply moving average smoothing to the landmarks.

    Args:
        landmarks (list): List of landmark coordinates (or None if no detection).
        window_size (int): Size of the smoothing window.
    Returns:
        Smoothed landmarks.
    """
    smoothed_landmarks = []
    for i in range(len(landmarks)):
        if landmarks[i] is None:
            smoothed_landmarks.append(None)
            continue
        
        start = max(0, i - window_size // 2)
        end = min(len(landmarks), i + window_size // 2 + 1)
        valid_frames = [landmarks[j] for j in range(start, end) if landmarks[j] is not None]
        
        if valid_frames:
            avg_landmarks = np.mean(valid_frames, axis=0).tolist()
            smoothed_landmarks.append(avg_landmarks)
        else:
            smoothed_landmarks.append(None)

    return smoothed_landmarks

def normalize_landmarks(landmarks, image_width, image_height):
    """
    Normalize landmarks to the range [0, 1] based on image dimensions.

    Args:
        landmarks (list): List of landmark coordinates.
        image_width (int): Width of the image frame.
        image_height (int): Height of the image frame.
    Returns:
        Normalized landmarks.
    """
    normalized_landmarks = []
    for frame_landmarks in landmarks:
        if frame_landmarks is None:
            normalized_landmarks.append(None)
        else:
            normalized_frame = [
                [lm[0] / image_width, lm[1] / image_height, lm[2]] for lm in frame_landmarks
            ]
            normalized_landmarks.append(normalized_frame)

    return normalized_landmarks

def preprocess_directory(input_dir, output_dir, window_size=5):
    """
    Process all video files in a directory.

    Args:
        input_dir (str): Directory containing video files.
        output_dir (str): Directory to save processed data.
        window_size (int): Smoothing window size.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith('.mp4') or file.endswith('.avi'):  # Support common video formats
            video_path = os.path.join(input_dir, file)
            landmarks = extract_landmarks_from_video(video_path)
            
            # Normalize and smooth
            smoothed = smooth_landmarks(landmarks, window_size=window_size)
            cap = cv2.VideoCapture(video_path)
            image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            normalized = normalize_landmarks(smoothed, image_width, image_height)
            
            # Save processed data
            output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '_processed.npy')
            np.save(output_file, normalized)
            print(f"Processed and saved: {output_file}")

if __name__ == "__main__":
    # Example usage
    input_directory = "path/to/input/videos"
    output_directory = "path/to/output/data"
    preprocess_directory(input_directory, output_directory, window_size=5)

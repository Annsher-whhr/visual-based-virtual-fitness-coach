import cv2
import mediapipe as mp
import numpy as np
import pickle

mp_pose = mp.solutions.pose

KEYPOINTS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27
}

def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = {}
            for name, idx in KEYPOINTS.items():
                keypoints[name] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
            return keypoints
    return None

def main():
    standard_frames = []
    for i in range(1, 5):
        path = f'video/0{i}.png'
        keypoints = extract_keypoints(path)
        if keypoints:
            standard_frames.append(keypoints)
    
    with open('trajectory_system/standard_trajectory.pkl', 'wb') as f:
        pickle.dump(standard_frames, f)
    
    print(f"提取完成，共{len(standard_frames)}帧")

if __name__ == '__main__':
    main()


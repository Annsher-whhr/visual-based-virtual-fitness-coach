import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import mediapipe as mp
import numpy as np
import pickle
import joblib
from trajectory_system.trajectory_matcher import TrajectoryMatcher

mp_pose = mp.solutions.pose

class TrajectoryEvaluator:
    def __init__(self):
        self.matcher = TrajectoryMatcher()
        try:
            self.model = joblib.load('trajectory_system/trajectory_model.pkl')
        except:
            self.model = None
    
    def extract_keypoints_from_frame(self, frame):
        KEYPOINTS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27
        }
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                keypoints = {}
                for name, idx in KEYPOINTS.items():
                    keypoints[name] = [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z]
                return keypoints
        return None
    
    def evaluate_video_frames(self, frames):
        user_frames = []
        for frame in frames:
            kp = self.extract_keypoints_from_frame(frame)
            if kp:
                user_frames.append(kp)
        
        if len(user_frames) != 4:
            return {
                'score': 0,
                'accuracy': 0,
                'similarities': {},
                'advice': ['无法检测到完整的4帧姿势']
            }
        
        overall, similarities, basic_advice = self.matcher.calculate_similarity(user_frames)
        detailed_advice = self.matcher.get_detailed_advice(similarities)
        
        all_advice = basic_advice + detailed_advice
        
        return {
            'score': overall,
            'accuracy': overall / 100,
            'similarities': similarities,
            'advice': all_advice if all_advice else ['动作标准，保持！']
        }


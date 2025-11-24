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
from scipy.spatial.distance import euclidean
from tqdm import tqdm

mp_pose = mp.solutions.pose

class SmartFrameSelector:
    def __init__(self, standard_path='trajectory_system/standard_trajectory.pkl'):
        with open(standard_path, 'rb') as f:
            self.standard_frames = pickle.load(f)
    
    def extract_keypoints(self, frame):
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
    
    def calculate_frame_similarity(self, frame_kp, standard_kp):
        if frame_kp is None:
            return float('inf')
        
        total_distance = 0
        count = 0
        
        for part in standard_kp.keys():
            if part in frame_kp:
                std_point = np.array(standard_kp[part][:2])
                frame_point = np.array(frame_kp[part][:2])
                total_distance += euclidean(std_point, frame_point)
                count += 1
        
        return total_distance / count if count > 0 else float('inf')
    
    def select_best_frames(self, video_frames, min_gap=5):
        n_frames = len(video_frames)
        if n_frames < 4:
            return list(range(n_frames))
        
        print(f"正在分析 {n_frames} 帧...")
        frame_keypoints = []
        for frame in tqdm(video_frames, desc="提取关键点", ncols=80):
            kp = self.extract_keypoints(frame)
            frame_keypoints.append(kp)
        
        print("计算相似度矩阵...")
        similarity_matrix = np.zeros((n_frames, 4))
        for i in tqdm(range(n_frames), desc="相似度计算", ncols=80):
            for j in range(4):
                similarity_matrix[i, j] = self.calculate_frame_similarity(
                    frame_keypoints[i], 
                    self.standard_frames[j]
                )
        
        print("寻找最佳帧序列...")
        best_indices = self._find_best_sequence(similarity_matrix, min_gap)
        
        return best_indices
    
    def _find_best_sequence(self, similarity_matrix, min_gap):
        n_frames = similarity_matrix.shape[0]
        
        candidates_stage1 = np.argsort(similarity_matrix[:, 0])[:max(10, n_frames//10)]
        
        best_score = float('inf')
        best_indices = None
        
        for idx1 in candidates_stage1:
            valid_idx2 = [i for i in range(idx1 + min_gap, n_frames) if i < n_frames]
            if len(valid_idx2) == 0:
                continue
            
            candidates_stage2 = sorted(valid_idx2, key=lambda x: similarity_matrix[x, 1])[:10]
            
            for idx2 in candidates_stage2:
                valid_idx3 = [i for i in range(idx2 + min_gap, n_frames) if i < n_frames]
                if len(valid_idx3) == 0:
                    continue
                
                candidates_stage3 = sorted(valid_idx3, key=lambda x: similarity_matrix[x, 2])[:10]
                
                for idx3 in candidates_stage3:
                    valid_idx4 = [i for i in range(idx3 + min_gap, n_frames) if i < n_frames]
                    if len(valid_idx4) == 0:
                        continue
                    
                    idx4 = min(valid_idx4, key=lambda x: similarity_matrix[x, 3])
                    
                    score = (similarity_matrix[idx1, 0] + 
                            similarity_matrix[idx2, 1] + 
                            similarity_matrix[idx3, 2] + 
                            similarity_matrix[idx4, 3])
                    
                    if score < best_score:
                        best_score = score
                        best_indices = [idx1, idx2, idx3, idx4]
        
        if best_indices is None:
            indices = np.linspace(0, n_frames-1, 4, dtype=int)
            best_indices = list(indices)
        
        return best_indices


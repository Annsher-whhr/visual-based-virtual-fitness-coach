import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pickle

class TrajectoryMatcher:
    def __init__(self, standard_path='trajectory_system/standard_trajectory.pkl'):
        with open(standard_path, 'rb') as f:
            self.standard_frames = pickle.load(f)
        
        self.body_parts = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
    
    def normalize_trajectory(self, trajectory):
        points = np.array(trajectory)
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points
    
    def calculate_similarity(self, user_frames):
        if len(user_frames) != 4:
            return 0.0, {}, []
        
        similarities = {}
        advice = []
        
        for part in self.body_parts:
            std_traj = [frame[part][:2] for frame in self.standard_frames]
            user_traj = [frame[part][:2] for frame in user_frames]
            
            std_norm = self.normalize_trajectory(std_traj)
            user_norm = self.normalize_trajectory(user_traj)
            
            distance, _ = fastdtw(std_norm, user_norm, dist=euclidean)
            similarity = max(0, 100 - distance * 20)
            similarities[part] = similarity
            
            if similarity < 70:
                advice.append(f"{part}轨迹偏差较大")
        
        overall = np.mean(list(similarities.values()))
        return overall, similarities, advice
    
    def get_detailed_advice(self, similarities):
        advice = []
        
        upper_body = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        lower_body = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        
        upper_sim = np.mean([similarities[p] for p in upper_body])
        lower_sim = np.mean([similarities[p] for p in lower_body])
        
        if upper_sim < 75:
            advice.append("上肢运动轨迹需要调整，注意手臂抬起的弧度和高度")
        if lower_sim < 75:
            advice.append("下肢运动轨迹需要调整，注意步伐和重心转移")
        
        left_arm_sim = np.mean([similarities['left_shoulder'], similarities['left_elbow'], similarities['left_wrist']])
        right_arm_sim = np.mean([similarities['right_shoulder'], similarities['right_elbow'], similarities['right_wrist']])
        
        if abs(left_arm_sim - right_arm_sim) > 10:
            advice.append("左右手臂动作不对称，注意保持平衡")
        
        return advice


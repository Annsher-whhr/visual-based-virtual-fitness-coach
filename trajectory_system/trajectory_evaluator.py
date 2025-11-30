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
    
    """
    从输入的视频帧中提取人体关键点坐标
    
    功能：
        使用MediaPipe Pose模型检测图像中的人体姿态，并提取预定义的11个关键点的3D坐标
    
    参数：
        frame: numpy.ndarray - 输入的BGR格式视频帧图像
    
    返回值：
        dict或None - 成功时返回包含关键点名称和对应3D坐标的字典，检测失败返回None
    
    关键点说明：
        提取11个主要身体关节点：左右肩、左右肘、左右腕、左右髋、左右膝和左脚踝
        每个关键点坐标为[x, y, z]三维数组，范围在0-1之间
    
    实现说明：
        1. 使用静态图像模式和0.5的最小检测置信度初始化MediaPipe Pose模型
        2. 将BGR图像转换为RGB格式并进行姿态检测
        3. 从检测结果中提取预定义关键点的x、y、z坐标
        4. 以字典形式返回关键点坐标数据
    """
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
    
    """
    评估视频帧序列中的动作质量和标准度
    
    功能：
        从输入的视频帧序列中提取关键点，计算与标准动作的相似度，并生成改进建议
    
    参数：
        frames: list - 视频帧序列，每个元素为BGR格式的numpy数组图像
    
    返回值：
        dict - 包含动作评估结果的字典，具有以下键：
            - score: float - 整体动作评分（0-100）
            - accuracy: float - 动作准确率（0-1）
            - similarities: dict - 各身体部位的相似度评分
            - advice: list - 动作改进建议列表
    
    实现说明：
        1. 遍历输入的视频帧序列，提取有效的关键点数据
        2. 验证是否成功提取了完整的4帧姿势数据
        3. 使用TrajectoryMatcher计算用户动作与标准动作的相似度
        4. 获取基于相似度的详细改进建议
        5. 合并基本建议和详细建议
        6. 构建并返回包含评分、准确率、各部位相似度和建议的评估结果字典
    """
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


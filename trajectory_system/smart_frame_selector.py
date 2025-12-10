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
        """
        从输入图像帧中提取人体关键点的3D坐标
        
        Args:
            frame (numpy.ndarray): OpenCV读取的BGR格式图像帧
            
        Returns:
            dict or None: 包含人体关键点3D坐标的字典，格式为{关键点名称: [x, y, z]}，
                         如果未检测到人体则返回None
            
        注意：
            - 坐标值已归一化到0-1范围
            - 使用MediaPipe Pose模型进行人体姿态检测
            - 仅提取预定义的11个关键身体关节点
        """
        # 定义需要提取的人体关键点及其在MediaPipe Pose模型中的索引
        KEYPOINTS = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27
        }
        
        # 初始化MediaPipe Pose模型
        # static_image_mode=True: 处理静态图像
        # min_detection_confidence=0.5: 检测置信度阈值
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            # 将输入帧从BGR格式转换为RGB格式（MediaPipe要求RGB输入）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 执行姿态检测
            results = pose.process(rgb_frame)
            
            # 检查是否成功检测到人体
            if results.pose_landmarks:
                # 获取所有检测到的关键点
                landmarks = results.pose_landmarks.landmark
                
                # 创建字典存储预定义关键点的坐标
                keypoints = {}
                
                # 遍历预定义的关键点列表，提取对应的3D坐标
                for name, idx in KEYPOINTS.items():
                    keypoints[name] = [
                        landmarks[idx].x,  # x坐标（归一化）
                        landmarks[idx].y,  # y坐标（归一化）
                        landmarks[idx].z   # z坐标（归一化，深度信息）
                    ]
                
                # 返回提取的关键点坐标字典
                return keypoints
        
        # 如果未检测到人体，返回None
        return None
    
    def calculate_frame_similarity(self, frame_kp, standard_kp):
        """
        计算当前帧关键点与标准动作关键点之间的相似度
        
        Args:
            frame_kp (dict or None): 当前帧的关键点坐标字典，格式为{部位名称: [x, y, z]}
            standard_kp (dict): 标准动作的关键点坐标字典，格式为{部位名称: [x, y, z]}
            
        Returns:
            float: 相似度分数（欧氏距离平均值），越小表示越相似
                   如果当前帧没有关键点数据或没有匹配的部位，返回正无穷大
        """
        # 如果当前帧没有检测到关键点，返回正无穷大表示不相似
        if frame_kp is None:
            return float('inf')
        
        # 初始化总距离和匹配部位计数
        total_distance = 0
        count = 0
        
        # 遍历标准动作的所有关键点部位
        for part in standard_kp.keys():
            # 检查当前帧是否包含相同的部位关键点
            if part in frame_kp:
                # 提取标准动作该部位的2D坐标（x, y）
                std_point = np.array(standard_kp[part][:2])
                # 提取当前帧该部位的2D坐标（x, y）
                frame_point = np.array(frame_kp[part][:2])
                # 计算欧氏距离并累加到总距离中
                total_distance += euclidean(std_point, frame_point)
                # 匹配部位计数加1
                count += 1
        
        # 返回平均距离作为相似度分数，如果没有匹配部位则返回正无穷大
        return total_distance / count if count > 0 else float('inf')
    
    def select_best_frames(self, video_frames, min_gap=5):
        """
        从视频帧序列中选择与标准动作最匹配的4个关键帧
        
        Args:
            video_frames (list): 视频帧序列，每个元素是OpenCV格式的图像帧
            min_gap (int): 选择的帧之间的最小间隔帧数，默认为5
            
        Returns:
            list: 选中的4个最佳帧的索引列表，按时间顺序排列
        """
        # 获取视频总帧数
        n_frames = len(video_frames)
        
        # 如果视频帧数不足4，直接返回所有帧的索引
        if n_frames < 4:
            return list(range(n_frames))
        
        # 打印日志：开始分析视频帧
        print(f"正在分析 {n_frames} 帧...")
        
        # 初始化列表存储所有帧的关键点数据
        frame_keypoints = []
        
        # 遍历所有视频帧，提取关键点（显示进度条）
        for frame in tqdm(video_frames, desc="提取关键点", ncols=80):
            # 调用extract_keypoints方法提取当前帧的关键点
            kp = self.extract_keypoints(frame)
            # 将提取的关键点添加到列表中
            frame_keypoints.append(kp)
        
        # 打印日志：开始计算相似度矩阵
        print("计算相似度矩阵...")
        
        # 初始化相似度矩阵：[帧数 × 标准帧数]，标准帧数固定为4
        similarity_matrix = np.zeros((n_frames, 4))
        
        # 遍历所有视频帧（显示进度条）
        for i in tqdm(range(n_frames), desc="相似度计算", ncols=80):
            # 遍历4个标准动作帧
            for j in range(4):
                # 计算当前视频帧与第j个标准帧的相似度
                similarity_matrix[i, j] = self.calculate_frame_similarity(
                    frame_keypoints[i],  # 当前视频帧的关键点
                    self.standard_frames[j]  # 第j个标准帧的关键点
                )
        
        # 打印日志：开始寻找最佳帧序列
        print("寻找最佳帧序列...")
        
        # 调用_find_best_sequence方法寻找满足最小间隔的最佳帧序列
        best_indices = self._find_best_sequence(similarity_matrix, min_gap)
        
        # 返回选中的最佳帧索引列表
        return best_indices
    
    def _find_best_sequence(self, similarity_matrix, min_gap):
        """
        从相似度矩阵中寻找与标准动作最匹配的4个关键帧序列
        
        Args:
            similarity_matrix (numpy.ndarray): 相似度矩阵，维度为[帧数×4]，
                                             其中每行代表一个视频帧与4个标准帧的相似度分数
            min_gap (int): 选择的帧之间的最小间隔帧数
            
        Returns:
            list: 选中的4个最佳帧的索引列表，按时间顺序排列
        """
        # 获取视频总帧数（相似度矩阵的行数）
        n_frames = similarity_matrix.shape[0]
        
        # 第一阶段：选择与第一个标准帧最相似的候选帧
        # 选取前max(10, 总帧数//10)个候选帧，确保有足够的选择空间
        candidates_stage1 = np.argsort(similarity_matrix[:, 0])[:max(10, n_frames//10)]
        
        # 初始化最佳分数和最佳索引
        best_score = float('inf')  # 初始化为正无穷大
        best_indices = None  # 初始化为None
        
        # 遍历第一阶段的所有候选帧
        for idx1 in candidates_stage1:
            # 寻找第二帧的有效候选：必须在第一帧之后，且满足最小间隔要求
            valid_idx2 = [i for i in range(idx1 + min_gap, n_frames) if i < n_frames]
            
            # 如果没有符合条件的第二帧候选，跳过当前第一帧
            if len(valid_idx2) == 0:
                continue
            
            # 第二阶段：从有效候选中选择与第二个标准帧最相似的前10个候选帧
            candidates_stage2 = sorted(valid_idx2, key=lambda x: similarity_matrix[x, 1])[:10]
            
            # 遍历第二阶段的所有候选帧
            for idx2 in candidates_stage2:
                # 寻找第三帧的有效候选：必须在第二帧之后，且满足最小间隔要求
                valid_idx3 = [i for i in range(idx2 + min_gap, n_frames) if i < n_frames]
                
                # 如果没有符合条件的第三帧候选，跳过当前第二帧
                if len(valid_idx3) == 0:
                    continue
                
                # 第三阶段：从有效候选中选择与第三个标准帧最相似的前10个候选帧
                candidates_stage3 = sorted(valid_idx3, key=lambda x: similarity_matrix[x, 2])[:10]
                
                # 遍历第三阶段的所有候选帧
                for idx3 in candidates_stage3:
                    # 寻找第四帧的有效候选：必须在第三帧之后，且满足最小间隔要求
                    valid_idx4 = [i for i in range(idx3 + min_gap, n_frames) if i < n_frames]
                    
                    # 如果没有符合条件的第四帧候选，跳过当前第三帧
                    if len(valid_idx4) == 0:
                        continue
                    
                    # 第四阶段：选择与第四个标准帧最相似的帧
                    idx4 = min(valid_idx4, key=lambda x: similarity_matrix[x, 3])
                    
                    # 计算当前序列的总相似度分数（分数越低表示越相似）
                    score = (similarity_matrix[idx1, 0] + 
                            similarity_matrix[idx2, 1] + 
                            similarity_matrix[idx3, 2] + 
                            similarity_matrix[idx4, 3])
                    
                    # 如果当前序列的分数优于最佳分数，更新最佳结果
                    if score < best_score:
                        best_score = score
                        best_indices = [idx1, idx2, idx3, idx4]
        
        # 如果没有找到合适的序列（通常是因为视频太短或动作不完整）
        if best_indices is None:
            # 均匀采样4个帧作为备选方案
            indices = np.linspace(0, n_frames-1, 4, dtype=int)
            best_indices = list(indices)
        
        # 返回选中的最佳帧索引列表
        return best_indices
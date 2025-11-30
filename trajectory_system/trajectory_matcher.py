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
            'left_ankle'
        ]
    
    def normalize_trajectory(self, trajectory):
        """
        对轨迹数据进行标准化处理，消除位置和尺度差异
        
        标准化过程包括：
        1. 中心化：将轨迹中心点移至原点
        2. 尺度归一化：将轨迹缩放到单位空间内
        
        参数:
            trajectory: list - 包含坐标点的轨迹数据列表，格式为 [[x1, y1], [x2, y2], ...]
        
        返回:
            numpy.ndarray - 标准化后的轨迹点数组，形状为 (n_points, 2)
        """
        # 将轨迹数据转换为NumPy数组便于后续计算
        points = np.array(trajectory)
        
        # 计算轨迹中心点坐标
        center = np.mean(points, axis=0)
        
        # 将轨迹中心点移至原点（中心化）
        points = points - center
        
        # 计算所有点到原点的最大欧几里得距离
        max_dist = np.max(np.linalg.norm(points, axis=1))
        
        # 尺度归一化：将所有点除以最大距离，使轨迹缩放到单位空间内
        # 添加条件检查避免除以零
        if max_dist > 0:
            points = points / max_dist
        
        # 返回标准化后的轨迹点数组
        return points
    
    def calculate_similarity(self, user_frames):
        """
        计算用户动作轨迹与标准轨迹之间的相似度
        
        参数:
            user_frames (list): 用户动作帧数据，包含关键点坐标的列表，应包含4个帧的数据
                每个帧应包含各身体部位的坐标信息
        
        返回:
            tuple: 包含三个元素的元组
                - overall (float): 整体相似度得分，范围0-100
                - similarities (dict): 各身体部位的相似度字典，键为身体部位名称，值为相似度得分
                - advice (list): 相似度低于阈值的身体部位建议列表
        
        处理流程:
            1. 验证用户帧数量是否为4，不符合要求则返回默认值
            2. 初始化相似度字典和建议列表
            3. 对每个身体部位分别计算轨迹相似度:
               - 提取标准轨迹和用户轨迹的坐标数据
               - 使用normalize_trajectory方法标准化轨迹
               - 应用fastdtw算法计算动态时间规整距离
               - 将距离转换为相似度得分(0-100)
               - 对相似度低于70的部位生成改进建议
            4. 计算所有身体部位的平均相似度
            5. 返回整体相似度、各部位相似度和改进建议
        """
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
        """
        根据各身体部位的相似度生成详细的动作改进建议
        
        参数:
            similarities (dict): 包含各身体部位相似度得分的字典，
                键为身体部位名称（如'left_shoulder'等），值为相似度得分(0-100)
        
        返回:
            list: 包含改进建议的字符串列表，每个字符串代表一条具体建议
        
        处理流程:
            1. 初始化空的建议列表
            2. 将身体部位划分为上肢和下肢两组
            3. 计算上肢和下肢的平均相似度
            4. 当下肢平均相似度低于75时，生成下肢调整建议
            5. 计算左右手臂各自的平均相似度
            6. 当左右手臂相似度差异大于10时，生成对称性建议
            7. 返回收集到的所有改进建议
        """
        advice = []
        
        upper_body = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        lower_body = ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle']
        
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
# -*- coding: utf-8 -*-
"""
智能关键帧选择器 - 基于特征相似度匹配标准训练帧
用于太极拳起势动作的4帧关键点提取
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import math


# 标准动作的4帧特征（从训练数据中获取）
STANDARD_FRAMES = [
    { "arm_height_ratio": 0.984, "avg_visibility": 0.964, "foot_distance": 0.077,
      "hand_distance": 0.328, "hip_balance_left": 0.067, "hip_balance_right": -0.067,
      "hip_width": 0.133, "knee_bend_average": 1.703, "left_elbow_angle": 167.309,
      "left_knee_angle": 177.937, "left_shoulder_angle": 29.407, "nose_center_offset": -0.001,
      "right_elbow_angle": 148.420, "right_knee_angle": 178.658, "right_shoulder_angle": 36.713,
      "shoulder_balance_left": 0.123, "shoulder_balance_right": -0.123,
      "shoulder_to_hip_y": 0.238, "shoulder_width": 0.245, "torso_angle": 0.256,
      "torso_vx": -0.001, "torso_vy": -0.238 },

    { "arm_height_ratio": 0.967, "avg_visibility": 0.980, "foot_distance": 0.405,
      "hand_distance": 0.480, "hip_balance_left": 0.101, "hip_balance_right": -0.101,
      "hip_width": 0.201, "knee_bend_average": 4.381, "left_elbow_angle": 169.214,
      "left_knee_angle": 176.670, "left_shoulder_angle": 32.429, "nose_center_offset": -0.012,
      "right_elbow_angle": 165.861, "right_knee_angle": 174.568, "right_shoulder_angle": 36.195,
      "shoulder_balance_left": 0.175, "shoulder_balance_right": -0.175,
      "shoulder_to_hip_y": 0.276, "shoulder_width": 0.350, "torso_angle": 0.592,
      "torso_vx": 0.003, "torso_vy": -0.276 },

    { "arm_height_ratio": -0.036, "avg_visibility": 0.993, "foot_distance": 0.252,
      "hand_distance": 0.358, "hip_balance_left": 0.066, "hip_balance_right": -0.066,
      "hip_width": 0.133, "knee_bend_average": 8.692, "left_elbow_angle": 70.112,
      "left_knee_angle": 174.491, "left_shoulder_angle": 68.890, "nose_center_offset": -0.008,
      "right_elbow_angle": 49.999, "right_knee_angle": 168.124, "right_shoulder_angle": 62.062,
      "shoulder_balance_left": 0.109, "shoulder_balance_right": -0.109,
      "shoulder_to_hip_y": 0.243, "shoulder_width": 0.218, "torso_angle": 0.303,
      "torso_vx": 0.001, "torso_vy": -0.243 },

    { "arm_height_ratio": 0.625, "avg_visibility": 0.994, "foot_distance": 0.285,
      "hand_distance": 0.310, "hip_balance_left": 0.070, "hip_balance_right": -0.070,
      "hip_width": 0.140, "knee_bend_average": 22.041, "left_elbow_angle": 140.754,
      "left_knee_angle": 154.360, "left_shoulder_angle": 35.928, "nose_center_offset": -0.003,
      "right_elbow_angle": 141.849, "right_knee_angle": 161.558, "right_shoulder_angle": 36.090,
      "shoulder_balance_left": 0.122, "shoulder_balance_right": -0.122,
      "shoulder_to_hip_y": 0.243, "shoulder_width": 0.244, "torso_angle": 1.074,
      "torso_vx": -0.005, "torso_vy": -0.243 }
]


# 各阶段的关键特征权重（用于相似度计算）
FEATURE_WEIGHTS = {
    # 第1帧（起始）：强调手臂高+脚近+头部居中+躯干稳定
    0: {
        'arm_height_ratio': 5.0,
        'foot_distance': 4.0,
        'hand_distance': 2.0,
        'torso_angle': 3.0,
        'nose_center_offset': 3.5,
        'hip_balance_left': 2.0,
        'hip_balance_right': 2.0,
        'avg_visibility': 1.0,
        'knee_bend_average': 1.5,
    },
    # 第2帧（重心转移）：脚距已稳定、手臂仍高、头部和骨盆回正
    1: {
        'arm_height_ratio': 5.5,
        'foot_distance': 5.5,
        'nose_center_offset': 7.0,
        'hip_balance_left': 4.0,
        'hip_balance_right': 4.0,
        'shoulder_balance_left': 3.0,
        'shoulder_balance_right': 3.0,
        'torso_angle': 3.5,
        'hand_distance': 2.0,
        'avg_visibility': 1.0,
    },
    # 第3帧（手臂下落）：强调手臂低+肘部弯曲+脚距大
    2: {
        'arm_height_ratio': 5.0,
        'foot_distance': 3.0,
        'left_elbow_angle': 4.0,
        'right_elbow_angle': 4.0,
        'hand_distance': 2.0,
        'torso_angle': 2.0,
        'nose_center_offset': 2.0,
        'avg_visibility': 1.0,
    },
    # 第4帧（完成）：强调脚距最大+膝盖弯曲+手臂中等
    3: {
        'foot_distance': 5.0,
        'knee_bend_average': 5.0,
        'arm_height_ratio': 3.0,
        'left_knee_angle': 3.0,
        'right_knee_angle': 3.0,
        'torso_angle': 2.0,
        'nose_center_offset': 2.0,
        'avg_visibility': 1.0,
    }
}


def calculate_feature_similarity(metrics: Dict, standard_frame: Dict, frame_idx: int) -> float:
    """
    计算metrics与标准帧之间的相似度分数（分数越高越相似）
    
    Args:
        metrics: 当前帧的特征字典
        standard_frame: 标准帧的特征字典
        frame_idx: 标准帧索引(0-3)，用于选择权重
    
    Returns:
        相似度分数（0-100，越高越好）
    """
    if not isinstance(metrics, dict) or not metrics:
        return 0.0
    
    # 可见性过滤
    vis = metrics.get('avg_visibility', 0.0)
    if vis < 0.3:
        return 0.0
    
    weights = FEATURE_WEIGHTS.get(frame_idx, {})
    eps = 1e-6

    # === 阶段特定的质量门槛（Hard constraints） ===
    if frame_idx == 0:
        arm = metrics.get('arm_height_ratio')
        if arm is not None and arm < 0.7:
            return 0.0
        fd = metrics.get('foot_distance')
        hip = metrics.get('hip_width')
        if fd is not None and hip is not None and hip > eps:
            if (fd / (hip + eps)) > 1.2:
                # 脚已经离开起始并明显打开，说明不是起始帧
                return 0.0
    elif frame_idx == 1:
        foot = metrics.get('foot_distance')
        hip = metrics.get('hip_width')
        if foot is None or hip is None or hip <= eps:
            return 0.0
        foot_ratio = float(foot) / (float(hip) + eps)
        # 阈值略低于标准 1.5，确保脚基本打开并落地
        if foot_ratio < 1.25:
            return 0.0

        arm = metrics.get('arm_height_ratio')
        if arm is not None and arm < 0.55:
            # 手臂已经掉下来，说明阶段已进入第三帧
            return 0.0

        shoulder_span = metrics.get('shoulder_width')
        if shoulder_span is None or shoulder_span <= eps:
            shoulder_span = hip  # 回退使用臀宽
        nose = metrics.get('nose_center_offset')
        if nose is not None:
            nose_norm = abs(float(nose)) / (float(shoulder_span) + eps)
            if nose_norm > 0.45:
                # 头部仍偏向一侧，说明重心未回中
                return 0.0

        # 髋部/肩部平衡过大表明身体仍在偏移
        hip_balances = [
            abs(float(metrics.get('hip_balance_left', 0.0))),
            abs(float(metrics.get('hip_balance_right', 0.0))),
        ]
        if any(val > 0.28 for val in hip_balances):
            return 0.0

        shoulder_balances = [
            abs(float(metrics.get('shoulder_balance_left', 0.0))),
            abs(float(metrics.get('shoulder_balance_right', 0.0))),
        ]
        if any(val > 0.35 for val in shoulder_balances):
            return 0.0

        torso_ang = metrics.get('torso_angle')
        if torso_ang is not None and abs(float(torso_ang)) > 10.0:
            return 0.0

    total_distance = 0.0
    total_weight = 0.0
    
    for feature, weight in weights.items():
        val_current = metrics.get(feature)
        val_standard = standard_frame.get(feature)
        
        if val_current is None or val_standard is None:
            continue
        
        # 归一化差异（不同特征的量纲不同，需要标准化）
        if feature in ['arm_height_ratio', 'foot_distance', 'hand_distance', 
                       'hip_width', 'shoulder_width']:
            # 距离类特征，相对差异
            scale = max(abs(val_standard), 0.01)
            diff = abs(val_current - val_standard) / scale
        elif feature in ['left_elbow_angle', 'right_elbow_angle', 
                         'left_knee_angle', 'right_knee_angle',
                         'left_shoulder_angle', 'right_shoulder_angle',
                         'knee_bend_average', 'torso_angle']:
            # 角度类特征，绝对差异除以180度
            diff = abs(val_current - val_standard) / 180.0
        elif feature == 'avg_visibility':
            # 可见性，直接差异
            diff = abs(val_current - val_standard)
        elif feature == 'nose_center_offset':
            # 头部居中偏移，相对于肩宽的归一化差异
            # 标准值接近0表示居中，差异越小越好
            shoulder_width = metrics.get('shoulder_width', 0.24)  # 使用当前帧的肩宽作为归一化基准
            if shoulder_width <= 1e-6:
                shoulder_width = 0.24  # 回退到标准值
            diff = abs(val_current - val_standard) / (shoulder_width + 1e-6)
        else:
            # 其他特征
            scale = max(abs(val_standard), 0.01)
            diff = abs(val_current - val_standard) / scale
        
        total_distance += weight * diff
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    # 将距离转换为相似度分数（0-100）
    avg_distance = total_distance / total_weight
    similarity = 100.0 * math.exp(-avg_distance)  # 指数衰减
    
    return similarity


def select_frames_by_similarity(
    metrics_list: List[Dict],
    min_frame_gap: int = 3,
    verbose: bool = True
) -> Tuple[List[int], Dict]:
    """
    使用动态规划算法，从视频中选择与标准4帧最相似的帧序列
    
    Args:
        metrics_list: 视频中所有帧的特征列表
        min_frame_gap: 相邻选中帧之间的最小间隔
        verbose: 是否打印详细信息
    
    Returns:
        (selected_indices, info_dict)
        - selected_indices: 选中的4个帧索引
        - info_dict: 包含相似度分数等详细信息
    """
    n = len(metrics_list)
    if n < 4:
        raise ValueError(f"视频帧数不足: {n} < 4")
    
    # 计算每一帧与每个标准帧的相似度矩阵
    similarity_matrix = np.zeros((n, 4))  # [video_frame, standard_frame]
    
    for i, metrics in enumerate(metrics_list):
        for j, std_frame in enumerate(STANDARD_FRAMES):
            similarity_matrix[i, j] = calculate_feature_similarity(metrics, std_frame, j)
    
    # 动态规划：找到总相似度最高的4帧序列
    # dp[i][k] = 在前i帧中选择k个标准帧的最大相似度
    # path[i][k] = 达到dp[i][k]的上一个选择的帧索引
    
    dp = np.full((n, 5), -np.inf)  # dp[i][k]: 到第i帧选了k个标准帧的最大相似度
    path = np.full((n, 5), -1, dtype=int)
    
    # 初始化：第0个标准帧可以从任何视频帧开始
    for i in range(n):
        dp[i][1] = similarity_matrix[i][0]
    
    # 动态规划填表
    for k in range(2, 5):  # 选择第k个标准帧 (k=2,3,4)
        for i in range((k-1) * min_frame_gap, n):  # 当前视频帧
            # 遍历可能的前一个选择
            for j in range((k-2) * min_frame_gap, i - min_frame_gap + 1):
                score = dp[j][k-1] + similarity_matrix[i][k-1]
                if score > dp[i][k]:
                    dp[i][k] = score
                    path[i][k] = j
    
    # 回溯找到最优路径
    max_score = -np.inf
    best_end = -1
    for i in range(3 * min_frame_gap, n):
        if dp[i][4] > max_score:
            max_score = dp[i][4]
            best_end = i
    
    if best_end == -1 or max_score == -np.inf:
        # 动态规划失败，使用回退策略
        if verbose:
            print("⚠️  动态规划选帧失败，使用均匀分布策略")
        return _fallback_uniform_selection(metrics_list, similarity_matrix)
    
    # 回溯路径
    selected_indices = []
    current = best_end
    for k in range(4, 0, -1):
        selected_indices.append(current)
        current = path[current][k]
    
    selected_indices.reverse()
    
    # 计算详细信息
    individual_scores = [
        similarity_matrix[idx][j] for j, idx in enumerate(selected_indices)
    ]
    avg_similarity = sum(individual_scores) / 4
    
    info_dict = {
        'total_score': max_score,
        'avg_similarity': avg_similarity,
        'individual_scores': individual_scores,
        'similarity_matrix': similarity_matrix,
        'selected_indices': selected_indices,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"智能选帧结果（基于特征相似度匹配）")
        print(f"{'='*60}")
        print(f"总相似度分数: {max_score:.2f}")
        print(f"平均相似度: {avg_similarity:.2f}%")
        print(f"\n各帧详情:")
        
        phase_names = ["起始姿态", "重心转移期", "手臂下落期", "完成姿态"]
        for j, (idx, score) in enumerate(zip(selected_indices, individual_scores)):
            metrics = metrics_list[idx]
            print(f"\n  标准帧{j+1} ({phase_names[j]}): 视频帧索引 {idx}")
            print(f"    相似度: {score:.2f}%")
            print(f"    arm_height_ratio: {metrics.get('arm_height_ratio', 'N/A'):.3f} "
                  f"(标准: {STANDARD_FRAMES[j]['arm_height_ratio']:.3f})")
            print(f"    foot_distance: {metrics.get('foot_distance', 'N/A'):.3f} "
                  f"(标准: {STANDARD_FRAMES[j]['foot_distance']:.3f})")
            if j == 2:  # 第3帧显示肘部角度
                print(f"    left_elbow_angle: {metrics.get('left_elbow_angle', 'N/A'):.1f}° "
                      f"(标准: {STANDARD_FRAMES[j]['left_elbow_angle']:.1f}°)")
            if j == 3:  # 第4帧显示膝盖弯曲
                print(f"    knee_bend_average: {metrics.get('knee_bend_average', 'N/A'):.1f}° "
                      f"(标准: {STANDARD_FRAMES[j]['knee_bend_average']:.1f}°)")
        
        print(f"\n{'='*60}\n")
    
    return selected_indices, info_dict


def _fallback_uniform_selection(
    metrics_list: List[Dict],
    similarity_matrix: np.ndarray
) -> Tuple[List[int], Dict]:
    """回退策略：均匀分布+相似度优化"""
    n = len(metrics_list)
    
    # 将视频分成4段，每段选相似度最高的帧
    segment_size = n // 4
    selected_indices = []
    individual_scores = []
    
    for j in range(4):
        start = j * segment_size
        end = (j + 1) * segment_size if j < 3 else n
        
        # 在该段内找到与标准帧j最相似的帧
        best_idx = start
        best_score = similarity_matrix[start][j]
        
        for i in range(start, end):
            if similarity_matrix[i][j] > best_score:
                best_score = similarity_matrix[i][j]
                best_idx = i
        
        selected_indices.append(best_idx)
        individual_scores.append(best_score)
    
    info_dict = {
        'total_score': sum(individual_scores),
        'avg_similarity': sum(individual_scores) / 4,
        'individual_scores': individual_scores,
        'similarity_matrix': similarity_matrix,
        'selected_indices': selected_indices,
        'fallback': True,
    }
    
    return selected_indices, info_dict


def visualize_similarity_heatmap(similarity_matrix: np.ndarray, selected_indices: List[int]):
    """
    可视化相似度热力图（需要matplotlib）
    
    Args:
        similarity_matrix: [n_frames, 4] 相似度矩阵
        selected_indices: 选中的4个帧索引
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(similarity_matrix.T, aspect='auto', cmap='YlOrRd', 
                      interpolation='nearest')
        
        ax.set_xlabel('视频帧索引', fontsize=12)
        ax.set_ylabel('标准帧', fontsize=12)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['标准帧1\n(起始)', '标准帧2\n(腿部打开)', 
                           '标准帧3\n(手臂下落)', '标准帧4\n(完成)'])
        ax.set_title('视频帧与标准帧的相似度热力图', fontsize=14, pad=20)
        
        # 标记选中的帧
        for j, idx in enumerate(selected_indices):
            ax.plot(idx, j, 'b*', markersize=20, markeredgecolor='white', 
                   markeredgewidth=2)
            ax.text(idx, j, f'{similarity_matrix[idx][j]:.1f}', 
                   ha='center', va='center', color='blue', fontweight='bold',
                   fontsize=10)
        
        plt.colorbar(im, ax=ax, label='相似度分数')
        plt.tight_layout()
        plt.savefig('frame_selection_heatmap.png', dpi=150, bbox_inches='tight')
        print("✅ 相似度热力图已保存到: frame_selection_heatmap.png")
        plt.close()
        
    except ImportError:
        print("⚠️  matplotlib未安装，跳过可视化")


# ============ 使用示例 ============
if __name__ == "__main__":
    # 示例：模拟一些metrics数据
    print("测试frame_selector模块...")
    
    # 模拟数据：创建一个渐进的动作序列
    test_metrics = []
    for i in range(20):
        t = i / 19  # 0 to 1
        
        # 模拟起势动作的变化
        arm_height = 0.98 - 0.5 * t + 0.2 * (t > 0.7)  # 高->低->中
        foot_dist = 0.08 + 0.21 * t  # 近->远
        knee_bend = 2 + 20 * (t > 0.6) * (t - 0.6) / 0.4  # 后期才弯曲
        
        test_metrics.append({
            'arm_height_ratio': arm_height,
            'foot_distance': foot_dist,
            'hand_distance': 0.32,
            'knee_bend_average': knee_bend,
            'left_elbow_angle': 165 - 100 * (t > 0.4) * (t - 0.4) / 0.3,
            'right_elbow_angle': 155 - 50 * (t > 0.4) * (t - 0.4) / 0.3,
            'torso_angle': 0.5,
            'avg_visibility': 0.95,
            'hip_width': 0.13,
            'shoulder_width': 0.24,
        })
    
    indices, info = select_frames_by_similarity(test_metrics, min_frame_gap=2, verbose=True)
    
    print(f"选中的帧索引: {indices}")
    print(f"对应的arm_height_ratio: {[test_metrics[i]['arm_height_ratio'] for i in indices]}")
    print(f"对应的foot_distance: {[test_metrics[i]['foot_distance'] for i in indices]}")


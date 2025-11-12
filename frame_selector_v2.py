# -*- coding: utf-8 -*-
"""
改进版关键帧选择器 - 支持多帧抽取
用于从视频中检测起势动作并抽取N帧进行评估
"""

import numpy as np
from typing import List, Dict, Tuple


def detect_action_start_end(metrics_list: List[Dict], fps=30, verbose=True, force_start_from_beginning=True) -> Tuple[int, int]:
    """
    自动检测起势动作的开始和结束帧
    
    检测逻辑：
    1. 开始: arm_height_ratio 从低位开始上升 + foot_distance 较小
    2. 结束: knee_bend_average 达到峰值 + foot_distance 稳定
    
    Args:
        metrics_list: 所有帧的特征列表
        fps: 视频帧率
        verbose: 是否打印详细信息
    
    Returns:
        (start_idx, end_idx): 动作的开始和结束帧索引
    """
    # 提取关键特征序列
    arm_heights = []
    foot_distances = []
    knee_bends = []
    valid_indices = []
    
    for i, metrics in enumerate(metrics_list):
        if not isinstance(metrics, dict) or not metrics:
            continue
        
        arm_h = metrics.get('arm_height_ratio')
        foot_d = metrics.get('foot_distance')
        knee_b = metrics.get('knee_bend_average')
        vis = metrics.get('avg_visibility', 0)
        
        # 只考虑可见性较高的帧
        if vis > 0.5 and arm_h is not None and foot_d is not None and knee_b is not None:
            arm_heights.append(arm_h)
            foot_distances.append(foot_d)
            knee_bends.append(knee_b)
            valid_indices.append(i)
    
    if len(valid_indices) < 10:
        if verbose:
            print("[WARNING] 有效帧数不足，使用全部帧")
        return 0, len(metrics_list) - 1
    
    # 转换为numpy数组
    arm_heights = np.array(arm_heights)
    foot_distances = np.array(foot_distances)
    knee_bends = np.array(knee_bends)
    
    # === 检测开始帧 ===
    if force_start_from_beginning:
        # 强制从第一帧开始（用于训练数据提取）
        start_idx = 0
        if verbose:
            print("[INFO] 使用完整动作：从第一帧开始")
    else:
        # 智能检测：寻找起势动作的起始点
        # 起势动作的特征：手臂从较低位置开始上升，脚距较小
        
        # 策略1: 寻找手臂高度从低到高开始持续上升的点
        arm_diff = np.diff(arm_heights)
        arm_diff_smooth = np.convolve(arm_diff, np.ones(7)/7, mode='same')  # 使用更大的窗口平滑
        
        # 策略2: 结合脚距特征（起势开始时脚距应该较小）
        start_candidates = []
        for i in range(len(arm_diff_smooth) - 10):
            # 条件1: 手臂开始持续上升（接下来10帧的平均变化率为正）
            arm_rising = arm_diff_smooth[i:i+10].mean() > 0.008
            
            # 条件2: 当前手臂高度较低（< 0.7），说明还没开始或刚开始
            arm_low = arm_heights[i] < 0.7
            
            # 条件3: 脚距较小（< 0.15），说明还没迈步或刚开始
            foot_small = foot_distances[i] < 0.15
            
            # 条件4: 膝盖弯曲度较小（< 10度），说明还是起始姿态
            knee_straight = knee_bends[i] < 10
            
            if arm_rising and arm_low and foot_small and knee_straight:
                start_candidates.append(i)
        
        if start_candidates:
            # 取第一个符合条件的点
            start_idx_local = start_candidates[0]
            if verbose:
                print(f"[INFO] 智能检测到起势动作开始：第 {valid_indices[start_idx_local]} 帧")
        else:
            # 回退策略：在前半段找到手臂高度最低的点
            search_range = len(arm_heights) // 2
            min_arm_idx = np.argmin(arm_heights[:search_range])
            
            # 进一步验证：该点附近脚距是否较小
            if foot_distances[min_arm_idx] < 0.2:
                start_idx_local = min_arm_idx
            else:
                # 如果脚距已经较大，往前找脚距较小的点
                for i in range(min_arm_idx, -1, -1):
                    if foot_distances[i] < 0.15:
                        start_idx_local = i
                        break
                else:
                    start_idx_local = min_arm_idx
            
            if verbose:
                print(f"[INFO] 使用回退策略：从第 {valid_indices[start_idx_local]} 帧开始")
        
        start_idx = valid_indices[start_idx_local]
    
    # === 检测结束帧 ===
    # 策略：找到起势动作完成的标志
    # 完成标志：膝盖弯曲达到峰值 + 脚距稳定 + 手臂高度在中等位置
    
    # 1. 找到膝盖弯曲的峰值区域
    knee_max_idx_local = np.argmax(knee_bends)
    knee_max_value = knee_bends[knee_max_idx_local]
    
    # 2. 从开始帧之后寻找结束帧（确保在动作区间内）
    # 注意：valid_indices 是原始帧索引，需要找到对应的 local 索引
    if not force_start_from_beginning:
        # 找到 start_idx 在 valid_indices 中的位置
        try:
            search_start_local = valid_indices.index(start_idx)
        except ValueError:
            search_start_local = 0
    else:
        search_start_local = 0
    
    # 3. 计算脚距变化率
    foot_diff = np.diff(foot_distances)
    
    # 4. 寻找结束帧候选点（从膝盖峰值往后）
    end_search_start = max(knee_max_idx_local, search_start_local + len(knee_bends) // 3)
    
    end_candidates = []
    for i in range(end_search_start, len(foot_diff) - 5):
        # 条件1: 脚距变化很小（稳定）
        foot_stable = np.abs(foot_diff[i:i+5]).max() < 0.02
        
        # 条件2: 膝盖弯曲度较高（> 15度，说明已经完成下蹲）
        knee_bent = knee_bends[i] > 15
        
        # 条件3: 手臂高度在中等位置（0.3-0.9之间，完成姿态）
        arm_medium = 0.3 < arm_heights[i] < 0.9
        
        if foot_stable and knee_bent and arm_medium:
            end_candidates.append(i)
    
    if end_candidates:
        # 取第一个符合条件的点
        end_idx_local = end_candidates[0]
        if verbose:
            print(f"[INFO] 智能检测到起势动作结束：第 {valid_indices[end_idx_local]} 帧")
    else:
        # 回退策略：使用膝盖弯曲峰值后几帧
        end_idx_local = min(knee_max_idx_local + 10, len(valid_indices) - 1)
        if verbose:
            print(f"[INFO] 使用回退策略：从膝盖峰值后确定结束帧")
    
    end_idx = valid_indices[end_idx_local]
    
    # 如果强制从开始（训练数据提取），确保结束帧不超过465
    if force_start_from_beginning and end_idx > 465:
        end_idx = min(465, len(metrics_list) - 1)
        if verbose:
            print(f"[INFO] 限制结束帧为465（完整动作结束）")
    
    # 确保动作区间足够长
    if end_idx - start_idx < 20:
        mid = (start_idx + end_idx) // 2
        start_idx = max(0, mid - 20)
        end_idx = min(len(metrics_list) - 1, mid + 20)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"起势动作边界检测")
        print(f"{'='*60}")
        print(f"开始帧: 第 {start_idx} 帧 (时间: {start_idx/fps:.2f}s)")
        print(f"结束帧: 第 {end_idx} 帧 (时间: {end_idx/fps:.2f}s)")
        print(f"动作时长: {end_idx - start_idx + 1} 帧 ({(end_idx - start_idx + 1)/fps:.2f}s)")
        
        # 显示关键特征
        if start_idx < len(metrics_list) and end_idx < len(metrics_list):
            start_m = metrics_list[start_idx]
            end_m = metrics_list[end_idx]
            
            print(f"\n开始帧特征:")
            print(f"  手臂高度: {start_m.get('arm_height_ratio', 'N/A'):.3f}")
            print(f"  脚距: {start_m.get('foot_distance', 'N/A'):.3f}")
            print(f"  膝盖弯曲: {start_m.get('knee_bend_average', 'N/A'):.1f}°")
            
            print(f"\n结束帧特征:")
            print(f"  手臂高度: {end_m.get('arm_height_ratio', 'N/A'):.3f}")
            print(f"  脚距: {end_m.get('foot_distance', 'N/A'):.3f}")
            print(f"  膝盖弯曲: {end_m.get('knee_bend_average', 'N/A'):.1f}°")
        
        print(f"{'='*60}\n")
    
    return start_idx, end_idx


def select_frames_evenly(metrics_list: List[Dict], n_frames=20, 
                         auto_detect_boundaries=True, fps=30, verbose=True, force_start_from_beginning=False) -> Tuple[List[int], List[Dict]]:
    """
    从视频中选择N帧关键帧
    
    Args:
        metrics_list: 所有帧的特征列表
        n_frames: 要选择的帧数（默认12）
        auto_detect_boundaries: 是否自动检测动作边界
        fps: 视频帧率
        verbose: 是否打印详细信息
    
    Returns:
        (selected_indices, selected_frames): 选中的帧索引和特征
    """
    if len(metrics_list) < n_frames:
        raise ValueError(f"视频帧数({len(metrics_list)})少于要求的帧数({n_frames})")
    
    # 自动检测起势动作边界
    if auto_detect_boundaries:
        start_idx, end_idx = detect_action_start_end(metrics_list, fps, verbose, force_start_from_beginning)
    else:
        start_idx = 0
        end_idx = len(metrics_list) - 1
    
    # 在动作区间内均匀采样
    action_length = end_idx - start_idx + 1
    
    if action_length < n_frames:
        # 动作区间太短，使用全部帧并补足
        selected_indices = list(range(start_idx, end_idx + 1))
        # 重复最后一帧补足
        while len(selected_indices) < n_frames:
            selected_indices.append(end_idx)
    else:
        # 均匀采样
        selected_indices = []
        for i in range(n_frames):
            ratio = i / (n_frames - 1) if n_frames > 1 else 0
            frame_idx = int(start_idx + ratio * (end_idx - start_idx))
            selected_indices.append(frame_idx)
    
    selected_frames = [metrics_list[idx] for idx in selected_indices]
    
    if verbose:
        print(f"选帧结果:")
        print(f"  选中帧数: {len(selected_indices)}")
        print(f"  帧索引: {selected_indices}")
        print(f"  时间跨度: {selected_indices[0]/fps:.2f}s - {selected_indices[-1]/fps:.2f}s\n")
    
    return selected_indices, selected_frames


# === 使用示例 ===
if __name__ == "__main__":
    print("测试改进版关键帧选择器...")
    
    # 模拟一些metrics数据（模拟起势动作）
    test_metrics = []
    for i in range(60):
        t = i / 59  # 0 to 1
        
        # 模拟起势动作的变化
        # 前期：手臂低 -> 中期：手臂高 -> 后期：手臂中等
        if t < 0.3:
            arm_height = 0.1 + t * 0.5
        elif t < 0.7:
            arm_height = 0.95 - (t - 0.3) * 0.6
        else:
            arm_height = 0.65 + (t - 0.7) * 0.4
        
        foot_dist = 0.08 + 0.20 * t  # 逐渐增大
        knee_bend = 2 + 25 * (t > 0.5) * (t - 0.5) / 0.5  # 后期弯曲
        
        test_metrics.append({
            'arm_height_ratio': arm_height,
            'foot_distance': foot_dist,
            'hand_distance': 0.32,
            'knee_bend_average': knee_bend,
            'left_elbow_angle': 165 - 100 * max(0, (t - 0.3)) / 0.4,
            'right_elbow_angle': 155 - 80 * max(0, (t - 0.3)) / 0.4,
            'left_knee_angle': 178 - 25 * (t > 0.5) * (t - 0.5) / 0.5,
            'right_knee_angle': 178 - 25 * (t > 0.5) * (t - 0.5) / 0.5,
            'torso_angle': 0.5 + t * 0.5,
            'avg_visibility': 0.95,
            'hip_width': 0.13,
            'shoulder_width': 0.24,
            'shoulder_to_hip_y': 0.24,
            'left_shoulder_angle': 35,
            'right_shoulder_angle': 35,
            'shoulder_balance_left': 0.12,
            'shoulder_balance_right': -0.12,
            'hip_balance_left': 0.07,
            'hip_balance_right': -0.07,
            'nose_center_offset': 0.0,
            'torso_vx': 0.0,
            'torso_vy': -0.24,
        })
    
    # 测试选帧
    indices, frames = select_frames_evenly(test_metrics, n_frames=12, fps=30, verbose=True)
    
    print(f"选中帧的arm_height_ratio:")
    for i, idx in enumerate(indices):
        print(f"  帧{i+1}: {test_metrics[idx]['arm_height_ratio']:.3f}")
    
    print(f"\n选中帧的foot_distance:")
    for i, idx in enumerate(indices):
        print(f"  帧{i+1}: {test_metrics[idx]['foot_distance']:.3f}")


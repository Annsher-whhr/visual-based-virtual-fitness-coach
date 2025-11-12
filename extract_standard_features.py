# -*- coding: utf-8 -*-
"""
从标准视频 qishi3.mp4 中提取所有帧的姿态特征
用于生成更丰富的训练数据
"""

import cv2
import numpy as np
import json
import os
from src.pose_estimation_v2 import estimate_pose
from src.action_recognition import recognize_action
from src.detection import detect_human


def extract_all_features_from_video(video_path, output_json=None, verbose=True):
    """
    从视频中提取所有帧的姿态特征
    
    Args:
        video_path: 视频文件路径
        output_json: 输出JSON文件路径（可选）
        verbose: 是否打印详细信息
    
    Returns:
        features_list: 所有帧的特征列表
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"从标准视频提取特征")
        print(f"{'='*60}")
        print(f"视频: {video_path}")
        print(f"FPS: {fps:.2f}")
        print(f"总帧数: {total_frames}")
        print(f"{'='*60}\n")
    
    features_list = []
    frame_count = 0
    valid_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测人体
        frame_with_detection = detect_human(frame)
        
        # 姿态估计
        _, results = estimate_pose(frame_with_detection)
        
        # 提取特征
        metrics = recognize_action(results)
        
        # 保存特征（即使为空也保存，以保持帧索引连续）
        features_list.append({
            'frame_index': frame_count,
            'timestamp': frame_count / fps if fps > 0 else 0,
            'metrics': metrics if isinstance(metrics, dict) else {}
        })
        
        if metrics and isinstance(metrics, dict):
            valid_count += 1
        
        frame_count += 1
        if verbose and frame_count % 10 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧 (有效: {valid_count})...", end='\r')
    
    cap.release()
    
    if verbose:
        print(f"\n\n[OK] 提取完成!")
        print(f"   总帧数: {frame_count}")
        print(f"   有效帧: {valid_count}")
        print(f"   有效率: {valid_count/frame_count*100:.1f}%\n")
    
    # 保存到JSON文件
    if output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(features_list, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"[OK] 特征已保存到: {output_json}\n")
    
    return features_list


def detect_action_boundaries(features_list, fps=30, verbose=True, force_start_from_beginning=True):
    """
    检测起势动作的开始和结束帧
    
    基于以下特征变化来检测：
    1. 开始: 从第一帧开始（完整动作）或自动检测
    2. 结束: knee_bend_average 达到峰值 + foot_distance 达到最大值并稳定
    
    Args:
        features_list: 所有帧的特征列表
        fps: 视频帧率（用于时间计算）
        verbose: 是否打印详细信息
        force_start_from_beginning: 是否强制从第一帧开始（默认True）
    
    Returns:
        (start_idx, end_idx): 起势动作的开始和结束帧索引
    """
    # 提取关键特征序列
    arm_heights = []
    foot_distances = []
    knee_bends = []
    torso_angles = []
    valid_indices = []
    
    for i, item in enumerate(features_list):
        metrics = item.get('metrics', {})
        if not metrics or not isinstance(metrics, dict):
            continue
        
        arm_h = metrics.get('arm_height_ratio')
        foot_d = metrics.get('foot_distance')
        knee_b = metrics.get('knee_bend_average')
        torso_a = metrics.get('torso_angle')
        vis = metrics.get('avg_visibility', 0)
        
        # 只考虑可见性较高的帧
        if vis > 0.5 and arm_h is not None and foot_d is not None and knee_b is not None:
            arm_heights.append(arm_h)
            foot_distances.append(foot_d)
            knee_bends.append(knee_b)
            torso_angles.append(torso_a if torso_a is not None else 0)
            valid_indices.append(i)
    
    if len(valid_indices) < 10:
        # 数据不足，使用全部帧
        if verbose:
            print("[WARNING] 有效帧数不足，使用全部帧")
        return 0, len(features_list) - 1
    
    # 转换为numpy数组便于分析（无论是否强制从开始，都需要这些数组来检测结束帧）
    arm_heights = np.array(arm_heights)
    foot_distances = np.array(foot_distances)
    knee_bends = np.array(knee_bends)
    torso_angles = np.array(torso_angles)
    
    # === 检测开始帧 ===
    if force_start_from_beginning:
        # 强制从第一帧开始（完整起势动作）
        start_idx = 0
        if verbose:
            print("[INFO] 使用完整动作：从第一帧开始")
    else:
        # 自动检测开始帧：基于身体直立和左腿开始张开
        # 起势动作开始的特征：
        # 1. 身体直立：膝盖较直（< 10°），躯干角度小（< 5°）
        # 2. 左腿开始往左张开：脚距较小（< 0.15）但即将开始增大（脚距变化率开始为正）
        # 3. 手臂保持不动：手臂高度较低（< 0.3）且稳定（变化不大）
        
        # 计算脚距变化率（检测左腿开始张开）
        foot_diff = np.diff(foot_distances)
        foot_diff_smooth = np.convolve(foot_diff, np.ones(5)/5, mode='same')
        
        # 计算手臂高度变化率（检测手臂是否稳定）
        arm_diff = np.diff(arm_heights)
        arm_diff_abs = np.abs(arm_diff)
        
        start_candidates = []
        for i in range(len(foot_diff_smooth) - 8):
            # 条件1: 身体直立
            knee_straight = knee_bends[i] < 10
            torso_upright = abs(torso_angles[i]) < 5
            
            # 条件2: 脚距较小，但即将开始增大（左腿开始张开）
            foot_small = foot_distances[i] < 0.15
            foot_starting_to_open = foot_diff_smooth[i:i+8].mean() > 0.003
            
            # 条件3: 手臂保持不动（较低且稳定）
            arm_low = arm_heights[i] < 0.3
            arm_stable = arm_diff_abs[max(0, i-3):i+1].mean() < 0.02 if i >= 3 else True
            
            if knee_straight and torso_upright and foot_small and foot_starting_to_open and arm_low and arm_stable:
                start_candidates.append(i)
        
        if start_candidates:
            start_idx_local = start_candidates[0]
            if verbose:
                print(f"[INFO] 智能检测到起势动作开始：第 {valid_indices[start_idx_local]} 帧")
        else:
            # 回退策略：在前半段找到脚距最小且身体直立的点
            search_range = len(foot_distances) // 2
            candidates = []
            for i in range(search_range):
                if knee_bends[i] < 10 and abs(torso_angles[i]) < 5 and foot_distances[i] < 0.15:
                    candidates.append((i, foot_distances[i]))
            
            if candidates:
                start_idx_local = min(candidates, key=lambda x: x[1])[0]
            else:
                start_idx_local = np.argmin(foot_distances[:search_range])
            
            if verbose:
                print(f"[INFO] 使用回退策略：从第 {valid_indices[start_idx_local]} 帧开始")
        
        start_idx = valid_indices[start_idx_local]
    
    # === 检测结束帧 ===
    # 标准：knee_bend_average 达到峰值 + foot_distance 达到最大并稳定
    
    # 找到膝盖弯曲度最大的区域
    knee_max_idx_local = np.argmax(knee_bends)
    
    # 从膝盖弯曲最大处往后找，foot_distance稳定的点
    end_search_start = max(knee_max_idx_local, len(knee_bends) // 2)
    
    # 计算脚距的变化率
    foot_diff = np.diff(foot_distances)
    
    # 找到脚距变化很小的点（动作完成并稳定）
    end_candidates = []
    for i in range(end_search_start, len(foot_diff) - 3):
        # 如果接下来3帧的脚距变化都很小
        if np.abs(foot_diff[i:i+3]).max() < 0.015:
            end_candidates.append(i)
    
    if end_candidates:
        end_idx_local = end_candidates[0]  # 取第一个稳定点
    else:
        # 回退：使用膝盖弯曲最大的点后几帧
        end_idx_local = min(knee_max_idx_local + 5, len(valid_indices) - 1)
    
    end_idx = valid_indices[end_idx_local]
    
    # 确保结束帧不超过465（根据用户要求）
    if end_idx > 465:
        end_idx = 465
        if verbose:
            print(f"[INFO] 限制结束帧为465（完整动作结束）")
    
    # 确保有足够的帧数
    if end_idx - start_idx < 10:
        # 动作区间太短，扩展一下
        mid = (start_idx + end_idx) // 2
        start_idx = max(0, mid - 15)
        end_idx = min(len(features_list) - 1, mid + 15)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"起势动作边界检测结果")
        print(f"{'='*60}")
        print(f"开始帧索引: {start_idx} (时间: {features_list[start_idx]['timestamp']:.2f}s)")
        print(f"结束帧索引: {end_idx} (时间: {features_list[end_idx]['timestamp']:.2f}s)")
        print(f"动作持续: {end_idx - start_idx + 1} 帧 ({(end_idx - start_idx + 1) / fps:.2f}s)")
        
        # 显示开始帧和结束帧的特征
        start_m = features_list[start_idx]['metrics']
        end_m = features_list[end_idx]['metrics']
        
        print(f"\n开始帧特征:")
        print(f"  arm_height_ratio: {start_m.get('arm_height_ratio', 'N/A'):.3f}")
        print(f"  foot_distance: {start_m.get('foot_distance', 'N/A'):.3f}")
        print(f"  knee_bend_average: {start_m.get('knee_bend_average', 'N/A'):.3f}")
        
        print(f"\n结束帧特征:")
        print(f"  arm_height_ratio: {end_m.get('arm_height_ratio', 'N/A'):.3f}")
        print(f"  foot_distance: {end_m.get('foot_distance', 'N/A'):.3f}")
        print(f"  knee_bend_average: {end_m.get('knee_bend_average', 'N/A'):.3f}")
        print(f"{'='*60}\n")
    
    return start_idx, end_idx


def extract_evenly_spaced_frames(features_list, start_idx, end_idx, n_frames=12):
    """
    从动作区间内均匀抽取N帧
    
    Args:
        features_list: 所有帧的特征列表
        start_idx: 开始帧索引
        end_idx: 结束帧索引  
        n_frames: 要抽取的帧数（默认12帧）
    
    Returns:
        selected_features: 选中的N帧特征列表
        selected_indices: 选中的帧索引列表
    """
    action_length = end_idx - start_idx + 1
    
    if action_length < n_frames:
        # 如果动作帧数少于目标帧数，使用全部帧
        selected_indices = list(range(start_idx, end_idx + 1))
    else:
        # 均匀采样
        selected_indices = []
        for i in range(n_frames):
            # 计算在动作区间内的相对位置
            ratio = i / (n_frames - 1) if n_frames > 1 else 0
            frame_idx = int(start_idx + ratio * (end_idx - start_idx))
            selected_indices.append(frame_idx)
    
    selected_features = [features_list[idx]['metrics'] for idx in selected_indices]
    
    return selected_features, selected_indices


if __name__ == "__main__":
    # 提取qishi3.mp4的特征
    video_path = "video/qishi3.mp4"
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频文件不存在: {video_path}")
        exit(1)
    
    print("步骤1: 提取视频所有帧的特征...")
    features_list = extract_all_features_from_video(
        video_path, 
        output_json=os.path.join("data", "standard", "qishi3_all_features.json"),
        verbose=True
    )
    
    # 获取视频FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    print("\n步骤2: 检测起势动作的开始和结束...")
    # 强制从第一帧开始，到第465帧结束（完整起势动作）
    start_idx, end_idx = detect_action_boundaries(
        features_list, 
        fps=fps, 
        verbose=True,
        force_start_from_beginning=True
    )
    
    # 确保结束帧为465（完整动作）
    if end_idx > 465:
        end_idx = 465
    elif end_idx < 465:
        # 如果检测到的结束帧小于465，使用465
        end_idx = min(465, len(features_list) - 1)
        print(f"[INFO] 调整结束帧为465（完整动作）")
    
    print("\n步骤3: 从动作区间均匀抽取20帧作为标准数据...")
    standard_features, standard_indices = extract_evenly_spaced_frames(
        features_list, start_idx, end_idx, n_frames=20
    )
    
    print(f"选中的帧索引: {standard_indices}")
    
    # 保存标准帧特征
    output = {
        'video_path': video_path,
        'fps': cap.get(cv2.CAP_PROP_FPS) if 'cap' in dir() else 30,
        'total_frames': len(features_list),
        'action_start_idx': start_idx,
        'action_end_idx': end_idx,
        'n_standard_frames': len(standard_features),
        'standard_frame_indices': standard_indices,
        'standard_features': standard_features
    }
    
    # 确保输出目录存在
    output_dir = os.path.join("data", "standard")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qishi3_standard_frames.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 标准帧特征已保存到: {output_path}")
    print(f"   共 {len(standard_features)} 帧标准数据")
    
    # 显示每一帧的关键特征
    print(f"\n{'='*60}")
    print(f"标准帧特征摘要")
    print(f"{'='*60}")
    for i, (idx, feat) in enumerate(zip(standard_indices, standard_features)):
        print(f"\n帧 {i+1}/{len(standard_features)} (视频帧索引: {idx}):")
        print(f"  arm_height_ratio: {feat.get('arm_height_ratio', 'N/A'):.3f}")
        print(f"  foot_distance: {feat.get('foot_distance', 'N/A'):.3f}")
        print(f"  knee_bend_average: {feat.get('knee_bend_average', 'N/A'):.3f}")
        print(f"  left_elbow_angle: {feat.get('left_elbow_angle', 'N/A'):.1f}°")
        print(f"  right_elbow_angle: {feat.get('right_elbow_angle', 'N/A'):.1f}°")
        print(f"  avg_visibility: {feat.get('avg_visibility', 'N/A'):.3f}")
    
    print(f"\n{'='*60}\n")


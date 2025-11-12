# -*- coding: utf-8 -*-
"""
改进版数据生成器 - 支持多帧训练数据
基于从qishi3.mp4提取的20帧标准数据（从0-465帧均匀采样）
"""

import os
import numpy as np
import json
import random

# === 统一的特征顺序（与 action_recognition.py 保持一致）===
FEATURE_ORDER = [
    'shoulder_width', 'hip_width', 'foot_distance', 'hand_distance',
    'shoulder_to_hip_y', 'left_knee_angle', 'right_knee_angle',
    'left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle',
    'right_shoulder_angle', 'knee_bend_average', 'torso_angle',
    'torso_vx', 'torso_vy', 'arm_height_ratio',
    'shoulder_balance_left', 'shoulder_balance_right',
    'hip_balance_left', 'hip_balance_right',
    'nose_center_offset', 'avg_visibility'
]

# === 从标准视频提取的标准帧特征（动态帧数，当前为20帧）===
def load_standard_frames():
    """从qishi3_standard_frames.json加载标准帧数据"""
    # 尝试从多个位置加载
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    
    candidates = [
        os.path.join(project_root, "qishi3_standard_frames.json"),
        "qishi3_standard_frames.json",
        os.path.join(base_dir, "qishi3_standard_frames.json"),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[OK] 已加载标准帧数据: {path}")
            print(f"     包含 {data['n_standard_frames']} 帧标准数据")
            return data['standard_features']
    
    raise FileNotFoundError(
        f"未找到标准帧数据文件 qishi3_standard_frames.json\n"
        f"请先运行 extract_standard_features.py 生成标准数据\n"
        f"查找路径: {candidates}"
    )

# 加载标准帧
try:
    standard_frames = load_standard_frames()
    N_FRAMES = len(standard_frames)
    print(f"     动作包含 {N_FRAMES} 个关键帧\n")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    # 使用默认的20帧空数据作为占位（实际应从JSON加载）
    standard_frames = [{} for _ in range(20)]
    N_FRAMES = 20
    print(f"[WARNING] 使用空数据占位，请先生成标准数据！\n")


def make_variation(base, scale=0.08):
    """
    对特征做小扰动（增加扰动幅度以提高数据多样性）
    
    Args:
        base: 基础特征字典
        scale: 扰动比例（默认8%）
    
    Returns:
        扰动后的特征字典
    """
    varied = {}
    for k, v in base.items():
        if v is None:
            varied[k] = None
        else:
            # 对不同类型的特征使用不同的扰动策略
            if 'angle' in k:
                # 角度特征：绝对扰动 ±5度
                varied[k] = v + np.random.uniform(-5, 5)
            elif 'visibility' in k:
                # 可见性：保持在[0, 1]范围内
                varied[k] = max(0.0, min(1.0, v * (1 + np.random.uniform(-scale/2, scale/2))))
            else:
                # 其他特征：相对扰动
                varied[k] = v * (1 + np.random.uniform(-scale, scale))
    
    return varied


def make_error(base, error_type):
    """
    根据错误类型施加特定扰动
    
    Args:
        base: 基础特征字典
        error_type: 错误类型
    
    Returns:
        包含错误的特征字典
    """
    e = base.copy()
    
    if error_type == "shoulder_high":
        # 肩膀不平：左肩高右肩低
        e["shoulder_balance_left"] = e.get("shoulder_balance_left", 0) + random.uniform(0.15, 0.35)
        e["shoulder_balance_right"] = e.get("shoulder_balance_right", 0) - random.uniform(0.15, 0.35)
    
    elif error_type == "knee_not_bent":
        # 屈膝不足：膝盖弯曲度减小
        current_bend = e.get("knee_bend_average", 20)
        e["knee_bend_average"] = max(0, current_bend - random.uniform(10, 18))
        # 相应地调整膝盖角度（更接近180度=直立）
        e["left_knee_angle"] = e.get("left_knee_angle", 160) + random.uniform(10, 20)
        e["right_knee_angle"] = e.get("right_knee_angle", 160) + random.uniform(10, 20)
    
    elif error_type == "arm_too_low":
        # 手臂偏低
        current_ratio = e.get("arm_height_ratio", 0.5)
        e["arm_height_ratio"] = current_ratio - random.uniform(0.25, 0.45)
    
    elif error_type == "arm_too_high":
        # 手臂偏高
        current_ratio = e.get("arm_height_ratio", 0.5)
        e["arm_height_ratio"] = current_ratio + random.uniform(0.25, 0.45)
    
    elif error_type == "torso_lean":
        # 躯干前倾或后仰
        e["torso_angle"] = e.get("torso_angle", 1) + random.uniform(8, 18)
    
    elif error_type == "foot_too_close":
        # 脚间距过窄
        e["foot_distance"] = e.get("foot_distance", 0.25) - random.uniform(0.08, 0.15)
    
    elif error_type == "foot_too_wide":
        # 脚间距过宽（新增）
        e["foot_distance"] = e.get("foot_distance", 0.25) + random.uniform(0.12, 0.25)
    
    elif error_type == "elbow_not_bent":
        # 肘部弯曲不足（新增）
        e["left_elbow_angle"] = min(175, e.get("left_elbow_angle", 140) + random.uniform(20, 35))
        e["right_elbow_angle"] = min(175, e.get("right_elbow_angle", 140) + random.uniform(20, 35))
    
    return e


def generate_dataset(n_correct=800, n_wrong=800):
    """
    生成训练数据集
    
    Args:
        n_correct: 正确样本数量
        n_wrong: 错误样本数量
    
    Returns:
        保存X.npy, y.npy, errors.json到taichi_ai目录
    """
    X, y, errors = [], [], []
    error_types = [
        "shoulder_high", "knee_not_bent", "arm_too_low", "arm_too_high", 
        "torso_lean", "foot_too_close", "foot_too_wide", "elbow_not_bent"
    ]
    
    print(f"正在生成训练数据...")
    print(f"  正确样本: {n_correct}")
    print(f"  错误样本: {n_wrong}")
    print(f"  每样本帧数: {N_FRAMES}")
    print(f"  每帧特征数: {len(FEATURE_ORDER)}")
    print(f"  总特征维度: {N_FRAMES * len(FEATURE_ORDER)}\n")
    
    # === 生成正确样本 ===
    for i in range(n_correct):
        frames = [make_variation(f, 0.08) for f in standard_frames]
        
        # 使用 FEATURE_ORDER 确保特征顺序一致
        frame_values = []
        for f in frames:
            frame_values.extend([f.get(key, 0.0) for key in FEATURE_ORDER])
        
        X.append(np.array(frame_values))
        y.append(1)
        errors.append([])
        
        if (i + 1) % 200 == 0:
            print(f"  生成正确样本: {i + 1}/{n_correct}")
    
    print(f"[OK] 正确样本生成完成\n")
    
    # === 生成错误样本 ===
    for i in range(n_wrong):
        # 随机选择1-2种错误类型组合
        n_errors = random.choice([1, 1, 1, 2])  # 多数为单一错误，少数为组合错误
        selected_errors = random.sample(error_types, n_errors)
        
        # 对所有帧应用错误
        frames = []
        for base_frame in standard_frames:
            frame = base_frame.copy()
            # 应用选中的所有错误类型
            for err_type in selected_errors:
                frame = make_error(frame, err_type)
            # 再加上随机扰动
            frame = make_variation(frame, 0.05)
            frames.append(frame)
        
        # 使用 FEATURE_ORDER 确保特征顺序一致
        frame_values = []
        for f in frames:
            frame_values.extend([f.get(key, 0.0) for key in FEATURE_ORDER])
        
        X.append(np.array(frame_values))
        y.append(0)
        errors.append(selected_errors)
        
        if (i + 1) % 200 == 0:
            print(f"  生成错误样本: {i + 1}/{n_wrong}")
    
    print(f"[OK] 错误样本生成完成\n")
    
    # === 打乱数据 ===
    indices = list(range(len(X)))
    random.shuffle(indices)
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    errors = [errors[i] for i in indices]
    
    # === 保存数据 ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(base_dir, "X.npy")
    y_path = os.path.join(base_dir, "y.npy")
    errors_path = os.path.join(base_dir, "errors.json")
    
    np.save(X_path, np.array(X))
    np.save(y_path, np.array(y))
    
    with open(errors_path, "w", encoding='utf-8') as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    
    print(f"{'='*60}")
    print(f"数据集生成完成!")
    print(f"{'='*60}")
    print(f"总样本数: {len(X)}")
    print(f"  - 正确样本: {sum(y)}")
    print(f"  - 错误样本: {len(y) - sum(y)}")
    print(f"特征维度: {X[0].shape[0]} ({N_FRAMES} 帧 × {len(FEATURE_ORDER)} 特征/帧)")
    print(f"\n保存位置:")
    print(f"  {X_path}")
    print(f"  {y_path}")
    print(f"  {errors_path}")
    print(f"{'='*60}\n")
    
    return X, y, errors


if __name__ == "__main__":
    generate_dataset(n_correct=800, n_wrong=800)


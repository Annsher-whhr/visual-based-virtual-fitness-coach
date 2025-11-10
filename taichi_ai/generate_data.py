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

# === 标准动作的4帧特征（按照统一顺序组织）===
standard_frames = [
    # 第1帧 - 起始姿势
    {
        'shoulder_width': 0.245, 'hip_width': 0.133, 'foot_distance': 0.077,
        'hand_distance': 0.328, 'shoulder_to_hip_y': 0.238,
        'left_knee_angle': 177.937, 'right_knee_angle': 178.658,
        'left_elbow_angle': 167.309, 'right_elbow_angle': 148.420,
        'left_shoulder_angle': 29.407, 'right_shoulder_angle': 36.713,
        'knee_bend_average': 1.703, 'torso_angle': 0.256,
        'torso_vx': -0.001, 'torso_vy': -0.238,
        'arm_height_ratio': 0.984,
        'shoulder_balance_left': 0.123, 'shoulder_balance_right': -0.123,
        'hip_balance_left': 0.067, 'hip_balance_right': -0.067,
        'nose_center_offset': -0.001, 'avg_visibility': 0.964
    },
    
    # 第2帧 - 重心转移期（更新后的标准值）
    {
        'shoulder_width': 0.350, 'hip_width': 0.201, 'foot_distance': 0.405,
        'hand_distance': 0.480, 'shoulder_to_hip_y': 0.276,
        'left_knee_angle': 176.670, 'right_knee_angle': 174.568,
        'left_elbow_angle': 169.214, 'right_elbow_angle': 165.861,
        'left_shoulder_angle': 32.429, 'right_shoulder_angle': 36.195,
        'knee_bend_average': 4.381, 'torso_angle': 0.592,
        'torso_vx': 0.003, 'torso_vy': -0.276,
        'arm_height_ratio': 0.967,
        'shoulder_balance_left': 0.175, 'shoulder_balance_right': -0.175,
        'hip_balance_left': 0.101, 'hip_balance_right': -0.101,
        'nose_center_offset': -0.012, 'avg_visibility': 0.980
    },
    
    # 第3帧 - 手臂下落期（更新后的标准值）
    {
        'shoulder_width': 0.218, 'hip_width': 0.133, 'foot_distance': 0.252,
        'hand_distance': 0.358, 'shoulder_to_hip_y': 0.243,
        'left_knee_angle': 174.491, 'right_knee_angle': 168.124,
        'left_elbow_angle': 70.112, 'right_elbow_angle': 49.999,
        'left_shoulder_angle': 68.890, 'right_shoulder_angle': 62.062,
        'knee_bend_average': 8.692, 'torso_angle': 0.303,
        'torso_vx': 0.001, 'torso_vy': -0.243,
        'arm_height_ratio': -0.036,
        'shoulder_balance_left': 0.109, 'shoulder_balance_right': -0.109,
        'hip_balance_left': 0.066, 'hip_balance_right': -0.066,
        'nose_center_offset': -0.008, 'avg_visibility': 0.993
    },
    
    # 第4帧 - 完成姿势
    {
        'shoulder_width': 0.244, 'hip_width': 0.140, 'foot_distance': 0.285,
        'hand_distance': 0.310, 'shoulder_to_hip_y': 0.243,
        'left_knee_angle': 154.360, 'right_knee_angle': 161.558,
        'left_elbow_angle': 140.754, 'right_elbow_angle': 141.849,
        'left_shoulder_angle': 35.928, 'right_shoulder_angle': 36.090,
        'knee_bend_average': 22.041, 'torso_angle': 1.074,
        'torso_vx': -0.005, 'torso_vy': -0.243,
        'arm_height_ratio': 0.625,
        'shoulder_balance_left': 0.122, 'shoulder_balance_right': -0.122,
        'hip_balance_left': 0.070, 'hip_balance_right': -0.070,
        'nose_center_offset': -0.003, 'avg_visibility': 0.994
    }
]

def make_variation(base, scale=0.05):
    """对特征做小扰动"""
    return {k: v * (1 + np.random.uniform(-scale, scale)) for k, v in base.items()}

def make_error(base, error_type):
    """根据错误类型施加特定扰动"""
    e = base.copy()
    if error_type == "shoulder_high":
        e["shoulder_balance_left"] += random.uniform(0.2, 0.4)
        e["shoulder_balance_right"] -= random.uniform(0.2, 0.4)
    elif error_type == "knee_not_bent":
        e["knee_bend_average"] -= random.uniform(8, 15)
    elif error_type == "arm_too_low":
        e["arm_height_ratio"] -= random.uniform(0.3, 0.5)
    elif error_type == "arm_too_high":
        e["arm_height_ratio"] += random.uniform(0.3, 0.5)
    elif error_type == "torso_lean":
        e["torso_angle"] += random.uniform(8, 15)
    elif error_type == "foot_too_close":
        e["foot_distance"] -= random.uniform(0.1, 0.2)
    return e

def generate_dataset(n_correct=500, n_wrong=500):
    X, y, errors = [], [], []
    error_types = ["shoulder_high", "knee_not_bent", "arm_too_low", "arm_too_high", "torso_lean", "foot_too_close"]

    # 正确样本 - 按照统一的特征顺序提取
    for _ in range(n_correct):
        frames = [make_variation(f, 0.05) for f in standard_frames]
        # 使用 FEATURE_ORDER 确保特征顺序一致
        frame_values = []
        for f in frames:
            frame_values.extend([f[key] for key in FEATURE_ORDER])
        X.append(np.array(frame_values))
        y.append(1)
        errors.append([])

    # 错误样本 - 按照统一的特征顺序提取
    for _ in range(n_wrong):
        err = random.choice(error_types)
        frames = [make_error(f, err) for f in standard_frames]
        # 使用 FEATURE_ORDER 确保特征顺序一致
        frame_values = []
        for f in frames:
            frame_values.extend([f[key] for key in FEATURE_ORDER])
        X.append(np.array(frame_values))
        y.append(0)
        errors.append([err])

    # 保存到脚本目录，确保从任意当前目录运行都能找到文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    X_path = os.path.join(base_dir, "X.npy")
    y_path = os.path.join(base_dir, "y.npy")
    errors_path = os.path.join(base_dir, "errors.json")
    np.save(X_path, np.array(X))
    np.save(y_path, np.array(y))
    json.dump(errors, open(errors_path, "w", encoding='utf-8'), ensure_ascii=False)
    print(f"✅ Generated dataset: {len(X)} samples with {len(FEATURE_ORDER)} features/frame × 4 frames")
    print(f"   Total features per sample: {len(FEATURE_ORDER) * 4}")
    print(f"Saved: {X_path}\n       {y_path}\n       {errors_path}")

if __name__ == "__main__":
    generate_dataset()

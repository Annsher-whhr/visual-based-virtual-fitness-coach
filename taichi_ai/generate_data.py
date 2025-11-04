import os
import numpy as np
import json
import random

# === 标准动作的4帧特征（你提供的） ===
standard_frames = [
    { "arm_height_ratio": 0.984, "avg_visibility": 0.964, "foot_distance": 0.077,
      "hand_distance": 0.328, "hip_balance_left": 0.067, "hip_balance_right": -0.067,
      "hip_width": 0.133, "knee_bend_average": 1.703, "left_elbow_angle": 167.309,
      "left_knee_angle": 177.937, "left_shoulder_angle": 29.407, "nose_center_offset": -0.001,
      "right_elbow_angle": 148.420, "right_knee_angle": 178.658, "right_shoulder_angle": 36.713,
      "shoulder_balance_left": 0.123, "shoulder_balance_right": -0.123,
      "shoulder_to_hip_y": 0.238, "shoulder_width": 0.245, "torso_angle": 0.256,
      "torso_vx": -0.001, "torso_vy": -0.238 },

    { "arm_height_ratio": 0.932, "avg_visibility": 0.977, "foot_distance": 0.187,
      "hand_distance": 0.315, "hip_balance_left": 0.062, "hip_balance_right": -0.062,
      "hip_width": 0.124, "knee_bend_average": 4.468, "left_elbow_angle": 166.809,
      "left_knee_angle": 176.304, "left_shoulder_angle": 29.215, "nose_center_offset": -0.002,
      "right_elbow_angle": 163.607, "right_knee_angle": 174.760, "right_shoulder_angle": 33.937,
      "shoulder_balance_left": 0.109, "shoulder_balance_right": -0.109,
      "shoulder_to_hip_y": 0.222, "shoulder_width": 0.218, "torso_angle": 0.957,
      "torso_vx": -0.004, "torso_vy": -0.222 },

    { "arm_height_ratio": -0.036, "avg_visibility": 0.993, "foot_distance": 0.252,
      "hand_distance": 0.358, "hip_balance_left": 0.066, "hip_balance_right": -0.066,
      "hip_width": 0.133, "knee_bend_average": 8.692, "left_elbow_angle": 33.873,
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

features = list(standard_frames[0].keys())

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

    # 正确样本
    for _ in range(n_correct):
        frames = [make_variation(f, 0.05) for f in standard_frames]
        X.append(np.concatenate([list(f.values()) for f in frames]))
        y.append(1)
        errors.append([])

    # 错误样本
    for _ in range(n_wrong):
        err = random.choice(error_types)
        frames = [make_error(f, err) for f in standard_frames]
        X.append(np.concatenate([list(f.values()) for f in frames]))
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
    print(f"✅ Generated dataset: {len(X)} samples with {len(features)} features/frame × 4 frames")
    print(f"Saved: {X_path}\n       {y_path}\n       {errors_path}")

if __name__ == "__main__":
    generate_dataset()

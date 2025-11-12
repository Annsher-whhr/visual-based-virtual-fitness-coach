# -*- coding: utf-8 -*-
"""
改进版预测模块 - 支持多帧输入
基于20帧标准数据进行动作质量评估（440维特征 = 20帧 × 22特征/帧）
"""

import os
import numpy as np
from tensorflow import keras
import joblib

# === 统一的特征顺序 ===
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

# === 加载模型和标准化器 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 优先使用v2版本的模型
MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
TRAINING_DIR = os.path.join(PROJECT_ROOT, 'data', 'training')
MODEL_PATH = os.path.join(MODELS_DIR, 'taichi_mlp_v2.h5')
SCALER_PATH = os.path.join(TRAINING_DIR, 'scaler.pkl')

# 如果v2不存在，回退到v1（向后兼容）
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'archive', 'v1', 'taichi_mlp.h5')
    SCALER_PATH = None  # v1可能没有scaler
    print("[WARNING] 未找到v2模型，使用v1模型（不支持标准化）")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

print(f"加载模型: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)

# 加载标准化器（如果存在）
scaler = None
if SCALER_PATH and os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print(f"加载标准化器: {SCALER_PATH}")
else:
    print("[WARNING] 未找到标准化器，跳过数据标准化")

# 获取模型期望的输入维度
expected_dim = model.input_shape[1]
n_frames_expected = expected_dim // len(FEATURE_ORDER)

print(f"[OK] 模型已加载")
print(f"     输入维度: {expected_dim}")
print(f"     期望帧数: {n_frames_expected}")
print(f"     每帧特征数: {len(FEATURE_ORDER)}\n")


# === 错误类型到文字建议 ===
ERROR_ADVICE = {
    "shoulder_high": "肩膀不平，请放松肩部并下沉。",
    "knee_not_bent": "屈膝不足，请下沉重心，膝盖微屈。",
    "arm_too_low": "手臂偏低，请抬高手臂保持与肩平行。",
    "arm_too_high": "手臂偏高，请下放手臂保持与肩平行。",
    "torso_lean": "躯干倾斜，请收腹挺胸保持中正。",
    "foot_too_close": "脚间距过窄，请适当分开双脚保持平衡。",
    "foot_too_wide": "脚间距过宽，请适当收拢双脚。",
    "elbow_not_bent": "肘部弯曲不足，请放松手臂自然下垂。"
}


def predict_quality(frames, verbose=False):
    """
    预测动作质量
    
    Args:
        frames: list of N 帧 dict 特征（N应等于n_frames_expected）
        verbose: 是否打印详细信息
    
    Returns:
        result: 包含score和advice的字典
    """
    # 检查帧数
    if len(frames) != n_frames_expected:
        print(f"[WARNING] 输入帧数({len(frames)})与模型期望({n_frames_expected})不符")
        # 尝试调整
        if len(frames) < n_frames_expected:
            # 重复最后一帧补足
            frames = frames + [frames[-1]] * (n_frames_expected - len(frames))
        else:
            # 均匀采样到期望帧数
            indices = np.linspace(0, len(frames) - 1, n_frames_expected).astype(int)
            frames = [frames[i] for i in indices]
        print(f"[INFO] 已调整为 {len(frames)} 帧")
    
    # 按照 FEATURE_ORDER 提取特征
    X_list = []
    for f in frames:
        frame_features = [f.get(key, 0.0) for key in FEATURE_ORDER]
        X_list.extend(frame_features)
    
    # 转换为模型输入格式
    X = np.array(X_list).reshape(1, -1)
    
    # 标准化（如果有scaler）
    if scaler is not None:
        X = scaler.transform(X)
    
    # 模型预测
    prob = float(model.predict(X, verbose=0)[0][0])
    
    # 分析特征以提供具体建议
    advice = analyze_errors(frames, prob, verbose)
    
    result = {
        "score": prob,
        "advice": advice,
        "is_correct": prob >= 0.7
    }
    
    if verbose:
        print(f"\n预测结果:")
        print(f"  质量分数: {prob*100:.2f}%")
        print(f"  判定: {'正确' if prob >= 0.7 else '需改进'}")
        print(f"  建议: {advice}\n")
    
    return result


def analyze_errors(frames, score, verbose=False):
    """
    基于特征分析可能的错误并生成建议
    
    Args:
        frames: 帧特征列表
        score: 模型预测分数
        verbose: 是否打印分析过程
    
    Returns:
        advice: 建议文本
    """
    if score >= 0.7:
        return "动作标准，请保持！"
    
    # 使用多个关键帧进行分析（而不仅仅是最后一帧）
    errors_detected = []
    
    # 分析所有帧的平均特征
    avg_features = {}
    for key in FEATURE_ORDER:
        values = [f.get(key, 0) for f in frames if f.get(key) is not None]
        if values:
            avg_features[key] = np.mean(values)
    
    # 也分析最后几帧（完成姿态）
    last_frames = frames[-3:]  # 最后3帧
    last_avg = {}
    for key in FEATURE_ORDER:
        values = [f.get(key, 0) for f in last_frames if f.get(key) is not None]
        if values:
            last_avg[key] = np.mean(values)
    
    # === 检测各种错误 ===
    
    # 1. 检查肩膀平衡
    shoulder_balance = max(
        abs(avg_features.get('shoulder_balance_left', 0)),
        abs(avg_features.get('shoulder_balance_right', 0))
    )
    if shoulder_balance > 0.25:
        errors_detected.append("shoulder_high")
    
    # 2. 检查膝盖弯曲（主要看最后几帧）
    knee_bend = last_avg.get('knee_bend_average', 20)
    if knee_bend < 15:
        errors_detected.append("knee_not_bent")
    
    # 3. 检查手臂高度（分阶段）
    # 前半段应该较高，后半段中等
    first_half = frames[:len(frames)//2]
    second_half = frames[len(frames)//2:]
    
    first_arm_avg = np.mean([f.get('arm_height_ratio', 0.5) for f in first_half 
                             if f.get('arm_height_ratio') is not None])
    second_arm_avg = np.mean([f.get('arm_height_ratio', 0.5) for f in second_half 
                              if f.get('arm_height_ratio') is not None])
    
    if first_arm_avg < 0.3:  # 前半段手臂太低
        errors_detected.append("arm_too_low")
    elif second_arm_avg > 0.9:  # 后半段手臂还很高
        errors_detected.append("arm_too_high")
    elif second_arm_avg < 0.3:  # 后半段手臂太低
        errors_detected.append("arm_too_low")
    
    # 4. 检查躯干角度
    torso_angle = abs(avg_features.get('torso_angle', 1))
    if torso_angle > 10:
        errors_detected.append("torso_lean")
    
    # 5. 检查脚距
    foot_dist = last_avg.get('foot_distance', 0.26)
    if foot_dist < 0.15:
        errors_detected.append("foot_too_close")
    elif foot_dist > 0.40:
        errors_detected.append("foot_too_wide")
    
    # 6. 检查肘部弯曲（中间阶段）
    mid_frames = frames[len(frames)//3:2*len(frames)//3]
    mid_elbow_avg = np.mean([
        (f.get('left_elbow_angle', 90) + f.get('right_elbow_angle', 90)) / 2
        for f in mid_frames
        if f.get('left_elbow_angle') is not None and f.get('right_elbow_angle') is not None
    ])
    if mid_elbow_avg > 150:  # 手臂下落阶段肘部应该弯曲
        errors_detected.append("elbow_not_bent")
    
    if verbose:
        print(f"检测到的错误: {errors_detected}")
        print(f"关键特征:")
        print(f"  平均手臂高度(前半): {first_arm_avg:.3f}")
        print(f"  平均手臂高度(后半): {second_arm_avg:.3f}")
        print(f"  平均膝盖弯曲: {knee_bend:.1f}°")
        print(f"  平均脚距: {foot_dist:.3f}")
        print(f"  平均躯干角度: {torso_angle:.1f}°\n")
    
    # 生成建议
    if not errors_detected:
        return "动作存在偏差，请注意整体协调性。"
    
    # 组合多个建议
    advice_list = [ERROR_ADVICE.get(err, "") for err in errors_detected[:2]]  # 最多提示2个
    advice = " ".join([a for a in advice_list if a])
    
    return advice if advice else "动作需要改进，请注意姿态标准。"


# === 示例测试 ===
if __name__ == "__main__":
    import json
    
    print("="*60)
    print("测试预测功能（20帧输入）")
    print("="*60)
    
    # 尝试加载标准帧数据进行测试
    standard_path = os.path.join(PROJECT_ROOT, "data", "standard", "qishi3_standard_frames.json")
    # 向后兼容旧路径
    if not os.path.exists(standard_path):
        standard_path = os.path.join(PROJECT_ROOT, "qishi3_standard_frames.json")
    
    if os.path.exists(standard_path):
        with open(standard_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_frames = data['standard_features']
        print(f"\n使用标准视频的{len(test_frames)}帧进行测试...\n")
        
        result = predict_quality(test_frames, verbose=True)
        
        print(f"测试完成！")
    else:
        print(f"\n[WARNING] 未找到标准帧数据: {standard_path}")
        print(f"请先运行: python extract_standard_features.py\n")
        
        # 使用模拟数据测试
        print("使用模拟数据测试...")
        test_frame = {
            'shoulder_width': 0.24, 'hip_width': 0.14, 'foot_distance': 0.25,
            'hand_distance': 0.3, 'shoulder_to_hip_y': 0.23,
            'left_knee_angle': 150, 'right_knee_angle': 150,
            'left_elbow_angle': 130, 'right_elbow_angle': 140,
            'left_shoulder_angle': 40, 'right_shoulder_angle': 38,
            'knee_bend_average': 5, 'torso_angle': 1,
            'torso_vx': 0, 'torso_vy': -0.2,
            'arm_height_ratio': 0.2,
            'shoulder_balance_left': 0.25, 'shoulder_balance_right': -0.25,
            'hip_balance_left': 0.1, 'hip_balance_right': -0.1,
            'nose_center_offset': 0, 'avg_visibility': 0.99
        }
        
        # 使用期望的帧数
        test_frames = [test_frame] * n_frames_expected
        result = predict_quality(test_frames, verbose=True)
    
    print("="*60)


import os
import numpy as np
from tensorflow import keras

# === 统一的特征顺序（与 action_recognition.py 和 generate_data.py 保持一致）===
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

# Resolve model path in this order:
# 1. environment variable TAICHI_MODEL_PATH
# 2. taichi_ai/taichi_mlp.h5 (same folder as this script)
# 3. project-root taichi_mlp.h5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.environ.get('TAICHI_MODEL_PATH')
candidates = []
if env_path:
    candidates.append(os.path.abspath(os.path.expanduser(env_path)))
candidates.append(os.path.join(BASE_DIR, 'taichi_mlp.h5'))
candidates.append(os.path.abspath(os.path.join(BASE_DIR, '..', 'taichi_mlp.h5')))

MODEL_PATH = None
for p in candidates:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Model file not found. Tried the following locations:\n  " + "\n  ".join(candidates) +
        "\n\nPlease either:\n - place your trained 'taichi_mlp.h5' into the 'taichi_ai' folder,\n - or set environment variable TAICHI_MODEL_PATH to the model file path,\n - or start main with --model <path> (the program will set TAICHI_MODEL_PATH for you)."
    )

model = keras.models.load_model(MODEL_PATH)

# === 错误类型到文字建议 ===
error_advice = {
    "shoulder_high": "肩膀不平，请放松肩部并下沉。",
    "knee_not_bent": "屈膝不足，请下沉重心，膝盖微屈约10°。",
    "arm_too_low": "手臂偏低，请抬高手臂保持与肩平行。",
    "arm_too_high": "手臂偏高，请下放手臂保持与肩平行。",
    "torso_lean": "躯干前倾，请收腹挺胸保持中正。",
    "foot_too_close": "脚间距过窄，请适当分开双脚保持平衡。"
}

def predict_quality(frames):
    """
    frames: list of 4 帧 dict 特征
    
    重要：按照统一的 FEATURE_ORDER 提取特征值，确保与训练时的顺序一致
    """
    # 按照 FEATURE_ORDER 提取每一帧的特征值
    X_list = []
    for f in frames:
        # 使用 get() 方法，如果特征不存在则使用默认值 0.0
        frame_features = [f.get(key, 0.0) for key in FEATURE_ORDER]
        X_list.extend(frame_features)
    
    # 转换为模型输入格式
    X = np.array(X_list).reshape(1, -1)
    
    # 模型预测
    prob = float(model.predict(X, verbose=0)[0][0])
    
    result = {"score": prob, "advice": "动作正确，请保持。"}
    
    if prob < 0.7:
        # 简单根据特征偏差推断主要错误
        last_frame = frames[-1]  # 使用最后一帧判断错误类型
        
        if last_frame.get("knee_bend_average", 0) < 10:
            err = "knee_not_bent"
        elif abs(last_frame.get("shoulder_balance_left", 0)) > 0.2:
            err = "shoulder_high"
        elif last_frame.get("arm_height_ratio", 0.5) < 0.2:
            err = "arm_too_low"
        elif last_frame.get("arm_height_ratio", 0.5) > 0.9:
            err = "arm_too_high"
        elif abs(last_frame.get("torso_angle", 0)) > 8:
            err = "torso_lean"
        elif last_frame.get("foot_distance", 0.2) < 0.1:
            err = "foot_too_close"
        else:
            err = "unknown"
        
        result["advice"] = error_advice.get(err, "动作有偏差，请检查姿态。")
    
    return result

# === 示例 ===
if __name__ == "__main__":
    # 创建一个测试样本（使用正确的特征顺序）
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
    
    print("测试预测功能...")
    result = predict_quality([test_frame] * 4)
    print(f"预测分数: {result['score']:.2%}")
    print(f"建议: {result['advice']}")

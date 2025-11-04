import os
import numpy as np
from tensorflow import keras

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
    """frames: list of 4 帧 dict 特征"""
    X = np.concatenate([list(f.values()) for f in frames]).reshape(1, -1)
    prob = float(model.predict(X)[0][0])
    result = {"score": prob, "advice": "动作正确，请保持。"}
    if prob < 0.7:
        # 简单根据特征偏差推断主要错误
        if frames[-1]["knee_bend_average"] < 10:
            err = "knee_not_bent"
        elif abs(frames[-1]["shoulder_balance_left"]) > 0.2:
            err = "shoulder_high"
        elif frames[-1]["arm_height_ratio"] < 0.2:
            err = "arm_too_low"
        elif frames[-1]["arm_height_ratio"] > 0.9:
            err = "arm_too_high"
        elif abs(frames[-1]["torso_angle"]) > 8:
            err = "torso_lean"
        elif frames[-1]["foot_distance"] < 0.1:
            err = "foot_too_close"
        else:
            err = "unknown"
        result["advice"] = error_advice.get(err, "动作有偏差，请检查姿态。")
    return result

# === 示例 ===
if __name__ == "__main__":
    import json
    std = json.load(open("errors.json","r"))
    print(predict_quality([
        # 示例：用标准帧稍微改动一下
        {"arm_height_ratio":0.2, "avg_visibility":0.99, "foot_distance":0.25, "hand_distance":0.3,
         "hip_balance_left":0.1,"hip_balance_right":-0.1,"hip_width":0.14,"knee_bend_average":5,
         "left_elbow_angle":130,"left_knee_angle":150,"left_shoulder_angle":40,"nose_center_offset":0,
         "right_elbow_angle":140,"right_knee_angle":150,"right_shoulder_angle":38,
         "shoulder_balance_left":0.25,"shoulder_balance_right":-0.25,
         "shoulder_to_hip_y":0.23,"shoulder_width":0.24,"torso_angle":1,"torso_vx":0,"torso_vy":-0.2}
        ]*4))

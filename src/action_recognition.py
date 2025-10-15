import math
from typing import Dict, Any
import mediapipe as mp


def _angle(a, b, c):
    """计算三点 a-b-c 的夹角，返回角度（度）。点是具有 x,y 属性的 landmark。"""
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_cb = math.hypot(cb[0], cb[1])
    if mag_ab * mag_cb == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ab * mag_cb)))
    return math.degrees(math.acos(cos_angle))


def _angle_with_vertical_up(x, y):
    """计算向量 (x,y) 与竖直向上向量 (0,-1) 的夹角（度）。
    返回角度，越小表示越接近直立朝上。"""
    mag = math.hypot(x, y)
    if mag == 0:
        return 180.0
    # vertical up vector is (0, -1)
    # dot = x*0 + y*(-1) = -y
    cos_a = (-y) / mag
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


def recognize_action(results) -> Dict[str, Any]:
    """识别是否为抬膝（knee raise）动作，并判断是否标准。

    返回结构化字典，示例：
    {
      'action': 'knee_raise' or 'none',
      'side': 'left'/'right'/None,
      'correct': True/False,
      'reason': '短文本说明原因',
      'metrics': {...}
    }
    """
    if not results or not results.pose_landmarks:
        return {'action': 'none', 'side': None, 'correct': False, 'reason': '未检测到人体', 'metrics': {}}

    lm = results.pose_landmarks.landmark
    P = mp.solutions.pose.PoseLandmark

    # 读取左右髋、膝、踝
    left_hip = lm[P.LEFT_HIP.value]
    left_knee = lm[P.LEFT_KNEE.value]
    left_ankle = lm[P.LEFT_ANKLE.value]

    right_hip = lm[P.RIGHT_HIP.value]
    right_knee = lm[P.RIGHT_KNEE.value]
    right_ankle = lm[P.RIGHT_ANKLE.value]

    # 基本高度判断（注意：坐标系中 y 值向下增大，抬高时 y 更小）
    # 放宽阈值，并增加基于膝角的判断以适配侧面/正面视角
    vertical_thresh = 0.04
    knee_angle_thresh = 160.0  # 膝角小于此值视为弯曲（抬膝的一种表现）

    left_knee_angle = _angle(left_hip, left_knee, left_ankle)
    right_knee_angle = _angle(right_hip, right_knee, right_ankle)

    left_knee_raised = (left_knee.y + vertical_thresh < left_hip.y) or (left_knee_angle < knee_angle_thresh)
    right_knee_raised = (right_knee.y + vertical_thresh < right_hip.y) or (right_knee_angle < knee_angle_thresh)

    # 计算躯干倾斜（肩中点到髋中点向量与竖直向上向量的夹角）作为稳定性参考
    left_shoulder = lm[P.LEFT_SHOULDER.value]
    right_shoulder = lm[P.RIGHT_SHOULDER.value]
    # 中点坐标（简单使用 float），避免依赖 landmark 类型构造
    mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
    mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2.0
    mid_hip_x = (left_hip.x + right_hip.x) / 2.0
    mid_hip_y = (left_hip.y + right_hip.y) / 2.0

    # torso vector from hip -> shoulder
    torso_vx = mid_shoulder_x - mid_hip_x
    torso_vy = mid_shoulder_y - mid_hip_y

    # angle between torso vector and vertical-up (0,-1): smaller means closer to upright
    torso_angle = _angle_with_vertical_up(torso_vx, torso_vy)

    action = 'none'
    side = None
    correct = False
    reason = ''
    metrics = {
        'left_knee_y': left_knee.y,
        'left_hip_y': left_hip.y,
        'left_knee_angle': left_knee_angle,
        'right_knee_y': right_knee.y,
        'right_hip_y': right_hip.y,
        'right_knee_angle': right_knee_angle,
        'torso_angle': torso_angle,
        'torso_vx': torso_vx,
        'torso_vy': torso_vy,
    }

    # 优先判断双侧是否同时抬高（这里优先取单侧抬膝）
    if left_knee_raised and not right_knee_raised:
        action = 'knee_raise'
        side = 'left'
    elif right_knee_raised and not left_knee_raised:
        action = 'knee_raise'
        side = 'right'
    elif left_knee_raised and right_knee_raised:
        # 双膝同时抬高也视为抬膝，但标记为双侧
        action = 'knee_raise'
        side = 'both'

    if action == 'knee_raise':
        # 判断标准：膝盖抬高（已经满足），且躯干接近竖直向上（torso_angle 越小越直立）
        # 放宽躯干容差以适配不同视角
        torso_thresh = 25.0
        if torso_angle < torso_thresh:
            correct = True
            reason = '抬膝动作标准'
        else:
            correct = False
            reason = '上半身倾斜，请保持躯干直立'
    else:
        correct = False
        reason = '未检测到抬膝动作'

    return {'action': action, 'side': side, 'correct': correct, 'reason': reason, 'metrics': metrics}

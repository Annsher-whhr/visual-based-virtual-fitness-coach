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
    """计算丰富的太极起势/收势相关 metrics 并返回 metrics 字典（不作判定）。

    输出仅包含数值型/尺度化的度量，便于后续单独做判定或传给 AI。
    如果未检测到人体或关键点缺失，则返回空 dict。
    """
    if not results or not results.pose_landmarks:
        return {}

    lm = results.pose_landmarks.landmark
    P = mp.solutions.pose.PoseLandmark

    # 主要关键点（使用安全访问）
    def safe(i):
        return lm[i.value]

    L_sh = safe(P.LEFT_SHOULDER)
    R_sh = safe(P.RIGHT_SHOULDER)
    L_hip = safe(P.LEFT_HIP)
    R_hip = safe(P.RIGHT_HIP)
    L_knee = safe(P.LEFT_KNEE)
    R_knee = safe(P.RIGHT_KNEE)
    L_ank = safe(P.LEFT_ANKLE)
    R_ank = safe(P.RIGHT_ANKLE)
    L_wrist = safe(P.LEFT_WRIST)
    R_wrist = safe(P.RIGHT_WRIST)
    L_elbow = safe(P.LEFT_ELBOW)
    R_elbow = safe(P.RIGHT_ELBOW)
    nose = safe(P.NOSE)

    # 基础尺寸/距离（使用归一化坐标，0..1）
    shoulder_width = abs(L_sh.x - R_sh.x)
    hip_width = abs(L_hip.x - R_hip.x)
    foot_distance = abs(L_ank.x - R_ank.x)
    hand_distance = math.hypot(L_wrist.x - R_wrist.x, L_wrist.y - R_wrist.y)
    shoulder_to_hip_y = abs(((L_sh.y + R_sh.y) / 2.0) - ((L_hip.y + R_hip.y) / 2.0))

    # 角度度量
    left_knee_angle = _angle(L_hip, L_knee, L_ank)
    right_knee_angle = _angle(R_hip, R_knee, R_ank)
    left_elbow_angle = _angle(L_sh, L_elbow, L_wrist)
    right_elbow_angle = _angle(R_sh, R_elbow, R_wrist)
    left_shoulder_angle = _angle(L_hip, L_sh, L_elbow)
    right_shoulder_angle = _angle(R_hip, R_sh, R_elbow)

    # 躯干向量与竖直角度
    mid_sh_x = (L_sh.x + R_sh.x) / 2.0
    mid_sh_y = (L_sh.y + R_sh.y) / 2.0
    mid_hip_x = (L_hip.x + R_hip.x) / 2.0
    mid_hip_y = (L_hip.y + R_hip.y) / 2.0
    torso_vx = mid_sh_x - mid_hip_x
    torso_vy = mid_sh_y - mid_hip_y
    torso_angle = _angle_with_vertical_up(torso_vx, torso_vy)

    # 手臂高度比（归一化）：0=臀部线，1=肩部线
    shoulder_y = (L_sh.y + R_sh.y) / 2.0
    hip_y = (L_hip.y + R_hip.y) / 2.0
    avg_wrist_y = (L_wrist.y + R_wrist.y) / 2.0
    arm_height_ratio = None
    if abs(shoulder_y - hip_y) > 1e-6:
        arm_height_ratio = 1.0 - ((avg_wrist_y - hip_y) / (shoulder_y - hip_y))

    # 上下肢对称性（横向）
    shoulder_balance = (L_sh.x - mid_sh_x, R_sh.x - mid_sh_x)
    hip_balance = (L_hip.x - mid_hip_x, R_hip.x - mid_hip_x)

    # 头部/鼻子相对中线偏移（用于判断身体是否转向/偏离）
    person_mid_x = (mid_sh_x + mid_hip_x) / 2.0
    nose_center_offset = nose.x - person_mid_x

    # 关键点平均可见性（如果 landmark 有 visibility 字段则统计）
    vis_vals = []
    for p in (L_sh, R_sh, L_hip, R_hip, L_knee, R_knee, L_ank, R_ank, L_wrist, R_wrist):
        v = getattr(p, 'visibility', None)
        if v is not None:
            vis_vals.append(v)
    avg_visibility = sum(vis_vals) / len(vis_vals) if vis_vals else None

    # 复合尺度（可直接当作特征向量）
    metrics = {
        'shoulder_width': shoulder_width,
        'hip_width': hip_width,
        'foot_distance': foot_distance,
        'hand_distance': hand_distance,
        'shoulder_to_hip_y': shoulder_to_hip_y,
        'left_knee_angle': left_knee_angle,
        'right_knee_angle': right_knee_angle,
        'left_elbow_angle': left_elbow_angle,
        'right_elbow_angle': right_elbow_angle,
        'left_shoulder_angle': left_shoulder_angle,
        'right_shoulder_angle': right_shoulder_angle,
        'knee_bend_average': (180 - (left_knee_angle + right_knee_angle) / 2.0),
        'torso_angle': torso_angle,
        'torso_vx': torso_vx,
        'torso_vy': torso_vy,
        'arm_height_ratio': arm_height_ratio,
        'shoulder_balance_left': shoulder_balance[0],
        'shoulder_balance_right': shoulder_balance[1],
        'hip_balance_left': hip_balance[0],
        'hip_balance_right': hip_balance[1],
        'nose_center_offset': nose_center_offset,
        'avg_visibility': avg_visibility,
    }

    return metrics

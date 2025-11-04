import math
from typing import Dict, Any


def _safe_get(metrics: Dict[str, Any], key: str, default=None):
    return metrics.get(key, default) if isinstance(metrics, dict) else default


def provide_feedback(detected_action_or_metrics) -> str:
    """基于 recognize_action 返回的 metrics 对太极“起势”给出规则化判定与建议。

    输入可以是两种形式：
    - 旧格式：字典包含 'metrics' 字段（例如来自旧的 recognize_action 返回）
    - 新格式：直接是 metrics 字典（recognize_action 现在直接返回 metrics）

    输出：中文多行文本，包含简要结论和逐条建议，便于终端/界面显示。
    """
    # 兼容旧格式
    if isinstance(detected_action_or_metrics, dict) and 'metrics' in detected_action_or_metrics:
        metrics = detected_action_or_metrics.get('metrics', {})
    elif isinstance(detected_action_or_metrics, dict):
        metrics = detected_action_or_metrics
    else:
        return '无法解析输入（期望 metrics 字典）。'

    # 读取常用度量（默认为 None，后续判断将优雅降级）
    shoulder_width = _safe_get(metrics, 'shoulder_width')
    hip_width = _safe_get(metrics, 'hip_width')
    foot_distance = _safe_get(metrics, 'foot_distance')
    hand_distance = _safe_get(metrics, 'hand_distance')
    shoulder_to_hip_y = _safe_get(metrics, 'shoulder_to_hip_y')
    left_knee_angle = _safe_get(metrics, 'left_knee_angle')
    right_knee_angle = _safe_get(metrics, 'right_knee_angle')
    knee_bend_average = _safe_get(metrics, 'knee_bend_average')
    torso_angle = _safe_get(metrics, 'torso_angle')
    arm_height_ratio = _safe_get(metrics, 'arm_height_ratio')
    avg_visibility = _safe_get(metrics, 'avg_visibility')

    issues = []
    advices = []

    # 可见性检查
    if avg_visibility is not None and avg_visibility < 0.5:
        issues.append('low_visibility')
        advices.append('部分关键点置信度较低，请确保光线充足且身体完整在画面内。')

    # 躯干姿态（越小越直立）
    if torso_angle is None:
        advices.append('无法获取躯干角度，确保肩髋关键点可见。')
    else:
        if torso_angle > 25:
            issues.append('torso_tilt')
            advices.append('上半身前倾或左右倾斜较明显，起势时请抬头挺胸、收紧腹部，保持上身直立。')
        elif torso_angle > 15:
            advices.append('躯干略有倾斜，注意保持脊柱中立位以提高动作稳固性。')

    # 手臂高度（相对于肩-髋区间的归一化）
    if arm_height_ratio is None:
        advices.append('无法获取手臂高度比例，检查手腕关键点是否被遮挡或被关闭。')
    else:
        # 起势期望手在胸前到肩位之间，范围可调
        if arm_height_ratio < 0.35:
            issues.append('arm_too_low')
            advices.append('起势时双手位置偏低，尝试抬高手臂至胸前或肩部高度，肘部略弯曲以放松肩部。')
        elif arm_height_ratio > 0.9:
            issues.append('arm_too_high')
            advices.append('手臂位置偏高，放松肩部并降低手位以接近胸前/肩部高度有利于控制。')

    # 手间距（相对于肩宽的比例）
    if hand_distance is None or shoulder_width is None:
        advices.append('无法精确判断手间距与肩宽的关系。')
    else:
        ratio = hand_distance / (shoulder_width + 1e-6)
        # 起势时手不宜过紧也不宜过开
        if ratio < 0.25:
            issues.append('hands_too_close')
            advices.append('双手过于靠近，起势时适度打开手臂以利于稳体和平衡。')
        elif ratio > 1.5:
            issues.append('hands_too_wide')
            advices.append('双手间距过大，建议收窄至接近肩宽的范围以利于动作连贯。')

    # 足距（与肩/髋宽度对比）
    if foot_distance is None:
        advices.append('无法读取足距信息。')
    else:
        ref = shoulder_width if shoulder_width is not None else hip_width
        if ref:
            fd_ratio = foot_distance / (ref + 1e-6)
            if fd_ratio > 1.5:
                issues.append('feet_too_wide')
                advices.append('站位过宽，收拢双脚至与肩同宽或略窄有助于稳定起势。')
            
    # 膝盖弯曲（起势要求轻微放松但不过度弯曲）
    if knee_bend_average is None:
        advices.append('无法计算膝盖弯曲程度。')
    else:
        if knee_bend_average > 15:
            issues.append('knees_too_bent')
            advices.append('膝盖弯曲过多，起势时保持膝盖微屈（轻微弯曲以缓冲），避免深蹲式下沉。')
        elif knee_bend_average < 1:
            advices.append('膝盖过于绷直，起势时放松膝盖以利于吸收重心与稳定。')

    # 汇总结论
    correct = len(issues) == 0

    # 生成文本反馈（可读的多行中文）
    lines = []
    lines.append('太极 起势 检查：')
    lines.append(f"结论：{'标准' if correct else '需调整'}")
    # 概览分项
    if torso_angle is not None:
        lines.append(f" - 躯干角度: {torso_angle:.1f}°")
    if arm_height_ratio is not None:
        lines.append(f" - 手臂高度比: {arm_height_ratio:.2f} (0=臀线,1=肩线)")
    if knee_bend_average is not None:
        lines.append(f" - 平均膝弯曲: {knee_bend_average:.1f}")
    if shoulder_width is not None and hand_distance is not None:
        lines.append(f" - 手间距/肩宽比: {(hand_distance/(shoulder_width+1e-6)):.2f}")

    # 将所有计算出的 metrics 列举出来，便于调试与记录
    lines.append('\n详细 metrics：')
    try:
        for k in sorted(metrics.keys()):
            v = metrics.get(k)
            if isinstance(v, (int, float)):
                # 对浮点数格式化为保留三位小数，整数直接显示
                if isinstance(v, float):
                    lines.append(f" - {k}: {v:.3f}")
                else:
                    lines.append(f" - {k}: {v}")
            else:
                lines.append(f" - {k}: {v}")
    except Exception:
        # 如果 metrics 不是 dict 或排序失败，忽略此处
        pass

    if correct:
        lines.append('\n建议：')
        lines.append(' - 起势姿势良好，继续保持缓慢、放松且连贯的动作。')
        lines.append(' - 关注呼吸与重心转移，保持上身直立、双手在胸前到肩位之间。')
    else:
        lines.append('\n存在问题与建议：')
        # 列出每条建议（去重）
        seen = set()
        for a in advices:
            if a not in seen:
                lines.append(f" - {a}")
                seen.add(a)

    # 附加说明
    lines.append('\n提示：本反馈基于 2D 关键点规则判断，受摄像头角度与遮挡影响。若想要更详尽的口语化指导，可开启 AI 建议模块。')

    return '\n'.join(lines)

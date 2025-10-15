import random
import os
from src import ai_feedback


def provide_feedback(detected_action):
    """
    根据 recognize_action 返回的结构化结果提供中文反馈。
    支持向后兼容：如果传入的是字符串，会使用简单映射。
    """
    # backward compatible: if detected_action is a plain string
    if isinstance(detected_action, str):
        if detected_action == "标准动作":
            return "干得好，继续保持。"
        else:
            return "请调整你的姿势。"

    # 期望 detected_action 为字典
    action = detected_action.get('action')
    side = detected_action.get('side')
    correct = detected_action.get('correct')
    reason = detected_action.get('reason', '')

    # 如果配置了 ARK_API_KEY，优先调用 AI 接口获取更丰富的反馈
    api_key = os.getenv('ARK_API_KEY')
    if api_key and isinstance(detected_action, dict):
        ai_resp = ai_feedback.call_deepseek_feedback(detected_action.get('metrics', {}))
        if ai_resp.get('ok') and isinstance(ai_resp.get('result'), dict):
            # 如果 AI 返回了 JSON 并包含 advice，使用之
            ai_result = ai_resp['result']
            advice = ai_result.get('advice') or ai_result.get('verdict')
            if advice:
                return advice

    if action == 'knee_raise':
        if correct:
            good_msgs = [
                '很好，抬膝动作标准，继续保持！',
                '动作不错，保持核心收紧，继续完成动作。',
                '很好，你的抬膝姿势很到位，继续加油。'
            ]
            both_msgs = [
                '很好，双侧抬膝动作标准。保持稳定！',
                '双膝表现很好，继续注意节奏和稳定性。'
            ]
            if side == 'both':
                return random.choice(both_msgs)
            return random.choice(good_msgs)
        else:
            # 根据 reason 给出更具体建议，提供多条备选提示
            if '倾斜' in reason:
                tilt_msgs = [
                    '检测到上半身有偏斜，试着抬头挺胸、收紧腹部。',
                    '上半身倾斜，请保持脊柱直立并收紧核心肌群。'
                ]
                return random.choice(tilt_msgs)
            insufficient_msgs = [
                '膝盖可能抬得不够高，请尝试把膝盖抬到接近髋部高度。',
                '动作识别到抬膝，但需要更明显的抬高和控制，注意慢速抬起并保持平衡。'
            ]
            return random.choice(insufficient_msgs)

    return '未检测到目标动作。请检查摄像头视角并确保身体在画面中完整可见。'

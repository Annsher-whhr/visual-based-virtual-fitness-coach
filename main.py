# -*- coding: utf-8 -*-

import argparse
import math
import os
import cv2

from src.action_recognition import recognize_action
from src.detection import detect_human
from src.feedback import provide_feedback
from src.pose_estimation_v2 import estimate_pose
from utils.utils import draw_chinese_text
from src.ai_worker import submit_metrics, get_latest_advice


def main():
    parser = argparse.ArgumentParser(description='Visual-based Virtual Fitness Coach')
    parser.add_argument('-v', '--video', help='视频文件路径（若不提供则使用摄像头）', default=None)
    parser.add_argument('-m', '--model', help='可选：训练好的模型文件路径（会覆盖默认 taichi_ai/taichi_mlp.h5）', default=None)
    args = parser.parse_args()

    # 如果通过 CLI 指定了模型路径，把它放到环境变量中，predict.py 会读取该变量
    if args.model:
        os.environ['TAICHI_MODEL_PATH'] = os.path.abspath(os.path.expanduser(args.model))

    # 打开视频源：如果提供了文件且存在则打开文件，否则打开摄像头
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"视频文件不存在: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
        # 尝试获取视频的FPS来设定显示延迟
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            delay = int(1000 / fps)
        else:
            delay = 30
    else:
        cap = cv2.VideoCapture(0)
        delay = 1

    metrics_list = []
    frames_for_display = []
    _predicted_once = False

    def is_tai_chi_start_candidate(m):
        if not isinstance(m, dict):
            return False
        arm_h = m.get('arm_height_ratio')
        torso_a = m.get('torso_angle')
        hand_d = m.get('hand_distance')
        sh_w = m.get('shoulder_width')
        vis = m.get('avg_visibility')
        # 基本阈值（可以后续调优）
        if arm_h is None or torso_a is None or hand_d is None or sh_w is None:
            return False
        if vis is not None and vis < 0.35:
            return False
        if arm_h >= 0.30 and arm_h <= 0.85 and torso_a <= 22.0:
            ratio = hand_d / (sh_w + 1e-6)
            if ratio >= 0.35 and ratio <= 1.6:
                return True
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人体
        frame_with_detection = detect_human(frame)
        # 姿态估计（返回绘制帧与关键点结果）
        pose_estimated_frame, results = estimate_pose(frame_with_detection)

        # 动作识别（基于关键点结果） - 现在返回的是 metrics dict
        metrics = recognize_action(results)
        metrics_list.append(metrics)
        frames_for_display.append(pose_estimated_frame)

        # 提供本地即时反馈（rules-based），provide_feedback 已兼容 metrics dict
        feedback = provide_feedback(metrics)
        print(feedback)

        # 简单启势候选判定：当手臂高度、躯干角度和手间距满足起势特征时，提交给 AI 异步处理
        if is_tai_chi_start_candidate(metrics):
            submit_metrics(metrics)

        # 检查是否有 AI 返回的最新建议（若有则覆盖反馈）
        ai_advice = get_latest_advice()
        if ai_advice:
            feedback = ai_advice
            print(f"AI advice: {feedback}")

        # 显示结果（实时）
        cv2.imshow('Fitness Coach', pose_estimated_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 如果是视频文件，尝试从整段视频的 metrics 中抽取 4 帧关键 metrics 以供模型预测
    if args.video:
        try:
            from taichi_ai.predict import predict_quality
            from frame_selector import select_frames_by_similarity

            def candidate_score(m):
                """整体质量评分。用于回退和调试，不再直接驱动选帧。"""
                if not isinstance(m, dict) or not m:
                    return 0.0
                eps = 1e-6

                vis = m.get('avg_visibility', 0.0) or 0.0
                if vis < 0.25:
                    return 0.0

                score = 0.0
                vis_score = max(0.0, (vis - 0.25) / (1.0 - 0.25))
                score += 2.0 * vis_score

                arm_h = m.get('arm_height_ratio')
                if arm_h is not None:
                    arm_score = max(0.0, 1.0 - abs(arm_h - 0.6) / 0.6)
                    score += 3.0 * arm_score

                torso_a = m.get('torso_angle')
                if torso_a is not None:
                    torso_score = max(0.0, 1.0 - float(torso_a) / 45.0)
                    score += 2.0 * torso_score

                hand_d = m.get('hand_distance')
                sh_w = m.get('shoulder_width')
                if hand_d is not None and sh_w is not None and sh_w > eps:
                    ratio = float(hand_d) / (float(sh_w) + eps)
                    ratio_score = max(0.0, 1.0 - abs(ratio - 0.75) / 1.5)
                    score += 1.5 * ratio_score

                foot_d = m.get('foot_distance')
                hip_w = m.get('hip_width')
                if foot_d is not None and hip_w is not None and hip_w > eps:
                    fd_ratio = float(foot_d) / (float(hip_w) + eps)
                    fd_score = max(0.0, 1.0 - abs(fd_ratio - 1.0) / 1.0)
                    score += 1.0 * fd_score

                le = m.get('left_elbow_angle')
                re = m.get('right_elbow_angle')
                elbow_scores = []
                for ang in (le, re):
                    if ang is not None:
                        elbow_scores.append(max(0.0, 1.0 - abs(float(ang) - 160.0) / 60.0))
                if elbow_scores:
                    score += 0.5 * (sum(elbow_scores) / len(elbow_scores))

                return round(float(score), 3)

            def leg_open_score(m):
                """腿部打开阶段评分：脚距越大越好，同时要求身体保持正中。"""
                if not isinstance(m, dict) or not m:
                    return float('-inf')

                foot = m.get('foot_distance')
                hip = m.get('hip_width')
                shoulder = m.get('shoulder_width')
                torso_angle = m.get('torso_angle')
                nose_offset = m.get('nose_center_offset')
                arm_height = m.get('arm_height_ratio')

                if foot is None or hip is None:
                    return float('-inf')
                hip = float(hip)
                if hip <= 1e-6:
                    return float('-inf')

                ratio = float(foot) / (hip + 1e-6)

                # 头部居中 + 躯干直立约束
                span = shoulder if shoulder is not None else hip
                span = float(span) if span not in (None, 0.0) else 1.0
                nose_norm = abs(float(nose_offset or 0.0)) / (span + 1e-6)
                torso_norm = abs(float(torso_angle or 0.0)) / 35.0  # 约束 35° 以上视为偏离较大
                arm_norm = 0.0
                if arm_height is not None:
                    # 手尚未抬起：arm_height_ratio 越接近 0.2 越好，超过 0.4 逐渐扣分
                    arm_val = max(0.0, float(arm_height) - 0.2)
                    arm_norm = arm_val / 0.8  # normalize to 0..1, 强调过早抬臂的惩罚

                penalty = (
                    0.5 * min(1.0, nose_norm) +
                    0.3 * min(1.0, torso_norm) +
                    0.2 * min(1.0, arm_norm)
                )

                return ratio - penalty

            def arm_raise_score(m):
                if not isinstance(m, dict) or not m:
                    return float('-inf')
                arm = m.get('arm_height_ratio')
                if arm is None:
                    return float('-inf')
                return float(arm)

            def foot_hip_ratio(m):
                if not isinstance(m, dict) or not m:
                    return float('nan')
                foot = m.get('foot_distance')
                hip = m.get('hip_width')
                if foot is None or hip is None or hip == 0:
                    return float('nan')
                return float(foot) / (float(hip) + 1e-6)

            # 如果总帧数少于4，则无法预测
            if len(metrics_list) < 4:
                print('视频帧数不足，无法生成 4 帧特征用于预测。')
                return

            valid_metrics = [m for m in metrics_list if isinstance(m, dict) and m]
            if len(valid_metrics) < 4:
                print('有效关键帧少于 4 个，无法触发模型预测。请检查视频中的人体检测质量。')
                return

            def select_frames_by_phase(metrics_list):
                """
                根据“起势分段”思路选帧：
                1. 第一帧：动作起始
                2. 第二帧：腿部打开幅度达到峰值
                3. 第三帧：手臂抬升幅度达到峰值（发生在腿部动作之后）
                4. 第四帧：动作结束（视频最后一帧）
                """
                n = len(metrics_list)

                def is_valid(idx):
                    return 0 <= idx < n and isinstance(metrics_list[idx], dict) and metrics_list[idx]

                valid_indices = [i for i in range(n) if is_valid(i)]
                if not valid_indices:
                    fallback = [0, max(0, n // 3), max(0, (2 * n) // 3), max(0, n - 1)]
                    phases = {'start': None, 'leg_peak': None, 'arm_peak': None, 'end': None}
                    return sorted(fallback), phases

                first_idx = valid_indices[0]
                last_idx = valid_indices[-1]
                phases = {
                    'start': first_idx,
                    'leg_peak': None,
                    'arm_peak': None,
                    'end': last_idx,
                }

                if first_idx >= last_idx:
                    fallback = [first_idx] * 4
                    return fallback, phases

                # 腿部峰值：脚距/臀宽比最大（排除首尾，预留空间给后续阶段）
                leg_idx = None
                search_leg = range(first_idx + 1, max(first_idx + 1, last_idx))
                if search_leg:
                    leg_candidates = [(i, leg_open_score(metrics_list[i])) for i in search_leg]
                    leg_candidates = [item for item in leg_candidates if item[1] != float('-inf')]
                    if leg_candidates:
                        max_leg = max(score for _, score in leg_candidates)
                        plateau = [i for i, score in leg_candidates if score >= max_leg * 0.97]
                        leg_idx = plateau[0] if plateau else max(leg_candidates, key=lambda x: x[1])[0]
                        if leg_idx >= last_idx:
                            leg_idx = None
                phases['leg_peak'] = leg_idx

                # 手臂峰值：在腿部之后查找 arm_height_ratio 最大
                arm_idx = None
                arm_start = (leg_idx + 1) if leg_idx is not None else (first_idx + 1)
                search_arm = range(max(arm_start, first_idx + 1), last_idx)
                if search_arm:
                    arm_candidates = [(i, arm_raise_score(metrics_list[i])) for i in search_arm]
                    arm_candidates = [item for item in arm_candidates if item[1] != float('-inf')]
                    if arm_candidates:
                        max_arm = max(score for _, score in arm_candidates)
                        plateau = [i for i, score in arm_candidates if score >= max_arm * 0.97]
                        arm_idx = plateau[0] if plateau else max(arm_candidates, key=lambda x: x[1])[0]
                        if arm_idx >= last_idx:
                            arm_idx = None
                phases['arm_peak'] = arm_idx

                # 组装初步索引列表（保持添加顺序，以便后续补足）
                ordered_indices = []

                def add_idx(idx):
                    if idx is None:
                        return
                    if not is_valid(idx):
                        return
                    if idx not in ordered_indices:
                        ordered_indices.append(idx)

                add_idx(first_idx)
                add_idx(leg_idx)
                add_idx(arm_idx)
                add_idx(last_idx)

                # 补足到 4 帧：优先使用高质量帧，其次按时间顺序补齐
                if len(ordered_indices) < 4:
                    candidates = [
                        i for i in range(first_idx, last_idx + 1)
                        if i not in ordered_indices and candidate_score(metrics_list[i]) > 0
                    ]
                    candidates.sort(key=lambda i: candidate_score(metrics_list[i]), reverse=True)
                    for idx in candidates:
                        add_idx(idx)
                        if len(ordered_indices) >= 4:
                            break

                if len(ordered_indices) < 4:
                    for offset in range(1, (last_idx - first_idx) + 1):
                        for base in (first_idx, leg_idx, arm_idx, last_idx):
                            if base is None:
                                continue
                            for idx in (base - offset, base + offset):
                                if 0 <= idx < n and idx not in ordered_indices and is_valid(idx):
                                    add_idx(idx)
                                if len(ordered_indices) >= 4:
                                    break
                            if len(ordered_indices) >= 4:
                                break
                        if len(ordered_indices) >= 4:
                            break

                if len(ordered_indices) < 4:
                    span = max(1, last_idx - first_idx)
                    for frac in (1 / 3, 2 / 3):
                        idx = int(round(first_idx + span * frac))
                        if idx >= last_idx:
                            idx = last_idx - 1
                        if idx > first_idx and idx not in ordered_indices and is_valid(idx):
                            add_idx(idx)
                        if len(ordered_indices) >= 4:
                            break

                if len(ordered_indices) < 4:
                    for i in range(first_idx, last_idx + 1):
                        if i not in ordered_indices and is_valid(i):
                            add_idx(i)
                        if len(ordered_indices) >= 4:
                            break

                while len(ordered_indices) < 4:
                    ordered_indices.append(last_idx)

                ordered_indices = ordered_indices[:4]
                ordered_indices.sort()

                return ordered_indices, phases

            if _predicted_once:
                print('预测已执行过一次，跳过重复预测。')
            else:
                # 使用基于特征相似度的智能选帧算法
                try:
                    indices, info = select_frames_by_similarity(
                        metrics_list, 
                        min_frame_gap=3,  # 相邻帧最小间隔
                        verbose=True
                    )
                    selected_frames = [metrics_list[i] for i in indices]
                    
                    # 可选：生成相似度热力图
                    try:
                        from frame_selector import visualize_similarity_heatmap
                        visualize_similarity_heatmap(info['similarity_matrix'], indices)
                    except:
                        pass
                    
                except Exception as e:
                    print(f"⚠️  智能选帧失败: {e}")
                    print("使用原有的阶段性选帧作为后备...")
                    indices, phases = select_frames_by_phase(metrics_list)
                    selected_frames = [metrics_list[i] for i in indices]

                print(f"\n最终用于预测的帧索引（按时间顺序）：{indices}")
                print('即将调用模型预测...')
                pred = predict_quality(selected_frames)
                print('\n' + '='*60)
                print('🎯 模型预测结果')
                print('='*60)
                print(f"动作质量分数: {pred.get('score', 0):.2%}")
                print(f"反馈建议: {pred.get('advice', '无')}")
                print('='*60 + '\n')
                _predicted_once = True
        except Exception as e:
            print('调用模型预测时出错：', e)


if __name__ == "__main__":
    main()

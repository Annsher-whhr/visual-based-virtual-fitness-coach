# -*- coding: utf-8 -*-

import argparse
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

            def candidate_score(m):
                if not isinstance(m, dict):
                    return 0
                s = 0
                arm_h = m.get('arm_height_ratio')
                torso_a = m.get('torso_angle')
                hand_d = m.get('hand_distance')
                sh_w = m.get('shoulder_width')
                vis = m.get('avg_visibility')
                if vis is not None and vis < 0.35:
                    return 0
                if arm_h is not None and 0.30 <= arm_h <= 0.85:
                    s += 1
                if torso_a is not None and torso_a <= 22.0:
                    s += 1
                if hand_d is not None and sh_w is not None:
                    ratio = hand_d / (sh_w + 1e-6)
                    if 0.35 <= ratio <= 1.6:
                        s += 1
                return s

            # 如果总帧数少于4，则无法预测
            if len(metrics_list) < 4:
                print('视频帧数不足，无法生成 4 帧特征用于预测。')
                return

            # 滑动窗口选择得分最高的连续 4 帧序列
            best_idx = 0
            best_score = -1
            for i in range(0, len(metrics_list) - 3):
                score = sum(candidate_score(m) for m in metrics_list[i:i+4])
                if score > best_score:
                    best_score = score
                    best_idx = i

            # 如果最高分为0（无明显候选），则均匀采样 4 帧
            if best_score <= 0:
                indices = [int(len(metrics_list) * (j+0.5) / 4.0) for j in range(4)]
            else:
                indices = list(range(best_idx, best_idx+4))

            selected_frames = [metrics_list[i] for i in indices]
            print(f"Selected frame indices for prediction: {indices}")
            pred = predict_quality(selected_frames)
            print('模型预测结果：', pred)
        except Exception as e:
            print('调用模型预测时出错：', e)


if __name__ == "__main__":
    main()

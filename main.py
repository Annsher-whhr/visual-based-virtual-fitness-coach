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
    args = parser.parse_args()

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

        # 提供本地即时反馈（rules-based），provide_feedback 已兼容 metrics dict
        feedback = provide_feedback(metrics)
        print(feedback)

        # 简单启势候选判定：当手臂高度、躯干角度和手间距满足起势特征时，提交给 AI 异步处理
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

        if is_tai_chi_start_candidate(metrics):
            submit_metrics(metrics)

        # 检查是否有 AI 返回的最新建议（若有则覆盖反馈）
        ai_advice = get_latest_advice()
        if ai_advice:
            feedback = ai_advice
            print(f"AI advice: {feedback}")

        # 显示结果
        cv2.imshow('Fitness Coach', pose_estimated_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

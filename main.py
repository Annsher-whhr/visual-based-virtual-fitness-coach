# -*- coding: utf-8 -*-

import argparse
import os
import cv2

from src.action_recognition import recognize_action
from src.detection import detect_human
from src.feedback import provide_feedback
from src.pose_estimation_v2 import estimate_pose
from utils.utils import draw_chinese_text


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

        # 姿态估计
        pose_estimated_frame = estimate_pose(frame_with_detection)

        # 动作识别
        action = recognize_action(pose_estimated_frame)

        # 提供反馈
        feedback = provide_feedback(action)
        print(feedback)

        # 显示结果
        cv2.imshow('Fitness Coach', pose_estimated_frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

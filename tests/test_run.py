# test_run.py — 临时测试脚本（处理最多 10 帧并打印结果，不弹出窗口）
import os
import cv2
from src.detection import detect_human
from src.pose_estimation_v2 import estimate_pose
from src.action_recognition import recognize_action
from src.feedback import provide_feedback

VIDEO = "jianshen1.mp4"  # 若不存在会回退到摄像头
MAX_FRAMES = 10


def run_test():
    if os.path.exists(VIDEO):
        cap = cv2.VideoCapture(VIDEO)
    else:
        cap = cv2.VideoCapture(0)

    frame_count = 0
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 检测人体、估计姿态
        frame_with_detection = detect_human(frame)
        pose_frame, results = estimate_pose(frame_with_detection)

        # 动作识别与反馈
        action_result = recognize_action(results)
        feedback = provide_feedback(action_result)

        print(f"Frame {frame_count}: action={action_result.get('action')}, side={action_result.get('side')}, correct={action_result.get('correct')}")
        print(f"  reason: {action_result.get('reason')}")
        print(f"  metrics: {action_result.get('metrics')}")
        print(f"  feedback: {feedback}\n")

    cap.release()


if __name__ == '__main__':
    run_test()

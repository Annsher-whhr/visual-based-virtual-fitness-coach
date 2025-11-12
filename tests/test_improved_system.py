# -*- coding: utf-8 -*-
"""
测试改进后的太极拳起势动作评估系统
演示完整的评估流程：视频 -> 特征提取 -> 智能抽帧 -> 模型评估
"""

import cv2
import argparse
import sys
from frame_selector_v2 import detect_action_start_end, select_frames_evenly
from src.pose_estimation_v2 import estimate_pose
from src.action_recognition import recognize_action
from src.detection import detect_human

# 使用v2版本的预测模块
from taichi_ai.predict_v2 import predict_quality


def evaluate_video(video_path, n_frames=20, visualize=False):
    """
    评估视频中的起势动作
    
    Args:
        video_path: 视频文件路径
        n_frames: 抽取的帧数（默认12，应与训练时一致）
        visualize: 是否可视化（可选）
    
    Returns:
        result: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"太极拳起势动作评估系统 v2.0")
    print(f"{'='*60}")
    print(f"视频: {video_path}")
    print(f"抽帧数: {n_frames}")
    print(f"{'='*60}\n")
    
    # === 步骤1: 提取所有帧的特征 ===
    print("步骤1: 提取视频特征...")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  FPS: {fps:.2f}")
    print(f"  总帧数: {total_frames}")
    
    metrics_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测 -> 姿态估计 -> 特征提取
        frame_with_detection = detect_human(frame)
        _, results = estimate_pose(frame_with_detection)
        metrics = recognize_action(results)
        
        metrics_list.append(metrics if isinstance(metrics, dict) else {})
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧...", end='\r')
    
    cap.release()
    print(f"\n[OK] 特征提取完成，共 {len(metrics_list)} 帧\n")
    
    # === 步骤2: 智能抽帧 ===
    print("步骤2: 智能检测动作边界并抽取关键帧...")
    
    try:
        selected_indices, selected_frames = select_frames_evenly(
            metrics_list, 
            n_frames=n_frames,
            auto_detect_boundaries=True,
            fps=fps,
            verbose=True,
            force_start_from_beginning=False  # 评估时智能识别，不强制从开始
        )
    except Exception as e:
        print(f"[ERROR] 抽帧失败: {e}")
        return None
    
    # === 步骤3: 模型评估 ===
    print("步骤3: 使用神经网络模型评估动作质量...")
    
    try:
        result = predict_quality(selected_frames, verbose=True)
    except Exception as e:
        print(f"[ERROR] 预测失败: {e}")
        return None
    
    # === 输出最终结果 ===
    print(f"\n{'='*60}")
    print(f"最终评估结果")
    print(f"{'='*60}")
    print(f"质量分数: {result['score']*100:.2f}%")
    print(f"动作判定: {'[OK] 标准' if result['is_correct'] else '[!] 需改进'}")
    print(f"改进建议: {result['advice']}")
    print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='测试改进后的太极拳起势动作评估系统')
    parser.add_argument('-v', '--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('-n', '--n-frames', type=int, default=20,
                       help='抽取的关键帧数量（默认20）')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图表')
    
    args = parser.parse_args()
    
    result = evaluate_video(args.video, args.n_frames, args.visualize)
    
    if result is None:
        print("[ERROR] 评估失败")
        return 1
    
    return 0


if __name__ == "__main__":
    # 如果没有命令行参数，使用测试视频
    if len(sys.argv) == 1:
        print("使用默认测试视频...")
        test_videos = [
            "video/qishi3.mp4",  # 标准视频
            "video/qishi1.mp4",  # 测试视频1
            "video/qishi2.mp4",  # 测试视频2
        ]
        
        for video in test_videos:
            if not cv2.os.path.exists(video):
                print(f"[SKIP] 视频不存在: {video}\n")
                continue
            
            print(f"\n{'#'*60}")
            print(f"测试视频: {video}")
            print(f"{'#'*60}")
            
            result = evaluate_video(video, n_frames=20)
            
            if result:
                print(f"[OK] {video} 评估完成\n")
            else:
                print(f"[ERROR] {video} 评估失败\n")
    else:
        sys.exit(main())


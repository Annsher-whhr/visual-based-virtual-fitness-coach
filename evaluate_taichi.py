# -*- coding: utf-8 -*-
"""
太极拳起势动作评估 - 主程序 v2.0
简化的评估接口，使用改进后的12帧系统
"""

import cv2
import argparse
import os
from frame_selector_v2 import detect_action_start_end, select_frames_evenly
from src.pose_estimation_v2 import estimate_pose
from src.action_recognition import recognize_action
from src.detection import detect_human

# 导入v2预测模块
from taichi_ai.predict_v2 import predict_quality


def extract_video_features(video_path, verbose=True):
    """从视频提取所有帧的姿态特征"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if verbose:
        print(f"正在分析视频...")
        print(f"  FPS: {fps:.2f}, 总帧数: {total_frames}")
    
    metrics_list = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_with_detection = detect_human(frame)
        _, results = estimate_pose(frame_with_detection)
        metrics = recognize_action(results)
        
        metrics_list.append(metrics if isinstance(metrics, dict) else {})
        
        frame_count += 1
        if verbose and frame_count % 30 == 0:
            print(f"  进度: {frame_count}/{total_frames} 帧", end='\r')
    
    cap.release()
    
    if verbose:
        print(f"\n[OK] 特征提取完成，共 {len(metrics_list)} 帧\n")
    
    return metrics_list, fps


def evaluate_taichi_video(video_path, n_frames=12, verbose=True):
    """
    评估太极拳起势动作
    
    Args:
        video_path: 视频文件路径
        n_frames: 抽取帧数（默认12，应与训练时一致）
        verbose: 是否显示详细信息
    
    Returns:
        dict: 包含score, is_correct, advice的评估结果
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"太极拳起势动作评估 v2.0")
        print(f"{'='*60}")
        print(f"视频: {video_path}\n")
    
    # 步骤1: 提取特征
    metrics_list, fps = extract_video_features(video_path, verbose)
    
    # 步骤2: 智能抽帧
    if verbose:
        print("检测动作边界并抽取关键帧...")
    
    selected_indices, selected_frames = select_frames_evenly(
        metrics_list,
        n_frames=n_frames,
        auto_detect_boundaries=True,
        fps=fps,
        verbose=verbose
    )
    
    # 步骤3: 模型评估
    if verbose:
        print("使用神经网络评估动作质量...\n")
    
    result = predict_quality(selected_frames, verbose=False)
    
    # 输出结果
    if verbose:
        print(f"{'='*60}")
        print(f"评估结果")
        print(f"{'='*60}")
        print(f"质量分数: {result['score']*100:.2f}%")
        
        if result['is_correct']:
            print(f"判定: [OK] 动作标准")
        else:
            print(f"判定: [!] 需要改进")
        
        print(f"建议: {result['advice']}")
        print(f"{'='*60}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='太极拳起势动作评估系统 v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python evaluate_taichi.py -v video/qishi1.mp4
  python evaluate_taichi.py -v video/qishi2.mp4 -n 12
  python evaluate_taichi.py -v video/qishi3.mp4 --quiet
        """
    )
    
    parser.add_argument('-v', '--video', type=str, required=True,
                       help='输入视频路径')
    parser.add_argument('-n', '--n-frames', type=int, default=12,
                       help='抽取的关键帧数量（默认12）')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='静默模式（只输出最终结果）')
    
    args = parser.parse_args()
    
    try:
        result = evaluate_taichi_video(
            args.video,
            n_frames=args.n_frames,
            verbose=not args.quiet
        )
        
        # 简化输出（用于其他程序调用）
        if args.quiet:
            print(f"{result['score']:.4f}|{int(result['is_correct'])}|{result['advice']}")
        
        return 0
    
    except Exception as e:
        print(f"[ERROR] 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


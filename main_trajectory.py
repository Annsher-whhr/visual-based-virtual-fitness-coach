import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import argparse
import numpy as np
from trajectory_system.trajectory_evaluator import TrajectoryEvaluator
from trajectory_system.smart_frame_selector import SmartFrameSelector

def process_video(video_path, output_path=None, use_smart_selection=True):
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0:
        return None
    
    if use_smart_selection:
        selector = SmartFrameSelector()
        indices = selector.select_best_frames(frames, min_gap=5)
        print(f"智能选帧: {indices}")
    else:
        indices = np.linspace(0, len(frames)-1, 4, dtype=int)
        print(f"均匀选帧: {list(indices)}")
    
    selected_frames = [frames[i] for i in indices]
    
    evaluator = TrajectoryEvaluator()
    result = evaluator.evaluate_video_frames(selected_frames)
    
    if output_path:
        for i, frame in enumerate(selected_frames):
            cv2.imwrite(f"{output_path}_frame_{i+1}.jpg", frame)
    
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--no-smart', action='store_true', help='不使用智能选帧')
    args = parser.parse_args()
    
    result = process_video(args.video, args.output, use_smart_selection=not args.no_smart)
    
    if result:
        print("\n" + "="*60)
        score = result['score']
        if score >= 90:
            level = "优秀"
        elif score >= 80:
            level = "良好"
        elif score >= 70:
            level = "中等"
        else:
            level = "需改进"
        
        print(f"整体评分: {score:.2f} ({level})")
        bar_length = int(score / 2)
        print(f"[{'#' * bar_length}{'-' * (50 - bar_length)}] {result['accuracy']:.1%}")
        
        print("\n各部位相似度:")
        for part, sim in result['similarities'].items():
            bar_len = int(sim / 5)
            status = "[OK]" if sim >= 70 else "[X]"
            print(f"  {status} {part:20s} [{'#' * bar_len}{'-' * (20 - bar_len)}] {sim:.1f}")
        
        print("\n改进建议:")
        for advice in result['advice']:
            print(f"  - {advice}")
        print("="*60)

if __name__ == '__main__':
    main()


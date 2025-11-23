import cv2
import argparse
import numpy as np
from trajectory_system.trajectory_evaluator import TrajectoryEvaluator

def process_video(video_path, output_path=None):
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
    
    indices = np.linspace(0, len(frames)-1, 4, dtype=int)
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
    args = parser.parse_args()
    
    result = process_video(args.video, args.output)
    
    if result:
        print(f"整体评分: {result['score']:.2f}")
        print(f"准确度: {result['accuracy']:.2%}")
        print("\n各部位相似度:")
        for part, sim in result['similarities'].items():
            print(f"  {part}: {sim:.2f}")
        print("\n建议:")
        for advice in result['advice']:
            print(f"  - {advice}")

if __name__ == '__main__':
    main()


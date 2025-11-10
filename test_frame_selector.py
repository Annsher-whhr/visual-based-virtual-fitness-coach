# -*- coding: utf-8 -*-
"""
测试智能关键帧选择器
演示如何从视频中选择与标准训练帧最匹配的4帧
"""

import cv2
import argparse
from frame_selector import select_frames_by_similarity, visualize_similarity_heatmap
from src.pose_estimation_v2 import estimate_pose
from src.action_recognition import recognize_action
from src.detection import detect_human


def extract_metrics_from_video(video_path):
    """从视频中提取所有帧的metrics"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    metrics_list = []
    frame_count = 0
    
    print(f"正在分析视频: {video_path}")
    print("提取关键点特征...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测人体
        frame_with_detection = detect_human(frame)
        # 姿态估计
        _, results = estimate_pose(frame_with_detection)
        # 动作识别（提取metrics）
        metrics = recognize_action(results)
        
        if isinstance(metrics, dict) and metrics:
            metrics_list.append(metrics)
        else:
            # 保持索引连续性，添加空字典
            metrics_list.append({})
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  已处理 {frame_count} 帧...", end='\r')
    
    cap.release()
    print(f"\n✅ 完成！共提取 {len(metrics_list)} 帧特征\n")
    
    return metrics_list


def main():
    parser = argparse.ArgumentParser(description='测试智能关键帧选择器')
    parser.add_argument('-v', '--video', required=True, help='视频文件路径')
    parser.add_argument('--min-gap', type=int, default=3, help='相邻关键帧的最小间隔')
    parser.add_argument('--no-viz', action='store_true', help='不生成可视化图表')
    args = parser.parse_args()
    
    # 1. 从视频提取metrics
    metrics_list = extract_metrics_from_video(args.video)
    
    if len(metrics_list) < 4:
        print(f"❌ 错误：视频帧数不足（需要至少4帧，实际{len(metrics_list)}帧）")
        return
    
    valid_count = sum(1 for m in metrics_list if isinstance(m, dict) and m)
    print(f"有效特征帧数: {valid_count}/{len(metrics_list)}")
    
    if valid_count < 4:
        print("❌ 错误：有效帧数不足4帧，请检查视频中的人体检测质量")
        return
    
    # 2. 智能选帧
    print("\n" + "="*60)
    print("开始智能选帧...")
    print("="*60)
    
    try:
        indices, info = select_frames_by_similarity(
            metrics_list,
            min_frame_gap=args.min_gap,
            verbose=True
        )
        
        # 3. 可视化（可选）
        if not args.no_viz:
            visualize_similarity_heatmap(info['similarity_matrix'], indices)
        
        # 4. 显示选中帧的详细特征对比
        print("\n" + "="*60)
        print("选中帧的特征详细对比")
        print("="*60)
        
        from frame_selector import STANDARD_FRAMES
        
        for j, idx in enumerate(indices):
            print(f"\n【标准帧 {j+1} ←→ 视频帧 {idx}】")
            print(f"  相似度分数: {info['individual_scores'][j]:.2f}%")
            
            metrics = metrics_list[idx]
            std_frame = STANDARD_FRAMES[j]
            
            # 显示关键特征对比
            key_features = ['arm_height_ratio', 'foot_distance', 'hand_distance', 
                          'knee_bend_average', 'torso_angle']
            
            if j == 2:  # 第3帧额外显示肘部角度
                key_features.extend(['left_elbow_angle', 'right_elbow_angle'])
            if j == 3:  # 第4帧额外显示膝盖角度
                key_features.extend(['left_knee_angle', 'right_knee_angle'])
            
            for feat in key_features:
                val_video = metrics.get(feat, None)
                val_std = std_frame.get(feat, None)
                
                if val_video is not None and val_std is not None:
                    diff = abs(val_video - val_std)
                    diff_pct = (diff / (abs(val_std) + 1e-6)) * 100
                    
                    print(f"  {feat:20s}: 视频={val_video:7.3f}  标准={val_std:7.3f}  "
                          f"差异={diff:6.3f} ({diff_pct:5.1f}%)")
        
        print("\n" + "="*60)
        print(f"✅ 智能选帧完成！")
        print(f"   选中的帧索引: {indices}")
        print(f"   平均相似度: {info['avg_similarity']:.2f}%")
        print("="*60 + "\n")
        
        # 5. 提示可以用于模型预测
        print("💡 提示：这4帧现在可以输入到训练好的模型中进行动作质量评估")
        print(f"   使用命令: python main.py -v {args.video}")
        
    except Exception as e:
        print(f"\n❌ 选帧过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


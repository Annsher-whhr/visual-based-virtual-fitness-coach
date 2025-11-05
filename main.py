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
                # 更细致的评分：返回浮点分数范围近似 0..10，较高的分数表示更接近理想的起势帧
                if not isinstance(m, dict):
                    return 0.0
                eps = 1e-6

                vis = m.get('avg_visibility', 0.0) or 0.0
                # 可见度过低直接返回0
                if vis < 0.25:
                    return 0.0

                score = 0.0

                # 1) 可见度贡献（0..2）——保证可见度越高越可靠
                vis_score = max(0.0, (vis - 0.25) / (1.0 - 0.25))
                score += 2.0 * vis_score

                # 2) 手臂高度比（arm_height_ratio），目标约为 0.6（0..1 normalized），权重 3
                arm_h = m.get('arm_height_ratio')
                if arm_h is not None:
                    arm_score = max(0.0, 1.0 - abs(arm_h - 0.6) / 0.6)
                    score += 3.0 * arm_score

                # 3) 躯干角度（越小越好），权重 2
                torso_a = m.get('torso_angle')
                if torso_a is not None:
                    torso_score = max(0.0, 1.0 - float(torso_a) / 45.0)
                    score += 2.0 * torso_score

                # 4) 手间距 / 肩宽比，目标约 0.75，权重 1.5
                hand_d = m.get('hand_distance')
                sh_w = m.get('shoulder_width')
                if hand_d is not None and sh_w is not None and sh_w > eps:
                    ratio = float(hand_d) / (float(sh_w) + eps)
                    ratio_score = max(0.0, 1.0 - abs(ratio - 0.75) / 1.5)
                    score += 1.5 * ratio_score

                # 5) 站位（脚距 / 臀宽），目标约 1.0，权重 1.0
                foot_d = m.get('foot_distance')
                hip_w = m.get('hip_width')
                if foot_d is not None and hip_w is not None and hip_w > eps:
                    fd_ratio = float(foot_d) / (float(hip_w) + eps)
                    fd_score = max(0.0, 1.0 - abs(fd_ratio - 1.0) / 1.0)
                    score += 1.0 * fd_score

                # 6) 肘部角度（越接近伸直越好，目标约160度），综合左右肘，权重 0.5
                le = m.get('left_elbow_angle')
                re = m.get('right_elbow_angle')
                elbow_scores = []
                for ang in (le, re):
                    if ang is not None:
                        elbow_scores.append(max(0.0, 1.0 - abs(float(ang) - 160.0) / 60.0))
                if elbow_scores:
                    score += 0.5 * (sum(elbow_scores) / len(elbow_scores))

                # 最终结果，保留三位小数
                return round(float(score), 3)

            # 如果总帧数少于4，则无法预测
            if len(metrics_list) < 4:
                print('视频帧数不足，无法生成 4 帧特征用于预测。')
                return

            # 新的帧选择策略：先找出多个高质量段（优先选择质量最好最多 4 段），然后从每个段中各选 1 帧。
            # 如果没有足够长的高质量段，则挑选分散的高分帧（保证最小间隔），最后回退到均匀采样。
            def select_frames_for_prediction(metrics_list):
                """
                策略：首先识别出所有高质量连续段（每帧 score >= seg_thresh），
                按段的质量排序并选择最多 4 段（优先长且平均分高的段），
                然后从每个段中选一个代表帧（段内得分最高帧）。

                如果段数量不足 4：
                - 先从最高分的段中各自选一帧（每段至多一帧），
                - 然后在剩余帧中按分散策略补齐至 4 帧（保证最小时间间隔）。

                最终返回按时间升序的 4 个帧索引。
                """
                scores = [candidate_score(m) for m in metrics_list]
                n = len(scores)

                # 参数（可调）
                seg_thresh = 2
                min_seg_len = 1  # 段长至少为1，以便捕获短但可靠的动作片段
                max_segments = 4

                # 找出得分>=阈值的连续段
                segments = []  # 列表条目为 (s, e)
                start = None
                for i, sc in enumerate(scores):
                    if sc >= seg_thresh:
                        if start is None:
                            start = i
                    else:
                        if start is not None:
                            segments.append((start, i - 1))
                            start = None
                if start is not None:
                    segments.append((start, n - 1))

                # 为每个段计算质量分并排序（优先长度、其次平均分）
                seg_info = []
                for (s, e) in segments:
                    length = e - s + 1
                    if length < min_seg_len:
                        continue
                    avg = sum(scores[s:e+1]) / length
                    quality = length * (1 + avg)
                    seg_info.append((quality, s, e, length, avg))

                seg_info.sort(reverse=True, key=lambda x: x[0])

                picked_indices = []

                # 从每个优先段中选 1 帧：选择段内得分最高的帧（代表性强）
                for info in seg_info[:max_segments]:
                    _, s, e, _, _ = info
                    # 找到段内得分最高的索引（若并列取中间那个）
                    best_idx = s
                    best_sc = scores[s]
                    for i in range(s, e + 1):
                        if scores[i] > best_sc:
                            best_sc = scores[i]
                            best_idx = i
                    picked_indices.append(best_idx)

                # 若选到的段少于 4，则在剩余帧中按分散策略补齐
                if len(picked_indices) < 4:
                    remaining_slots = 4 - len(picked_indices)
                    already = set(picked_indices)
                    # 构建候选：排除已选择的帧后按得分降序
                    candidates = [i for i in sorted(range(n), key=lambda i: scores[i], reverse=True) if i not in already and scores[i] > 0]
                    # 保证最小时间间隔，基于视频长度自适应
                    min_gap = max(1, n // 12)
                    for idx in candidates:
                        if all(abs(idx - p) >= min_gap for p in picked_indices):
                            picked_indices.append(idx)
                        if len(picked_indices) >= 4:
                            break

                    # 最后若仍不足，用均匀采样补齐（避免重复）
                    if len(picked_indices) < 4:
                        for ai in [int(round((n - 1) * (i + 0.5) / 4.0)) for i in range(4)]:
                            if ai not in picked_indices:
                                picked_indices.append(ai)
                            if len(picked_indices) >= 4:
                                break

                # 保证唯一并按时间排序，裁剪到 4
                picked = sorted(list(dict.fromkeys(picked_indices)))[:4]
                if len(picked) < 4:
                    # 极端回退：均匀采样
                    picked = [int(round((n - 1) * (i + 0.5) / 4.0)) for i in range(4)]
                return picked

            indices = select_frames_for_prediction(metrics_list)
            selected_frames = [metrics_list[i] for i in indices]
            print(f"Selected frame indices for prediction: {indices}")
            pred = predict_quality(selected_frames)
            print('模型预测结果：', pred)
        except Exception as e:
            print('调用模型预测时出错：', e)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import os
import cv2
import json
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import threading
import time

from src.action_recognition import recognize_action
from src.detection import detect_human
from src.feedback import provide_feedback
from src.pose_estimation_v2 import estimate_pose

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局变量用于存储处理状态
processing_status = {
    'is_processing': False,
    'current_frame': 0,
    'total_frames': 0,
    'metrics': [],
    'feedback': [],
    'final_result': None
}

# 摄像头状态
camera_active = False
camera = None
camera_lock = threading.Lock()

# 摄像头分析状态 - 用于记录动作并分析
camera_analysis_status = {
    'is_recording': False,      # 是否正在记录动作
    'metrics': [],              # 存储所有帧的特征数据
    'feedback': [],             # 存储每帧的反馈信息
    'start_time': None,         # 开始记录时间
    'final_result': None        # 最终分析结果
}
camera_analysis_lock = threading.Lock()


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_file(video_path):
    """处理视频文件并返回分析结果"""
    global processing_status

    processing_status['is_processing'] = True
    processing_status['current_frame'] = 0
    processing_status['metrics'] = []
    processing_status['feedback'] = []
    processing_status['final_result'] = None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processing_status['total_frames'] = total_frames

    metrics_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人体
        frame_with_detection = detect_human(frame)
        # 姿态估计
        pose_estimated_frame, results = estimate_pose(frame_with_detection)
        # 动作识别
        metrics = recognize_action(results)
        metrics_list.append(metrics)

        # 提供即时反馈
        feedback = provide_feedback(metrics)

        processing_status['current_frame'] += 1
        processing_status['metrics'].append(metrics)
        processing_status['feedback'].append(feedback)

    cap.release()

    # 视频处理完成后，尝试调用模型预测
    final_result = None
    if len(metrics_list) >= 4:
        try:
            from taichi_ai.predict_v2 import predict_quality
            from frame_selector_v2 import select_frames_evenly

            # 使用均匀选帧（默认20帧）
            # 获取视频FPS
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

            indices, selected_frames = select_frames_evenly(
                metrics_list,
                n_frames=20,
                auto_detect_boundaries=True,
                fps=fps,
                verbose=False
            )

            # 调用模型预测
            pred = predict_quality(selected_frames)
            final_result = {
                'score': float(pred.get('score', 0)),
                'advice': pred.get('advice', '无'),
                'selected_frame_indices': indices
            }
        except Exception as e:
            print(f'模型预测出错: {e}')
            final_result = {'error': str(e)}

    processing_status['final_result'] = final_result
    processing_status['is_processing'] = False

    return final_result


def generate_camera_frames():
    """生成摄像头帧流"""
    global camera, camera_active, camera_analysis_status

    while camera_active:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret:
            break

        # 检测人体
        frame_with_detection = detect_human(frame)
        # 姿态估计
        pose_estimated_frame, results = estimate_pose(frame_with_detection)
        # 动作识别
        metrics = recognize_action(results)
        # 提供即时反馈
        feedback = provide_feedback(metrics)

        # 如果正在记录动作,保存metrics和feedback
        with camera_analysis_lock:
            if camera_analysis_status['is_recording']:
                camera_analysis_status['metrics'].append(metrics)
                camera_analysis_status['feedback'].append(feedback)

        # 在帧上绘制反馈信息和录制状态
        from utils.utils import draw_chinese_text
        if feedback:
            draw_chinese_text(pose_estimated_frame, feedback, (10, 30))

        # 显示录制状态
        if camera_analysis_status['is_recording']:
            frame_count = len(camera_analysis_status['metrics'])
            status_text = f"正在记录动作... 已记录 {frame_count} 帧"
            draw_chinese_text(pose_estimated_frame, status_text, (10, 70),
                            color=(0, 0, 255))  # 红色提示

        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', pose_estimated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """处理视频上传"""
    if 'video' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400

    # 保存文件
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # 在后台处理视频
    thread = threading.Thread(target=process_video_file, args=(filepath,))
    thread.start()

    return jsonify({
        'message': '视频上传成功，正在处理中...',
        'filename': filename
    })


@app.route('/api/status')
def get_status():
    """获取处理状态"""
    return jsonify(processing_status)


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """启动摄像头"""
    global camera, camera_active

    with camera_lock:
        if camera_active:
            return jsonify({'error': '摄像头已启动'}), 400

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'error': '无法打开摄像头'}), 500

        camera_active = True

    return jsonify({'message': '摄像头已启动'})


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """停止摄像头"""
    global camera, camera_active

    with camera_lock:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None

    return jsonify({'message': '摄像头已停止'})


@app.route('/api/camera/feed')
def camera_feed():
    """摄像头视频流"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/evaluate', methods=['POST'])
def evaluate_action():
    """评估动作（用于实时摄像头）"""
    data = request.json
    # 这里可以根据需要实现实时评估逻辑
    return jsonify({'message': '功能开发中'})


@app.route('/api/camera/start_recording', methods=['POST'])
def start_recording():
    """开始记录摄像头动作"""
    global camera_analysis_status

    with camera_analysis_lock:
        if not camera_active:
            return jsonify({'error': '请先启动摄像头'}), 400

        if camera_analysis_status['is_recording']:
            return jsonify({'error': '已经在记录中'}), 400

        # 重置并开始记录
        camera_analysis_status = {
            'is_recording': True,
            'metrics': [],
            'feedback': [],
            'start_time': time.time(),
            'final_result': None
        }

    return jsonify({'message': '开始记录动作'})


@app.route('/api/camera/stop_recording', methods=['POST'])
def stop_recording():
    """停止记录摄像头动作"""
    global camera_analysis_status

    with camera_analysis_lock:
        if not camera_analysis_status['is_recording']:
            return jsonify({'error': '当前没有在记录'}), 400

        camera_analysis_status['is_recording'] = False
        frame_count = len(camera_analysis_status['metrics'])

    return jsonify({
        'message': '停止记录动作',
        'frame_count': frame_count
    })


@app.route('/api/camera/analyze', methods=['POST'])
def analyze_camera_action():
    """分析已记录的摄像头动作"""
    global camera_analysis_status

    with camera_analysis_lock:
        if camera_analysis_status['is_recording']:
            return jsonify({'error': '请先停止记录'}), 400

        metrics_list = camera_analysis_status['metrics']

        if len(metrics_list) < 4:
            return jsonify({'error': f'记录的帧数太少({len(metrics_list)}帧),至少需要4帧'}), 400

    # 在后台线程中进行分析
    def analyze_in_background():
        global camera_analysis_status
        try:
            from taichi_ai.predict_v2 import predict_quality
            from frame_selector_v2 import select_frames_evenly

            # 使用均匀选帧(默认20帧)
            # 假设摄像头FPS为30
            fps = 30

            indices, selected_frames = select_frames_evenly(
                metrics_list,
                n_frames=20,
                auto_detect_boundaries=True,
                fps=fps,
                verbose=False
            )

            # 调用模型预测
            pred = predict_quality(selected_frames)
            final_result = {
                'score': float(pred.get('score', 0)),
                'advice': pred.get('advice', '无'),
                'selected_frame_indices': indices,
                'total_frames': len(metrics_list)
            }

            with camera_analysis_lock:
                camera_analysis_status['final_result'] = final_result

        except Exception as e:
            print(f'摄像头动作分析出错: {e}')
            with camera_analysis_lock:
                camera_analysis_status['final_result'] = {'error': str(e)}

    thread = threading.Thread(target=analyze_in_background)
    thread.start()

    return jsonify({
        'message': '正在分析动作...',
        'frame_count': len(metrics_list)
    })


@app.route('/api/camera/analysis_status')
def get_camera_analysis_status():
    """获取摄像头分析状态"""
    with camera_analysis_lock:
        return jsonify({
            'is_recording': camera_analysis_status['is_recording'],
            'frame_count': len(camera_analysis_status['metrics']),
            'final_result': camera_analysis_status['final_result']
        })


@app.route('/api/camera/reset', methods=['POST'])
def reset_camera_analysis():
    """重置摄像头分析数据"""
    global camera_analysis_status

    with camera_analysis_lock:
        camera_analysis_status = {
            'is_recording': False,
            'metrics': [],
            'feedback': [],
            'start_time': None,
            'final_result': None
        }

    return jsonify({'message': '摄像头分析数据已重置'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

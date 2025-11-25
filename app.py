import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import uuid
import threading
from trajectory_system.trajectory_evaluator import TrajectoryEvaluator
from trajectory_system.smart_frame_selector import SmartFrameSelector

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

progress_store = {}
progress_lock = threading.Lock()

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    with progress_lock:
        progress = progress_store.get(task_id, {'progress': 0, 'message': '等待开始'})
    return jsonify(progress)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    task_id = str(uuid.uuid4())
    with progress_lock:
        progress_store[task_id] = {'progress': 0, 'message': '开始处理...'}
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    def process_task():
        try:
            update_progress(task_id, 10, '读取视频文件...')
            result = process_video(filepath, task_id)
            update_progress(task_id, 100, '处理完成')
            with progress_lock:
                progress_store[task_id] = {'progress': 100, 'message': '完成', 'result': result}
            os.remove(filepath)
        except Exception as e:
            with progress_lock:
                progress_store[task_id] = {'progress': 0, 'message': '错误', 'error': str(e)}
            if os.path.exists(filepath):
                os.remove(filepath)
    
    thread = threading.Thread(target=process_task)
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/api/process_frames', methods=['POST'])
def process_frames():
    data = request.json
    frames_data = data.get('frames', [])
    
    if len(frames_data) == 0:
        return jsonify({'error': '没有提供帧数据'}), 400
    
    task_id = str(uuid.uuid4())
    with progress_lock:
        progress_store[task_id] = {'progress': 0, 'message': '开始处理...'}
    
    def process_task():
        try:
            update_progress(task_id, 10, '解码图像帧...')
            frames = []
            total = len(frames_data)
            for i, frame_data in enumerate(frames_data):
                header, encoded = frame_data.split(',', 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frames.append(frame)
                update_progress(task_id, 10 + int(30 * (i + 1) / total), f'解码帧 {i+1}/{total}...')
            
            update_progress(task_id, 50, '选择关键帧...')
            result = process_video_frames(frames, task_id)
            update_progress(task_id, 100, '处理完成')
            with progress_lock:
                progress_store[task_id] = {'progress': 100, 'message': '完成', 'result': result}
        except Exception as e:
            with progress_lock:
                progress_store[task_id] = {'progress': 0, 'message': '错误', 'error': str(e)}
    
    thread = threading.Thread(target=process_task)
    thread.start()
    
    return jsonify({'task_id': task_id})

def update_progress(task_id, progress, message):
    with progress_lock:
        if task_id in progress_store:
            progress_store[task_id]['progress'] = progress
            progress_store[task_id]['message'] = message

def process_video(video_path, task_id=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
        if task_id and frame_count % 10 == 0:
            progress = 10 + int(40 * frame_count / max(total_frames, 1))
            update_progress(task_id, min(progress, 50), f'读取视频帧 {frame_count}...')
    
    cap.release()
    
    if len(frames) == 0:
        return {'error': '无法读取视频帧'}
    
    return process_video_frames(frames, task_id)

def process_video_frames(frames, task_id=None):
    if len(frames) < 4:
        return {
            'score': 0,
            'accuracy': 0,
            'similarities': {},
            'advice': ['视频帧数不足，需要至少4帧'],
            'error': '帧数不足'
        }
    
    if task_id:
        update_progress(task_id, 50, '选择关键帧...')
    
    selector = SmartFrameSelector()
    indices = selector.select_best_frames(frames, min_gap=5)
    selected_frames = [frames[i] for i in indices]
    
    if task_id:
        update_progress(task_id, 60, '提取关键点...')
    
    evaluator = TrajectoryEvaluator()
    
    if task_id:
        update_progress(task_id, 70, '计算轨迹相似度...')
    
    result = evaluator.evaluate_video_frames(selected_frames)
    
    if task_id:
        update_progress(task_id, 90, '生成评估报告...')
    
    return result

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

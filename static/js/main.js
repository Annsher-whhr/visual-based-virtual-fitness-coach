// 全局变量
let isProcessing = false;
let isCameraActive = false;
let isCameraRecording = false;  // 摄像头录制状态
let statusCheckInterval = null;
let cameraAnalysisInterval = null;  // 摄像头分析状态轮询

// DOM元素
const videoUpload = document.getElementById('videoUpload');
const startCameraBtn = document.getElementById('startCamera');
const stopCameraBtn = document.getElementById('stopCamera');
const cameraFeed = document.getElementById('cameraFeed');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const progressSection = document.getElementById('progressSection');
const progressText = document.getElementById('progressText');
const progressPercent = document.getElementById('progressPercent');
const progressFill = document.getElementById('progressFill');
const feedbackContent = document.getElementById('feedbackContent');
const evaluationContent = document.getElementById('evaluationContent');
const metricsContent = document.getElementById('metricsContent');
const scoreDisplay = document.getElementById('scoreDisplay');
const scoreNumber = document.getElementById('scoreNumber');
const scoreProgress = document.getElementById('scoreProgress');
const adviceContent = document.getElementById('adviceContent');
const evaluationPlaceholder = document.getElementById('evaluationPlaceholder');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// 设置事件监听器
function setupEventListeners() {
    videoUpload.addEventListener('change', handleVideoUpload);
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
}

// 处理视频上传
async function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // 检查文件类型
    if (!file.type.startsWith('video/')) {
        showNotification('请选择视频文件', 'error');
        return;
    }

    // 检查文件大小 (100MB)
    if (file.size > 100 * 1024 * 1024) {
        showNotification('文件大小不能超过100MB', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('video', file);

    try {
        showNotification('正在上传视频...', 'info');

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showNotification(data.message, 'success');
            startStatusPolling();
        } else {
            showNotification(data.error || '上传失败', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('上传失败，请重试', 'error');
    }
}

// 启动摄像头
async function startCamera() {
    try {
        const response = await fetch('/api/camera/start', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            isCameraActive = true;
            cameraFeed.src = '/api/camera/feed?' + new Date().getTime();
            cameraFeed.style.display = 'block';
            videoPlaceholder.style.display = 'none';
            startCameraBtn.style.display = 'none';
            stopCameraBtn.style.display = 'block';
            showNotification(data.message, 'success');
        } else {
            showNotification(data.error || '启动摄像头失败', 'error');
        }
    } catch (error) {
        console.error('Start camera error:', error);
        showNotification('启动摄像头失败', 'error');
    }
}

// 停止摄像头
async function stopCamera() {
    try {
        const response = await fetch('/api/camera/stop', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            isCameraActive = false;
            cameraFeed.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
            startCameraBtn.style.display = 'block';
            stopCameraBtn.style.display = 'none';
            showNotification(data.message, 'success');
        } else {
            showNotification(data.error || '停止摄像头失败', 'error');
        }
    } catch (error) {
        console.error('Stop camera error:', error);
        showNotification('停止摄像头失败', 'error');
    }
}

// 开始状态轮询
function startStatusPolling() {
    isProcessing = true;
    progressSection.style.display = 'block';
    evaluationPlaceholder.style.display = 'block';
    scoreDisplay.style.display = 'none';
    adviceContent.innerHTML = '';

    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }

    statusCheckInterval = setInterval(checkStatus, 500);
}

// 检查处理状态
async function checkStatus() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();

        if (status.is_processing) {
            updateProgress(status);
        } else if (status.final_result) {
            stopStatusPolling();
            displayResults(status);
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

// 停止状态轮询
function stopStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
    isProcessing = false;
    progressSection.style.display = 'none';
}

// 更新进度
function updateProgress(status) {
    const percent = status.total_frames > 0
        ? Math.round((status.current_frame / status.total_frames) * 100)
        : 0;

    progressText.textContent = `处理中... (${status.current_frame}/${status.total_frames})`;
    progressPercent.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;

    // 显示最新的反馈
    if (status.feedback && status.feedback.length > 0) {
        const latestFeedback = status.feedback[status.feedback.length - 1];
        displayFeedback(latestFeedback);
    }

    // 显示最新的指标
    if (status.metrics && status.metrics.length > 0) {
        const latestMetrics = status.metrics[status.metrics.length - 1];
        displayMetrics(latestMetrics);
    }
}

// 显示结果
function displayResults(status) {
    const result = status.final_result;

    if (result && !result.error) {
        evaluationPlaceholder.style.display = 'none';
        scoreDisplay.style.display = 'block';

        // 显示分数
        const score = Math.round(result.score * 100);
        animateScore(score);

        // 显示建议
        if (result.advice) {
            adviceContent.innerHTML = `
                <div class="advice-item">
                    <strong>AI建议：</strong><br>
                    ${result.advice}
                </div>
            `;
        }

        showNotification('视频分析完成！', 'success');
    } else if (result && result.error) {
        evaluationPlaceholder.textContent = `分析出错: ${result.error}`;
        showNotification('分析过程中出现错误', 'error');
    }
}

// 动画显示分数
function animateScore(targetScore) {
    const circumference = 2 * Math.PI * 45;
    scoreProgress.style.strokeDasharray = circumference;
    scoreProgress.style.strokeDashoffset = circumference;

    let currentScore = 0;
    const duration = 1500;
    const steps = 60;
    const increment = targetScore / steps;
    const stepTime = duration / steps;

    const interval = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(interval);
        }

        scoreNumber.textContent = Math.round(currentScore);

        const offset = circumference - (currentScore / 100) * circumference;
        scoreProgress.style.strokeDashoffset = offset;

        // 根据分数改变颜色
        if (currentScore >= 80) {
            scoreProgress.style.stroke = '#52c41a';
        } else if (currentScore >= 60) {
            scoreProgress.style.stroke = '#faad14';
        } else {
            scoreProgress.style.stroke = '#ff4d4f';
        }
    }, stepTime);
}

// 显示反馈信息
function displayFeedback(feedback) {
    if (!feedback || feedback === '无') {
        feedbackContent.innerHTML = '<p class="placeholder-text">暂无反馈信息</p>';
        return;
    }

    // 判断反馈类型
    let feedbackClass = 'feedback-message';
    if (feedback.includes('良好') || feedback.includes('正确')) {
        feedbackClass += ' success';
    } else if (feedback.includes('注意') || feedback.includes('建议')) {
        feedbackClass += ' warning';
    } else if (feedback.includes('错误') || feedback.includes('不足')) {
        feedbackClass += ' error';
    }

    feedbackContent.innerHTML = `<div class="${feedbackClass}">${feedback}</div>`;
}

// 显示动作指标
function displayMetrics(metrics) {
    if (!metrics || typeof metrics !== 'object') {
        metricsContent.innerHTML = '<p class="placeholder-text">暂无数据</p>';
        return;
    }

    const metricLabels = {
        'arm_height_ratio': '手臂高度比',
        'torso_angle': '躯干角度',
        'hand_distance': '双手距离',
        'shoulder_width': '肩宽',
        'foot_distance': '脚距',
        'hip_width': '臀宽',
        'left_elbow_angle': '左肘角度',
        'right_elbow_angle': '右肘角度',
        'avg_visibility': '平均可见度'
    };

    let html = '<div class="metrics-grid">';

    for (const [key, label] of Object.entries(metricLabels)) {
        if (metrics[key] !== undefined && metrics[key] !== null) {
            let value = metrics[key];
            let unit = '';

            if (key.includes('angle')) {
                unit = '°';
                value = Math.round(value);
            } else if (key === 'avg_visibility') {
                value = (value * 100).toFixed(1);
                unit = '%';
            } else {
                value = value.toFixed(2);
            }

            html += `
                <div class="metric-item">
                    <div class="metric-label">${label}</div>
                    <div class="metric-value">${value}${unit}</div>
                </div>
            `;
        }
    }

    html += '</div>';
    metricsContent.innerHTML = html;
}

// 显示通知
function showNotification(message, type = 'info') {
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideInRight 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    `;

    // 设置背景色
    switch (type) {
        case 'success':
            notification.style.background = '#52c41a';
            break;
        case 'error':
            notification.style.background = '#ff4d4f';
            break;
        case 'warning':
            notification.style.background = '#faad14';
            break;
        default:
            notification.style.background = '#4a90e2';
    }

    document.body.appendChild(notification);

    // 3秒后自动移除
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// 添加动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
`;
document.head.appendChild(style);

// ==================== 摄像头分析功能 ====================

// 开始记录摄像头动作
async function startCameraRecording() {
    try {
        const response = await fetch('/api/camera/start_recording', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            isCameraRecording = true;
            const recordBtn = document.getElementById('startRecordBtn');
            const stopRecordBtn = document.getElementById('stopRecordBtn');
            if (recordBtn) recordBtn.style.display = 'none';
            if (stopRecordBtn) stopRecordBtn.style.display = 'block';
            showNotification('开始记录动作...', 'success');
        } else {
            showNotification(data.error || '开始记录失败', 'error');
        }
    } catch (error) {
        console.error('Start recording error:', error);
        showNotification('开始记录失败', 'error');
    }
}

// 停止记录摄像头动作
async function stopCameraRecording() {
    try {
        const response = await fetch('/api/camera/stop_recording', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            isCameraRecording = false;
            const recordBtn = document.getElementById('startRecordBtn');
            const stopRecordBtn = document.getElementById('stopRecordBtn');
            const analyzeBtn = document.getElementById('analyzeCameraBtn');
            if (recordBtn) recordBtn.style.display = 'block';
            if (stopRecordBtn) stopRecordBtn.style.display = 'none';
            if (analyzeBtn) analyzeBtn.disabled = false;
            showNotification(`停止记录,已记录${data.frame_count}帧`, 'success');
        } else {
            showNotification(data.error || '停止记录失败', 'error');
        }
    } catch (error) {
        console.error('Stop recording error:', error);
        showNotification('停止记录失败', 'error');
    }
}

// 分析摄像头动作
async function analyzeCameraAction() {
    try {
        const response = await fetch('/api/camera/analyze', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            showNotification('正在分析动作,请稍候...', 'info');
            const analyzeBtn = document.getElementById('analyzeCameraBtn');
            if (analyzeBtn) analyzeBtn.disabled = true;

            // 开始轮询分析状态
            startCameraAnalysisPolling();
        } else {
            showNotification(data.error || '分析失败', 'error');
        }
    } catch (error) {
        console.error('Analyze camera action error:', error);
        showNotification('分析失败', 'error');
    }
}

// 开始轮询摄像头分析状态
function startCameraAnalysisPolling() {
    if (cameraAnalysisInterval) {
        clearInterval(cameraAnalysisInterval);
    }

    evaluationPlaceholder.style.display = 'block';
    evaluationPlaceholder.textContent = '正在分析中...';
    scoreDisplay.style.display = 'none';
    adviceContent.innerHTML = '';

    cameraAnalysisInterval = setInterval(checkCameraAnalysisStatus, 1000);
}

// 检查摄像头分析状态
async function checkCameraAnalysisStatus() {
    try {
        const response = await fetch('/api/camera/analysis_status');
        const status = await response.json();

        if (status.final_result) {
            stopCameraAnalysisPolling();
            displayCameraAnalysisResults(status.final_result);
        }
    } catch (error) {
        console.error('Check camera analysis status error:', error);
    }
}

// 停止轮询摄像头分析状态
function stopCameraAnalysisPolling() {
    if (cameraAnalysisInterval) {
        clearInterval(cameraAnalysisInterval);
        cameraAnalysisInterval = null;
    }
}

// 显示摄像头分析结果
function displayCameraAnalysisResults(result) {
    if (result && !result.error) {
        evaluationPlaceholder.style.display = 'none';
        scoreDisplay.style.display = 'block';

        // 显示分数
        const score = Math.round(result.score * 100);
        animateScore(score);

        // 显示建议
        if (result.advice) {
            adviceContent.innerHTML = `
                <div class="advice-item">
                    <strong>AI建议：</strong><br>
                    ${result.advice}
                </div>
                <div class="advice-item" style="margin-top: 10px;">
                    <small>分析帧数: ${result.total_frames} 帧，选取: ${result.selected_frame_indices ? result.selected_frame_indices.length : 20} 帧</small>
                </div>
            `;
        }

        showNotification('动作分析完成！', 'success');

        // 重新启用分析按钮
        const analyzeBtn = document.getElementById('analyzeCameraBtn');
        if (analyzeBtn) analyzeBtn.disabled = false;
    } else if (result && result.error) {
        evaluationPlaceholder.textContent = `分析出错: ${result.error}`;
        showNotification('分析过程中出现错误', 'error');

        const analyzeBtn = document.getElementById('analyzeCameraBtn');
        if (analyzeBtn) analyzeBtn.disabled = false;
    }
}

// 重置摄像头分析数据
async function resetCameraAnalysis() {
    try {
        const response = await fetch('/api/camera/reset', {
            method: 'POST'
        });

        const data = await response.json();

        if (response.ok) {
            // 清空显示
            scoreDisplay.style.display = 'none';
            evaluationPlaceholder.style.display = 'block';
            evaluationPlaceholder.textContent = '请先记录动作，然后点击分析按钮';
            adviceContent.innerHTML = '';

            const analyzeBtn = document.getElementById('analyzeCameraBtn');
            if (analyzeBtn) analyzeBtn.disabled = true;

            showNotification('已重置分析数据', 'success');
        } else {
            showNotification(data.error || '重置失败', 'error');
        }
    } catch (error) {
        console.error('Reset camera analysis error:', error);
        showNotification('重置失败', 'error');
    }
}

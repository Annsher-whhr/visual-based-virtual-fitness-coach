let currentStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let selectedFile = null;
let capturedFrames = [];

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const videoInput = document.getElementById('videoInput');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const removeFile = document.getElementById('removeFile');
    const processBtn = document.getElementById('processBtn');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const startCamera = document.getElementById('startCamera');
    const stopCamera = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('captureBtn');
    const stopCapture = document.getElementById('stopCapture');
    const cameraVideo = document.getElementById('cameraVideo');
    const recordingIndicator = document.getElementById('recordingIndicator');

    uploadArea.addEventListener('click', () => videoInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('dragleave', handleDragLeave);

    videoInput.addEventListener('change', handleFileSelect);
    removeFile.addEventListener('click', handleRemoveFile);
    processBtn.addEventListener('click', handleProcess);

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    startCamera.addEventListener('click', startCameraStream);
    stopCamera.addEventListener('click', stopCameraStream);
    captureBtn.addEventListener('click', startRecording);
    stopCapture.addEventListener('click', stopRecording);

    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('video/')) {
            handleFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    }

    function handleFile(file) {
        if (!file.type.startsWith('video/')) {
            alert('请选择视频文件');
            return;
        }
        selectedFile = file;
        fileName.textContent = file.name;
        fileInfo.style.display = 'flex';
        uploadArea.style.display = 'none';
        processBtn.disabled = false;
    }

    function handleRemoveFile() {
        selectedFile = null;
        videoInput.value = '';
        fileInfo.style.display = 'none';
        uploadArea.style.display = 'block';
        processBtn.disabled = true;
    }

    function switchTab(tab) {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });

        document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
        document.getElementById(`${tab}-tab`).classList.add('active');

        if (tab === 'camera') {
            processBtn.disabled = true;
        } else {
            processBtn.disabled = !selectedFile;
        }
    }

    async function startCameraStream() {
        try {
            currentStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 }
            });
            cameraVideo.srcObject = currentStream;
            startCamera.style.display = 'none';
            stopCamera.style.display = 'inline-block';
            captureBtn.style.display = 'inline-block';
        } catch (err) {
            alert('无法访问摄像头: ' + err.message);
        }
    }

    function stopCameraStream() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
            cameraVideo.srcObject = null;
            startCamera.style.display = 'inline-block';
            stopCamera.style.display = 'none';
            captureBtn.style.display = 'none';
            stopCapture.style.display = 'none';
            recordingIndicator.style.display = 'none';
            capturedFrames = [];
            processBtn.disabled = true;
        }
    }

    function startRecording() {
        capturedFrames = [];
        const canvas = document.getElementById('cameraCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = cameraVideo.videoWidth;
        canvas.height = cameraVideo.videoHeight;

        const captureInterval = setInterval(() => {
            ctx.drawImage(cameraVideo, 0, 0);
            const frameData = canvas.toDataURL('image/jpeg', 0.8);
            capturedFrames.push(frameData);
        }, 200);

        captureBtn.style.display = 'none';
        stopCapture.style.display = 'inline-block';
        recordingIndicator.style.display = 'flex';

        window.captureInterval = captureInterval;
    }

    function stopRecording() {
        if (window.captureInterval) {
            clearInterval(window.captureInterval);
            window.captureInterval = null;
        }

        if (capturedFrames.length >= 4) {
            processBtn.disabled = false;
            alert(`已录制 ${capturedFrames.length} 帧，可以开始分析了`);
        } else {
            alert('录制帧数不足，至少需要4帧');
        }

        captureBtn.style.display = 'inline-block';
        stopCapture.style.display = 'none';
        recordingIndicator.style.display = 'none';
    }

    async function handleProcess() {
        const loading = document.getElementById('loading');
        const resultSection = document.getElementById('resultSection');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');

        loading.style.display = 'block';
        resultSection.style.display = 'none';
        processBtn.disabled = true;
        progressFill.style.width = '0%';
        progressText.textContent = '开始处理...';

        try {
            let taskId;
            const activeTab = document.querySelector('.tab-btn.active').dataset.tab;

            if (activeTab === 'upload') {
                if (!selectedFile) {
                    throw new Error('请先选择视频文件');
                }
                taskId = await uploadAndProcess(selectedFile);
            } else {
                if (capturedFrames.length < 4) {
                    throw new Error('录制帧数不足，至少需要4帧');
                }
                taskId = await processFrames(capturedFrames);
            }

            const result = await pollProgress(taskId, progressFill, progressText);
            displayResult(result);
        } catch (error) {
            alert('处理失败: ' + error.message);
        } finally {
            loading.style.display = 'none';
            processBtn.disabled = false;
        }
    }

    async function pollProgress(taskId, progressFill, progressText) {
        return new Promise((resolve, reject) => {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/api/progress/${taskId}`);
                    const data = await response.json();

                    progressFill.style.width = data.progress + '%';
                    progressText.textContent = data.message || `处理中... ${data.progress}%`;

                    if (data.result) {
                        clearInterval(interval);
                        resolve(data.result);
                    } else if (data.error) {
                        clearInterval(interval);
                        reject(new Error(data.error));
                    }
                } catch (error) {
                    clearInterval(interval);
                    reject(error);
                }
            }, 500);

            setTimeout(() => {
                clearInterval(interval);
                reject(new Error('处理超时'));
            }, 300000);
        });
    }

    async function uploadAndProcess(file) {
        const formData = new FormData();
        formData.append('video', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '上传失败');
        }

        const data = await response.json();
        return data.task_id;
    }

    async function processFrames(frames) {
        const response = await fetch('/api/process_frames', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ frames: frames })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || '处理失败');
        }

        const data = await response.json();
        return data.task_id;
    }

    function displayResult(result) {
        if (result.error) {
            throw new Error(result.error);
        }

        const resultSection = document.getElementById('resultSection');
        const scoreValue = document.getElementById('scoreValue');
        const scoreLevel = document.getElementById('scoreLevel');
        const similarities = document.getElementById('similarities');
        const adviceList = document.getElementById('adviceList');

        const score = Math.round(result.score || 0);
        scoreValue.textContent = score;

        let level = '需改进';
        let levelColor = '#e74c3c';
        if (score >= 90) {
            level = '优秀';
            levelColor = '#27ae60';
        } else if (score >= 80) {
            level = '良好';
            levelColor = '#3498db';
        } else if (score >= 70) {
            level = '中等';
            levelColor = '#f39c12';
        }

        scoreLevel.textContent = level;
        scoreLevel.style.color = levelColor;

        similarities.innerHTML = '';
        similarities.innerHTML = '';

        const bodyPartTranslations = {
            'left_shoulder': '左肩',
            'right_shoulder': '右肩',
            'left_elbow': '左肘',
            'right_elbow': '右肘',
            'left_wrist': '左腕',
            'right_wrist': '右腕',
            'left_hip': '左髋',
            'right_hip': '右髋',
            'left_knee': '左膝',
            'right_knee': '右膝',
            'left_ankle': '左脚踝',
            'right_ankle': '右脚踝'
        };

        if (result.similarities) {
            Object.entries(result.similarities).forEach(([part, value]) => {
                const item = document.createElement('div');
                item.className = 'similarity-item';
                // Use translation or fallback to formatted English
                const partName = bodyPartTranslations[part] || part.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                // Main bar
                const mainContent = `
                    <div class="similarity-label">
                        <span>${partName}</span>
                        <span>${Math.round(value)}%</span>
                    </div>
                    <div class="similarity-bar">
                        <div class="similarity-fill" style="width: ${value}%">${Math.round(value)}%</div>
                    </div>
                `;

                // Details section
                let detailsContent = '<div class="similarity-details">';

                // Standard Frames
                detailsContent += '<div class="frames-row"><h4>标准动作</h4><div class="frames-grid">';
                for (let i = 0; i < 4; i++) {
                    const kp = result.standard_frames && result.standard_frames[i] ? result.standard_frames[i][part] : null;
                    const markerStyle = kp ? `left: ${kp[0] * 100}%; top: ${kp[1] * 100}%;` : 'display: none;';
                    // Standard frames are 1-indexed in filename: 01.png, 02.png...
                    detailsContent += `
                        <div class="frame-container">
                            <img src="/video/0${i + 1}.png" alt="Standard ${i + 1}">
                            <div class="keypoint-marker" style="${markerStyle}"></div>
                        </div>
                    `;
                }
                detailsContent += '</div></div>';

                // User Frames
                detailsContent += '<div class="frames-row"><h4>你的动作</h4><div class="frames-grid">';
                for (let i = 0; i < 4; i++) {
                    const kp = result.user_frames && result.user_frames[i] ? result.user_frames[i][part] : null;
                    const markerStyle = kp ? `left: ${kp[0] * 100}%; top: ${kp[1] * 100}%;` : 'display: none;';
                    const imgSrc = result.user_images && result.user_images[i] ? `data:image/jpeg;base64,${result.user_images[i]}` : '';
                    detailsContent += `
                        <div class="frame-container">
                            <img src="${imgSrc}" alt="User ${i + 1}">
                            <div class="keypoint-marker" style="${markerStyle}"></div>
                        </div>
                    `;
                }
                detailsContent += '</div></div>';

                detailsContent += '</div>'; // End details

                item.innerHTML = mainContent + detailsContent;

                item.addEventListener('click', (e) => {
                    // Prevent toggling if clicking inside details
                    if (e.target.closest('.similarity-details')) {
                        return;
                    }

                    const details = item.querySelector('.similarity-details');
                    const wasActive = details.classList.contains('active');

                    // Close all others
                    document.querySelectorAll('.similarity-details').forEach(d => d.classList.remove('active'));

                    if (!wasActive) {
                        details.classList.add('active');
                    }
                });

                similarities.appendChild(item);
            });
        }

        adviceList.innerHTML = '';
        if (result.advice && result.advice.length > 0) {
            result.advice.forEach(advice => {
                const li = document.createElement('li');
                li.textContent = advice;
                adviceList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = '动作标准，保持！';
            adviceList.appendChild(li);
        }

        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
});


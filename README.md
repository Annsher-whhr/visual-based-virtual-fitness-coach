# 基于视觉的虚拟健身教练系统

一个基于计算机视觉和姿态估计技术的智能健身动作评估系统，专注于太极拳起势动作的质量评估和改进建议。

## 📋 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [核心算法](#核心算法)
- [API 文档](#api-文档)
- [常见问题](#常见问题)
- [许可证](#许可证)

## 🎯 项目简介

本系统通过分析用户上传的视频或实时摄像头画面，自动提取人体关键点，选择关键动作帧，并与标准动作进行轨迹相似度匹配，从而评估用户动作的标准程度，并提供个性化的改进建议。

### 主要应用场景

- **健身训练指导**：实时评估动作质量，提供专业反馈
- **太极拳学习**：帮助初学者纠正动作，提高练习效果
- **运动康复**：监测康复训练动作的准确性
- **体育教学**：辅助体育教师进行动作标准化教学

## ✨ 功能特性

### 核心功能

1. **智能关键帧选择**
   - 自动从视频中识别并选择最具代表性的4个关键动作帧
   - 基于与标准动作的相似度进行智能匹配
   - 支持最小帧间隔设置，确保动作序列的完整性

2. **姿态估计与关键点提取**
   - 使用 MediaPipe Pose 模型进行高精度人体姿态检测
   - 提取11个关键身体关节点（肩、肘、腕、髋、膝、踝）
   - 支持3D坐标提取，提供更准确的空间位置信息

3. **轨迹相似度匹配**
   - 使用 FastDTW 算法进行动态时间规整
   - 计算用户动作与标准动作的轨迹相似度
   - 提供各身体部位的详细相似度评分

4. **动作质量评估**
   - 综合评分系统（0-100分）
   - 多维度相似度分析（各身体部位独立评分）
   - 智能生成个性化改进建议

5. **双模式使用**
   - **Web 应用模式**：通过浏览器上传视频或使用摄像头实时录制
   - **命令行模式**：适合批量处理和自动化脚本

### 技术亮点

- 🎯 **高精度姿态检测**：基于 MediaPipe 的实时人体姿态估计
- 🔍 **智能帧选择算法**：多阶段优化搜索，确保选择最佳关键帧
- 📊 **轨迹匹配算法**：FastDTW 动态时间规整，处理不同速度的动作
- 🚀 **异步处理**：支持大视频文件的异步处理，实时进度反馈
- 💡 **实时反馈**：Web 界面实时显示处理进度和评估结果

## 🛠 技术栈

### 后端技术

- **Python 3.7+**
- **Flask**：Web 框架
- **OpenCV**：视频处理和图像操作
- **MediaPipe**：人体姿态估计
- **NumPy**：数值计算
- **SciPy**：科学计算（FastDTW 算法）
- **scikit-learn**：机器学习工具
- **joblib**：模型序列化

### 前端技术

- **HTML5**：页面结构
- **CSS3**：样式设计
- **JavaScript (ES6+)**：交互逻辑
- **Canvas API**：图像处理

### 核心算法

- **MediaPipe Pose**：Google 的人体姿态估计模型
- **FastDTW**：快速动态时间规整算法
- **欧几里得距离**：关键点距离计算
- **动态规划**：最优帧序列搜索

## 📦 安装指南

### 系统要求

- Python 3.7 或更高版本
- 操作系统：Windows 10/11, macOS, Linux
- 内存：建议 4GB 以上
- 存储空间：至少 2GB 可用空间

### 安装步骤

1. **克隆项目**

```bash
git clone <repository-url>
cd visual-based-virtual-fitness-coach
```

2. **创建虚拟环境（推荐）**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **验证安装**

```bash
python -c "import cv2, mediapipe, flask; print('安装成功！')"
```

### 依赖说明

主要依赖包及其用途：

- `opencv-python`：视频读取和图像处理
- `mediapipe`：人体姿态估计
- `flask`：Web 服务器
- `flask-cors`：跨域请求支持
- `numpy`：数值计算
- `scipy`：科学计算库
- `fastdtw`：动态时间规整算法
- `scikit-learn`：机器学习工具
- `joblib`：模型序列化
- `tensorflow`：深度学习框架（MediaPipe 依赖）

## 🚀 快速开始

### Web 应用模式

1. **启动服务器**

```bash
python app.py
```

2. **访问应用**

打开浏览器访问：`http://localhost:5000`

3. **使用步骤**

   - **上传视频模式**：
     1. 点击"上传视频"标签
     2. 选择或拖拽视频文件（支持 MP4, AVI, MOV, MKV, WEBM）
     3. 等待处理完成
     4. 查看评估结果

   - **摄像头模式**：
     1. 点击"摄像头拍摄"标签
     2. 点击"启动摄像头"授权摄像头访问
     3. 点击"开始录制"开始录制动作
     4. 完成动作后点击"停止录制"
     5. 系统自动处理并显示结果

### 命令行模式

```bash
# 基本用法
python main_trajectory.py --video video/qishi1.mp4

# 保存选中的关键帧
python main_trajectory.py --video video/qishi1.mp4 --output result

# 不使用智能选帧（均匀采样）
python main_trajectory.py --video video/qishi1.mp4 --no-smart
```

## 📖 使用说明

### Web 应用详细使用

#### 上传视频模式

1. **准备视频**
   - 视频格式：MP4, AVI, MOV, MKV, WEBM
   - 视频大小：建议不超过 500MB
   - 视频内容：包含完整的太极拳起势动作
   - 拍摄建议：
     - 确保人物完整出现在画面中
     - 光线充足，背景简洁
     - 人物与背景对比明显
     - 动作清晰，无遮挡

2. **上传和处理**
   - 点击上传区域或拖拽文件
   - 系统自动开始处理
   - 实时显示处理进度：
     - 读取视频文件
     - 选择关键帧
     - 提取关键点
     - 计算轨迹相似度
     - 生成评估报告

3. **查看结果**
   - **整体评分**：0-100 分的综合评分
   - **准确率**：动作准确度百分比
   - **各部位相似度**：各身体部位的详细评分
   - **改进建议**：针对性的动作改进建议

#### 摄像头实时模式

1. **启动摄像头**
   - 点击"启动摄像头"按钮
   - 浏览器会请求摄像头权限，请允许
   - 确认画面正常显示

2. **录制动作**
   - 点击"开始录制"开始录制
   - 完成完整的太极拳起势动作
   - 点击"停止录制"结束录制
   - 系统自动处理录制的视频

3. **查看结果**
   - 处理完成后自动显示评估结果
   - 结果格式与上传视频模式相同

### 命令行模式详细使用

#### 基本命令

```bash
python main_trajectory.py --video <视频路径>
```

#### 参数说明

- `--video`（必需）：输入视频文件路径
- `--output`（可选）：输出选中帧的保存路径前缀
- `--no-smart`（可选）：不使用智能选帧，改为均匀采样

#### 输出示例

```
智能选帧: [5, 23, 45, 67]

============================================================
整体评分: 85.32 (良好)
[##################################################--] 85.3%

各部位相似度:
  [OK] left_shoulder      [####################] 92.5
  [OK] right_shoulder     [####################] 91.2
  [OK] left_elbow         [##################--] 88.7
  [OK] right_elbow        [##################--] 87.3
  [OK] left_wrist         [####################] 90.1
  [OK] right_wrist        [####################] 89.5
  [OK] left_hip           [####################] 93.2
  [OK] right_hip          [####################] 92.8
  [OK] left_knee          [##################--] 86.4
  [OK] right_knee         [##################--] 85.9
  [OK] left_ankle         [####################] 91.7

改进建议:
  - 注意保持躯干直立，避免过度前倾
  - 手臂下落时保持肘部角度稳定
============================================================
```

## 📁 项目结构

```
visual-based-virtual-fitness-coach/
│
├── app.py                          # Web 应用主程序
├── main_trajectory.py              # 命令行工具
├── requirements.txt                # Python 依赖包列表
├── LICENSE                         # 许可证文件
│
├── trajectory_system/              # 核心轨迹评估系统
│   ├── __init__.py
│   ├── smart_frame_selector.py    # 智能关键帧选择器
│   ├── trajectory_evaluator.py    # 轨迹评估器
│   ├── trajectory_matcher.py      # 轨迹匹配算法
│   ├── standard_trajectory.pkl    # 标准动作轨迹数据
│   └── trajectory_model.pkl        # 轨迹模型（可选）
│
├── templates/                      # HTML 模板
│   └── index.html                 # 主页面
│
├── static/                         # 静态资源
│   ├── style.css                  # 样式表
│   └── script.js                  # 前端脚本
│
├── uploads/                        # 上传文件临时目录（自动创建）
│
└── video/                          # 示例视频目录
    ├── qishi1.mp4
    ├── qishi2.mp4
    └── ...
```

### 核心模块说明

#### `trajectory_system/smart_frame_selector.py`

智能关键帧选择器，负责从视频中选择最具代表性的4个关键帧。

**主要功能**：
- 提取视频帧中的人体关键点
- 计算每帧与标准动作的相似度
- 使用动态规划算法选择最优帧序列

**关键方法**：
- `extract_keypoints(frame)`: 提取单帧关键点
- `select_best_frames(video_frames, min_gap=5)`: 选择最佳帧序列

#### `trajectory_system/trajectory_evaluator.py`

轨迹评估器，负责评估用户动作质量。

**主要功能**：
- 从视频帧中提取关键点
- 调用轨迹匹配器计算相似度
- 生成评估报告和改进建议

**关键方法**：
- `extract_keypoints_from_frame(frame)`: 提取关键点
- `evaluate_video_frames(frames)`: 评估视频帧序列

#### `trajectory_system/trajectory_matcher.py`

轨迹匹配器，负责计算用户动作与标准动作的相似度。

**主要功能**：
- 轨迹标准化处理
- 使用 FastDTW 算法进行轨迹匹配
- 计算各身体部位的相似度
- 生成改进建议

**关键方法**：
- `normalize_trajectory(trajectory)`: 轨迹标准化
- `calculate_similarity(user_frames)`: 计算相似度
- `get_detailed_advice(similarities)`: 获取详细建议

## 🔬 核心算法

### 1. 智能关键帧选择算法

系统使用多阶段优化搜索算法选择关键帧：

1. **关键点提取**：对视频中每一帧提取11个关键身体关节点
2. **相似度计算**：计算每帧与4个标准动作帧的相似度
3. **动态规划搜索**：使用动态规划找到总相似度最高的4帧序列
4. **约束条件**：确保相邻选中帧之间有最小间隔（默认5帧）

**算法复杂度**：O(n × m)，其中 n 是视频帧数，m 是标准帧数（4）

### 2. 轨迹相似度匹配算法

使用 FastDTW（Fast Dynamic Time Warping）算法进行轨迹匹配：

1. **轨迹提取**：从4个关键帧中提取各身体部位的轨迹
2. **轨迹标准化**：
   - 中心化：将轨迹中心移至原点
   - 尺度归一化：缩放到单位空间
3. **DTW 匹配**：使用动态时间规整对齐不同长度的轨迹
4. **相似度计算**：基于对齐后的轨迹计算欧几里得距离

**优势**：
- 处理不同速度的动作
- 对时间偏移不敏感
- 计算效率高

### 3. 评分系统

综合评分 = 各身体部位相似度的加权平均

- **整体评分**：0-100 分
- **部位评分**：每个身体部位独立评分
- **等级划分**：
  - 90-100：优秀
  - 80-89：良好
  - 70-79：中等
  - <70：需改进

## 📡 API 文档

### Web API 端点

#### 1. 上传视频

**端点**：`POST /api/upload`

**请求**：
- Content-Type: `multipart/form-data`
- 参数：`video` (文件)

**响应**：
```json
{
  "task_id": "uuid-string"
}
```

#### 2. 查询进度

**端点**：`GET /api/progress/<task_id>`

**响应**：
```json
{
  "progress": 50,
  "message": "正在处理...",
  "result": {
    "score": 85.32,
    "accuracy": 0.8532,
    "similarities": {...},
    "advice": [...]
  }
}
```

#### 3. 处理图像帧

**端点**：`POST /api/process_frames`

**请求**：
```json
{
  "frames": ["data:image/jpeg;base64,...", ...]
}
```

**响应**：
```json
{
  "task_id": "uuid-string"
}
```

### Python API

#### TrajectoryEvaluator

```python
from trajectory_system.trajectory_evaluator import TrajectoryEvaluator

evaluator = TrajectoryEvaluator()
result = evaluator.evaluate_video_frames(frames)

# result 包含：
# - score: 整体评分
# - accuracy: 准确率
# - similarities: 各部位相似度
# - advice: 改进建议
```

#### SmartFrameSelector

```python
from trajectory_system.smart_frame_selector import SmartFrameSelector

selector = SmartFrameSelector()
indices = selector.select_best_frames(frames, min_gap=5)
```

## ❓ 常见问题

### 安装问题

**Q: 安装 MediaPipe 失败？**

A: MediaPipe 需要特定的系统依赖。请参考 [MediaPipe 官方文档](https://google.github.io/mediapipe/getting_started/install.html)。

**Q: 提示 TensorFlow 相关错误？**

A: 确保安装了正确版本的 TensorFlow。如果不需要 GPU 支持，可以安装 CPU 版本：
```bash
pip install tensorflow-cpu
```

### 使用问题

**Q: 视频上传后处理失败？**

A: 检查以下几点：
- 视频格式是否支持（MP4, AVI, MOV, MKV, WEBM）
- 视频中是否包含完整的人体
- 光线是否充足
- 视频大小是否超过 500MB

**Q: 检测不到人体关键点？**

A: 可能原因：
- 人物在画面中不完整
- 背景与人物对比度低
- 光线不足
- 人物被遮挡

**Q: 评分总是很低？**

A: 可能原因：
- 动作不完整或不标准
- 视频质量差
- 关键帧选择不当
- 建议检查视频质量和动作完整性

**Q: 摄像头无法启动？**

A: 检查：
- 浏览器是否允许摄像头权限
- 摄像头是否被其他程序占用
- 浏览器是否支持 WebRTC（Chrome、Firefox、Edge 都支持）

### 性能问题

**Q: 处理速度慢？**

A: 优化建议：
- 使用较小的视频文件
- 确保有足够的系统内存
- 关闭其他占用资源的程序
- 考虑使用 GPU 加速（需要配置 CUDA）

**Q: 内存占用高？**

A: 系统会：
- 自动清理临时文件
- 限制同时处理的视频数量
- 处理完成后释放内存

## 🔧 高级配置

### 修改服务器端口

编辑 `app.py`：

```python
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)  # 修改端口
```

### 调整关键帧选择参数

在 `app.py` 中修改：

```python
indices = selector.select_best_frames(frames, min_gap=10)  # 增加最小间隔
```

### 自定义标准动作

替换 `trajectory_system/standard_trajectory.pkl` 文件，包含4个标准动作帧的关键点数据。

## 📝 开发说明

### 添加新的动作类型

1. 准备标准动作数据（4个关键帧）
2. 更新 `standard_trajectory.pkl`
3. 调整评估阈值和建议规则

### 扩展评估功能

1. 修改 `trajectory_matcher.py` 中的相似度计算逻辑
2. 更新 `get_detailed_advice` 方法添加新的建议规则
3. 调整评分权重

## 📄 许可证

查看 [LICENSE](LICENSE) 文件了解详细信息。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**注意**：本系统专注于太极拳起势动作评估，如需评估其他动作，需要准备相应的标准动作数据。


# 智能健身指导系统 (Visual-based Virtual Fitness Coach)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)

智能健身指导系统利用计算机视觉和深度学习技术，通过摄像头捕捉健身者的动作，实时提供健身动作的识别和指导反馈，帮助用户以更科学、安全的方式进行个人健身训练。

## 📋 目录

- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
- [使用方法](#使用方法)
  - [实时摄像头模式](#实时摄像头模式)
  - [本地视频模式](#本地视频模式)
  - [高级选项](#高级选项)
- [核心功能说明](#核心功能说明)
  - [帧选择与评分策略](#帧选择与评分策略)
  - [模型预测](#模型预测)
- [配置与调优](#配置与调优)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)
- [许可证](#许可证)
- [致谢](#致谢)

## ✨ 功能特性

- 🎥 **实时动作捕捉**：支持摄像头实时视频流和本地视频文件输入
- 🤖 **智能动作识别**：基于 MediaPipe 的人体姿态估计，提取关键动作特征
- 📊 **动作质量评估**：使用机器学习模型对动作质量进行量化评分
- 💡 **即时反馈系统**：基于规则和 AI 的双重反馈机制，提供实时改进建议
- 🎯 **精准帧选择**：智能选择关键帧进行分析，避免冗余计算
- 🔄 **异步 AI 处理**：支持异步调用外部 AI 服务进行深度分析
- 📈 **多维度评估**：综合考虑可见度、手臂高度、躯干角度、手间距等多个维度

## 🛠 技术栈

- **计算机视觉**：OpenCV, MediaPipe
- **深度学习**：TensorFlow/Keras (MLP 模型)
- **数据处理**：NumPy, Pillow
- **异步处理**：Python asyncio
- **语言**：Python 3.7+

## 📁 项目结构

```
visual-based-virtual-fitness-coach/
├── data/                          # 数据与模型文件
│   └── models/                    # 预训练模型
│       └── haarcascade_fullbody.xml
├── src/                           # 源代码
│   ├── action_recognition.py      # 动作识别：生成每帧的 metrics（太极相关特征）
│   ├── detection.py               # 人体检测
│   ├── pose_estimation_v1.py      # 姿态估计（旧版本）
│   ├── pose_estimation_v2.py      # 姿态估计（当前版本，已去掉脸部关键点绘制）
│   ├── feedback.py                # 基于规则的即时反馈
│   ├── ai_worker.py               # 异步 AI/外部服务调用
│   ├── ai_feedback.py             # AI 反馈处理
│   └── capture.py                 # 视频捕获工具
├── taichi_ai/                     # 太极相关的数据生成、训练与预测脚本
│   ├── generate_data.py           # 生成训练数据
│   ├── train_model.py             # 训练 MLP 模型
│   ├── predict.py                 # 模型预测接口
│   ├── X.npy                      # 训练特征数据
│   ├── y.npy                      # 训练标签数据
│   └── errors.json                # 错误记录
├── utils/                         # 工具函数
│   └── utils.py                   # 通用工具（如中文字体绘制）
├── video/                         # 示例视频文件
├── main.py                        # 主入口：支持摄像头与本地视频，后处理调用模型预测
├── requirements.txt               # 项目依赖
├── LICENSE                        # MIT 许可证
└── README.md                      # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- Python 3.7 或更高版本
- 摄像头（实时模式）或视频文件（视频模式）
- Windows/Linux/macOS

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Annsher-whhr/visual-based-virtual-fitness-coach
   cd visual-based-virtual-fitness-coach
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **（可选）准备模型文件**
   - 默认模型路径：`taichi_ai/taichi_mlp.h5`
   - 如果模型文件不存在，需要先训练模型（见[开发指南](#开发指南)）

## 📖 使用方法

### 实时摄像头模式

使用默认摄像头进行实时动作捕捉和分析：

```powershell
python main.py
```

**操作说明**：
- 按 `q` 键退出程序
- 系统会实时显示姿态估计结果和反馈信息

### 本地视频模式

使用本地视频文件进行分析，视频播放完毕后会自动进行模型预测：

```powershell
python main.py --video "D:\path\to\your_video.mp4"
```

**特点**：
- 视频播放过程中进行实时分析
- 视频结束后自动选择 4 帧关键帧进行模型预测
- 输出动作质量评分和改进建议

### 高级选项

#### 指定自定义模型

```powershell
python main.py --video "video.mp4" --model "D:\models\taichi_mlp.h5"
```

#### 完整参数说明

```bash
python main.py [-v VIDEO] [-m MODEL]

参数说明：
  -v, --video   视频文件路径（若不提供则使用摄像头）
  -m, --model   训练好的模型文件路径（覆盖默认 taichi_ai/taichi_mlp.h5）
```

## 🔍 核心功能说明

### 帧选择与评分策略

为了让预测输入与训练集中的标准四帧（`QS1.jpg` ~ `QS4.jpg`）保持阶段匹配，系统采用**阶段划分式选帧**：

#### 阶段划分式选帧 (`select_frames_by_phase`)

1. **起始帧**：选取第一帧可用的 metrics 作为起势开始 (`QS1`)
2. **腿部峰值**：在起始之后、结束之前，查找“脚距/臀宽比最大且头部位于双脚中点上方、躯干接近竖直”的帧，代表双腿展开到位 (`QS2`)
3. **手臂峰值**：自腿部峰值往后，选取手臂高度比达到峰值的帧，代表双臂抬起到位 (`QS3`)
4. **收势帧**：保留最后一帧可用的 metrics 作为动作结束 (`QS4`)

如果某阶段检测不到有效帧，会自动回退到邻近的高质量帧或使用末帧补齐，保证始终输出 4 帧供模型预测。

#### 帧质量评分 (`candidate_score`)

`candidate_score` 仍然用于辅助回退逻辑与调试，采用 0 ~ 10 的连续评分衡量单帧质量，评分项包括：

| 评估项 | 目标值 | 权重 | 说明 |
|--------|--------|------|------|
| 可见度 (avg_visibility) | 越高越好 | 2.0 | 保证关键点检测的可靠性 |
| 手臂高度比 (arm_height_ratio) | ~0.6 | 3.0 | 评估手臂抬升高度 |
| 躯干角度 (torso_angle) | 越小越好 | 2.0 | 评估身体是否保持中正 |
| 手间距/肩宽比 | ~0.75 | 1.5 | 评估双臂展开幅度 |
| 脚距/臀宽比 | ~1.0 | 1.0 | 评估双脚站位稳定性 |
| 肘部角度 | ~160° | 0.5 | 评估手臂伸展程度 |

选帧过程会优先满足阶段要求，若某阶段没有可靠帧，`candidate_score` 会帮助挑选最佳备选帧。

### 模型预测

- **默认模型路径**：`taichi_ai/taichi_mlp.h5`
- **输入特征**：从选定的 4 帧中提取的 metrics 特征
- **输出结果**：动作质量评分和改进建议

**重要提示**：确保 `taichi_ai/predict.py` 使用与训练数据一致的特征顺序。若使用 `dict.values()` 式的无序构建，可能会导致特征错位。

## ⚙️ 配置与调优

### 调参建议

如果阶段判定与实际动作轨迹存在偏差，可以尝试以下调优：

1. **调整腿部峰值判定**  
   - 修改 `main.py` 中 `leg_open_score` 的计算方式，例如叠加膝盖角度或采用移动平均  
   - 调整判定时的“峰值平台”阈值（当前为 `max_leg * 0.97`）

2. **调整手臂峰值判定**  
   - 修改 `arm_raise_score`，例如同时考虑 `hand_distance` 或肩角度  
   - 调整峰值搜索区间，限制最小/最大间隔

3. **微调候选评分**  
   - 在 `candidate_score` 中改变各项权重或阈值，用于控制回退时的优先级  
   - 可结合调试输出观察得分变化

### 参数位置

当前所有参数都在 `main.py` 文件中：
- `leg_open_score` / `arm_raise_score`：控制腿部/手臂峰值判定
- `candidate_score`：帧质量评分及权重
- `select_frames_by_phase`：阶段匹配及回退逻辑

如需将这些参数暴露为命令行参数，可以在 `main.py` 中扩展 `argparse` 配置。

### 调试模式

在 `main.py` 视频处理结束处添加调试输出：

```python
print("阶段性选帧结果：")
for phase_name, idx in phases.items():
    if idx is None:
        continue
    print(
        f"{phase_name:>7} -> idx={idx}, "
        f"腿部开度={leg_open_score(metrics_list[idx]):.3f}, "
        f"手臂高度={arm_raise_score(metrics_list[idx]):.3f}, "
        f"质量={candidate_score(metrics_list[idx]):.3f}"
    )
print(f"最终帧索引: {indices}")
```

### 可视化选定的帧

可以将选定的 4 帧保存为图片以便人工复查：

```python
import os
output_dir = "output/selected_frames"
os.makedirs(output_dir, exist_ok=True)
for i, idx in enumerate(indices):
    cv2.imwrite(f"{output_dir}/frame_{i+1}_idx_{idx}.jpg", frames_for_display[idx])
```

## 👨‍💻 开发指南

### 训练模型

1. **生成训练数据**
   ```bash
   python taichi_ai/generate_data.py
   ```
   这会生成 `X.npy` 和 `y.npy` 文件。

2. **训练模型**
   ```bash
   python taichi_ai/train_model.py
   ```
   训练完成后会生成 `taichi_ai/taichi_mlp.h5` 模型文件。

### 扩展新的动作类型

1. 在 `src/action_recognition.py` 中添加新的特征提取逻辑
2. 更新 `taichi_ai/generate_data.py` 以支持新动作的数据生成
3. 重新训练模型并更新预测接口

### 代码规范

- 使用类型提示（Type Hints）
- 遵循 PEP 8 代码风格
- 添加必要的文档字符串

## ❓ 常见问题

### Q: 视频播放完毕后没有进行预测？

A: 检查以下几点：
- 确保视频文件存在且可读
- 检查模型文件路径是否正确
- 查看控制台是否有错误信息
- 确保视频帧数 >= 4

### Q: 预测结果置信度过高？

A: 参考[配置与调优](#配置与调优)章节，调整 `seg_thresh` 和评分权重。

### Q: 如何提高检测准确性？

A: 
- 确保光照充足
- 人体在画面中完整可见
- 背景尽可能简洁
- 距离摄像头保持适当距离（2-3米）

### Q: 支持哪些视频格式？

A: OpenCV 支持的所有格式，常见的有：`.mp4`, `.avi`, `.mov`, `.mkv` 等。

### Q: 如何自定义反馈规则？

A: 修改 `src/feedback.py` 中的 `provide_feedback` 函数，添加或修改判断逻辑。

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 贡献流程

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 报告问题

在 [GitHub Issues](https://github.com/Annsher-whhr/visual-based-virtual-fitness-coach/issues) 中提交问题，请包含：
- 问题描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（Python 版本、操作系统等）

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目的支持：

- [MediaPipe](https://mediapipe.dev/) - 人体姿态估计
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [TensorFlow](https://www.tensorflow.org/) - 深度学习框架
- [NumPy](https://numpy.org/) - 数值计算库

---

**注意**：本项目目前主要针对太极起势动作进行优化，其他动作类型的支持正在开发中。

如有问题或建议，欢迎提交 Issue 或 Pull Request！
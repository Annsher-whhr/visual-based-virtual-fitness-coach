# 智能健身指导系统

智能健身指导系统利用计算机视觉和深度学习技术，通过摄像头捕捉健身者的动作，实时提供健身动作的识别和指导反馈，帮助用户以更科学、安全的方式进行个人健身训练。

## 特性

# 智能健身指导系统 (Visual-based Virtual Fitness Coach)

本项目通过摄像头或本地视频，结合人体检测、姿态估计与简单的规则/机器学习模型，实时为健身动作（当前聚焦：太极起势等）提供质量判定与改进建议。

README 重点更新：本仓库现在支持把视频作为输入（`--video`），并在视频结束后从整段动作中选取 4 帧特征做模型评估。选帧策略与评分逻辑已改为更鲁棒的多段抽样与连续加权评分，以匹配训练时使用的“分散的关键帧”样式（见“帧选择与评分”一节）。

目录一览（核心）

```
visual-based-virtual-fitness-coach/
├── data/                  # 数据与模型（可选的本地存放）
├── src/                   # 源代码
│   ├── action_recognition.py  # 生成每帧的 metrics（太极相关特征）
│   ├── detection.py           # 人体检测
│   ├── pose_estimation_v2.py  # 姿态估计（已去掉脸部关键点绘制）
│   ├── feedback.py            # 基于 rules 的即时反馈
│   └── ai_worker.py           # 异步 AI/外部服务调用（可选）
├── taichi_ai/              # 太极相关的数据生成、训练与预测脚本
├── utils/                  # 工具函数
├── requirements.txt
└── main.py                 # 入口：支持摄像头与本地视频，后处理调用模型预测
```

快速开始
```
git clone https://github.com/Annsher-whhr/visual-based-virtual-fitness-coach
cd visual-based-virtual-fitness-coach
pip install -r requirements.txt
```

运行

- 使用摄像头（实时）：

```powershell
python .\main.py
```

- 使用本地视频并在结束后运行模型预测（示例）：

```powershell
python .\main.py --video "D:\path\to\your_video.mp4"
```

- 可选：指定模型文件（覆盖默认路径 taichi_ai/taichi_mlp.h5）：

```powershell
python .\main.py --video "video.mp4" --model "D:\models\taichi_mlp.h5"
```

帧选择与评分（重要）

为避免将高度相似的连续帧（例如同一完美姿态的 4 帧）作为模型输入并产生过高置信，我们实现了新的选帧策略：

- candidate_score(m)：将原来的离散评分替换为连续的加权评分（约 0..10），评分项包括：
	- avg_visibility（可见度，权重 2）
	- arm_height_ratio（理想值 ~0.6，权重 3）
	- torso_angle（越小越好，权重 2）
	- hand_distance / shoulder_width（理想 ~0.75，权重 1.5）
	- foot_distance / hip_width（理想 ~1.0，权重 1.0）
	- left/right elbow angles（理想 ~160°，权重 0.5）

- select_frames_for_prediction(metrics_list)：
	1. 识别所有得分 >= seg_thresh 的连续段（seg_thresh 为阈值，代码默认 2，建议与新分数量级对齐，可设置为 4）；
	2. 按段质量（长度与平均分）排序，选取最多 4 段；
	3. 从每段选取一个“代表帧”（段内 candidate_score 最高）；
	4. 若段数不足 4，则从剩余高分且分散的帧中补齐；若仍不足则均匀采样补齐。

该策略更贴合训练时“从若干高质量且分散的动作段中各取一帧”的做法（参见仓库中 `taichi_ai/generate_data.py` 的数据生成方式）。

调参建议

- 若模型仍然偏向过高置信：
	- 增大 `seg_thresh`（使得仅非常“好”的帧被视为段的一部分）；
	- 增大 segment 选择优先级（更偏重长度或更高的 avg_score）；
	- 调整 candidate_score 各项权重（例如降低 arm_height 的权重或提高 torso_angle 的惩罚）。

- 当前参数都写在 `main.py` 内的 `select_frames_for_prediction` 与 `candidate_score` 中；如需我把这些参数暴露为命令行参数（例如 `--seg-thresh`），我可以继续实现并添加示例。

模型与预测

- 默认模型路径：`taichi_ai/taichi_mlp.h5`。
- 你可以通过 `--model <path>` 指定自定义模型文件，程序会将其路径传递给预测器。
- 强烈建议确认 `taichi_ai/predict.py` 使用与训练数据一致的特征顺序（仓库中已有说明）；若使用 `dict.values()` 式的无序构建，可能会导致训练/预测时特征错位，从而产生不可靠的评分。若需要，我可以帮你把 `predict.py` 强制为固定特征顺序并加入输入长度校验。

调试与可视化建议

- 开启 debug 模式（临时在代码中打印）：在 `main.py` 视频处理结束处打印候选段（start/end/avg_score/length）和最终选帧索引及每帧 score，便于观察与调参。
- 可将 `frames_for_display` 中选定的 4 帧保存为图片以便人工复查：示例代码可在选帧后把对应帧写入 `output/selected_frames/`。

开发者说明

- 若要训练或重训练模型，请查看 `taichi_ai/` 下的 `generate_data.py`（合成或整理训练样本）、`train_model.py`（训练 MLP 并保存为 h5）。

贡献与许可

欢迎提交 issue 与 PR。本项目采用 MIT 许可证（详见 LICENSE）。

致谢

感谢 MediaPipe、OpenCV、TensorFlow 等开源项目。 

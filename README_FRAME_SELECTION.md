# 智能关键帧选择功能 - 使用说明

## 🆕 新功能：智能关键帧选择

针对太极拳起势动作评估，新增了基于**特征相似度匹配**的智能选帧算法，能够自动从视频中选择与训练数据最匹配的4个关键帧。

---

## 🎯 为什么需要这个功能？

### 问题背景

你的模型是基于**4个标准关键帧**训练的，这4帧代表起势动作的不同阶段：

| 阶段 | 特征描述 | arm_height_ratio | foot_distance |
|------|----------|------------------|---------------|
| 第1帧 | 起始姿态：手臂高举，双脚并拢 | 0.984 | 0.077 |
| 第2帧 | 腿部打开初期：手臂仍高，双脚开始分开 | 0.932 | 0.187 |
| 第3帧 | 手臂下落期：手臂下落并弯曲肘部 | -0.036 | 0.252 |
| 第4帧 | 完成姿态：手臂中等，双脚最大距离，膝盖弯曲 | 0.625 | 0.285 |

### 核心挑战

如何从任意视频中自动选择出与这4个标准帧**最匹配**的帧序列？

---

## ✅ 解决方案

### 智能选帧算法

使用**基于特征相似度的动态规划算法**：

```
视频帧 → 特征提取 → 相似度计算 → 动态规划优化 → 最优4帧
```

**核心优势**：
- ✅ 数据驱动，自动匹配标准帧
- ✅ 考虑不同阶段的关键特征权重
- ✅ 保证帧的时间顺序和最小间隔
- ✅ 提供详细的相似度分数
- ✅ 支持可视化分析

---

## 🚀 快速开始

### 方式1：直接运行（推荐）

```bash
python main.py -v video/qishi2.mp4
```

**程序会自动**：
1. 提取视频中每一帧的特征
2. 计算与标准帧的相似度
3. 使用智能算法选择最匹配的4帧
4. 输入模型进行动作质量评估
5. 给出分数和改进建议

**输出示例**：
```
============================================================
智能选帧结果（基于特征相似度匹配）
============================================================
总相似度分数: 345.67
平均相似度: 86.42%

各帧详情:
  标准帧1 (起始姿态): 视频帧索引 5
    相似度: 92.34%
    arm_height_ratio: 0.975 (标准: 0.984)
    foot_distance: 0.082 (标准: 0.077)
  ...

============================================================
🎯 模型预测结果
============================================================
动作质量分数: 85.30%
反馈建议: 动作正确，请保持。
============================================================
```

### 方式2：详细分析（用于调试）

```bash
python test_frame_selector.py -v video/qishi2.mp4
```

**会生成**：
- 📊 详细的选帧报告
- 📈 各帧特征对比分析
- 🎨 相似度热力图（`frame_selection_heatmap.png`）

### 方式3：查看快速入门

```bash
python quick_start_example.py
```

显示详细的使用说明和示例。

---

## 📊 工作原理

### 1. 特征相似度计算

对于视频的每一帧，计算其与每个标准帧的相似度：

```python
相似度分数 = 100 × exp(-加权特征距离)
```

**不同阶段的关键特征权重**：

- **第1帧（起始）**：强调 `arm_height_ratio`(权重5.0) 和 `foot_distance`(4.0)
- **第2帧（腿部打开）**：强调 `foot_distance`(5.0) 和 `arm_height_ratio`(4.0)
- **第3帧（手臂下落）**：强调 `arm_height_ratio`(5.0) 和肘部角度(4.0)
- **第4帧（完成）**：强调 `foot_distance`(5.0) 和 `knee_bend_average`(5.0)

### 2. 动态规划优化

使用动态规划找到总相似度最高的4帧序列：

```
dp[i][k] = 在前i帧中选择k个标准帧的最大相似度
```

**约束条件**：
- 相邻选中帧之间必须有最小间隔（默认3帧）
- 帧的时间顺序必须递增

### 3. 智能回退策略

如果动态规划失败（视频太短或质量差），自动使用均匀分布策略。

---

## 📁 新增文件说明

| 文件 | 功能 |
|------|------|
| **frame_selector.py** | 核心选帧模块，包含相似度计算和动态规划算法 |
| **test_frame_selector.py** | 独立测试脚本，用于详细分析和调试 |
| **quick_start_example.py** | 快速入门示例，显示详细使用说明 |
| **FRAME_SELECTION_GUIDE.md** | 完整的使用指南和技术文档 |
| **SOLUTION_SUMMARY.md** | 解决方案总结 |
| **README_FRAME_SELECTION.md** | 本文档 |

**修改的文件**：
- `main.py`：集成了新的智能选帧算法

---

## ⚙️ 参数调整

### 调整最小帧间隔

```bash
# 视频动作较快时，增大间隔
python test_frame_selector.py -v video/qishi2.mp4 --min-gap 5

# 视频动作较慢时，减小间隔
python test_frame_selector.py -v video/qishi2.mp4 --min-gap 2
```

**建议**：
- 视频较快：5-8
- 正常速度：3（默认）
- 视频较慢：1-2

### 调整特征权重

编辑 `frame_selector.py` 中的 `FEATURE_WEIGHTS` 字典：

```python
FEATURE_WEIGHTS = {
    0: {  # 第1帧的权重
        'arm_height_ratio': 5.0,  # 增大此值会更强调手臂高度
        'foot_distance': 4.0,     # 增大此值会更强调脚距
        'hand_distance': 2.0,
        'torso_angle': 2.0,
        'avg_visibility': 1.0,
        'knee_bend_average': 1.5,
    },
    # ... 其他3帧
}
```

---

## 📈 相似度热力图

运行测试脚本会自动生成相似度热力图：

```bash
python test_frame_selector.py -v video/qishi2.mp4
```

生成的 `frame_selection_heatmap.png` 显示：
- **横轴**：视频帧索引
- **纵轴**：4个标准帧
- **颜色**：相似度分数（越红越相似）
- **蓝色星号**：算法选中的帧

---

## 🔍 效果验证

### 1. 查看相似度分数

**正常范围**：60-95%

- **> 85%**：优秀，高度匹配
- **70-85%**：良好，基本匹配
- **60-70%**：一般，有一定差异
- **< 60%**：较差，视频动作与标准动作差异较大

### 2. 查看特征对比

测试脚本会输出详细的特征对比：

```
【标准帧 1 ←→ 视频帧 5】
  相似度分数: 92.34%
  arm_height_ratio     : 视频=  0.975  标准=  0.984  差异= 0.009 ( 0.9%)
  foot_distance        : 视频=  0.082  标准=  0.077  差异= 0.005 ( 6.5%)
  ...
```

### 3. 可视化热力图

查看生成的热力图，验证：
- 选中的帧是否在高相似度区域
- 是否存在更好的帧被忽略
- 时间顺序是否合理

---

## 💡 使用技巧

### 技巧1：针对特定视频优化参数

```bash
# 先运行测试脚本，查看默认效果
python test_frame_selector.py -v your_video.mp4

# 根据输出调整参数
python test_frame_selector.py -v your_video.mp4 --min-gap 5

# 满意后运行完整评估
python main.py -v your_video.mp4
```

### 技巧2：批量处理多个视频

```python
import os
from frame_selector import select_frames_by_similarity
# ... 提取metrics_list的代码 ...

video_dir = "video/"
for video_file in os.listdir(video_dir):
    if video_file.endswith('.mp4'):
        # 提取metrics
        metrics_list = extract_metrics(os.path.join(video_dir, video_file))
        
        # 选帧
        indices, info = select_frames_by_similarity(metrics_list)
        
        print(f"{video_file}: 平均相似度 {info['avg_similarity']:.2f}%")
```

### 技巧3：保存选帧结果

```python
import pickle

# 选帧
indices, info = select_frames_by_similarity(metrics_list)

# 保存
result = {
    'video': 'qishi2.mp4',
    'indices': indices,
    'similarity_scores': info['individual_scores'],
    'selected_frames': [metrics_list[i] for i in indices]
}
with open('frame_selection_result.pkl', 'wb') as f:
    pickle.dump(result, f)
```

---

## 🐛 常见问题排查

### 问题1：相似度分数很低（< 60%）

**可能原因**：
- 视频中的动作与标准起势差异较大
- 人体检测质量不佳
- 光线、遮挡等因素影响

**解决方法**：
1. 检查 `avg_visibility` 是否过低
2. 确保人体完整出现在画面中
3. 查看特征对比，找出差异最大的特征
4. 考虑重新录制标准视频

### 问题2：选中的帧不符合预期

**可能原因**：
- `min_frame_gap` 设置不合适
- 特征权重需要调整
- 视频动作速度异常

**解决方法**：
1. 查看相似度热力图
2. 调整 `min_frame_gap` 参数
3. 根据实际情况调整特征权重

### 问题3：程序运行很慢

**可能原因**：
- 视频太长或分辨率太高
- 电脑性能不足

**解决方法**：
1. 使用跳帧处理：`metrics_list[::2]`（每隔2帧）
2. 降低视频分辨率
3. 缓存中间结果避免重复计算

---

## 📚 详细文档

- **完整技术文档**：`FRAME_SELECTION_GUIDE.md`
- **解决方案总结**：`SOLUTION_SUMMARY.md`
- **快速入门示例**：运行 `python quick_start_example.py`

---

## 🎯 总结

这个智能选帧功能通过以下方式解决了关键帧选择问题：

✅ **准确性**：基于特征相似度，数据驱动的匹配  
✅ **可靠性**：动态规划算法保证最优解  
✅ **灵活性**：支持参数调整和特征权重定制  
✅ **可视化**：提供详细的分数和热力图  
✅ **易用性**：无缝集成到现有系统  

**立即试用**：
```bash
python main.py -v video/qishi2.mp4
```

---

**🎉 享受更准确的太极拳动作评估吧！**


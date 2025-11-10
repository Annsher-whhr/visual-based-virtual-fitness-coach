# 太极拳起势动作关键帧选择 - 解决方案总结

## 🎯 问题

你的模型基于**4个标准关键帧**训练，需要从视频中选择出与这些标准帧最匹配的4帧来进行评估。

**标准4帧的特征**：
- **第1帧**：手臂高举(0.984)，脚距很近(0.077)，头部居中(-0.001) - **起始姿态**
- **第2帧**：手臂仍高(0.932)，脚距已打开(0.187)，骨盆/肩部回正，**头部居中在两脚正上方(-0.002)** - **重心转移期**  
- **第3帧**：手臂下落(-0.036)，肘部弯曲(34°/50°)，脚距继续增大(0.252) - **手臂下落期**
- **第4帧**：手臂中等(0.625)，脚距最大(0.285)，膝盖弯曲(22°) - **完成姿态**

为避免算法把“迈步未落地”或“重心仍偏向一侧”的画面误判为第二帧，我们在第2阶段加入了**硬阈值筛选**：
- `foot_distance / hip_width ≥ 1.25`：保证脚已经稳定落地并打开。
- `arm_height_ratio ≥ 0.55`：手臂仍保持高举状态。
- `nose_center_offset / shoulder_width ≤ 0.45`：头部回到两脚中点正上方。
- `|hip_balance| ≤ 0.28`、`|shoulder_balance| ≤ 0.35`：骨盆和肩部已回正。
- `|torso_angle| ≤ 10°`：躯干保持中正，不再侧倾。

## ✅ 解决方案

开发了**基于特征相似度匹配的智能选帧算法**：

### 核心算法
1. **特征相似度计算**：计算视频每帧与标准帧的加权特征距离
2. **动态规划优化**：找到总相似度最高的4帧序列
3. **智能回退策略**：处理边界情况

### 关键特性
✅ 针对不同阶段使用不同的特征权重  
✅ 保证选中帧的时间顺序和最小间隔  
✅ 提供详细的相似度分数和可视化  
✅ 无缝集成到现有系统  

---

## 📁 新增文件

| 文件 | 说明 |
|------|------|
| `frame_selector.py` | 核心选帧模块 |
| `test_frame_selector.py` | 独立测试脚本 |
| `FRAME_SELECTION_GUIDE.md` | 详细使用文档 |
| `quick_start_example.py` | 快速入门示例 |
| `SOLUTION_SUMMARY.md` | 本文档 |

**修改文件**：`main.py` - 集成了新的选帧算法

---

## 🚀 快速开始

### 1️⃣ 最简单方式 - 直接运行

```bash
python main.py -v video/qishi2.mp4
```

程序会自动：
- ✅ 提取视频特征
- ✅ 智能选择4帧
- ✅ 模型评估打分
- ✅ 给出改进建议

### 2️⃣ 详细分析 - 测试脚本

```bash
python test_frame_selector.py -v video/qishi2.mp4
```

生成内容：
- 📊 详细选帧报告
- 📈 特征对比分析
- 🎨 相似度热力图

### 3️⃣ 查看使用说明

```bash
python quick_start_example.py
```

---

## 📊 输出示例

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

  标准帧2 (腿部打开初期): 视频帧索引 15
    相似度: 88.50%
    arm_height_ratio: 0.920 (标准: 0.932)
    foot_distance: 0.195 (标准: 0.187)

  ...

============================================================

🎯 模型预测结果
============================================================
动作质量分数: 85.30%
反馈建议: 动作正确，请保持。
============================================================
```

---

## ⚙️ 参数调整

### 最小帧间隔

```python
select_frames_by_similarity(metrics_list, min_frame_gap=3)
```

- 视频较快：增大到 5-8
- 视频较慢：减小到 1-2

### 特征权重

编辑 `frame_selector.py` 中的 `FEATURE_WEIGHTS` 字典：

```python
FEATURE_WEIGHTS = {
    0: {  # 第1帧
        'arm_height_ratio': 5.0,  # 增大=更强调该特征
        'foot_distance': 4.0,
        ...
    },
    ...
}
```

---

## 🎓 算法优势

| 特性 | 原方法 | 新方法 |
|------|--------|--------|
| 匹配准确度 | 中等 | **高** ✓ |
| 适应性 | 依赖规则 | **数据驱动** ✓ |
| 可解释性 | 较强 | **强**（分数+可视化）✓ |
| 容错性 | 弱 | **强**（回退策略）✓ |

---

## 🔍 常见问题

### Q: 相似度分数很低怎么办？

**正常范围**：60-95%  
**如果 < 60%**：视频动作与标准动作差异较大

**解决方法**：
1. 检查视频中的动作是否是标准起势
2. 确保人体完整出现在画面中
3. 查看特征对比，找出差异最大的特征
4. 调整特征权重或最小帧间隔

### Q: 如何验证选帧效果？

1. ✅ 查看相似度热力图
2. ✅ 对比输出的特征值
3. ✅ 查看模型预测准确率
4. ✅ 使用测试脚本详细分析

### Q: 能否用于其他动作？

**可以！** 只需修改 `STANDARD_FRAMES` 和 `FEATURE_WEIGHTS`

---

## 📈 性能优化建议

### 针对长视频 - 跳帧处理

```python
step = 2  # 每隔2帧提取一次
metrics_list_sparse = metrics_list[::step]
indices = select_frames_by_similarity(metrics_list_sparse)[0]
# 转换回原始索引
indices = [i * step for i in indices]
```

### 缓存中间结果

```python
import pickle

# 保存
with open('metrics_cache.pkl', 'wb') as f:
    pickle.dump(metrics_list, f)

# 加载
with open('metrics_cache.pkl', 'rb') as f:
    metrics_list = pickle.load(f)
```

---

## 📚 详细文档

- **完整使用指南**：`FRAME_SELECTION_GUIDE.md`
- **快速入门示例**：`quick_start_example.py`
- **测试脚本帮助**：`python test_frame_selector.py -h`

---

## 🎯 工作流程图

```
视频输入 (qishi2.mp4)
    ↓
提取每一帧的特征 (metrics_list)
    ↓
计算与标准帧的相似度 (similarity_matrix)
    ↓
动态规划选择最优4帧序列 (indices)
    ↓
输入模型进行评估 (predict_quality)
    ↓
输出质量分数和建议 (result)
```

---

## ✨ 核心代码示例

```python
# 从视频选择最匹配的4帧并评估
from frame_selector import select_frames_by_similarity
from taichi_ai.predict import predict_quality

# 1. 提取视频特征（已在main.py中实现）
# metrics_list = [...]

# 2. 智能选帧
indices, info = select_frames_by_similarity(
    metrics_list, 
    min_frame_gap=3,
    verbose=True
)

# 3. 获取选中的帧
selected_frames = [metrics_list[i] for i in indices]

# 4. 模型评估
result = predict_quality(selected_frames)

# 5. 输出结果
print(f"动作质量: {result['score']:.1%}")
print(f"建议: {result['advice']}")
```

---

## 📞 下一步

1. **运行测试**：
   ```bash
   python test_frame_selector.py -v video/qishi2.mp4
   ```

2. **查看热力图**：
   ```bash
   # 会自动生成 frame_selection_heatmap.png
   ```

3. **运行完整系统**：
   ```bash
   python main.py -v video/qishi2.mp4
   ```

4. **调整参数**（如需要）：
   - 修改 `min_frame_gap`
   - 调整 `FEATURE_WEIGHTS`

---

## 🎉 总结

这个解决方案通过**智能特征匹配**，解决了从视频中选择与训练数据最匹配的关键帧的问题。它：

✅ 提高了帧选择的准确性  
✅ 提供了详细的相似度分析  
✅ 支持参数调优和可视化  
✅ 无缝集成到现有系统  
✅ 具有良好的容错性和扩展性  

**立即试用**: `python main.py -v video/qishi2.mp4` 🚀


# 智能关键帧选择器 - 使用指南

## 📋 问题描述

你的太极拳起势动作评估模型是基于**4个标准关键帧**训练的，这4帧代表了起势动作的不同阶段：

1. **第1帧（起始姿态）**：手臂高举，双脚并拢，头部居中
2. **第2帧（重心转移期）**：手臂仍高，双脚已打开，**头部移到两脚中点正上方**
3. **第3帧（手臂下落期）**：手臂下落并弯曲肘部，双脚继续分开
4. **第4帧（完成姿态）**：手臂中等高度，双脚最大距离，膝盖弯曲

**核心挑战**：如何从任意视频中自动选择出与这4个标准帧**最匹配**的帧序列？

---

## 🎯 解决方案

我开发了一个**基于特征相似度匹配的智能选帧算法**，它能够：

✅ 计算视频每一帧与标准帧之间的特征相似度  
✅ 使用动态规划找到最优的4帧序列  
✅ 考虑各阶段的关键特征权重  
✅ 保证选中的帧时间顺序合理（有最小间隔约束）  
✅ 提供详细的相似度分数和可视化

---

## 🔧 核心算法原理

### 1. 特征相似度计算

对于视频中的每一帧，算法会计算它与每个标准帧的相似度分数（0-100分）：

```python
similarity_score = 100 × exp(-weighted_distance)
```

其中 `weighted_distance` 是加权特征距离，不同阶段关注不同特征：

| 阶段 | 关键特征（权重） |
|------|------------------|
| 第1帧（起始） | `arm_height_ratio`(5.0), `foot_distance`(4.0), `nose_center_offset`(3.5), `torso_angle`(3.0), `hip_balance`(2.0) |
| 第2帧（重心转移） | `nose_center_offset`(7.0), `arm_height_ratio`(5.5), `foot_distance`(5.5), `hip_balance`(4.0), `shoulder_balance`(3.0), `torso_angle`(3.5) |
| 第3帧（手臂下落） | `arm_height_ratio`(5.0), `left/right_elbow_angle`(4.0) |
| 第4帧（完成） | `foot_distance`(5.0), `knee_bend_average`(5.0) |

为确保算法不会把“迈步尚未落地”的画面误识别为第二帧，我们还在特征匹配前增加了**阶段性硬阈值**：

- 第2帧只接受 `foot_distance / hip_width ≥ 1.25`，确保脚已经打开并落地。
- 第2帧要求 `arm_height_ratio ≥ 0.55`（手臂仍然高举）。
- 第2帧要求 `nose_center_offset / shoulder_width ≤ 0.45`（头部回到两脚中点正上方）。
- 第2帧限制 `hip_balance` ≤ 0.28、`shoulder_balance` ≤ 0.35、`|torso_angle|` ≤ 10°，避免身体仍在侧移或倾斜。

### 2. 动态规划选帧

使用动态规划算法找到总相似度最高的4帧序列：

```
dp[i][k] = 在前i帧中选择k个标准帧的最大相似度总和
```

约束条件：
- 相邻选中帧之间必须有最小间隔（默认3帧）
- 帧的时间顺序必须递增

### 3. 回退策略

如果动态规划失败（例如视频太短或质量太差），系统会自动使用均匀分布策略：
- 将视频分成4段
- 每段选择与对应标准帧相似度最高的帧

---

## 📁 文件说明

### 新增文件

1. **`frame_selector.py`** - 核心选帧模块
   - `select_frames_by_similarity()` - 主选帧函数
   - `calculate_feature_similarity()` - 计算相似度
   - `visualize_similarity_heatmap()` - 生成可视化热力图

2. **`test_frame_selector.py`** - 测试脚本
   - 独立测试选帧算法
   - 显示详细的特征对比
   - 生成相似度热力图

3. **`FRAME_SELECTION_GUIDE.md`** - 本文档

### 修改文件

- **`main.py`** - 集成了新的选帧算法
  - 原有的 `select_frames_by_phase()` 保留作为后备
  - 优先使用 `select_frames_by_similarity()`

---

## 🚀 使用方法

### 方法1：通过主程序运行（推荐）

```bash
# 直接运行主程序，会自动使用智能选帧
python main.py -v video/qishi2.mp4

# 输出示例：
# ============================================================
# 智能选帧结果（基于特征相似度匹配）
# ============================================================
# 总相似度分数: 345.67
# 平均相似度: 86.42%
# 
# 各帧详情:
#   标准帧1 (起始姿态): 视频帧索引 5
#     相似度: 92.34%
#     arm_height_ratio: 0.975 (标准: 0.984)
#     foot_distance: 0.082 (标准: 0.077)
#   ...
```

### 方法2：使用测试脚本（用于调试）

```bash
# 测试选帧算法并生成详细报告
python test_frame_selector.py -v video/qishi2.mp4

# 自定义最小帧间隔
python test_frame_selector.py -v video/qishi2.mp4 --min-gap 5

# 不生成可视化图表（加快速度）
python test_frame_selector.py -v video/qishi2.mp4 --no-viz
```

### 方法3：在代码中调用

```python
from frame_selector import select_frames_by_similarity

# 假设你已经有了metrics_list（从视频提取的特征列表）
indices, info = select_frames_by_similarity(
    metrics_list,
    min_frame_gap=3,  # 相邻帧最小间隔
    verbose=True      # 打印详细信息
)

# 获取选中的4帧
selected_frames = [metrics_list[i] for i in indices]

# 查看相似度分数
print(f"平均相似度: {info['avg_similarity']:.2f}%")
print(f"各帧分数: {info['individual_scores']}")
```

---

## 📊 输出说明

### 1. 控制台输出

运行后会显示：

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

  标准帧3 (手臂下落期): 视频帧索引 28
    相似度: 81.23%
    arm_height_ratio: -0.028 (标准: -0.036)
    foot_distance: 0.240 (标准: 0.252)
    left_elbow_angle: 38.5° (标准: 33.9°)

  标准帧4 (完成姿态): 视频帧索引 42
    相似度: 83.60%
    arm_height_ratio: 0.635 (标准: 0.625)
    foot_distance: 0.278 (标准: 0.285)
    knee_bend_average: 20.5° (标准: 22.0°)

============================================================
```

### 2. 可视化热力图

如果安装了 `matplotlib`，会自动生成 `frame_selection_heatmap.png`：

- **横轴**：视频帧索引
- **纵轴**：4个标准帧
- **颜色**：相似度分数（越红越相似）
- **蓝色星号**：选中的帧

这个热力图可以帮助你：
- 直观看到哪些视频帧与标准帧最匹配
- 验证选帧结果是否合理
- 调试和优化算法参数

---

## ⚙️ 参数调优

### 最小帧间隔 (`min_frame_gap`)

```python
select_frames_by_similarity(metrics_list, min_frame_gap=3)
```

- **默认值**: 3帧
- **作用**: 避免选中过于接近的帧
- **调整建议**:
  - 视频较快：增大到5-8
  - 视频较慢：减小到1-2
  - 视频很短：设为1

### 特征权重 (`FEATURE_WEIGHTS`)

在 `frame_selector.py` 中修改：

```python
FEATURE_WEIGHTS = {
    0: {  # 第1帧
        'arm_height_ratio': 5.0,  # 增大此值会更强调手臂高度
        'foot_distance': 4.0,     # 增大此值会更强调脚距
        ...
    },
    ...
}
```

根据你的实际数据调整权重：
- 某特征很重要但总是匹配不好 → 增大权重
- 某特征不重要 → 减小权重或删除

---

## 🔍 常见问题

### Q1: 为什么选出的帧与预期不符？

**可能原因**:
1. 视频动作与标准动作差异较大
2. 人体检测质量不佳（光线、遮挡等）
3. 动作速度过快或过慢

**解决方法**:
1. 检查 `avg_visibility` 是否过低（应 > 0.3）
2. 调整 `min_frame_gap` 参数
3. 查看相似度热力图，找出问题帧
4. 考虑调整特征权重

### Q2: 相似度分数很低怎么办？

**正常范围**: 60-95%  
**如果 < 60%**: 说明视频动作与标准动作差异较大

**解决方法**:
1. 检查视频中的动作是否是标准起势
2. 确保人体完整出现在画面中
3. 查看控制台输出的各帧特征对比，找出差异最大的特征

### Q3: 能否用于其他太极拳动作？

**可以！** 只需修改 `frame_selector.py` 中的：
1. `STANDARD_FRAMES` - 改为新动作的标准帧特征
2. `FEATURE_WEIGHTS` - 根据新动作的特点调整权重

### Q4: 如何验证选帧效果？

1. 查看相似度热力图
2. 对比输出的特征值与标准值
3. 将选中的帧可视化显示
4. 查看模型预测的准确率

---

## 🎓 算法优势

相比于原有的 `select_frames_by_phase()` 方法：

| 特性 | 原方法（阶段式） | 新方法（相似度匹配） |
|------|------------------|----------------------|
| 匹配准确度 | 中等 | **高** |
| 适应性 | 依赖启发式规则 | **数据驱动** |
| 可解释性 | 较强 | **强**（有相似度分数） |
| 容错性 | 弱 | **强**（有回退策略） |
| 可视化 | 无 | **有**（热力图） |
| 可调优 | 难 | **易**（权重调整） |

---

## 📈 性能优化建议

### 1. 跳帧处理（针对长视频）

```python
# 每隔N帧提取一次特征，加快处理速度
step = 2  # 跳帧步长
metrics_list_sparse = metrics_list[::step]
indices_sparse, _ = select_frames_by_similarity(metrics_list_sparse)
# 转换回原始索引
indices = [i * step for i in indices_sparse]
```

### 2. 并行处理

```python
from multiprocessing import Pool

def process_frame(frame):
    # 提取单帧特征
    return recognize_action(estimate_pose(detect_human(frame)))

with Pool() as pool:
    metrics_list = pool.map(process_frame, video_frames)
```

### 3. 缓存中间结果

```python
import pickle

# 保存提取的特征
with open('metrics_cache.pkl', 'wb') as f:
    pickle.dump(metrics_list, f)

# 下次直接加载
with open('metrics_cache.pkl', 'rb') as f:
    metrics_list = pickle.load(f)
```

---

## 🎯 下一步工作

1. **收集更多标准动作样本**
   - 不同人、不同角度的标准起势动作
   - 更新 `STANDARD_FRAMES` 为多组标准的平均值

2. **在线调优**
   - 根据模型预测结果自动调整特征权重
   - 使用强化学习优化选帧策略

3. **多动作支持**
   - 扩展到其他太极拳动作（单鞭、白鹤亮翅等）
   - 自动识别动作类型并选择对应的标准帧

4. **实时选帧**
   - 在视频播放过程中实时计算相似度
   - 提示用户何时达到标准姿态

---

## 📞 技术支持

如有问题或建议，请：
1. 检查本文档的"常见问题"部分
2. 运行 `test_frame_selector.py` 生成详细报告
3. 查看相似度热力图分析问题
4. 调整算法参数并重新测试

---

## 📝 更新日志

### v1.0 (2024)
- ✅ 实现基于特征相似度的智能选帧
- ✅ 支持动态规划优化
- ✅ 添加可视化热力图
- ✅ 集成到主程序
- ✅ 提供独立测试脚本

---

**祝你的太极拳动作评估系统运行顺利！** 🥋


# 项目文件结构说明

## 📁 目录结构

```
visual-based-virtual-fitness-coach/
├── archive/                    # 旧版本代码归档
│   └── v1/                     # v1版本代码（已废弃）
│       ├── frame_selector.py
│       ├── generate_data.py
│       ├── train_model.py
│       ├── predict.py
│       ├── test_model.py
│       ├── taichi_mlp.h5
│       └── pose_estimation_v1.py
│
├── data/                       # 数据文件
│   ├── models/                 # 训练好的模型
│   │   ├── taichi_mlp_v2.h5   # v2模型（当前使用）
│   │   └── haarcascade_fullbody.xml
│   ├── standard/               # 标准数据
│   │   ├── qishi3_standard_frames.json    # 20帧标准帧数据
│   │   └── qishi3_all_features.json       # 完整视频特征
│   └── training/               # 训练数据
│       ├── X.npy              # 训练特征（1600样本×440维）
│       ├── y.npy              # 训练标签
│       ├── errors.json        # 错误类型记录
│       └── scaler.pkl         # 数据标准化器
│
├── docs/                       # 文档
│   ├── README.md              # 主文档
│   ├── README_v2.md           # v2版本说明
│   ├── 系统改进总结_v2.md      # 系统改进文档
│   └── ...                    # 其他文档
│
├── reports/                    # 报告和图表
│   ├── model_evaluation_report_v2.png    # 模型评估报告
│   ├── system_comparison_v1_v2.png       # 系统对比图
│   └── ...
│
├── scripts/                    # 工具脚本
│   └── generate_comparison_chart.py      # 生成对比图表
│
├── tests/                      # 测试文件
│   ├── test_model_v2.py        # 模型测试
│   ├── test_improved_system.py # 系统测试
│   └── ...
│
├── examples/                   # 示例代码
│   ├── demo_v2_system.py       # v2系统演示
│   └── quick_start_example.py  # 快速开始示例
│
├── taichi_ai/                  # 核心AI模块（v2版本）
│   ├── __init__.py
│   ├── generate_data_v2.py    # 数据生成器
│   ├── train_model_v2.py      # 模型训练
│   └── predict_v2.py          # 预测模块
│
├── src/                        # 源代码
│   ├── action_recognition.py   # 动作识别
│   ├── pose_estimation_v2.py  # 姿态估计（v2）
│   ├── detection.py            # 人体检测
│   └── ...
│
├── video/                      # 视频文件
│   ├── qishi1.mp4
│   ├── qishi2.mp4
│   └── qishi3.mp4              # 标准视频
│
├── frame_selector_v2.py        # 关键帧选择器（v2）
├── extract_standard_features.py # 标准特征提取
├── evaluate_taichi.py          # 评估主程序
├── main.py                     # 主入口
└── requirements.txt            # 依赖包
```

## 🔄 主要工作流程

### 1. 数据准备
```bash
# 提取标准视频特征
python extract_standard_features.py
# 输出: data/standard/qishi3_standard_frames.json

# 生成训练数据
python taichi_ai/generate_data_v2.py
# 输出: data/training/X.npy, y.npy, errors.json
```

### 2. 模型训练
```bash
python taichi_ai/train_model_v2.py
# 输出: data/models/taichi_mlp_v2.h5, data/training/scaler.pkl
```

### 3. 评估视频
```bash
python evaluate_taichi.py -v video/qishi2.mp4
```

## 📝 重要文件说明

### 核心模块
- `taichi_ai/` - v2版本的核心AI模块
- `frame_selector_v2.py` - 智能关键帧选择
- `extract_standard_features.py` - 标准数据提取
- `evaluate_taichi.py` - 评估主程序

### 数据文件
- `data/models/taichi_mlp_v2.h5` - 训练好的模型（20帧输入，440维特征）
- `data/training/` - 训练数据和标准化器
- `data/standard/` - 标准动作数据

### 文档
- `docs/README_v2.md` - v2版本使用说明
- `docs/系统改进总结_v2.md` - 改进详情

## ⚠️ 注意事项

1. **路径更新**: 所有代码已更新为使用新的文件夹结构
2. **向后兼容**: 代码会优先查找新路径，如果不存在会尝试旧路径
3. **数据迁移**: 旧数据已移动到 `data/` 目录下
4. **旧版本**: v1版本代码已归档到 `archive/v1/`

## 🚀 快速开始

1. 确保数据文件存在：
   - `data/standard/qishi3_standard_frames.json`
   - `data/models/taichi_mlp_v2.h5`
   - `data/training/scaler.pkl`

2. 运行评估：
   ```bash
   python evaluate_taichi.py -v video/qishi2.mp4
   ```


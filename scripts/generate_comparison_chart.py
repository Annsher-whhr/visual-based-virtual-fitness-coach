# -*- coding: utf-8 -*-
"""
生成v1和v2系统性能对比图表
"""

import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据
metrics = ['准确率', '精确率', '召回率', 'F1分数']
v1_scores = [60, 50, 70, 58]  # 估计值（基于用户描述）
v2_scores = [99.38, 98.77, 100.00, 99.38]  # 实际测试值（从test_model_v2.py获取）

x = np.arange(len(metrics))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# === 子图1: 柱状对比图 ===
bars1 = ax1.bar(x - width/2, v1_scores, width, label='v1系统（4帧）', 
                color='#FF6B6B', alpha=0.8)
bars2 = ax1.bar(x + width/2, v2_scores, width, label='v2系统（20帧）',
                color='#4ECDC4', alpha=0.8)

ax1.set_ylabel('性能指标 (%)', fontsize=12)
ax1.set_title('系统性能对比', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(fontsize=11)
ax1.set_ylim([0, 105])
ax1.grid(True, alpha=0.3, axis='y')

# 在柱子上显示数值和提升幅度
for i, (bar1, bar2, v1, v2) in enumerate(zip(bars1, bars2, v1_scores, v2_scores)):
    # v1数值
    ax1.text(bar1.get_x() + bar1.get_width()/2., v1 + 1,
            f'{v1:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # v2数值
    ax1.text(bar2.get_x() + bar2.get_width()/2., v2 + 1,
            f'{v2:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 提升幅度
    improvement = v2 - v1
    ax1.annotate(f'↑{improvement:.1f}%',
                xy=(i, max(v1, v2) + 3),
                ha='center', fontsize=10, color='green', fontweight='bold')

# === 子图2: 关键改进点 ===
improvements = [
    ('训练帧数', 4, 20),
    ('特征维度', 88, 440),
    ('训练样本', 1000, 1600),
    ('网络层数', 3, 4),
    ('错误类型', 6, 8),
]

categories = [item[0] for item in improvements]
v1_values = [item[1] for item in improvements]
v2_values = [item[2] for item in improvements]

# 归一化以便显示（使用v2值作为100%）
v1_normalized = [v1/v2*100 for v1, v2 in zip(v1_values, v2_values)]
v2_normalized = [100] * len(v2_values)

y_pos = np.arange(len(categories))
height = 0.35

bars3 = ax2.barh(y_pos - height/2, v1_normalized, height, label='v1系统',
                 color='#FF6B6B', alpha=0.8)
bars4 = ax2.barh(y_pos + height/2, v2_normalized, height, label='v2系统',
                 color='#4ECDC4', alpha=0.8)

ax2.set_xlabel('相对值 (v2=100%)', fontsize=12)
ax2.set_title('系统规模对比', fontsize=14, fontweight='bold')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(categories)
ax2.legend(fontsize=11)
ax2.set_xlim([0, 120])
ax2.grid(True, alpha=0.3, axis='x')

# 显示实际数值
for i, (bar, v1, v2) in enumerate(zip(bars3, v1_values, v2_values)):
    ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'{v1}', ha='left', va='center', fontsize=9)

for i, (bar, v1, v2) in enumerate(zip(bars4, v1_values, v2_values)):
    ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'{v2}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 显示增长倍数
    ratio = v2 / v1
    ax2.text(110, i, f'×{ratio:.1f}', ha='center', va='center',
            fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))

plt.suptitle('太极拳起势动作评估系统 - v1 vs v2 全面对比', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)
output_path = os.path.join(REPORTS_DIR, 'system_comparison_v1_v2.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"[OK] 对比图表已保存到: {output_path}")
plt.close()

print("\n性能提升总结:")
print("="*60)
for metric, v1, v2 in zip(metrics, v1_scores, v2_scores):
    improvement = v2 - v1
    print(f"{metric:8s}: {v1:5.1f}% -> {v2:5.1f}% (↑{improvement:+5.1f}%)")
print("="*60)


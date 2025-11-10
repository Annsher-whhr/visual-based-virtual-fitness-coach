import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# 配置中文字体 - 使用系统字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# === 加载数据和模型 ===
# 第13-20行，修改为：
# === 加载数据和模型 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载数据（在 taichi_ai 文件夹）
X = np.load(os.path.join(BASE_DIR, 'taichi_ai', 'X.npy'))
y = np.load(os.path.join(BASE_DIR, 'taichi_ai', 'y.npy'))

# 加载模型（在项目根目录）
model = keras.models.load_model(os.path.join(BASE_DIR, 'taichi_mlp.h5'))

# === 划分数据集 ===
# 使用相同的随机种子以确保可重复性
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("📊 数据集信息")
print("=" * 60)
print(f"总样本数: {len(X)}")
print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")
print(f"特征维度: {X.shape[1]}")
print(f"正样本: {np.sum(y == 1)} | 负样本: {np.sum(y == 0)}")
print()

# === 模型预测 ===
y_train_pred_prob = model.predict(X_train, verbose=0).flatten()
y_test_pred_prob = model.predict(X_test, verbose=0).flatten()

# 使用0.5作为阈值
y_train_pred = (y_train_pred_prob >= 0.5).astype(int)
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

# === 评估指标计算 ===
print("=" * 60)
print("📈 模型性能评估")
print("=" * 60)

# 训练集性能
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

print("【训练集表现】")
print(f"  准确率 (Accuracy):  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  精确率 (Precision): {train_precision:.4f}")
print(f"  召回率 (Recall):    {train_recall:.4f}")
print(f"  F1分数 (F1-Score):  {train_f1:.4f}")
print()

# 测试集性能
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("【测试集表现】")
print(f"  准确率 (Accuracy):  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  精确率 (Precision): {test_precision:.4f}")
print(f"  召回率 (Recall):    {test_recall:.4f}")
print(f"  F1分数 (F1-Score):  {test_f1:.4f}")
print()

# === 过拟合检查 ===
print("【过拟合检查】")
overfit_gap = train_accuracy - test_accuracy
if overfit_gap > 0.1:
    print(f"  ⚠️  可能存在过拟合！训练集准确率比测试集高 {overfit_gap*100:.2f}%")
elif overfit_gap > 0.05:
    print(f"  ⚡ 轻微过拟合，训练集准确率比测试集高 {overfit_gap*100:.2f}%")
else:
    print(f"  ✅ 模型泛化良好，训练集和测试集准确率差异仅 {overfit_gap*100:.2f}%")
print()

# === 混淆矩阵 ===
print("=" * 60)
print("🔍 混淆矩阵（测试集）")
print("=" * 60)
cm = confusion_matrix(y_test, y_test_pred)
print("           预测: 错误  预测: 正确")
print(f"实际: 错误    {cm[0][0]:>4}      {cm[0][1]:>4}    (TN, FP)")
print(f"实际: 正确    {cm[1][0]:>4}      {cm[1][1]:>4}    (FN, TP)")
print()
print(f"  真负例 (TN): {cm[0][0]} - 正确识别错误动作")
print(f"  假正例 (FP): {cm[0][1]} - 错误识别为正确动作")
print(f"  假负例 (FN): {cm[1][0]} - 正确动作被识别为错误")
print(f"  真正例 (TP): {cm[1][1]} - 正确识别正确动作")
print()

# === 详细分类报告 ===
print("=" * 60)
print("📋 详细分类报告（测试集）")
print("=" * 60)
print(classification_report(y_test, y_test_pred, 
                          target_names=['错误动作', '正确动作']))

# === ROC曲线和AUC ===
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)
roc_auc = auc(fpr, tpr)
print("=" * 60)
print(f"📊 AUC值（测试集）: {roc_auc:.4f}")
print("=" * 60)
if roc_auc >= 0.9:
    print("  ✅ 优秀！模型判别能力很强")
elif roc_auc >= 0.8:
    print("  ✅ 良好，模型表现不错")
elif roc_auc >= 0.7:
    print("  ⚡ 一般，还有提升空间")
else:
    print("  ⚠️  较差，建议重新训练或调整模型")
print()

# === 预测概率分布分析 ===
print("=" * 60)
print("📊 预测概率分布分析（测试集）")
print("=" * 60)
correct_probs = y_test_pred_prob[y_test == 1]
wrong_probs = y_test_pred_prob[y_test == 0]
print(f"正确动作的平均预测概率: {np.mean(correct_probs):.4f} ± {np.std(correct_probs):.4f}")
print(f"错误动作的平均预测概率: {np.mean(wrong_probs):.4f} ± {np.std(wrong_probs):.4f}")
print()

# === 可视化 ===
print("正在生成可视化图表...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 混淆矩阵热图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('混淆矩阵 (Confusion Matrix)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('实际标签')
axes[0, 0].set_xlabel('预测标签')
axes[0, 0].set_xticklabels(['错误', '正确'])
axes[0, 0].set_yticklabels(['错误', '正确'])

# 2. ROC曲线
axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate')
axes[0, 1].set_ylabel('True Positive Rate')
axes[0, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(alpha=0.3)

# 3. 预测概率分布
axes[1, 0].hist(correct_probs, bins=20, alpha=0.6, label='正确动作', color='green')
axes[1, 0].hist(wrong_probs, bins=20, alpha=0.6, label='错误动作', color='red')
axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='阈值=0.5')
axes[1, 0].set_xlabel('预测概率')
axes[1, 0].set_ylabel('样本数量')
axes[1, 0].set_title('预测概率分布', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. 性能指标对比
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
train_metrics = [train_accuracy, train_precision, train_recall, train_f1]
test_metrics = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics_names))
width = 0.35

axes[1, 1].bar(x - width/2, train_metrics, width, label='训练集', color='skyblue')
axes[1, 1].bar(x + width/2, test_metrics, width, label='测试集', color='orange')
axes[1, 1].set_ylabel('分数')
axes[1, 1].set_title('性能指标对比', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics_names, rotation=15)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')
axes[1, 1].set_ylim([0, 1.1])

# 在柱状图上添加数值
for i, (train_val, test_val) in enumerate(zip(train_metrics, test_metrics)):
    axes[1, 1].text(i - width/2, train_val + 0.02, f'{train_val:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    axes[1, 1].text(i + width/2, test_val + 0.02, f'{test_val:.3f}', 
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('model_evaluation_report.png', dpi=300, bbox_inches='tight')
print("✅ 可视化图表已保存为: model_evaluation_report.png")
print()

# === 总结建议 ===
print("=" * 60)
print("💡 模型评估总结与建议")
print("=" * 60)
if test_accuracy >= 0.95:
    print("✅ 模型表现优秀！可以放心使用。")
elif test_accuracy >= 0.85:
    print("✅ 模型表现良好，但仍有优化空间。")
else:
    print("⚠️  模型表现一般，建议考虑以下改进：")

if test_precision < 0.85:
    print("  - 精确率较低：考虑增加更多正样本或调整分类阈值")
if test_recall < 0.85:
    print("  - 召回率较低：可能需要更多训练数据或调整模型结构")
if overfit_gap > 0.1:
    print("  - 存在过拟合：考虑增加Dropout、减少模型复杂度或增加训练数据")
if roc_auc < 0.85:
    print("  - AUC值偏低：考虑特征工程或使用更复杂的模型")

print()
print("=" * 60)
print("测试完成！")
print("=" * 60)
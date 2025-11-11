# -*- coding: utf-8 -*-
"""
改进版模型训练 - 支持多帧输入
基于12帧标准数据训练深度神经网络
"""

import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

layers = tf.keras.layers

# 加载数据集
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(BASE_DIR, "X.npy")
Y_PATH = os.path.join(BASE_DIR, "y.npy")

if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    raise FileNotFoundError(
        f"未找到训练数据文件！\n"
        f"请先运行: python taichi_ai/generate_data_v2.py\n"
        f"预期文件:\n  {X_PATH}\n  {Y_PATH}\n"
    )

print("正在加载训练数据...")
X = np.load(X_PATH)
y = np.load(Y_PATH)

print(f"[OK] 数据加载完成")
print(f"     样本数: {len(X)}")
print(f"     特征维度: {X.shape[1]}")
print(f"     正样本: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"     负样本: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)\n")

# 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本\n")

# === 构建改进的神经网络模型 ===
print("构建神经网络模型...")

model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    
    # 第一层：特征提取
    layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    # 第二层：特征组合
    layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    # 第三层：深度特征
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    # 输出层
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()]
)

print(f"[OK] 模型构建完成\n")
model.summary()

# === 训练模型 ===
print(f"\n{'='*60}")
print(f"开始训练...")
print(f"{'='*60}\n")

# 添加早停和学习率调整
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=80,
    validation_split=0.15,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# === 评估模型 ===
print(f"\n{'='*60}")
print(f"模型评估")
print(f"{'='*60}\n")

# 在测试集上评估
test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test, verbose=0)

print(f"测试集性能:")
print(f"  准确率 (Accuracy): {test_acc*100:.2f}%")
print(f"  精确率 (Precision): {test_prec*100:.2f}%")
print(f"  召回率 (Recall): {test_rec*100:.2f}%")
print(f"  F1分数: {2*test_prec*test_rec/(test_prec+test_rec)*100:.2f}%\n")

# 详细分类报告
y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=['错误动作', '正确动作']))

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  真负例: {cm[0,0]}, 假正例: {cm[0,1]}")
print(f"  假负例: {cm[1,0]}, 真正例: {cm[1,1]}\n")

# === 保存模型 ===
model_path = os.path.join(os.path.dirname(BASE_DIR), "taichi_mlp_v2.h5")
model.save(model_path)
print(f"[OK] 模型已保存到: {model_path}\n")

# 同时保存标准化器
import joblib
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"[OK] 标准化器已保存到: {scaler_path}\n")

# === 生成训练报告图表 ===
print("生成训练报告...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 训练历史 - 损失
axes[0, 0].plot(history.history['loss'], label='训练损失', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='验证损失', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('损失')
axes[0, 0].set_title('训练与验证损失')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 训练历史 - 准确率
axes[0, 1].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('准确率')
axes[0, 1].set_title('训练与验证准确率')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 混淆矩阵
im = axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['错误动作', '正确动作'])
axes[1, 0].set_yticklabels(['错误动作', '正确动作'])
axes[1, 0].set_xlabel('预测标签')
axes[1, 0].set_ylabel('真实标签')
axes[1, 0].set_title('混淆矩阵')

# 在每个格子中显示数值
for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm[i, j],
                              ha="center", va="center", color="black", fontsize=16)
plt.colorbar(im, ax=axes[1, 0])

# 4. 性能指标对比
metrics_names = ['准确率', '精确率', '召回率', 'F1分数']
metrics_values = [
    test_acc * 100,
    test_prec * 100,
    test_rec * 100,
    2 * test_prec * test_rec / (test_prec + test_rec) * 100
]

bars = axes[1, 1].bar(metrics_names, metrics_values, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
axes[1, 1].set_ylabel('百分比 (%)')
axes[1, 1].set_title('模型性能指标')
axes[1, 1].set_ylim([0, 100])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# 在柱子上显示数值
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle(f'太极拳起势动作评估模型训练报告 (12帧输入)', fontsize=16, y=0.995)
plt.tight_layout()

report_path = os.path.join(os.path.dirname(BASE_DIR), "model_evaluation_report_v2.png")
plt.savefig(report_path, dpi=150, bbox_inches='tight')
print(f"[OK] 训练报告已保存到: {report_path}\n")
plt.close()

print(f"{'='*60}")
print(f"训练完成！")
print(f"{'='*60}\n")


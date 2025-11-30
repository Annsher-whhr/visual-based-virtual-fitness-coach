import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model():
    
    """
训练轨迹动作质量评估模型的函数

功能：
    - 从预先生成的pickle文件加载训练和测试数据
    - 使用随机森林回归算法训练模型，评估动作质量
    - 计算模型性能指标(MSE和R2)
    - 保存训练好的模型供后续使用

参数：
    无

返回值：
    无

实现步骤：
    1. 加载存储在'trajectory_system/training_data.pkl'中的数据
    2. 提取训练集和测试集的特征(X)和标签(y)
    3. 初始化随机森林回归模型，设置100个决策树，最大深度20，使用所有可用CPU核心
    4. 在训练集上拟合模型
    5. 在测试集上进行预测
    6. 计算均方误差(MSE)和决定系数(R2)评估模型性能
    7. 输出模型评估指标
    8. 将训练好的模型保存到'trajectory_system/trajectory_model.pkl'
"""

    with open('trajectory_system/training_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    
    joblib.dump(model, 'trajectory_system/trajectory_model.pkl')
    print("模型已保存")

if __name__ == '__main__':
    train_model()


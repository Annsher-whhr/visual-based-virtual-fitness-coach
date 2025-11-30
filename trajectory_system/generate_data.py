import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def generate_synthetic_data(standard_frames, num_samples=500):

    """
    基于标准动作轨迹生成合成训练数据，用于机器学习模型训练
    
    参数:
        standard_frames (list): 标准动作帧数据，包含多个帧，每个帧是字典格式，
            键为身体部位名称，值为包含坐标的列表
        num_samples (int): 要生成的合成样本数量，默认为500
    
    返回:
        tuple: 包含两个numpy数组的元组
            - X (numpy.ndarray): 特征数据，形状为(num_samples, 特征维度)，
              每个样本是所有帧所有身体部位坐标的一维展开
            - y (numpy.ndarray): 标签数据，形状为(num_samples,)，
              对应每个样本的动作质量评分(0.6-1.0)
    
    处理流程:
        1. 初始化特征列表X和标签列表y
        2. 获取身体部位列表（从第一帧中提取）
        3. 循环生成指定数量的样本：
           a. 随机生成噪声水平(0-0.3之间的均匀分布)
           b. 初始化当前样本的特征数据列表
           c. 对每个标准帧：
              - 初始化该帧的数据列表
              - 对每个身体部位：
                 - 提取2D坐标
                 - 生成高斯噪声并根据噪声水平缩放
                 - 将噪声添加到坐标上，模拟不同质量的动作
                 - 将带噪声的坐标添加到帧数据列表
              - 将帧数据合并到样本特征中
           d. 将样本特征添加到特征列表X
           e. 根据噪声水平确定动作质量评分(噪声越小，评分越高)
           f. 将质量评分添加到标签列表y
        4. 将特征和标签转换为numpy数组并返回
    
    用途:
        该函数通过向标准动作添加不同程度的噪声来模拟不同质量的用户动作，
        生成的数据集可用于训练动作质量评估模型，使模型能够识别不同质量水平的动作执行
    """
    
    X = []
    y = []
    
    body_parts = list(standard_frames[0].keys())
    
    for _ in range(num_samples):
        noise_level = np.random.uniform(0, 0.3)
        sample = []
        
        for frame in standard_frames:
            frame_data = []
            for part in body_parts:
                coords = np.array(frame[part][:2])
                noise = np.random.randn(2) * noise_level
                noisy_coords = coords + noise
                frame_data.extend(noisy_coords)
            sample.extend(frame_data)
        
        X.append(sample)
        
        if noise_level < 0.05:
            quality = 1.0
        elif noise_level < 0.1:
            quality = 0.9
        elif noise_level < 0.15:
            quality = 0.8
        elif noise_level < 0.2:
            quality = 0.7
        else:
            quality = 0.6
        
        y.append(quality)
    
    return np.array(X), np.array(y)

def main():
    with open('trajectory_system/standard_trajectory.pkl', 'rb') as f:
        standard_frames = pickle.load(f)
    
    X, y = generate_synthetic_data(standard_frames, num_samples=1000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with open('trajectory_system/training_data.pkl', 'wb') as f:
        pickle.dump({'X_train': X_train, 'X_test': X_test, 
                     'y_train': y_train, 'y_test': y_test}, f)
    
    print(f"生成数据：训练集{len(X_train)}条，测试集{len(X_test)}条")

if __name__ == '__main__':
    main()


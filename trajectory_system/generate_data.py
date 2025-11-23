import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def generate_synthetic_data(standard_frames, num_samples=500):
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


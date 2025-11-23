"""
模型训练脚本
使用收集的数据训练手势识别模型
"""
import numpy as np
import os
from gesture_classifier import GestureClassifier


def train_model():
    """训练手势识别模型"""
    data_dir = 'training_data'
    
    # 检查数据文件是否存在
    features_path = os.path.join(data_dir, 'features.npy')
    labels_path = os.path.join(data_dir, 'labels.npy')
    
    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("错误: 未找到训练数据文件")
        print("请先运行 data_collector.py 收集数据")
        print("\n或者使用示例数据生成器创建模拟数据...")
        
        # 生成示例数据用于演示
        response = input("是否生成示例数据？(y/n): ")
        if response.lower() == 'y':
            generate_sample_data(data_dir)
        else:
            return
    
    # 加载数据
    print("加载训练数据...")
    X = np.load(features_path)
    y = np.load(labels_path)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"标签类别数: {len(np.unique(y))}")
    
    # 创建分类器并训练
    classifier = GestureClassifier()
    classifier.train(X, y)
    
    print("\n模型训练完成！")


def generate_sample_data(data_dir):
    """生成示例数据（用于演示）"""
    import os
    os.makedirs(data_dir, exist_ok=True)
    
    print("生成示例数据...")
    print("注意: 这是模拟数据，实际使用时请使用真实的手势数据")
    
    # 为每个手势生成一些随机特征
    classifier = GestureClassifier()
    n_samples_per_class = 20
    # 注意：新特征提取方法会产生更多特征（约77维）
    # 这里使用一个合理的默认值
    n_features = 77
    
    features_list = []
    labels_list = []
    
    for label in classifier.LABELS:
        # 为每个标签生成随机特征（实际应用中应该使用真实数据）
        for _ in range(n_samples_per_class):
            # 生成随机特征向量
            features = np.random.randn(n_features)
            features_list.append(features)
            labels_list.append(label)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # 保存数据
    np.save(os.path.join(data_dir, 'features.npy'), X)
    np.save(os.path.join(data_dir, 'labels.npy'), y)
    
    print(f"已生成示例数据: {X.shape[0]} 个样本")


if __name__ == '__main__':
    train_model()


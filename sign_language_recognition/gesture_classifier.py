"""
手势分类器模块
使用机器学习模型识别手语字母
"""
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class GestureClassifier:
    """手势分类器"""
    
    # 30个手语字母标签
    LABELS = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'ZH', 'CH', 'SH', 'NG'
    ]
    
    def __init__(self, model_path='gesture_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载训练好的模型"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"模型已加载: {self.model_path}")
        else:
            print(f"模型文件不存在: {self.model_path}")
            print("请先运行 train_model.py 训练模型")
            # 创建一个改进的随机森林分类器（增加树的数量和深度）
            self.model = RandomForestClassifier(
                n_estimators=300,  # 增加树的数量
                max_depth=20,      # 增加树的深度
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )
    
    def save_model(self):
        """保存模型"""
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"模型已保存: {self.model_path}")
    
    def train(self, X, y):
        """
        训练分类器
        
        Args:
            X: 特征矩阵
            y: 标签向量
        """
        print("开始训练模型...")
        
        # 检查并处理数据
        print(f"原始数据形状: X={X.shape}, y={y.shape}")
        
        # 移除包含NaN或Inf的样本
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"清理后数据形状: X={X.shape}, y={y.shape}")
        
        # 划分训练集和测试集（使用分层抽样确保每个类别都有样本）
        from sklearn.model_selection import StratifiedShuffleSplit
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(split.split(X, y))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"训练集: {X_train.shape[0]} 个样本")
        print(f"测试集: {X_test.shape[0]} 个样本")
        
        # 如果模型未初始化，创建新模型
        if self.model is None:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        
        # 训练模型
        print("正在训练模型（这可能需要一些时间）...")
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n模型准确率: {accuracy:.2%}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.LABELS, zero_division=0))
        
        # 显示特征重要性（前10个）
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            print("\n最重要的10个特征索引:")
            for idx in top_indices:
                print(f"  特征 {idx}: {importances[idx]:.4f}")
        
        # 保存模型
        self.save_model()
    
    def predict(self, features):
        """
        预测手势
        
        Args:
            features: 特征向量
            
        Returns:
            label: 预测的手势标签
            confidence: 置信度
        """
        if self.model is None or features is None:
            return None, 0.0
        
        # 确保特征是一维数组
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 预测
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def get_label_index(self, label):
        """获取标签的索引"""
        try:
            return self.LABELS.index(label.upper())
        except ValueError:
            return -1


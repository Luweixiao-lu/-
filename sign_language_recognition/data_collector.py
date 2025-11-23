"""
数据收集工具
用于收集和保存手语字母的训练数据
"""
import cv2
import numpy as np
import os
import json
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier


class DataCollector:
    """数据收集器"""
    
    def __init__(self, data_dir='training_data'):
        self.data_dir = data_dir
        self.detector = HandLandmarkDetector()
        self.classifier = GestureClassifier()
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 为每个标签创建子目录
        for label in self.classifier.LABELS:
            os.makedirs(os.path.join(data_dir, label), exist_ok=True)
    
    def collect_data(self):
        """交互式数据收集"""
        cap = cv2.VideoCapture(0)
        
        current_label_idx = 0
        sample_count = 0
        samples_per_label = 100  # 每个手势收集100个样本（增加数据量以提高准确率）
        
        print("=" * 50)
        print("手语字母数据收集工具")
        print("=" * 50)
        print(f"\n当前手势: {self.classifier.LABELS[current_label_idx]}")
        print(f"样本数量: {sample_count}/{samples_per_label}")
        print("\n操作说明:")
        print("  - 按 's' 键保存当前手势样本")
        print("  - 按 'n' 键切换到下一个手势")
        print("  - 按 'q' 键退出")
        print("=" * 50)
        
        features_list = []
        labels_list = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测手部关键点
            landmarks, annotated_frame = self.detector.detect(frame)
            
            # 显示当前手势和样本数
            cv2.putText(annotated_frame, 
                       f"Gesture: {self.classifier.LABELS[current_label_idx]}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame,
                       f"Samples: {sample_count}/{samples_per_label}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示提示
            if landmarks is not None:
                cv2.putText(annotated_frame, "Press 's' to save", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No hand detected", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Data Collection', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and landmarks is not None:
                # 提取特征
                features = self.detector.extract_features(landmarks)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(self.classifier.LABELS[current_label_idx])
                    sample_count += 1
                    print(f"已保存样本 {sample_count}/{samples_per_label} - {self.classifier.LABELS[current_label_idx]}")
                    
                    if sample_count >= samples_per_label:
                        print(f"\n完成手势 {self.classifier.LABELS[current_label_idx]} 的数据收集")
                        current_label_idx += 1
                        sample_count = 0
                        
                        if current_label_idx >= len(self.classifier.LABELS):
                            print("\n所有手势数据收集完成！")
                            break
                        else:
                            print(f"\n切换到手势: {self.classifier.LABELS[current_label_idx]}")
            elif key == ord('n'):
                # 跳过当前手势
                current_label_idx += 1
                sample_count = 0
                if current_label_idx >= len(self.classifier.LABELS):
                    break
                print(f"\n切换到手势: {self.classifier.LABELS[current_label_idx]}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 保存数据
        if features_list:
            X = np.array(features_list)
            y = np.array(labels_list)
            
            np.save(os.path.join(self.data_dir, 'features.npy'), X)
            np.save(os.path.join(self.data_dir, 'labels.npy'), y)
            
            print(f"\n数据已保存:")
            print(f"  - 特征: {X.shape}")
            print(f"  - 标签: {y.shape}")
            print(f"  - 保存路径: {self.data_dir}")


def main():
    """主函数，用于命令行入口"""
    collector = DataCollector()
    collector.collect_data()


if __name__ == '__main__':
    main()


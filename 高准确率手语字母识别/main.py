"""
手语识别软件主程序
实时识别手语字母
"""
import cv2
import numpy as np
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier


class SignLanguageRecognizer:
    """手语识别器主类"""
    
    def __init__(self):
        self.detector = HandLandmarkDetector()
        self.classifier = GestureClassifier()
        self.cap = None
        
        # 用于平滑预测结果
        self.prediction_history = []
        self.history_size = 5
        
    def start(self):
        """启动识别程序"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("错误: 无法打开摄像头")
            return
        
        print("=" * 50)
        print("手语字母识别软件")
        print("=" * 50)
        print("\n操作说明:")
        print("  - 将手放在摄像头前")
        print("  - 做出手语字母手势")
        print("  - 按 'q' 键退出")
        print("=" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 水平翻转图像（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 检测手部关键点
            landmarks, annotated_frame = self.detector.detect(frame)
            
            # 识别手势
            prediction = None
            confidence = 0.0
            
            if landmarks is not None:
                # 提取特征
                features = self.detector.extract_features(landmarks)
                
                if features is not None:
                    # 预测手势
                    prediction, confidence = self.classifier.predict(features)
                    
                    # 使用历史记录平滑预测
                    self.prediction_history.append(prediction)
                    if len(self.prediction_history) > self.history_size:
                        self.prediction_history.pop(0)
                    
                    # 使用最常见的预测结果
                    if len(self.prediction_history) >= 3:
                        from collections import Counter
                        most_common = Counter(self.prediction_history).most_common(1)[0]
                        prediction = most_common[0]
                        confidence = most_common[1] / len(self.prediction_history)
            
            # 显示结果
            self._draw_results(annotated_frame, prediction, confidence)
            
            cv2.imshow('Sign Language Recognition', annotated_frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def _draw_results(self, frame, prediction, confidence):
        """在图像上绘制识别结果"""
        # 绘制标题
        cv2.putText(frame, "Sign Language Recognition",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if prediction is not None:
            # 显示预测结果
            text = f"Gesture: {prediction}"
            confidence_text = f"Confidence: {confidence:.1%}"
            
            # 根据置信度选择颜色
            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            
            cv2.putText(frame, text,
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, confidence_text,
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "No hand detected",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Please show your hand",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n程序已退出")


def main():
    """主函数"""
    recognizer = SignLanguageRecognizer()
    recognizer.start()


if __name__ == '__main__':
    main()


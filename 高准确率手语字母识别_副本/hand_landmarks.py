"""
手部关键点检测模块
使用MediaPipe进行手部检测和关键点提取
"""
import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 安全导入核心库
try:
    import numpy as np
    logger.info("成功导入numpy")
except ImportError as e:
    logger.error(f"导入numpy失败: {str(e)}")
    raise

try:
    import cv2
    logger.info("成功导入cv2")
except ImportError as e:
    logger.error(f"导入cv2失败: {str(e)}")
    raise

try:
    import mediapipe as mp
    logger.info("成功导入mediapipe")
except ImportError as e:
    logger.error(f"导入mediapipe失败: {str(e)}")
    raise


class HandLandmarkDetector:
    """手部关键点检测器"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, image):
        """
        检测图像中的手部关键点
        
        Args:
            image: BGR格式的图像
            
        Returns:
            landmarks: 手部关键点坐标（21个点，每个点有x, y, z坐标）
            annotated_image: 标注了关键点的图像
        """
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(rgb_image)
        
        landmarks = None
        annotated_image = image.copy()
        
        if results.multi_hand_landmarks:
            # 获取第一个检测到的手
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 绘制关键点
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # 提取关键点坐标
            landmarks = []
            h, w, _ = image.shape
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x * w, landmark.y * h, landmark.z])
            landmarks = np.array(landmarks)
        
        return landmarks, annotated_image
    
    def extract_features(self, landmarks):
        """
        从关键点提取特征向量（改进版）
        
        Args:
            landmarks: 手部关键点坐标
            
        Returns:
            features: 特征向量（包含相对坐标、角度、距离等）
        """
        if landmarks is None or len(landmarks) == 0:
            return None
        
        features_list = []
        
        # 1. 使用手腕（landmark 0）作为参考点的相对坐标
        wrist = landmarks[0]
        relative_landmarks = landmarks - wrist
        
        # 归一化（使用手部尺寸）
        hand_size = np.linalg.norm(landmarks[5] - landmarks[17])  # 使用食指根到小指根的距离
        if hand_size > 0:
            relative_landmarks = relative_landmarks / hand_size
        
        # 添加相对坐标特征（只使用x和y，z坐标通常变化较大）
        features_list.extend(relative_landmarks[:, :2].flatten())  # 42维
        
        # 2. 关键点之间的距离特征
        key_distances = []
        # 手指长度（指尖到指根）
        finger_tips = [4, 8, 12, 16, 20]  # 拇指、食指、中指、无名指、小指指尖
        finger_mcps = [2, 5, 9, 13, 17]   # 对应的指根
        
        for tip, mcp in zip(finger_tips, finger_mcps):
            if tip < len(landmarks) and mcp < len(landmarks):
                dist = np.linalg.norm(landmarks[tip] - landmarks[mcp])
                key_distances.append(dist)
        
        # 归一化距离
        if len(key_distances) > 0:
            max_dist = max(key_distances) if max(key_distances) > 0 else 1
            key_distances = [d / max_dist for d in key_distances]
        
        features_list.extend(key_distances)  # 5维
        
        # 3. 手指之间的角度特征
        angles = []
        # 计算相邻手指之间的角度
        for i in range(len(finger_tips) - 1):
            tip1 = landmarks[finger_tips[i]]
            tip2 = landmarks[finger_tips[i + 1]]
            wrist_pos = landmarks[0]
            
            # 计算两个向量
            vec1 = tip1 - wrist_pos
            vec2 = tip2 - wrist_pos
            
            # 计算角度（使用2D投影）
            if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                cos_angle = np.dot(vec1[:2], vec2[:2]) / (np.linalg.norm(vec1[:2]) * np.linalg.norm(vec2[:2]))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                angles.append(angle)
        
        features_list.extend(angles)  # 4维
        
        # 4. 手指是否伸直（指尖到手腕的距离 vs 指根到手腕的距离）
        finger_straightness = []
        for tip, mcp in zip(finger_tips, finger_mcps):
            if tip < len(landmarks) and mcp < len(landmarks):
                tip_to_wrist = np.linalg.norm(landmarks[tip] - landmarks[0])
                mcp_to_wrist = np.linalg.norm(landmarks[mcp] - landmarks[0])
                if mcp_to_wrist > 0:
                    ratio = tip_to_wrist / mcp_to_wrist
                    finger_straightness.append(ratio)
        
        features_list.extend(finger_straightness)  # 5维
        
        # 5. 手部中心到各关键点的距离（归一化）
        hand_center = np.mean(landmarks, axis=0)
        center_distances = []
        for landmark in landmarks:
            dist = np.linalg.norm(landmark - hand_center)
            center_distances.append(dist)
        
        if len(center_distances) > 0:
            max_center_dist = max(center_distances) if max(center_distances) > 0 else 1
            center_distances = [d / max_center_dist for d in center_distances]
        
        features_list.extend(center_distances)  # 21维
        
        # 转换为numpy数组
        features = np.array(features_list)
        
        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features


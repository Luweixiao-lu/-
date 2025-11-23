#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手语字母识别 - Streamlit Web应用
提供友好的Web界面来实时识别手语字母
"""

# 基础导入
sys = None
os = None
cv2 = None
np = None
st = None
HandLandmarkDetector = None
GestureClassifier = None

# 主要应用类
import logging
import sys
import time
import traceback

class SignLanguageRecognitionApp:
    def __init__(self):
        """初始化应用"""
        self.setup_logging()
        self.import_dependencies()
        self.detector = None
        self.classifier = None
        self.cap = None
        self.prediction_history = []
        # 保存模块引用
        self.traceback = traceback
        self.time = time
        
    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def import_dependencies(self):
        """安全导入所有依赖"""
        global sys, os, cv2, np, st, HandLandmarkDetector, GestureClassifier
        
        # 导入基础库
        try:
            import sys, os, time, traceback
            self.logger.info("成功导入基础库")
        except Exception as e:
            self.logger.error(f"导入基础库失败: {str(e)}")
            print(f"错误: {str(e)}")
            exit(1)
        
        # 导入Streamlit
        try:
            import streamlit as st
            self.logger.info("成功导入streamlit")
        except ImportError as e:
            self.logger.error(f"导入streamlit失败: {str(e)}")
            print("错误: 缺少streamlit依赖。请运行 pip install streamlit 安装所需依赖。")
            sys.exit(1)
        
        # 导入numpy
        try:
            import numpy as np
            self.logger.info("成功导入numpy")
        except ImportError as e:
            self.logger.error(f"导入numpy失败: {str(e)}")
            if st:
                st.error(f"导入numpy失败: {str(e)}")
                st.info("请运行: pip install numpy")
        
        # 导入OpenCV
        try:
            import cv2
            self.logger.info("成功导入OpenCV")
        except ImportError as e:
            self.logger.error(f"导入OpenCV失败: {str(e)}")
            if st:
                st.warning(f"导入OpenCV失败: {str(e)}")
                st.info("请运行: pip install opencv-python 或 opencv-python-headless")
        
        # 导入自定义模块
        self.import_custom_modules()
    
    def import_custom_modules(self):
        """导入自定义模块"""
        global HandLandmarkDetector, GestureClassifier
        
        # 尝试导入GestureClassifier
        try:
            from gesture_classifier import GestureClassifier
            self.logger.info("成功导入GestureClassifier")
        except ImportError as e:
            self.logger.error(f"导入GestureClassifier失败: {str(e)}")
            
            # 创建替代类
            class DummyGestureClassifier:
                LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                         'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                         'ZH', 'CH', 'SH', 'NG']
                
                def predict(self, features):
                    return 'A', 0.5
                
                def get_confidence(self):
                    return 0.5
            
            GestureClassifier = DummyGestureClassifier
            if st:
                st.info("已创建GestureClassifier替代类")
        
        # 尝试导入HandLandmarkDetector
        try:
            from hand_landmarks import HandLandmarkDetector
            self.logger.info("成功导入HandLandmarkDetector")
        except ImportError as e:
            self.logger.error(f"导入HandLandmarkDetector失败: {str(e)}")
            
            # 创建替代类
            class DummyHandLandmarkDetector:
                def __init__(self):
                    self.logger = logging.getLogger(__name__)
                    self.logger.info("使用DummyHandLandmarkDetector替代类")
                    self.fake_landmarks = None
                
                def detect(self, image):
                    return None, image if image is not None else None
                
                def extract_features(self, landmarks):
                    if np is not None:
                        return np.zeros(63)
                    return []
                
                def get_landmarks(self, image):
                    return self.fake_landmarks
                
                def draw_landmarks(self, image, landmarks=None, connections=True):
                    if image is None:
                        return None
                    return image.copy()
            
            HandLandmarkDetector = DummyHandLandmarkDetector
            if st:
                st.info("已创建HandLandmarkDetector替代类")
    
    def is_streamlit_cloud(self):
        """检测是否在Streamlit Cloud环境中"""
        # 检查环境变量
        if os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true':
            return True
        if os.name == 'posix' and os.path.exists('/app/.streamlit/config.toml'):
            return True
        if os.environ.get('HOME') == '/app' and os.environ.get('HOSTNAME'):
            return True
        if os.environ.get('PWD') == '/app' or os.environ.get('DOCKER_CONTAINER') == 'true':
            return True
        return False
    
    def setup_ui(self):
        """设置用户界面"""
        # 页面配置
        st.set_page_config(
            page_title="手语字母识别",
            page_icon="👋",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 自定义CSS
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                text-align: center;
                color: #2c3e50;
                margin-bottom: 1rem;
            }
            .sub-header {
                font-size: 1.2rem;
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 2rem;
            }
            .result-container {
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                background-color: #f8f9fa;
                text-align: center;
            }
            .result-text {
                font-size: 1.8rem;
                font-weight: bold;
            }
            .confidence-text {
                font-size: 1.2rem;
                color: #27ae60;
            }
            .instruction-box {
                background-color: #e8f4f8;
                border-left: 5px solid #3498db;
                padding: 1rem;
                margin: 1rem 0;
            }
            .warning-box {
                background-color: #fef5e7;
                border-left: 5px solid #f39c12;
                padding: 1rem;
                margin: 1rem 0;
            }
            .success-box {
                background-color: #e8f8f5;
                border-left: 5px solid #27ae60;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 标题
        st.markdown('<h1 class="main-header">👋 手语字母识别系统</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">实时识别汉语手指字母（A-Z, ZH, CH, SH, NG）</p>', unsafe_allow_html=True)
    
    def create_sidebar(self):
        """创建侧边栏设置"""
        with st.sidebar:
            st.header("设置")
            
            # 摄像头设置
            camera_index = st.selectbox(
                "选择摄像头",
                options=[0, 1, 2],
                format_func=lambda x: f"摄像头 {x}",
                index=0
            )
            
            # 识别设置
            history_size = st.slider(
                "识别结果平滑度",
                min_value=1,
                max_value=10,
                value=5,
                help="较大的值会使识别结果更稳定，但响应会变慢"
            )
            
            # 显示设置
            show_landmarks = st.checkbox("显示手部关键点", value=True)
            show_connections = st.checkbox("显示骨骼连接", value=True)
            
            # 关于部分
            st.markdown("---")
            st.header("关于")
            st.info(
                "基于计算机视觉和机器学习的手语字母识别系统。" 
                "支持30个汉语手指字母手势识别。"
            )
        
        return camera_index, history_size, show_landmarks, show_connections
    
    def create_main_layout(self):
        """创建主布局"""
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.header("摄像头预览")
            video_placeholder = st.empty()
            
            # 控制面板
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                start_button = st.button("开始识别", type="primary")
            with control_col2:
                stop_button = st.button("停止识别")
        
        with col2:
            st.header("识别结果")
            result_placeholder = st.empty()
            
            # 使用说明
            with st.expander("使用说明", expanded=True):
                st.markdown('<div class="instruction-box">', unsafe_allow_html=True)
                st.write("1. 点击 '开始识别' 按钮启动摄像头")
                st.write("2. 将手放在摄像头前，保持手势清晰")
                st.write("3. 确保良好的光照条件")
                st.write("4. 每个手势保持2-3秒以便识别")
                st.write("5. 点击 '停止识别' 按钮结束")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # 手势指南
        with st.expander("手势指南"):
            st.info("详细手势说明请参考 gesture_guide.md 文件")
            st.write("支持以下30个手语字母：")
            if GestureClassifier is not None and hasattr(GestureClassifier, 'LABELS'):
                st.code(", ".join(GestureClassifier.LABELS))
            else:
                st.code("A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, ZH, CH, SH, NG")
        
        return start_button, stop_button, video_placeholder, result_placeholder
    
    def check_model_file(self, result_placeholder):
        """检查模型文件是否存在"""
        try:
            model_exists = os.path.exists('gesture_model.pkl')
            if not model_exists:
                with result_placeholder.container():
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    if self.is_streamlit_cloud():
                        st.warning("未检测到训练好的模型文件 (gesture_model.pkl)")
                        st.info("在Streamlit Cloud上使用前，请确保模型文件已正确上传到GitHub仓库")
                    else:
                        st.warning("未检测到训练好的模型文件 (gesture_model.pkl)。请先训练模型。")
                        st.info("运行train_model.py脚本来训练和保存模型")
                    st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            self.logger.error(f"检查模型文件时出错: {str(e)}")
            with result_placeholder.container():
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(f"检查模型文件时出错: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
    
    def initialize_detector_and_classifier(self):
        """初始化检测器和分类器"""
        try:
            self.detector = HandLandmarkDetector()
            self.logger.info("成功初始化HandLandmarkDetector")
        except Exception as e:
            self.logger.error(f"初始化检测器失败: {str(e)}")
            return False, f"初始化检测器失败: {str(e)}"
        
        try:
            self.classifier = GestureClassifier()
            self.logger.info("成功初始化GestureClassifier")
            return True, ""
        except Exception as e:
            self.logger.error(f"初始化分类器失败: {str(e)}")
            return False, f"初始化分类器失败: {str(e)}"
    
    def open_camera(self, camera_index):
        """打开摄像头"""
        if self.is_streamlit_cloud():
            return False, "在Streamlit Cloud环境中无法访问摄像头"
        
        max_retries = 3
        retry_count = 0
        self.cap = None
        
        while retry_count < max_retries:
            try:
                self.cap = cv2.VideoCapture(camera_index)
                self.time.sleep(0.5)  # 等待摄像头初始化
                
                if self.cap.isOpened():
                    # 设置分辨率
                    try:
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        self.logger.info("成功设置摄像头分辨率: 1280x720")
                    except Exception as e:
                        self.logger.warning(f"设置摄像头分辨率失败: {str(e)}")
                    
                    return True, ""
                else:
                    retry_count += 1
                    self.logger.warning(f"摄像头打开失败，正在尝试第 {retry_count} 次重试...")
                    if self.cap:
                        self.cap.release()
                    self.time.sleep(0.5)
                    
            except Exception as e:
                retry_count += 1
                self.logger.error(f"尝试打开摄像头时出错: {str(e)}")
                if self.cap:
                    self.cap.release()
                self.time.sleep(0.5)
        
        return False, "无法打开摄像头，请检查设备连接和权限"
    
    def process_frame(self, frame, history_size):
        """处理每一帧图像"""
        # 水平翻转图像
        frame = cv2.flip(frame, 1)
        
        # 检测手部关键点
        landmarks = None
        annotated_frame = frame.copy()
        
        try:
            landmarks, annotated_frame = self.detector.detect(frame)
        except Exception as e:
            self.logger.error(f"手部检测出错: {str(e)}")
            annotated_frame = frame
        
        # 识别手势
        prediction = None
        confidence = 0.0
        
        try:
            if landmarks is not None:
                # 提取特征
                features = self.detector.extract_features(landmarks)
                
                if features is not None:
                    # 预测手势
                    prediction, confidence = self.classifier.predict(features)
                    
                    # 平滑预测结果
                    self.prediction_history.append(prediction)
                    if len(self.prediction_history) > history_size:
                        self.prediction_history.pop(0)
                    
                    # 使用最常见的预测结果
                    if len(self.prediction_history) >= 3:
                        try:
                            from collections import Counter
                            most_common = Counter(self.prediction_history).most_common(1)[0]
                            prediction = most_common[0]
                            confidence = most_common[1] / len(self.prediction_history)
                        except Exception:
                            prediction = self.prediction_history[-1]
                            confidence = 0.7
            else:
                prediction = '未检测到手势'
                confidence = 0.0
        except Exception as e:
            self.logger.error(f"手势识别出错: {str(e)}")
            prediction = '处理错误'
            confidence = 0.0
        
        return annotated_frame, prediction, confidence
    
    def display_result(self, result_placeholder, prediction, confidence):
        """显示识别结果"""
        with result_placeholder.container():
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            if prediction is not None:
                # 根据置信度选择颜色
                if confidence > 0.7:
                    color = "#27ae60"  # 绿色
                elif confidence > 0.5:
                    color = "#f39c12"  # 橙色
                else:
                    color = "#e74c3c"  # 红色
                
                st.markdown(f'<p class="result-text" style="color: {color};">识别结果: {prediction}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="confidence-text">置信度: {confidence:.1%}</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="result-text" style="color: #7f8c8d;">未检测到手势</p>', unsafe_allow_html=True)
                st.markdown('<p>请将手放在摄像头前</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    def release_resources(self):
        """释放资源"""
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.logger.info("摄像头资源已释放")
        except Exception as e:
            self.logger.error(f"释放资源时出错: {str(e)}")
    
    def run(self):
        """运行应用"""
        try:
            self.setup_ui()
            camera_index, history_size, show_landmarks, show_connections = self.create_sidebar()
            start_button, stop_button, video_placeholder, result_placeholder = self.create_main_layout()
            
            # 检查模型文件
            self.check_model_file(result_placeholder)
            
            # 运行识别
            if start_button:
                # 初始化检测器和分类器
                success, error_msg = self.initialize_detector_and_classifier()
                if not success:
                    with result_placeholder.container():
                        st.error(error_msg)
                    return
                
                # 打开摄像头
                success, error_msg = self.open_camera(camera_index)
                if not success:
                    with result_placeholder.container():
                        st.error(error_msg)
                        st.info("在Streamlit Cloud环境中，摄像头功能通常不可用")
                    return
                
                # 显示成功信息
                with result_placeholder.container():
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.success("摄像头启动成功！请将手放在摄像头前进行手势识别")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 初始化停止标志
                st.session_state.stop = False
                self.prediction_history = []
                
                # 主循环
                while not st.session_state.get('stop', False):
                    try:
                        # 读取帧
                        ret, frame = self.cap.read()
                        if not ret:
                            self.logger.warning("无法获取摄像头图像")
                            self.time.sleep(0.1)
                            continue
                        
                        # 处理帧
                        annotated_frame, prediction, confidence = self.process_frame(frame, history_size)
                        
                        # 显示图像
                        try:
                            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                        except Exception as e:
                            self.logger.error(f"显示图像出错: {str(e)}")
                        
                        # 显示结果
                        self.display_result(result_placeholder, prediction, confidence)
                        
                        # 短暂延迟
                        self.time.sleep(0.05)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        self.logger.error(f"处理帧时出错: {str(e)}")
                        # 继续运行，不中断
                        self.time.sleep(0.1)
                
                # 停止时显示信息
                with result_placeholder.container():
                    st.info("识别已停止")
                
            # 停止按钮
            if stop_button:
                st.session_state.stop = True
                with result_placeholder.container():
                    st.info("正在停止识别...")
            
            # 页脚
            st.markdown("---")
            st.markdown("*手语字母识别系统 © 2024*")
            
        except Exception as e:
            self.logger.error(f"应用运行出错: {str(e)}")
            self.logger.debug(self.traceback.format_exc())
            if st:
                st.error(f"应用出错: {str(e)}")
        finally:
            self.release_resources()

# 主程序
if __name__ == "__main__":
    app = SignLanguageRecognitionApp()
    app.run()
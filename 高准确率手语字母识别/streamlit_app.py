#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手语字母识别 - Streamlit Web应用
提供友好的Web界面来实时识别手语字母
"""

# 添加错误处理导入
# 基础导入
import streamlit as st
import numpy as np
import os
import time
import traceback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义函数检测是否在Streamlit Cloud环境中
def is_streamlit_cloud():
    """检测当前运行环境是否为Streamlit Cloud"""
    # 检查环境变量
    if os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true':
        return True
    # 检查是否在Linux环境且有特定路径
    if os.name == 'posix' and os.path.exists('/app/.streamlit/config.toml'):
        return True
    # 检查其他Streamlit Cloud特有的环境变量
    if os.environ.get('HOME') == '/app' and os.environ.get('HOSTNAME'):
        return True
    return False

# 检测当前环境
IN_STREAMLIT_CLOUD = is_streamlit_cloud()
logger.info(f"当前运行环境: {'Streamlit Cloud' if IN_STREAMLIT_CLOUD else '本地环境'}")

# 安全导入必要的依赖
critical_missing = False

try:
    import streamlit as st
except ImportError:
    logger.error("无法导入必要的依赖: streamlit")
    critical_missing = True

try:
    import numpy as np
except ImportError:
    logger.error("无法导入必要的依赖: numpy")

# 其他基础依赖
try:
    import time
except ImportError:
    logger.error("无法导入必要的依赖: time")

try:
    import os
except ImportError:
    logger.error("无法导入必要的依赖: os")

# 尝试导入OpenCV，添加更健壮的错误处理
cv2 = None
try:
    import cv2
    logger.info("成功导入OpenCV")
except ImportError:
    logger.error("无法导入必要的依赖: cv2/opencv-python")
    if IN_STREAMLIT_CLOUD:
        st.warning("在Streamlit Cloud上无法使用摄像头功能，这是正常的限制。请在本地环境运行以体验完整功能。")
    else:
        st.warning("无法导入OpenCV库，可能无法使用摄像头功能。请确保已安装opencv-python包。")

# 如果缺少关键依赖，优雅退出
if critical_missing:
    print("错误: 缺少关键依赖包streamlit。请安装所有依赖后重试。")
    print("提示: 运行 'pip install -r requirements.txt' 安装所需依赖。")
    sys.exit(1)

# 安全导入自定义模块
HandLandmarkDetector = None
GestureClassifier = None

# 处理macOS上mediapipe的兼容性问题
import platform
IS_MACOS = platform.system() == 'Darwin'

# 首先尝试导入GestureClassifier，因为它依赖较少
try:
    from gesture_classifier import GestureClassifier
    logger.info("成功导入GestureClassifier")
except ImportError as e:
    logger.error(f"导入GestureClassifier失败: {str(e)}")
    # 创建替代类
    class DummyGestureClassifier:
        LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ZH', 'CH', 'SH', 'NG']
    
    GestureClassifier = DummyGestureClassifier
    st.info("已创建GestureClassifier替代类，部分功能可能受限")

# 尝试导入HandLandmarkDetector
try:
    from hand_landmarks import HandLandmarkDetector
    logger.info("成功导入HandLandmarkDetector")
except ImportError as e:
    logger.error(f"导入HandLandmarkDetector失败: {str(e)}")
    
    # 检查错误是否与libGL相关（常见于macOS）
    if 'libGL.so.1' in str(e) or 'GL' in str(e).upper():
        st.warning("在macOS上检测到mediapipe依赖问题，这是已知的兼容性问题。")
        st.info("您可以尝试安装额外的依赖来解决此问题:")
        st.code("# 方法1: 安装特定版本的mediapipe\npip install mediapipe-silicon")
        st.code("# 方法2: 使用conda安装\nconda install -c menpo opencv")
    else:
        st.warning(f"无法加载HandLandmarkDetector: {str(e)}")
    
    # 创建一个简单的替代类
    class DummyHandLandmarkDetector:
        def __init__(self):
            pass
        
        def detect(self, image):
            return None, image
        
        def extract_features(self, landmarks):
            return np.zeros(63)  # 返回一个零向量作为特征
    
    HandLandmarkDetector = DummyHandLandmarkDetector
    st.info("已创建HandLandmarkDetector替代类，摄像头功能将不可用")

logger.info("自定义模块导入尝试完成")

# 设置页面配置
# 检查st是否已成功导入
if 'st' in globals() and st is not None:
    st.set_page_config(
        page_title="手语字母识别",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
else:
    logger.error("streamlit未成功导入，无法设置页面配置")
    sys.exit(1)

# 自定义CSS样式
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

# 应用标题
st.markdown('<h1 class="main-header">👋 手语字母识别系统</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">实时识别汉语手指字母（A-Z, ZH, CH, SH, NG）</p>', unsafe_allow_html=True)

# 侧边栏设置
with st.sidebar:
    st.header("设置")
    
    # 选择摄像头
    camera_index = st.selectbox(
        "选择摄像头",
        options=[0, 1, 2],
        format_func=lambda x: f"摄像头 {x}",
        index=0
    )
    
    # 平滑设置
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

# 主要内容区域
col1, col2 = st.columns([3, 2])

with col1:
    st.header("摄像头预览")
    # 用于显示摄像头流的占位符
    video_placeholder = st.empty()
    
    # 控制面板
    control_col1, control_col2 = st.columns(2)
    with control_col1:
        start_button = st.button("开始识别", type="primary")
    with control_col2:
        stop_button = st.button("停止识别")

with col2:
    st.header("识别结果")
    # 用于显示识别结果的占位符
    result_placeholder = st.empty()
    
    # 显示使用说明
    with st.expander("使用说明", expanded=True):
        st.markdown('<div class="instruction-box">', unsafe_allow_html=True)
        st.write("1. 点击 '开始识别' 按钮启动摄像头")
        st.write("2. 将手放在摄像头前，保持手势清晰")
        st.write("3. 确保良好的光照条件")
        st.write("4. 每个手势保持2-3秒以便识别")
        st.write("5. 点击 '停止识别' 按钮结束")
        st.markdown('</div>', unsafe_allow_html=True)

# 加载手势指南图片
with st.expander("手势指南"):
    st.info("详细手势说明请参考 gesture_guide.md 文件")
    st.write("支持以下30个手语字母：")
    # 添加条件检查，确保GestureClassifier不为None
    if GestureClassifier is not None and hasattr(GestureClassifier, 'LABELS'):
        st.code(", ".join(GestureClassifier.LABELS))
    else:
        st.code("A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, ZH, CH, SH, NG")
        st.info("提示：部分功能可能受限，正在使用默认标签列表")

# 在应用开始时显示环境信息
with col2:
    if IN_STREAMLIT_CLOUD:
        with st.expander("关于运行环境", expanded=True):
            st.info("当前正在Streamlit Cloud环境中运行")
            st.info("📷 摄像头功能在Cloud环境中不可用")
            st.info("🧠 模型文件需要提前上传到GitHub仓库")
            st.info("💻 本地运行可获得完整功能体验")

# 检查模型文件是否存在
try:
    model_exists = os.path.exists('gesture_model.pkl')
    if not model_exists:
        with result_placeholder.container():
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            if IN_STREAMLIT_CLOUD:
                st.warning("未检测到训练好的模型文件 (gesture_model.pkl)")
                st.info("在Streamlit Cloud上使用前，请确保模型文件已正确上传到GitHub仓库")
                st.info("提示：您需要在本地训练模型并将gesture_model.pkl文件添加到仓库中")
            else:
                st.warning("未检测到训练好的模型文件 (gesture_model.pkl)。请先训练模型。")
                st.info("运行train_model.py脚本来训练和保存模型")
            st.markdown('</div>', unsafe_allow_html=True)
except Exception as e:
    logger.error(f"检查模型文件时出错: {str(e)}")
    with result_placeholder.container():
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"检查模型文件时出错: {str(e)}")
        if IN_STREAMLIT_CLOUD:
            st.info("请确保GitHub仓库中包含有效的gesture_model.pkl文件")
        st.markdown('</div>', unsafe_allow_html=True)

# 主应用逻辑
if start_button:
    # 检查必要模块是否已导入
    if HandLandmarkDetector is None or GestureClassifier is None:
        with result_placeholder.container():
            st.error("无法加载必要的自定义模块。请检查hand_landmarks.py和gesture_classifier.py文件是否存在且没有错误。")
        st.stop()
    
    # 检查OpenCV是否可用
    if cv2 is None:
        with result_placeholder.container():
            st.error("OpenCV库未成功加载，无法使用摄像头功能。")
            if IN_STREAMLIT_CLOUD:
                st.info("在Streamlit Cloud上，摄像头功能受限是正常的限制。")
                st.info("🔧 解决方案：将应用克隆到本地并运行以使用完整功能")
                st.code("git clone 您的仓库URL\ncd 仓库目录\npip install -r requirements.txt\nstreamlit run streamlit_app.py")
            else:
                st.info("🔧 解决方案：安装OpenCV库")
                st.code("pip install opencv-python")
        st.stop()
    
    try:
        # 初始化检测器和分类器
        detector = HandLandmarkDetector()
        
        # 安全初始化分类器，处理可能的模型加载错误
        try:
            classifier = GestureClassifier()
        except Exception as e:
            logger.error(f"加载手势分类器失败: {str(e)}")
            with result_placeholder.container():
                st.error(f"加载模型失败: {str(e)}")
                st.info("请确保gesture_model.pkl文件存在且没有损坏")
            st.stop()
        
        # 用于平滑预测结果
        prediction_history = []
        
        # 打开摄像头 (在Streamlit Cloud上可能无法访问摄像头，添加条件检查)
        try:
            # 首先检查是否在Streamlit Cloud环境
            if os.environ.get('STREAMLIT_CLOUD', 'false').lower() == 'true':
                with result_placeholder.container():
                    st.info("注意: 在Streamlit Cloud上运行时，摄像头访问受限。请在本地环境测试完整功能。")
                    st.image("https://via.placeholder.com/800x600?text=Streamlit+Cloud+%E4%B8%8A%E6%97%A0%E6%B3%95%E8%AE%BF%E9%97%AE%E6%91%84%E5%83%8F%E5%A4%B4", use_column_width=True)
                st.stop()
            
            # 尝试打开摄像头，添加重试机制
            max_retries = 3
            retry_count = 0
            cap = None
            
            while retry_count < max_retries:
                try:
                    cap = cv2.VideoCapture(camera_index)
                    # 等待摄像头初始化
                    time.sleep(0.5)
                    
                    if cap.isOpened():
                        break
                    else:
                        logger.warning(f"摄像头打开失败，正在尝试第 {retry_count + 1} 次重试...")
                        if cap is not None:
                            cap.release()
                        retry_count += 1
                        time.sleep(0.5)
                except Exception as inner_e:
                    logger.error(f"尝试打开摄像头时出错: {str(inner_e)}")
                    retry_count += 1
                    time.sleep(0.5)
            
            # 检查摄像头是否成功打开
            if cap is None or not cap.isOpened():
                with result_placeholder.container():
                    st.error("无法打开摄像头，请检查设备连接、权限或尝试选择其他摄像头索引。")
                    st.info("可能的解决方案：")
                    st.info("1. 确保摄像头未被其他应用占用")
                    st.info("2. 检查应用是否有摄像头访问权限")
                    st.info("3. 在侧边栏尝试选择其他摄像头索引")
                    st.info("4. 重启应用或计算机后重试")
                st.stop()
            
            # 设置摄像头分辨率
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                logger.info(f"成功设置摄像头分辨率: 1280x720")
            except Exception as e:
                logger.warning(f"设置摄像头分辨率失败: {str(e)}，使用默认分辨率")
            
            # 初始化停止标志
            st.session_state.stop = False
            
            # 显示成功启动信息
            with result_placeholder.container():
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("摄像头启动成功！请将手放在摄像头前进行手势识别")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # 主循环
            while not st.session_state.get('stop', False):
                try:
                    # 读取帧
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("无法获取摄像头图像，尝试重新连接...")
                        # 尝试重新初始化摄像头
                        cap.release()
                        cap = cv2.VideoCapture(camera_index)
                        time.sleep(0.5)
                        continue
                    
                    # 水平翻转图像（镜像效果）
                    frame = cv2.flip(frame, 1)
                    
                    # 安全检测手部关键点
                    try:
                        landmarks, annotated_frame = detector.detect(frame)
                    except Exception as e:
                        logger.error(f"手部检测出错: {str(e)}")
                        # 使用原始帧继续
                        annotated_frame = frame
                        landmarks = None
                    
                    # 识别手势
                    prediction = None
                    confidence = 0.0
                    
                    if landmarks is not None:
                        try:
                            # 提取特征
                            features = detector.extract_features(landmarks)
                            
                            if features is not None:
                                # 预测手势
                                prediction, confidence = classifier.predict(features)
                                
                                # 使用历史记录平滑预测
                                prediction_history.append(prediction)
                                if len(prediction_history) > history_size:
                                    prediction_history.pop(0)
                                
                                # 使用最常见的预测结果
                                if len(prediction_history) >= 3:
                                    try:
                                        from collections import Counter
                                        most_common = Counter(prediction_history).most_common(1)[0]
                                        prediction = most_common[0]
                                        confidence = most_common[1] / len(prediction_history)
                                    except:
                                        # 简单地使用最新的预测结果
                                        prediction = prediction_history[-1]
                                        confidence = 0.7
                        except Exception as e:
                            logger.error(f"手势识别出错: {str(e)}")
                    
                    # 安全转换和显示图像
                    try:
                        # 将BGR图像转换为RGB格式以便Streamlit显示
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # 显示摄像头流
                        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    except Exception as e:
                        logger.error(f"图像处理出错: {str(e)}")
                        # 使用备用图像
                        video_placeholder.error("图像处理出错")
                    
                    # 显示识别结果
                    with result_placeholder.container():
                        if prediction is not None:
                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            # 根据置信度选择颜色
                            if confidence > 0.7:
                                color = "#27ae60"  # 绿色
                            elif confidence > 0.5:
                                color = "#f39c12"  # 橙色
                            else:
                                color = "#e74c3c"  # 红色
                            
                            st.markdown(f'<p class="result-text" style="color: {color};">识别结果: {prediction}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="confidence-text">置信度: {confidence:.1%}</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="result-container">', unsafe_allow_html=True)
                            st.markdown('<p class="result-text" style="color: #7f8c8d;">未检测到手势</p>', unsafe_allow_html=True)
                            st.markdown('<p>请将手放在摄像头前</p>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 模拟实时性，添加小延迟
                    time.sleep(0.05)
                    
                except KeyboardInterrupt:
                    # 处理用户中断
                    break
                except Exception as e:
                    # 捕获其他所有异常
                    logger.error(f"识别循环中出错: {str(e)}")
                    error_msg = traceback.format_exc()
                    logger.debug(error_msg)
                    # 显示友好错误信息但继续运行
                    with result_placeholder.container():
                        st.warning(f"处理过程中出现小错误: {str(e)}。程序将继续运行。")
                    # 短暂暂停后继续
                    time.sleep(1)
        except Exception as e:
            logger.error(f"摄像头处理出错: {str(e)}")
            with result_placeholder.container():
                st.error(f"无法初始化摄像头: {str(e)}")
        finally:
            # 确保资源被释放
            try:
                if 'cap' in locals() and cap is not None:
                    cap.release()
            except:
                pass
            with result_placeholder.container():
                st.info("识别已停止")
    except Exception as e:
        logger.error(f"应用程序出错: {str(e)}")
        error_msg = traceback.format_exc()
        logger.debug(error_msg)
        with result_placeholder.container():
            st.error(f"应用程序出错: {str(e)}")
            st.info("请刷新页面重试")

if stop_button:
    st.session_state.stop = True
    with result_placeholder.container():
        st.info("正在停止识别...")

# 页脚信息
st.markdown("---")
st.markdown("*手语字母识别系统 © 2024*")

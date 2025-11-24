#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
手语字母识别 - Streamlit Web应用
提供友好的Web界面来实时识别手语字母
"""

# 必须首先导入sys，用于可能的优雅退出
import sys

# 添加错误处理导入
# 基础导入
import os
import time
import traceback
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保streamlit存在
st = None
try:
    import streamlit as st
    logger.info("成功导入streamlit")
except ImportError as e:
    logger.error(f"导入streamlit失败: {str(e)}")
    print("错误: 缺少streamlit依赖。请运行 pip install -r requirements.txt 安装所需依赖。")
    sys.exit(1)

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
    # 检查是否有Streamlit Cloud特有的环境变量
    if os.environ.get('PWD') == '/app' or os.environ.get('DOCKER_CONTAINER') == 'true':
        return True
    return False

# 检测当前环境
IN_STREAMLIT_CLOUD = is_streamlit_cloud()
logger.info(f"当前运行环境: {'Streamlit Cloud' if IN_STREAMLIT_CLOUD else '本地环境'}")

# 安全导入必要的依赖
critical_missing = False

# numpy是必需的
np = None
try:
    import numpy as np
    logger.info("成功导入numpy")
except ImportError:
    logger.error("无法导入必要的依赖: numpy")
    critical_missing = True

# 尝试导入OpenCV，添加更健壮的错误处理
cv2 = None
try:
    # 设置环境变量以禁用OpenGL（如果可能）
    import os
    os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '0')
    # 尝试导入OpenCV
    import cv2
    # 测试基本功能是否可用
    try:
        # 尝试创建一个简单的测试，确保OpenCV可用
        test_img = cv2.imread.__name__  # 只是检查函数是否存在
        logger.info("成功导入OpenCV")
    except Exception as test_e:
        logger.warning(f"OpenCV已导入但功能可能受限: {str(test_e)}")
except (ImportError, OSError, RuntimeError, AttributeError) as e:
    error_msg = str(e)
    logger.error(f"无法导入cv2/opencv-python: {error_msg}")
    # 无论在什么环境，都显示友好的错误信息
    if st is not None:
        if 'libGL' in error_msg or 'GL' in error_msg.upper():
            st.error("⚠️ 无法导入OpenCV：缺少OpenGL系统库 (libGL.so.1)")
            if IN_STREAMLIT_CLOUD:
                st.warning("在Streamlit Cloud环境中，这是已知的限制。")
                st.info("💡 解决方案：应用将使用替代模式运行，摄像头功能不可用。")
            else:
                st.info("💡 推荐解决方案：")
                st.code("pip install opencv-python-headless", language="bash")
                st.info("或安装系统库（Linux）：")
                st.code("apt-get update && apt-get install -y libgl1-mesa-glx", language="bash")
        else:
            st.error("无法导入OpenCV，可能是缺少图像功能，请确保已安装opencv-python-headless包。")
            st.info("推荐安装命令: `pip install opencv-python-headless`")
            if IN_STREAMLIT_CLOUD:
                st.info("在Streamlit Cloud环境中，摄像头功能通常不可用。这是云平台的安全限制。")

# 如果缺少关键依赖，优雅退出
if critical_missing:
    print("错误: 缺少关键依赖包。请安装所有依赖后重试。")
    print("提示: 运行 'pip install -r requirements.txt' 安装所需依赖。")
    sys.exit(1)

# 安全导入自定义模块
HandLandmarkDetector = None
GestureClassifier = None

# 处理不同平台的兼容性问题
import platform
IS_MACOS = platform.system() == 'Darwin'
IS_LINUX = platform.system() == 'Linux'
IS_WINDOWS = platform.system() == 'Windows'

logger.info(f"操作系统类型: {platform.system()}")

# 首先尝试导入GestureClassifier，因为它依赖较少
try:
    from gesture_classifier import GestureClassifier
    logger.info("成功导入GestureClassifier")
except ImportError as e:
    logger.error(f"导入GestureClassifier失败: {str(e)}")
    # 创建替代类
    class DummyGestureClassifier:
        LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ZH', 'CH', 'SH', 'NG']
        
        def predict(self, features):
            # 返回默认手势和置信度
            return 'A', 0.5
        
        def get_confidence(self):
            # 返回默认置信度
            return 0.5
    
    GestureClassifier = DummyGestureClassifier
    if st is not None:
        st.info("已创建GestureClassifier替代类，部分功能可能受限")

# 尝试导入HandLandmarkDetector
try:
    from hand_landmarks import HandLandmarkDetector
    logger.info("成功导入HandLandmarkDetector")
    # 尝试初始化以检查mediapipe是否可用
    try:
        test_detector = HandLandmarkDetector()
        if not hasattr(test_detector, 'available') or not test_detector.available:
            logger.warning("HandLandmarkDetector已导入，但MediaPipe不可用，将使用受限模式")
            if st is not None and IN_STREAMLIT_CLOUD:
                st.info("⚠️ MediaPipe在Streamlit Cloud上不可用（缺少OpenGL库），应用将以演示模式运行。")
    except Exception as init_e:
        logger.warning(f"HandLandmarkDetector初始化测试失败: {str(init_e)}")
except (ImportError, OSError, RuntimeError) as e:
    error_msg = str(e)
    logger.error(f"导入HandLandmarkDetector失败: {error_msg}")
    
    # 为不同环境提供适当的错误信息
    if st is not None:
        # 检查错误类型并提供相应的解决方案
        if 'libGL.so.1' in error_msg or 'GL' in error_msg.upper():
            if IN_STREAMLIT_CLOUD:
                st.warning("在Streamlit Cloud环境中检测到OpenGL依赖问题，这是已知的限制。")
                st.info("在Streamlit Cloud上，我们使用了替代方案以避免此错误。虽然识别功能不可用，但应用可以正常启动。")
                st.success("应用将使用替代模式继续运行，您仍可以查看界面和了解功能。")
            elif IS_MACOS:
                st.warning("在macOS上检测到mediapipe依赖问题，这是已知的兼容性问题。")
                st.info("您可以尝试安装额外的依赖来解决此问题:")
                st.code("# 方法1: 安装特定版本的mediapipe\npip install mediapipe-silicon")
                st.code("# 方法2: 使用conda安装\nconda install -c menpo opencv")
            else:
                st.warning(f"无法加载HandLandmarkDetector，缺少必要的系统库: {error_msg}")
                st.info('建议在Linux系统上运行: `apt-get update && apt-get install -y libgl1-mesa-glx`')
                st.info('或使用无GUI版本: `pip install opencv-python-headless`')
        else:
            st.warning(f"无法加载HandLandmarkDetector: {error_msg}")
            st.info("这可能是因为缺少必要的依赖或环境配置问题。")
            st.info("请确保requirements.txt中的所有依赖已正确安装。")
    
    # 创建一个功能更完善的替代类
    class DummyHandLandmarkDetector:
        def __init__(self):
            # 初始化时记录信息
            logger.info("使用DummyHandLandmarkDetector替代类")
            self.fake_landmarks = None  # 模拟手部关键点
            
        def detect(self, image):
            # 返回None和原图，确保与主代码逻辑兼容
            return None, image if image is not None else None
        
        def extract_features(self, landmarks):
            # 确保返回有效的numpy数组
            if np is not None:
                return np.zeros(63)  # 返回一个零向量作为特征
            return []  # 返回空列表作为后备
        
        def get_landmarks(self, image):
            # 模拟获取关键点
            return self.fake_landmarks
        
        def draw_landmarks(self, image, landmarks=None, connections=True):
            # 如果没有提供图像，返回None
            if image is None:
                return None
            # 返回原始图像（不在替代类中绘制）
            return image.copy()
    
    HandLandmarkDetector = DummyHandLandmarkDetector
    if st is not None:
        st.info("已创建HandLandmarkDetector替代类，部分功能可能受限")

logger.info("自定义模块导入尝试完成")

# 设置页面配置
# 检查st是否已成功导入
if st is not None:
    try:
        st.set_page_config(
            page_title="手语字母识别",
            page_icon="👋",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        logger.info("成功设置页面配置")
    except Exception as e:
        logger.error(f"设置页面配置失败: {str(e)}")
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
    st.header("输入方式")
    
    # 如果OpenCV不可用，只显示文件上传选项
    if cv2 is None:
        st.warning("⚠️ 摄像头功能不可用，请使用图片上传功能")
        uploaded_file = st.file_uploader(
            "上传手语图片进行识别",
            type=['png', 'jpg', 'jpeg'],
            help="支持 PNG、JPG、JPEG 格式的图片"
        )
        start_button = uploaded_file is not None
        stop_button = False
        video_placeholder = st.empty()
    else:
        # 使用标签页让用户选择输入方式
        input_tab1, input_tab2 = st.tabs(["📷 摄像头", "📁 上传图片"])
        
        with input_tab1:
            st.subheader("摄像头预览")
            video_placeholder = st.empty()
            # 控制面板
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                start_button = st.button("开始识别", type="primary", key="start_camera")
            with control_col2:
                stop_button = st.button("停止识别", key="stop_camera")
            uploaded_file = None
        
        with input_tab2:
            st.subheader("图片上传")
            uploaded_file = st.file_uploader(
                "上传手语图片进行识别",
                type=['png', 'jpg', 'jpeg'],
                help="支持 PNG、JPG、JPEG 格式的图片",
                key="image_upload"
            )
            if uploaded_file is not None:
                start_button = True
                stop_button = False
            else:
                start_button = False
                stop_button = False
            video_placeholder = st.empty()

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
    
    # 如果使用图片上传，OpenCV不是必需的（但最好有）
    # 如果使用摄像头，OpenCV是必需的
    use_uploaded_image = uploaded_file is not None if 'uploaded_file' in locals() else False
    
    if not use_uploaded_image and cv2 is None:
        with result_placeholder.container():
            st.error("❌ OpenCV库未成功加载，无法使用摄像头功能。")
            st.markdown("---")
            if IN_STREAMLIT_CLOUD:
                st.warning("⚠️ 在Streamlit Cloud上的限制")
                st.info("Streamlit Cloud环境通常不支持摄像头访问和OpenGL库，这是平台的安全限制。")
                st.markdown("**💡 解决方案：**")
                st.info("1. 使用图片上传功能（上方标签页）")
                st.info("2. 将应用克隆到本地环境运行")
                st.code("git clone 您的仓库URL\ncd 仓库目录\npip install -r requirements.txt\nstreamlit run streamlit_app.py", language="bash")
            else:
                st.warning("⚠️ OpenCV安装问题")
                st.markdown("**💡 解决方案：**")
                st.info("**方法1：安装无GUI版本（推荐）**")
                st.code("pip install opencv-python-headless", language="bash")
                st.info("**方法2：安装标准版本**")
                st.code("pip install opencv-python", language="bash")
                st.info("**方法3：如果缺少系统库（Linux），安装OpenGL库**")
                st.code("sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx", language="bash")
                st.markdown("---")
                st.info("安装后请刷新页面或重启应用。")
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
        
        # 检查是否使用上传的图片
        if 'uploaded_file' in locals() and uploaded_file is not None:
            # 处理上传的图片
            try:
                # 读取上传的图片
                import io
                from PIL import Image
                
                # 将上传的文件转换为numpy数组
                image_bytes = uploaded_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # 转换为numpy数组（RGB格式）
                import numpy as np
                frame_rgb = np.array(pil_image)
                
                # 如果是RGBA，转换为RGB
                if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:
                    # 使用PIL转换RGBA到RGB
                    pil_image = pil_image.convert('RGB')
                    frame_rgb = np.array(pil_image)
                
                # 转换为BGR格式（OpenCV使用BGR，但如果OpenCV不可用，使用RGB）
                if cv2 is not None:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                else:
                    frame = frame_rgb
                
                # 显示原始图片
                video_placeholder.image(pil_image, caption="上传的图片", use_column_width=True)
                
                # 检测手部关键点（需要OpenCV或MediaPipe，但我们已经处理了这些情况）
                try:
                    landmarks, annotated_frame = detector.detect(frame)
                except Exception as detect_e:
                    logger.error(f"手部检测过程出错: {str(detect_e)}")
                    landmarks = None
                    annotated_frame = frame_rgb if cv2 is None else frame
                
                # 识别手势
                prediction = None
                confidence = 0.0
                
                # 检查检测器是否可用
                detector_available = hasattr(detector, 'available') and detector.available
                
                if not detector_available:
                    with result_placeholder.container():
                        st.warning("⚠️ 手部检测功能不可用")
                        st.info("MediaPipe 或 OpenCV 未正确加载，无法进行手部检测。")
                        if IN_STREAMLIT_CLOUD:
                            st.info("💡 在 Streamlit Cloud 上，这是已知的限制。")
                            st.info("建议在本地环境运行以获得完整功能。")
                        else:
                            st.info("💡 请检查依赖是否正确安装：")
                            st.code("pip install opencv-python-headless mediapipe", language="bash")
                elif landmarks is not None:
                    # 提取特征
                    try:
                        features = detector.extract_features(landmarks)
                        
                        if features is not None:
                            # 预测手势
                            prediction, confidence = classifier.predict(features)
                    except Exception as predict_e:
                        logger.error(f"特征提取或预测过程出错: {str(predict_e)}")
                        prediction = None
                        confidence = 0.0
                
                # 显示结果
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
                        
                        st.markdown(f'<p class="result-text" style="color: {color}; font-size: 24px; font-weight: bold;">识别结果: {prediction}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="confidence-text">置信度: {confidence:.1%}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif detector_available:
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.markdown('<p class="result-text" style="color: #7f8c8d;">未检测到手势</p>', unsafe_allow_html=True)
                        st.markdown('<p>请确保图片中包含清晰的手部手势</p>', unsafe_allow_html=True)
                        st.markdown('<p>💡 提示：确保手部清晰可见，背景简洁，光照充足</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # 如果检测到关键点且OpenCV可用，显示标注后的图片
                if landmarks is not None:
                    if cv2 is not None and annotated_frame is not None:
                        try:
                            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(annotated_rgb, caption="检测结果（已标注关键点）", use_column_width=True)
                        except Exception as e:
                            logger.warning(f"无法转换标注图片: {str(e)}")
                            video_placeholder.image(pil_image, caption="上传的图片（检测成功但无法显示标注）", use_column_width=True)
                    else:
                        # 即使OpenCV不可用，也显示检测成功的消息
                        st.success("✅ 成功检测到手部关键点！")
                        if cv2 is None:
                            st.info("💡 由于 OpenCV 不可用，无法显示关键点标注。")
                
            except Exception as e:
                logger.error(f"处理上传图片时出错: {str(e)}")
                import traceback
                error_details = traceback.format_exc()
                logger.debug(error_details)
                with result_placeholder.container():
                    st.error(f"处理图片时出错: {str(e)}")
                    st.info("请确保上传的是有效的图片文件（PNG、JPG、JPEG格式）")
                    if IN_STREAMLIT_CLOUD:
                        st.info("💡 提示：在Streamlit Cloud上，某些功能可能受限。")
            st.stop()
        
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
                    
                    try:
                        if landmarks is not None:
                            # 提取特征
                            features = detector.extract_features(landmarks)
                            
                            if features is not None:
                                try:
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
                                        except Exception as inner_e:
                                            # 简单地使用最新的预测结果
                                            prediction = prediction_history[-1]
                                            confidence = 0.7
                                except Exception as e:
                                    logger.error(f"预测过程中出错: {str(e)}")
                                    st.warning(f'预测过程中出错: {str(e)}')
                                    prediction = '无法识别'
                                    confidence = 0.0
                        elif hasattr(detector, 'extract_features'):
                            # 对于替代实现，也尝试进行预测
                            try:
                                features = detector.extract_features(None)
                                if features is not None and len(features) > 0:
                                    prediction, confidence = classifier.predict(features)
                                else:
                                    prediction = '未检测到手势'
                                    confidence = 0.0
                            except Exception as e:
                                prediction = '无法识别'
                                confidence = 0.0
                    except Exception as e:
                        logger.error(f"手势识别出错: {str(e)}")
                        prediction = '处理错误'
                        confidence = 0.0
                    
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
            except Exception:
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

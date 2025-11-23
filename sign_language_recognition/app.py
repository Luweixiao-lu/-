"""
手语识别Web应用
基于Streamlit构建的Web界面
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
from hand_landmarks import HandLandmarkDetector
from gesture_classifier import GestureClassifier
import os

# 页面配置
st.set_page_config(
    page_title="手语识别系统",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 现代化UI设计
st.markdown("""
<style>
    /* 全局样式 */
    :root {
        --primary-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --secondary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --success-color: #00bf63;
        --danger-color: #ff4757;
        --warning-color: #ffa502;
        --info-color: #0984e3;
        --light-bg: #f8f9fa;
        --dark-bg: #2d3436;
        --card-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        --transition: all 0.3s ease;
    }
    
    /* 主标题样式 */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 1.5rem 0;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        position: relative;
        z-index: 1;
    }
    
    /* 预测框样式 - 现代卡片设计 */
    .prediction-box {
        background: var(--secondary-gradient);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: var(--card-shadow);
        transition: var(--transition);
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .prediction-text {
        font-size: 5rem;
        font-weight: 900;
        margin: 0.5rem 0;
        letter-spacing: -2px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .confidence-text {
        font-size: 1.5rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* 按钮样式 - 现代化设计 */
    .stButton > button {
        width: 100%;
        background: var(--primary-gradient);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: var(--transition);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
        transition: var(--transition);
        z-index: -1;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* 状态框样式 */
    .stInfo {
        background-color: #e3f2fd;
        border-left: 4px solid var(--info-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #ffebee;
        border-left: 4px solid var(--danger-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fff3e0;
        border-left: 4px solid var(--warning-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 4px solid var(--success-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: var(--light-bg);
        border-right: 1px solid #e0e0e0;
    }
    
    /* 标题样式 */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700;
        color: var(--dark-bg);
    }
    
    /* 卡片样式 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 1.5rem;
        transition: var(--transition);
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: var(--primary-gradient);
        border-radius: 5px;
        height: 10px;
    }
    
    /* 响应式设计 */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .prediction-text {
            font-size: 3rem;
        }
        
        .prediction-box {
            padding: 1.5rem;
        }
        
        .stButton > button {
            font-size: 1rem;
            padding: 0.6rem 1.2rem;
        }
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
    
    /* 页面加载动画 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stApp > header {
        animation: fadeIn 0.5s ease-out;
    }
    
    .stApp > main {
        animation: fadeIn 0.5s ease-out 0.1s both;
    }
</style>
""", unsafe_allow_html=True)

# 初始化Session State
if 'recognizer' not in st.session_state:
    st.session_state.recognizer = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

def find_model_file():
    """查找模型文件，尝试多个可能的位置"""
    possible_paths = [
        Path('gesture_model.pkl'),  # 当前工作目录
        Path(__file__).parent / 'gesture_model.pkl',  # app.py所在目录
        Path.cwd() / 'gesture_model.pkl',  # 当前工作目录（明确）
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

def initialize_components():
    """初始化检测器和分类器"""
    if st.session_state.detector is None:
        with st.spinner("正在初始化手部检测器..."):
            st.session_state.detector = HandLandmarkDetector()
    
    if st.session_state.classifier is None:
        with st.spinner("正在加载手势分类模型..."):
            # 尝试查找模型文件
            model_path = find_model_file()
            if model_path:
                st.session_state.classifier = GestureClassifier(model_path=model_path)
            else:
                st.session_state.classifier = GestureClassifier()
            
            if st.session_state.classifier.model is None:
                st.error("⚠️ 模型文件不存在！")
                st.info("💡 提示：请确保 `gesture_model.pkl` 已上传到GitHub仓库")
                # 显示调试信息
                with st.expander("🔍 调试信息"):
                    st.write("尝试查找的路径：")
                    for path in [Path('gesture_model.pkl'), Path(__file__).parent / 'gesture_model.pkl']:
                        exists = path.exists()
                        st.write(f"- `{path}`: {'✅ 存在' if exists else '❌ 不存在'}")
                    st.write(f"当前工作目录: {os.getcwd()}")
                    st.write(f"app.py位置: {Path(__file__).parent}")
                return False
    return True

def smooth_prediction(prediction):
    """平滑预测结果"""
    st.session_state.prediction_history.append(prediction)
    if len(st.session_state.prediction_history) > 5:
        st.session_state.prediction_history.pop(0)
    
    if len(st.session_state.prediction_history) >= 3:
        from collections import Counter
        most_common = Counter(st.session_state.prediction_history).most_common(1)[0]
        return most_common[0], most_common[1] / len(st.session_state.prediction_history)
    
    return prediction, 0.5

def main():
    """主应用函数"""
    # 标题
    st.markdown('<h1 class="main-header">✋ 手语识别系统</h1>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 模型状态检查
        model_path = find_model_file()
        if model_path:
            st.success(f"✅ 模型已找到: {model_path}")
        else:
            st.error("❌ 模型未找到")
            st.info("请确保 `gesture_model.pkl` 已上传到GitHub仓库")
            with st.expander("🔍 查看调试信息"):
                st.write("尝试查找的路径：")
                for path in [Path('gesture_model.pkl'), Path(__file__).parent / 'gesture_model.pkl']:
                    exists = path.exists()
                    st.write(f"- `{path}`: {'✅ 存在' if exists else '❌ 不存在'}")
                st.write(f"当前工作目录: {os.getcwd()}")
                st.write(f"app.py位置: {Path(__file__).parent}")
        
        st.markdown("---")
        
        # 功能选择
        st.subheader("📋 功能")
        page = st.radio(
            "选择功能",
            ["实时识别", "数据收集", "模型训练", "使用说明"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("ℹ️ 关于")
        st.info("""
        **手语识别系统 v1.0**
        
        支持30个手语字母识别：
        - A-Z (26个字母)
        - ZH, CH, SH, NG
        
        基于MediaPipe和机器学习
        """)
    
    # 主内容区域
    if page == "实时识别":
        show_recognition_page()
    elif page == "数据收集":
        show_data_collection_page()
    elif page == "模型训练":
        show_training_page()
    elif page == "使用说明":
        show_instructions_page()

def show_recognition_page():
    """显示实时识别页面"""
    st.header("🎥 实时手语识别")
    
    # 添加使用提示
    st.info("""
    💡 **使用提示**：
    1. 点击"开始识别"按钮启动实时摄像头
    2. 将手放在摄像头前，做出手语字母手势
    3. 系统会实时识别并显示结果
    4. 支持识别30个手语字母（A-Z, ZH, CH, SH, NG）
    5. 点击"停止识别"可关闭摄像头
    """)
    
    if not initialize_components():
        return
    
    # 初始化Session State
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'camera' not in st.session_state:
        st.session_state.camera = None
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    # 创建布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 实时摄像头画面")
        # 创建开始/停止按钮
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 开始识别", key="start_button"):
                st.session_state.running = True
        with col_btn2:
            if st.button("🛑 停止识别", key="stop_button"):
                st.session_state.running = False
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
        
        # 创建图像占位符
        image_placeholder = st.empty()
        
        # 实时视频处理
        if st.session_state.running:
            try:
                # 打开摄像头
                    if st.session_state.camera is None:
                        # 显示详细初始化状态
                        image_placeholder.info("正在尝试打开摄像头...请确保已授予应用摄像头访问权限")
                        print("开始初始化摄像头...")
                        
                        # 根据测试结果，只尝试索引0，并使用AVFOUNDATION后端
                        camera_idx = 0
                        backend = cv2.CAP_AVFOUNDATION  # 优先使用AVFoundation后端
                        camera_opened = False
                        
                        try:
                            print(f"尝试使用后端 {backend} 打开摄像头索引 {camera_idx}")
                            
                            # 创建VideoCapture对象
                            st.session_state.camera = cv2.VideoCapture(camera_idx, backend)
                            
                            # 检查摄像头是否成功打开
                            if st.session_state.camera.isOpened():
                                # 尝试获取一帧来验证
                                ret, test_frame = st.session_state.camera.read()
                                if ret:
                                    # 设置摄像头参数
                                    st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                    st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                    
                                    # 获取实际的摄像头参数
                                    actual_width = st.session_state.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                                    actual_height = st.session_state.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                    actual_fps = st.session_state.camera.get(cv2.CAP_PROP_FPS)
                                    
                                    camera_opened = True
                                    print(f"成功打开摄像头，分辨率: {actual_width}x{actual_height}, FPS: {actual_fps}")
                                    image_placeholder.info(f"摄像头初始化成功! 分辨率: {int(actual_width)}x{int(actual_height)}")
                                    time.sleep(1)  # 给用户时间看到状态
                                else:
                                    # 无法读取帧
                                    print(f"摄像头已打开但无法读取帧")
                                    st.session_state.camera.release()
                                    st.session_state.camera = None
                        except Exception as e:
                            print(f"打开摄像头时出错: {str(e)}")
                            if st.session_state.camera is not None:
                                st.session_state.camera.release()
                                st.session_state.camera = None
                        
                        if not camera_opened:
                            error_msg = "无法打开摄像头，请检查:\n1. 摄像头连接是否正确\n2. 应用是否有摄像头访问权限\n3. 其他程序是否占用了摄像头"
                            print("错误: " + error_msg)
                            image_placeholder.error(error_msg)
                            st.session_state.running = False
                            return
                
                # 显示加载信息
                image_placeholder.info("正在初始化摄像头...")
                
                # 初始化预测结果
                last_prediction = None
                prediction_count = 0
                prediction_history = []
                
                # 主循环 - 使用while而非无限循环，避免Streamlit崩溃
                import threading
                import queue
                
                # 创建队列用于传递帧和结果
                frame_queue = queue.Queue(maxsize=1)
                result_queue = queue.Queue(maxsize=1)
                
                def process_frames():
                    """在单独线程中处理视频帧"""
                    # 确保在访问前检查键是否存在
                    while True:
                        if 'running' not in st.session_state or not st.session_state.running:
                            break
                        if frame_queue.empty():
                            time.sleep(0.01)
                            continue
                        
                        try:
                            frame = frame_queue.get(timeout=0.1)
                            
                            # 检测手部关键点
                            landmarks, annotated_frame = st.session_state.detector.detect(frame)
                            
                            prediction = None
                            confidence = 0.0
                            
                            if landmarks is not None:
                                # 提取特征
                                features = st.session_state.detector.extract_features(landmarks)
                                
                                if features is not None:
                                    # 预测手势
                                    pred, conf = st.session_state.classifier.predict(features)
                                    prediction, confidence = smooth_prediction(pred)
                            
                            # 在图像上绘制结果
                            if prediction:
                                cv2.putText(annotated_frame, f"手势: {prediction}",
                                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                                cv2.putText(annotated_frame, f"置信度: {confidence:.1%}",
                                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            else:
                                cv2.putText(annotated_frame, "未检测到手",
                                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(annotated_frame, "请将手放在摄像头前",
                                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # 转换为RGB
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            # 将结果放入队列
                            if not result_queue.full():
                                result_queue.put((annotated_frame_rgb, prediction, confidence), timeout=0.1)
                        except Exception as e:
                            print(f"处理帧时出错: {str(e)}")
                
                # 启动处理线程
                processing_thread = threading.Thread(target=process_frames)
                processing_thread.daemon = True
                processing_thread.start()
                
                # 主显示循环
                try:
                    frame_count = 0
                    error_count = 0
                    
                    while st.session_state.running:
                        # 读取摄像头帧
                        try:
                            ret, frame = st.session_state.camera.read()
                            frame_count += 1
                            
                            if not ret:
                                error_count += 1
                                print(f"无法读取摄像头帧，计数: {frame_count}, 错误数: {error_count}")
                                
                                # 如果连续多次读取失败，认为摄像头出现问题
                                if error_count >= 5:
                                    image_placeholder.error("摄像头读取持续失败，请检查摄像头连接")
                                    break
                                
                                # 短暂等待后重试
                                time.sleep(0.1)
                                continue
                            else:
                                # 重置错误计数
                                error_count = 0
                                
                                # 如果是第1帧，记录成功信息
                                if frame_count == 1:
                                    print(f"成功读取第一帧，分辨率: {frame.shape[1]}x{frame.shape[0]}")
                        except Exception as e:
                            error_count += 1
                            print(f"读取摄像头帧时发生异常: {str(e)}")
                            if error_count >= 3:
                                image_placeholder.error(f"摄像头读取异常: {str(e)}")
                                break
                            time.sleep(0.1)
                            continue
                        
                        # 水平翻转（镜像效果）
                        frame = cv2.flip(frame, 1)
                        
                        # 将帧放入队列
                        if not frame_queue.full():
                            try:
                                frame_queue.put(frame, timeout=0.1)
                            except queue.Full:
                                print("帧队列已满，丢弃一帧")
                        else:
                            print("帧队列已满，丢弃一帧")
                        
                        # 从结果队列获取处理后的图像
                        if not result_queue.empty():
                            try:
                                annotated_frame_rgb, prediction, confidence = result_queue.get(timeout=0.1)
                                
                                # 更新图像
                                image_placeholder.image(annotated_frame_rgb, channels="RGB")
                            except Exception as e:
                                # 处理可能的队列异常
                                print(f"从结果队列获取数据时出错: {str(e)}")
                                prediction = None
                                confidence = 0
                                
                                # 更新预测历史
                                if prediction:
                                    prediction_history.append(prediction)
                                    if len(prediction_history) > 10:
                                        prediction_history.pop(0)
                                    
                                    # 简单的预测平滑 - 取最近出现最多的预测
                                    from collections import Counter
                                    if len(prediction_history) >= 3:
                                        counter = Counter(prediction_history[-3:])
                                        most_common = counter.most_common(1)[0]
                                        if most_common[1] >= 2:  # 如果至少出现2次
                                            last_prediction = most_common[0]
                                            
                                            # 更新检测历史
                                            if last_prediction not in st.session_state.detection_history[-5:]:
                                                st.session_state.detection_history.append(last_prediction)
                                                if len(st.session_state.detection_history) > 20:
                                                    st.session_state.detection_history = st.session_state.detection_history[-20:]
                        
                        # 添加短暂延迟以避免CPU占用过高
                        time.sleep(0.05)
                        
                except Exception as e:
                    st.error(f"视频处理出错: {str(e)}")
                finally:
                    # 清理资源
                    st.session_state.running = False
                    if st.session_state.camera is not None:
                        st.session_state.camera.release()
                        st.session_state.camera = None
            except Exception as e:
                st.error(f"启动摄像头失败: {str(e)}")
                st.session_state.running = False
                if st.session_state.camera is not None:
                    st.session_state.camera.release()
                    st.session_state.camera = None
        else:
            # 显示默认提示图像
            default_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(default_image, "点击'开始识别'启动摄像头",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            default_image_rgb = cv2.cvtColor(default_image, cv2.COLOR_BGR2RGB)
            image_placeholder.image(default_image_rgb, channels="RGB")
    
    with col2:
        st.subheader("📊 识别结果")
        
        # 显示最新预测结果
        if 'prediction_history' in locals() and last_prediction:
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-text">{last_prediction}</div>
                <div class="confidence-text">置信度: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-text">--</div>
                <div class="confidence-text">等待识别...</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 显示识别历史
        st.subheader("📝 最近识别历史")
        if st.session_state.detection_history:
            for i, pred in enumerate(reversed(st.session_state.detection_history[-10:]), 1):
                st.write(f"{i}. {pred}")
        else:
            st.info("暂无识别记录")
        
        # 清空历史按钮
        if st.button("🧹 清空历史", key="clear_history"):
            st.session_state.detection_history = []
            st.experimental_rerun()

def show_data_collection_page():
    """显示数据收集页面"""
    st.header("📚 数据收集工具")
    st.info("💡 数据收集功能需要在本地运行。请使用命令行工具：`python data_collector.py`")
    
    st.markdown("""
    ### 使用步骤：
    1. 运行数据收集工具：`python data_collector.py`
    2. 按照提示收集每个手势的样本
    3. 每个手势建议收集100个样本
    4. 收集完成后运行训练脚本：`python train_model.py`
    """)

def show_training_page():
    """显示模型训练页面"""
    st.header("🤖 模型训练")
    st.info("💡 模型训练功能需要在本地运行。请使用命令行工具：`python train_model.py`")
    
    st.markdown("""
    ### 训练步骤：
    1. 确保已收集训练数据（运行 `data_collector.py`）
    2. 运行训练脚本：`python train_model.py`
    3. 等待训练完成，模型将保存为 `gesture_model.pkl`
    
    ### 训练参数：
    - 算法：随机森林分类器
    - 树的数量：300
    - 最大深度：20
    - 测试集比例：20%
    """)
    
    # 检查数据文件
    st.subheader("📁 数据文件检查")
    data_dir = "training_data"
    
    col1, col2 = st.columns(2)
    with col1:
        features_exists = os.path.exists(os.path.join(data_dir, "features.npy"))
        if features_exists:
            st.success("✅ features.npy 存在")
        else:
            st.error("❌ features.npy 不存在")
    
    with col2:
        labels_exists = os.path.exists(os.path.join(data_dir, "labels.npy"))
        if labels_exists:
            st.success("✅ labels.npy 存在")
        else:
            st.error("❌ labels.npy 不存在")
    
    model_exists = os.path.exists("gesture_model.pkl")
    if model_exists:
        st.success("✅ 已训练模型存在")
    else:
        st.warning("⚠️ 未找到训练好的模型")

def show_instructions_page():
    """显示使用说明页面"""
    st.header("📖 使用说明")
    
    st.markdown("""
    ## 🎯 如何使用
    
    ### 第一步：允许摄像头访问
    1. 点击"实时识别"页面
    2. 浏览器会请求摄像头权限，请点击"允许"
    3. 确保摄像头正常工作
    
    ### 第二步：开始识别
    1. 将手放在摄像头前
    2. 做出手语字母手势
    3. 系统会自动识别并显示结果
    
    ## 📋 支持的手势
    
    软件支持识别以下30个手语字母：
    - **A-Z**（26个英文字母）
    - **ZH, CH, SH, NG**（4个汉语拼音声母）
    
    ## 💡 使用提示
    
    1. ✅ **光照条件**：在明亮、均匀的光照下使用效果最佳
    2. ✅ **手势清晰**：保持手势清晰，手指完全展开或弯曲
    3. ✅ **保持稳定**：每个手势保持2-3秒，避免快速移动
    4. ✅ **完整显示**：确保手部完全在摄像头视野内
    5. ✅ **背景简洁**：使用简洁的背景，避免干扰
    
    ## 🔧 故障排除
    
    ### 摄像头无法访问
    - ✅ 检查浏览器是否已授权摄像头权限
    - ✅ 确保没有其他程序占用摄像头
    - ✅ 尝试刷新页面或使用Chrome/Firefox浏览器
    - ✅ 检查系统摄像头设置
    
    ### 识别不准确
    - ✅ 确保光照充足且均匀
    - ✅ 保持手势清晰稳定
    - ✅ 确保手部完全在画面中
    - ✅ 尝试调整手与摄像头的距离（约30-50cm）
    
    ### 没有识别结果
    - ✅ 检查是否检测到手部（查看画面中的绿色线条）
    - ✅ 确保手势正确
    - ✅ 尝试重新调整手的位置
    
    ## 🌐 关于此应用
    
    这是一个基于MediaPipe和机器学习的手语识别系统。
    
    - **技术栈**：OpenCV, MediaPipe, scikit-learn, Streamlit
    - **识别算法**：随机森林分类器
    - **手部检测**：MediaPipe Hands
    
    ## 📞 获取帮助
    
    如果遇到问题，请：
    1. 查看本页面的故障排除部分
    2. 检查浏览器控制台是否有错误信息
    3. 尝试刷新页面重新加载
    """)
    
    # 添加部署信息（如果是云端部署）
    st.markdown("---")
    st.info("""
    💡 **提示**：此应用已部署到云端，任何人都可以通过链接访问使用。
    无需安装任何软件，只需要浏览器和摄像头即可！
    """)

if __name__ == "__main__":
    main()


"""
手语识别Web应用
基于Streamlit构建的Web界面
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
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

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .prediction-text {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
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

def initialize_components():
    """初始化检测器和分类器"""
    if st.session_state.detector is None:
        with st.spinner("正在初始化手部检测器..."):
            st.session_state.detector = HandLandmarkDetector()
    
    if st.session_state.classifier is None:
        with st.spinner("正在加载手势分类模型..."):
            st.session_state.classifier = GestureClassifier()
            if st.session_state.classifier.model is None:
                st.error("⚠️ 模型文件不存在！请先训练模型。")
                st.info("💡 提示：运行 `python train_model.py` 来训练模型")
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
        model_exists = os.path.exists('gesture_model.pkl')
        if model_exists:
            st.success("✅ 模型已加载")
        else:
            st.error("❌ 模型未找到")
            st.info("请先运行 `python train_model.py` 训练模型")
        
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
    """显示识别页面"""
    st.header("🎥 实时手语识别")
    
    # 添加使用提示
    st.info("""
    💡 **使用提示**：
    1. 点击下方的摄像头按钮，允许浏览器访问摄像头
    2. 将手放在摄像头前，做出手语字母手势
    3. 系统会自动识别并显示结果
    4. 支持识别30个手语字母（A-Z, ZH, CH, SH, NG）
    """)
    
    if not initialize_components():
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 摄像头画面")
        
        # 使用Streamlit的相机输入组件
        camera_input = st.camera_input(
            "请将手放在摄像头前，然后做出手势",
            key="recognition_camera",
            help="点击此按钮允许浏览器访问您的摄像头"
        )
        
        if camera_input is not None:
            # 将PIL图像转换为numpy数组
            img_array = np.array(camera_input)
            # PIL图像是RGB格式，转换为BGR供OpenCV使用
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)
            
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
                cv2.putText(annotated_frame, f"Gesture: {prediction}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(annotated_frame, f"Confidence: {confidence:.1%}",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(annotated_frame, "No hand detected",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(annotated_frame, "Please show your hand",
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 转换为RGB并显示
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
    
    with col2:
        st.subheader("📊 识别结果")
        
        if camera_input is not None:
            # 重新处理以获取最新结果
            img_array = np.array(camera_input)
            frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
            landmarks, _ = st.session_state.detector.detect(frame)
            
            prediction = None
            confidence = 0.0
            
            if landmarks is not None:
                features = st.session_state.detector.extract_features(landmarks)
                if features is not None:
                    pred, conf = st.session_state.classifier.predict(features)
                    prediction, confidence = smooth_prediction(pred)
            
            # 显示结果
            if prediction:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-text">{prediction}</div>
                    <div class="confidence-text">置信度: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # 显示置信度条
                st.progress(confidence)
            else:
                st.info("👋 请将手放在摄像头前")
                st.warning("确保手部完全在画面中")
        
        # 显示最近识别历史
        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown("**最近识别历史：**")
            recent = st.session_state.prediction_history[-10:]
            # 显示为列表
            for i, pred in enumerate(reversed(recent[-5:]), 1):
                st.write(f"{i}. {pred}")
        
        # 清空历史按钮
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()

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


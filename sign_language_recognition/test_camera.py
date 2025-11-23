import cv2
import time

print("OpenCV版本:", cv2.__version__)
print("开始测试摄像头功能...")

# 尝试不同的摄像头索引和后端
camera_indices = [0, 1, 2]
backends = [
    (cv2.CAP_ANY, "CAP_ANY"),
    (cv2.CAP_V4L2, "CAP_V4L2"),
    (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION")
]

for backend_id, backend_name in backends:
    print(f"\n尝试使用后端: {backend_name} ({backend_id})")
    
    for idx in camera_indices:
        print(f"  尝试打开摄像头索引: {idx}")
        
        try:
            # 创建VideoCapture对象
            cap = cv2.VideoCapture(idx, backend_id)
            
            # 检查是否成功打开
            if cap.isOpened():
                print(f"  ✓ 成功打开摄像头 {idx}")
                
                # 获取摄像头属性
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  摄像头属性 - 分辨率: {width}x{height}, FPS: {fps}")
                
                # 尝试读取一帧
                ret, frame = cap.read()
                if ret:
                    print(f"  ✓ 成功读取一帧，帧大小: {frame.shape[1]}x{frame.shape[0]}")
                    # 设置摄像头参数
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    # 再次检查参数
                    new_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    new_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    print(f"  设置后的分辨率: {new_width}x{new_height}")
                    # 再次读取帧验证
                    ret, frame2 = cap.read()
                    if ret:
                        print(f"  ✓ 再次成功读取帧，新帧大小: {frame2.shape[1]}x{frame2.shape[0]}")
                else:
                    print(f"  ✗ 无法读取帧")
                
                # 释放摄像头
                cap.release()
            else:
                print(f"  ✗ 无法打开摄像头 {idx}")
                cap.release()
                
        except Exception as e:
            print(f"  ✗ 打开摄像头时出错: {str(e)}")

print("\n摄像头测试完成")
print("\n故障排除建议:")
print("1. 检查摄像头是否被其他程序占用")
print("2. 确认系统权限设置")
print("3. 尝试重启计算机")
print("4. 检查OpenCV版本是否兼容")
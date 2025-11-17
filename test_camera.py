# -*- coding: utf-8 -*-
"""
摄像头诊断脚本 - 测试摄像头是否能正常工作
"""
import cv2
import sys

def test_camera():
    """测试摄像头"""
    print("=" * 50)
    print("摄像头诊断测试")
    print("=" * 50)

    # 尝试不同的摄像头索引
    for camera_index in [0, 1, 2]:
        print(f"\n测试摄像头索引 {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

        if cap.isOpened():
            print(f"✓ 摄像头 {camera_index} 打开成功")

            # 尝试读取一帧
            ret, frame = cap.read()
            if ret:
                print(f"✓ 成功读取帧")
                print(f"  - 帧尺寸: {frame.shape}")
                print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                print(f"  - 宽度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
                print(f"  - 高度: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

                # 显示5秒钟
                print("\n显示摄像头画面5秒钟(按'q'退出)...")
                import time
                start_time = time.time()
                frame_count = 0

                while time.time() - start_time < 5:
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        cv2.imshow(f'Camera {camera_index} Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                cv2.destroyAllWindows()
                print(f"✓ 成功显示 {frame_count} 帧")
                cap.release()

                print(f"\n推荐使用摄像头索引: {camera_index}")
                return camera_index
            else:
                print(f"✗ 无法读取帧")
                cap.release()
        else:
            print(f"✗ 摄像头 {camera_index} 打开失败")

    print("\n" + "=" * 50)
    print("未找到可用的摄像头!")
    print("请检查:")
    print("1. 摄像头是否已连接")
    print("2. 摄像头驱动是否已安装")
    print("3. 其他程序是否占用了摄像头")
    print("4. 系统权限设置")
    print("=" * 50)
    return None

if __name__ == '__main__':
    result = test_camera()
    if result is not None:
        print(f"\n✓ 摄像头测试成功! 使用索引 {result}")
        sys.exit(0)
    else:
        print("\n✗ 摄像头测试失败!")
        sys.exit(1)

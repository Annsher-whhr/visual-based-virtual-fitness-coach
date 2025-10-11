import cv2
import mediapipe as mp

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)
#
# cap = cv2.VideoCapture(0)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # 转换图像颜色空间
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image.flags.writeable = False
#
#     # 姿态检测
#     results = pose.process(image)
#
#     # 在原图上绘制姿态注释
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
#     cv2.imshow('MediaPipe Pose', image)
#     if cv2.waitKey(5) & 0xFF == 27:  # 按ESC键退出
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# 初始化MediaPipe Pose组件。
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


def estimate_pose(frame):
    """
    使用MediaPipe进行姿态估计。

    参数:
    - frame: 输入的图像帧

    返回:
    - frame: 经过姿态估计绘制了姿态关键点的图像帧
    """

    # 转换颜色空间从BGR到RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 设置图像为不可写，以提高性能
    frame_rgb.flags.writeable = False

    # 进行姿态估计
    results = pose.process(frame_rgb)

    # 设置图像为可写，以便在其上绘制关键点
    frame_rgb.flags.writeable = True

    # 转换颜色空间回BGR，以便与OpenCV兼容
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # 绘制姿态关键点（但去掉脸部关键点）
    mp_drawing = mp.solutions.drawing_utils

    if results.pose_landmarks:
        # 定义被认为是“面部”的关键点索引集合（MediaPipe Pose 的部分索引）
        face_indices = {
            mp_pose.PoseLandmark.NOSE.value,
            mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
            mp_pose.PoseLandmark.LEFT_EYE.value,
            mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
            mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
            mp_pose.PoseLandmark.RIGHT_EYE.value,
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
            mp_pose.PoseLandmark.LEFT_EAR.value,
            mp_pose.PoseLandmark.RIGHT_EAR.value,
            mp_pose.PoseLandmark.MOUTH_LEFT.value,
            mp_pose.PoseLandmark.MOUTH_RIGHT.value,
        }

        h, w, _ = frame.shape

        # 手动绘制连接：仅绘制连接两端都不是面部关键点的连线
        for connection in mp_pose.POSE_CONNECTIONS:
            # connection 可能包含 enum 或整数
            a = connection[0].value if hasattr(connection[0], 'value') else connection[0]
            b = connection[1].value if hasattr(connection[1], 'value') else connection[1]
            if a in face_indices or b in face_indices:
                continue
            lm_a = results.pose_landmarks.landmark[a]
            lm_b = results.pose_landmarks.landmark[b]
            # 仅在两点都有检测到合理坐标时绘制
            if (0 <= lm_a.x <= 1 and 0 <= lm_a.y <= 1) and (0 <= lm_b.x <= 1 and 0 <= lm_b.y <= 1):
                ax, ay = int(lm_a.x * w), int(lm_a.y * h)
                bx, by = int(lm_b.x * w), int(lm_b.y * h)
                cv2.line(frame, (ax, ay), (bx, by), (0, 255, 0), 2)

        # 绘制非面部关键点（圆点）
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in face_indices:
                continue
            if 0 <= lm.x <= 1 and 0 <= lm.y <= 1:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    return frame

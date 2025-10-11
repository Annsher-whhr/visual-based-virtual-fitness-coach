import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def resize_frame(frame, target_width=720):
    """
    调整帧的大小，保持纵横比不变。

    参数:
    - frame: 输入的图像帧
    - target_width: 目标宽度

    返回:
    - resized_frame: 调整大小后的图像帧
    """
    height, width = frame.shape[:2]
    scale = target_width / width
    target_height = int(height * scale)
    resized_frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized_frame


def draw_text(frame, text, position=(20, 30), font_scale=1, color=(0, 255, 0), thickness=2):
    """
    在图像帧上绘制文本。

    参数:
    - frame: 输入的图像帧
    - text: 要绘制的文本字符串
    - position: 文本在图像上的位置
    - font_scale: 字体大小
    - color: 文本颜色
    - thickness: 文本线条粗细

    返回:
    - frame: 绘制了文本的图像帧
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame


def convert_to_grayscale(frame):
    """
    将图像帧转换为灰度图。

    参数:
    - frame: 输入的彩色图像帧

    返回:
    - grayscale_frame: 转换为灰度的图像帧
    """
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return grayscale_frame


def overlay_image(background, overlay, position=(0, 0), opacity=1):
    """
    将一个图像覆盖到另一个图像上。

    参数:
    - background: 背景图像
    - overlay: 覆盖图像
    - position: 覆盖图像在背景图像上的位置
    - opacity: 覆盖图像的不透明度

    返回:
    - output: 合成的图像
    """
    x, y = position
    overlay_height, overlay_width = overlay.shape[:2]
    background[y:y + overlay_height, x:x + overlay_width] = cv2.addWeighted(
        background[y:y + overlay_height, x:x + overlay_width], 1 - opacity, overlay, opacity, 0)
    return background

def draw_chinese_text(img, text, position, font_size, color):
    """
    在OpenCV图像上绘制支持中文的文本。

    :param img: opencv格式的图像 (numpy array)
    :param text: 要绘制的文本字符串
    :param position: 文本左上角的坐标 (x, y)
    :param font_size: 字体大小
    :param color: 文本颜色 (B, G, R)
    :return: 绘制了文本的图像 (numpy array)
    """
    # 将OpenCV图像格式转换为PIL图像格式
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个绘制对象
    draw = ImageDraw.Draw(img_pil)

    # 加载字体文件（请确保字体文件与脚本在同一目录下或提供完整路径）
    try:
        font = ImageFont.truetype("SourceHanSansSC-Regular.otf", font_size, encoding="utf-8")
    except IOError:
        print("字体文件'SourceHanSansSC-Regular.otf'未找到，请检查路径。")
        return img # 返回原图

    # 在图像上绘制文本
    draw.text(position, text, font=font, fill=color[::-1]) # PIL使用RGB颜色，所以需要反转BGR

    # 将PIL图像格式转换回OpenCV图像格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img
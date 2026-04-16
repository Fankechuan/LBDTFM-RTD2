import cv2
import numpy as np

def draw_boxes_on_image(image, boxes, labels, label_map):
    """
    在RGB图像上绘制边框和标签

    参数：
        image (np.ndarray): RGB格式，形状(H,W,3)，uint8
        boxes (list of list): [[xc,yc,w,h], ...]，归一化坐标（0~1）
        labels (list of int): 类别索引列表
        label_map (dict): 类别id到名称映射

    返回：
        np.ndarray: 绘制好框和标签的RGB图像
    """

    img_h, img_w = image.shape[:2]

    # 预定义颜色（BGR格式，因为cv2.draw用BGR）
    color_map = {
        0: (0, 0, 255),     # 闪络 - 红色
        1: (0, 255, 255),   # 破损 - 黄色
        2: (0, 255, 0),     # 正常 - 绿色
    }

    image = image.copy()

    for box, label in zip(boxes, labels):
        # 归一化转像素坐标
        xc, yc, w, h = box
        x1 = int((xc - w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w)
        y2 = int((yc + h/2) * img_h)

        # 防止越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        color = color_map.get(label, (255, 255, 255))
        label_text = label_map.get(label, str(label))

        # 画矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 画标签背景
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)

        # 写文字（白色）
        cv2.putText(image, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return image

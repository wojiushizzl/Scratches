import os
from glob import glob
import cv2
import numpy as np

def process_image_auto_adapt(image_path):
    """处理单张图像，自动适应边缘面积阈值"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 使用Sobel算子提取边缘
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    # Otsu阈值分割
    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学闭运算（细长结构，横向为主）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 去除小区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    min_area = 50
    mask = np.zeros_like(morph)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255


    return mask


# 示例：处理一个文件夹中的所有图像
def batch_process_images(input_folder, output_folder, view_folder, image_exts=('*.jpg', '*.png', '*.bmp')):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(view_folder, exist_ok=True)
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    for path in image_paths:
        mask = process_image_auto_adapt(path)
        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, f"mask_{filename}")
        cv2.imwrite(save_path, mask)

        # 原图和mask图合并为一张图，利用mask高亮划痕区域，保存到view文件夹
        img = cv2.imread(path)
        # img = cv2.resize(img, (512, 512))
        # mask = cv2.resize(mask, (512, 512))
        # 将mask转换为3通道图像
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # 将mask_3channel中划痕区域设置为红色
        mask_3channel[mask == 255] = [0, 0, 255]
        # 将mask_3channel和img合并
        merged_img = cv2.addWeighted(img, 0.5, mask_3channel, 0.5, 0)

        cv2.imwrite(os.path.join(view_folder, f"view_{filename}"), merged_img)

    return len(image_paths)









# 示例调用代码（路径可替换）
processed_count = batch_process_images("./dataset/images", "./dataset/masks", "./dataset/view")
print(f"Processed {processed_count} images.")


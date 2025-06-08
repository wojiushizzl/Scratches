import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image_path = "C:/Users/ZZL/OneDrive/桌面/ZZL_projects/Scratches/Scratches/dataset/images/Image_20250510161943245.bmp"
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
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

# 去除小区域
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
min_area = 50
mask = np.zeros_like(morph)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        mask[labels == i] = 255

# 可视化原图和mask
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(sobel, cmap='gray')
axes[1].set_title("Sobel Edge")
axes[1].axis("off")

axes[2].imshow(mask, cmap='gray')
axes[2].set_title("Scratch Mask")
axes[2].axis("off")

plt.tight_layout()
plt.show()

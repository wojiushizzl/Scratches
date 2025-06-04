import cv2
import numpy as np
import os

# ==== 路径设置 ====
input_dir = './dataset/images'      # 原图路径（如含裂缝的图片）
output_dir = './dataset/masks'      # 输出的 mask 图路径（黑底白斑）

os.makedirs(output_dir, exist_ok=True)

# ==== 处理所有图像 ====
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
        continue

    # 读取图像
    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to load image: {img_path}")
        continue

    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==== 阈值选择：提取“暗区域”（裂缝、划痕等） ====
    # 可根据图像实际调整 upper 值（如 80 ~ 120）
    lower = 0
    upper = 150
    mask = cv2.inRange(gray, lower, upper)  # 黑色/灰色变白，其它变黑

    # # ==== 可选：形态学处理，去除小噪声 ====
    # kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # ==== 保存 mask 图 ====
    out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + '.png')
    cv2.imwrite(out_path, mask)

    print(f"[✔] Saved mask: {out_path}")

print("\n✅ 所有图像处理完成。")

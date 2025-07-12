# 将 dataset/videos/images1 的图片批量转换为灰度图并保存到 dataset/videos/images11

import os
import cv2

input_folder = "dataset/videos/images1"
output_folder = "dataset/videos/images11"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.png', '.bmp')):
        img = cv2.imread(os.path.join(input_folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, filename), gray)
print(f"Converted images saved to {output_folder}")
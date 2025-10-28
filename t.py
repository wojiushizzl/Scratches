import cv2
import numpy as np




import os
import cv2
import numpy as np
import time

# 读取原图
script_dir = os.path.dirname(os.path.abspath(__file__))
name_src = os.path.join(script_dir, "d.jpeg")
# name_src = os.path.join(script_dir, "temp_frame_53.jpg")
image_src = cv2.imread(name_src, cv2.IMREAD_COLOR)
cv2.imshow("16.18(a) - Original Image", image_src)
cv2.waitKey(0)

# 阈值分割
_, image_dst = cv2.threshold(image_src, 110, 255, cv2.THRESH_BINARY)
cv2.imshow("16.18(b) - Binarize", image_dst)
cv2.waitKey(0)
# 读取阴影图像
image_shadow = cv2.GaussianBlur(image_src, (181, 181), 0)
cv2.imshow("16.18(c) - Shadow Image", image_shadow)
cv2.waitKey(0)
# 图像减法消除阴影 (R_SUB + R_SATURATION)
image_dst = cv2.subtract(image_src, image_shadow)
cv2.imshow("16.18(d) - Shadow Removed", image_dst)
cv2.waitKey(0)
# 确保输入图像为灰度图像
if len(image_dst.shape) == 3:  # 检查是否为多通道图像
    image_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2GRAY)
# 自适应二值化
image_dst = cv2.adaptiveThreshold(image_dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("16.18(e) - Adaptive Threshold", image_dst)
cv2.waitKey(0)
# 高斯平滑
start_time = time.time()
image_shadow_gauss = cv2.GaussianBlur(image_src, (181, 31), 0)
print("Gaussian Blur time:", time.time() - start_time)
cv2.imshow("16.18(f) - Gaussian Blur", image_shadow_gauss)
cv2.waitKey(0)
# 图像减法消除阴影 (R_SUB + R_SATURATION)
image_dst = cv2.subtract(image_shadow_gauss, image_src)
cv2.imshow("16.18(g) - Shadow Removed After Gaussian", image_dst)
cv2.waitKey(0)

# 取反
image_dst = cv2.bitwise_not(image_dst)
cv2.imshow("16.18(g.1) - Inverted Image", image_dst)
cv2.waitKey(0)


# 确保输入图像为灰度图像
if len(image_dst.shape) == 3:  # 检查是否为多通道图像
    image_dst = cv2.cvtColor(image_dst, cv2.COLOR_BGR2GRAY)
# 再次自适应二值化
image_dst = cv2.adaptiveThreshold(image_dst, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)
cv2.imshow("16.18(h) - Adaptive Threshold After", image_dst)
cv2.waitKey(0)
# 亮度修正 (Rectify)
image_dst = cv2.convertScaleAbs(image_dst, alpha=1.1, beta=0)
cv2.imshow("16.18(i) - Brightness Adjustment", image_dst)
cv2.waitKey(0)



# 释放内存
cv2.waitKey(0)
cv2.destroyAllWindows()
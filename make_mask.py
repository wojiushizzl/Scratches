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

    # 填充小洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 膨胀
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 腐蚀
    mask = cv2.erode(mask, kernel, iterations=1)

    # 去除mask上下的边界，以及链接的裂缝
    height = mask.shape[0]
    
    # 自动找到边界高度
    def find_boundary_height(mask, threshold=0.1, window_size=10):
        """
        自动检测图像上下边界位置
        :param mask: 输入的二值图像
        :param threshold: 判断为边界的阈值（白色像素比例）
        :param window_size: 滑动窗口大小
        :return: 上边界和下边界的位置
        """
        # 计算每行的白色像素比例
        row_ratios = np.sum(mask == 255, axis=1) / mask.shape[1]
        
        # 使用滑动窗口平滑处理
        smoothed_ratios = np.convolve(row_ratios, np.ones(window_size)/window_size, mode='same')
        
        # 从上往下找第一个超过阈值的行
        top_boundary = 0
        for i in range(height):
            if smoothed_ratios[i] > threshold:
                top_boundary = i
                break
        
        # 从下往上找第一个超过阈值的行
        bottom_boundary = height - 1
        for i in range(height-1, -1, -1):
            if smoothed_ratios[i] > threshold:
                bottom_boundary = i
                break
        
        return top_boundary+10, bottom_boundary-10
    
    # 获取边界位置
    top_boundary, bottom_boundary = find_boundary_height(mask)
    print(f"top_boundary: {top_boundary}, bottom_boundary: {bottom_boundary}")  # 打印边界位置
    
    # 去除上下边界
    mask[:top_boundary, :] = 0  # 去除上边界
    mask[bottom_boundary:, :] = 0  # 去除下边界

    # 去除与边界相连的裂缝
    # 标记所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # 创建新的mask
    new_mask = np.zeros_like(mask)
    
    # 遍历所有连通区域
    for i in range(1, num_labels):  # 从1开始，跳过背景
        # 获取当前连通区域的边界框
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # 检查是否与边界相连
        if y <= top_boundary or y + h >= bottom_boundary:
            continue  # 跳过与边界相连的区域
        
        # 保留不与边界相连的区域
        new_mask[labels == i] = 255

    mask = new_mask

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











if __name__ == "__main__":
    task = "batch"

    if task == "batch":
        # 批量处理图片，生成mask和view
        processed_count = batch_process_images("./dataset/images", "./dataset/masks", "./dataset/view")
        print(f"Processed {processed_count} images.")

    elif task == "single":
        # 处理单个图片
        image_path = "./dataset/images/Image_20250510161058460.bmp"
        mask = process_image_auto_adapt(image_path)

        # 可视化结果
        img = cv2.imread(image_path)
        
        # 创建掩码的彩色版本（红色）
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_3channel[mask == 255] = [0, 0, 255]
        
        # 合并原图和掩码
        merged_img = cv2.addWeighted(img, 0.5, mask_3channel, 0.5, 0)
        
        # 调整图像大小以便显示
        scale = 0.5
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
        mask_resized = cv2.resize(mask, None, fx=scale, fy=scale)
        merged_resized = cv2.resize(merged_img, None, fx=scale, fy=scale)
        
        # 水平拼接图像
        combined = np.hstack((img_resized, cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR), merged_resized))
        
        # 显示结果 图像大小为 3072*1024
        combined = cv2.resize(combined, (3072, 1024))
        cv2.imshow('Original | Mask | Overlay', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        output_dir = "./dataset/view"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "visualization_result.jpg"), combined)

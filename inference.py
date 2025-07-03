import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import os
from unet_model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model_path = "best_unet_model.pth"
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def post_process_mask(pred_np_resized, original_image_np, threshold=0.1, min_pixel=1000):
    binary_mask = (pred_np_resized > threshold).astype(np.uint8)
    result = "OK" if np.sum(binary_mask) < min_pixel else "NG"
    overlay = original_image_np.copy()
    overlay[binary_mask == 1] = [255, 0, 0]  # 红色标记
    return binary_mask, overlay, result

def single_inference(image_path, threshold=0.1, min_pixel=1000, show_result=True):
    image_size = (448, 448)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # 加载并预处理图像
    image = Image.open(image_path).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(1.0)
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推理及耗时统计
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    end_time = time.time()
    print(f"[{os.path.basename(image_path)}] 推理耗时: {(end_time - start_time) * 1000:.2f} ms")

    # 还原为原始大小
    pred_np_resized = cv2.resize(pred.squeeze(), original_size, interpolation=cv2.INTER_NEAREST)
    image_np = np.array(image)

    # 后处理
    binary_mask, overlay, result = post_process_mask(pred_np_resized, image_np, threshold, min_pixel)
    print(f"判定结果: {result} (前景像素数: {np.sum(binary_mask)})")

    # 可视化显示
    if show_result:
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(image_np)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred_np_resized, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

def batch_inference(folder_path, threshold=0.1, min_pixel=1000):
    for image_name in sorted(os.listdir(folder_path)):
        if image_name.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, image_name)
            single_inference(image_path, threshold, min_pixel, show_result=False)

def video_inference(video_path, threshold=0.1, min_pixel=1000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频")
        return

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 保存帧图像到临时路径
        temp_path = f"temp_frame_{frame_id}.jpg"
        cv2.imwrite(temp_path, frame)
        print(f"\n[视频帧 {frame_id}]")
        single_inference(temp_path, threshold, min_pixel, show_result=False)
        os.remove(temp_path)
        frame_id += 1

    cap.release()

# === 主程序入口 ===
if __name__ == "__main__":
    # 单张图像推理
    image_path = "dataset/images/Image_20250510161147250.bmp"
    single_inference(image_path)

    # 批量推理
    folder_path = "dataset/images"
    batch_inference(folder_path)

    # 视频推理
    video_path = "dataset/videos/video.mp4"
    video_inference(video_path)

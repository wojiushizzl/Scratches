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

class UNetInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = UNet().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def post_process_mask(self, pred_np_resized, original_image_np, threshold=0.1, min_pixel=1000):
        binary_mask = (pred_np_resized > threshold).astype(np.uint8)
        result = "OK" if np.sum(binary_mask) < min_pixel else "NG"
        overlay = original_image_np.copy()
        overlay[binary_mask == 1] = [255, 0, 0]  # 红色标记
        
        # result 为 OK 或 NG，标注再图片，正方形，边长为 100，字体为白色，OK为绿色正方形，NG为红色正方形
        # 正方形填充红或者绿色，边长为 100，字体为白色，右上角，OK为绿色正方形，NG为红色正方形，字体为白色
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 5
        text_size, _ = cv2.getTextSize(result, font, font_scale, thickness)
        text_x = 10
        text_y = 10
        if result == "OK":
            cv2.rectangle(overlay, (text_x, text_y), (text_x + 100, text_y + 100), (0, 255, 0), -1)
        else:
            cv2.rectangle(overlay, (text_x, text_y), (text_x + 100, text_y + 100), (255, 0, 0), -1)
        cv2.putText(overlay, result, (text_x+10, text_y+60), font, font_scale, (255, 255, 255), thickness)
        return binary_mask, overlay, result
    
    def single_inference(self, image_path, threshold=0.1, min_pixel=1000, show_result=True):
        image_size = (448, 448)

        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        # 加载并预处理图像，image可以是图片路径，也可以是cv2读取的图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
            # # SAVE TEMP IMAGE
            # image.save("temp_image1.jpg")
        else:
            image = image_path.copy()
            image = Image.fromarray(image)
            image = image.convert("RGB")
            # # SAVE TEMP IMAGE
            # image.save("temp_image2.jpg")


        image = ImageEnhance.Brightness(image).enhance(1.0)
        original_size = image.size
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 推理及耗时统计
        start_time = time.time()
        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
        end_time = time.time()
        if isinstance(image_path, str):
            print(f"[{os.path.basename(image_path)}] 推理耗时: {(end_time - start_time) * 1000:.2f} ms")
        else:
            print(f"推理耗时: {(end_time - start_time) * 1000:.2f} ms")

        # 还原为原始大小
        pred_np_resized = cv2.resize(pred.squeeze(), original_size, interpolation=cv2.INTER_NEAREST)
        image_np = np.array(image)

        # 后处理
        binary_mask, overlay, result = self.post_process_mask(pred_np_resized, image_np, threshold, min_pixel)
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
        
        return overlay,result
    

    def batch_inference(self, folder_path, threshold=0.1, min_pixel=1000):
        for image_name in sorted(os.listdir(folder_path)):
            if image_name.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, image_name)
                self.single_inference(image_path, threshold, min_pixel, show_result=False)

    def video_inference(self, video_path, threshold=0.1, min_pixel=1000):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频")
            return

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 视频流显示推理结果
            overlay ,result= self.single_inference(frame, threshold, min_pixel, show_result=False)
            cv2.imshow("Video", overlay)
            cv2.waitKey(1)
            # cv2.imwrite(f"temp_frame_{frame_id}.jpg", overlay)
            frame_id += 1

        cap.release()


if __name__ == "__main__":
    unet_inference = UNetInference("best_unet_model.pth")
    image_path = "dataset/images/Image_20250510161147250.bmp"
    # image_path = "temp_frame_53.jpg"
    unet_inference.single_inference(image_path)

    # 批量推理
    # folder_path = "dataset/images"
    # batch_inference(folder_path)

    # 视频推理
    video_path = "dataset/videos/Video_20250627093322672.avi"
    unet_inference.video_inference(video_path)







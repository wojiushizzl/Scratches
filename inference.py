import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from unet_model import UNet
import cv2
from PIL import ImageEnhance
import time

# TODO
# 批量推理
# 推理速度 20ms/image
# 数据增强训练
# 优化数据集
# 后处理

# 设置
# image_path = "./dataset\images\Image_20250510163304998.bmp"
image_path = "C:/Users/ZZL/OneDrive/桌面/MV-CS050-10GM+DA5818283/Image_20250510162029539.bmp"
model_path = "best_unet_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = (448, 448)

# 预处理
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# 加载图片
image = Image.open(image_path).convert("RGB")
# 增加图片的亮度
image = ImageEnhance.Brightness(image).enhance(1)
original_size = image.size  # (width, height)

input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

# 加载模型
model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 推理
for i in range(100):
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    end_time = time.time()
    print(f"推理时间: {(end_time - start_time)*1000} ms")

# 1. 还原 pred 为原图大小
pred_np = pred.squeeze()  # shape: (H, W)
pred_np_resized = cv2.resize(pred_np, original_size, interpolation=cv2.INTER_NEAREST)

# 2. 二值化（你可能已经有 binary_mask，也可以在这里生成）
binary_mask = (pred_np_resized > 0.5).astype(np.uint8)

# 3. 生成 overlay
image_np = np.array(image)  # shape: (H, W, 3)
overlay = image_np.copy()
overlay[binary_mask == 1] = [255, 0, 0]  # 红色覆盖区域

# 4. 显示三图
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

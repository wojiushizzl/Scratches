import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from tqdm import tqdm

# 数据集类
class CrackDetectionDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(448, 448)):
        self.images = sorted([
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if f.endswith(('.jpg', '.png','.bmp'))
        ])
        self.masks = sorted([
            os.path.join(masks_dir, f) for f in os.listdir(masks_dir)
            if f.endswith(('.jpg', '.png','.bmp'))
        ])
        self.img_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = (mask > 0.5).float()

        return img, mask


# UNet 模型定义
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=32):
        super(UNet, self).__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.encoder1 = block(in_channels, features)
        self.encoder2 = block(features, features*2)
        self.encoder3 = block(features*2, features*4)
        self.encoder4 = block(features*4, features*8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(features*8, features*16)

        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, 2, 2)
        self.decoder4 = block(features*16, features*8)
        self.upconv3 = nn.ConvTranspose2d(features*8, features*4, 2, 2)
        self.decoder3 = block(features*8, features*4)
        self.upconv2 = nn.ConvTranspose2d(features*4, features*2, 2, 2)
        self.decoder2 = block(features*4, features*2)
        self.upconv1 = nn.ConvTranspose2d(features*2, features, 2, 2)
        self.decoder1 = block(features*2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.decoder4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.decoder3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.final_conv(d1)


# Dice 评估指标
def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean()


# 训练主逻辑
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CrackDetectionDataset(
        "./dataset/images/", "./dataset/masks/", image_size=(448, 448)
    )
    train_set, val_set = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 30
    best_dice = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # 验证
        model.eval()
        dice_total = 0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)
                dice_total += dice_score(pred, mask).item()

        avg_loss = total_loss / len(train_loader)
        avg_dice = dice_total / len(val_loader)
        print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), "best_unet_model.pth")
            print(f"🎯 Best model saved. Dice = {best_dice:.4f}")

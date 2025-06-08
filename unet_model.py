import torch
import torch.nn as nn
from collections import OrderedDict


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

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F 

from src.core.logger_config import setup_application_logger


# Improved U-Net Architecture (keeping the original implementation)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, dilation: bool=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        dilation_layer = (
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False) if dilation else
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            dilation_layer,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int, dilation:bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dilation:bool=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dilation=dilation) #, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dilation=dilation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, 
                 n_channels: int = 3, 
                 n_classes: int = 1, 
                 bilinear: bool = False, 
                 architecture: str = "standard",
                 app_logger:Optional[logging.Logger]=None):
        super(UNet, self).__init__()
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('UNet')
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.architecture = architecture

        if architecture == "standard":
            self.logger.info("Using UNet - Standard architecture")
            self._build_standard_architecture()
        elif architecture == "lightweight":
            self.logger.info("Using UNet - Lightweight architecture")
            self._build_lightweight_architecture()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_standard_architecture(self):
        """Standard U-Net architecture"""
        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def _build_lightweight_architecture(self):
        """Lightweight U-Net architecture for limited resources"""
        self.inc = DoubleConv(self.n_channels, 24, dilation=True)
        self.down1 = Down(24, 32, dilation=True)
        self.down2 = Down(32, 40, dilation=True)
        self.down3 = Down(40, 48, dilation=True)
        self.up1 = Up(88, 32, self.bilinear, dilation=True)
        self.up2 = Up(64, 32, self.bilinear, dilation=True)
        self.up3 = Up(56, 32, self.bilinear, dilation=True)
        self.outc = OutConv(32, self.n_classes)

    def forward(self, x):
        if self.architecture == "standard":
            return self._forward_standard(x)
        else:
            return self._forward_lightweight(x)

    def _forward_standard(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def _forward_lightweight(self, x):
        x1 = self.inc(x) # 1/24
        x2 = self.down1(x1) # 24/32
        x3 = self.down2(x2) # 32/40
        x4 = self.down3(x3) # 40/48
        x = self.up1(x4, x3) # 88/32
        x = self.up2(x, x2) # 64/32
        x = self.up3(x, x1) # 56/32
        logits = self.outc(x) # 32/1
        return logits

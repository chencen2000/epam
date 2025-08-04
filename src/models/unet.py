import logging
from typing import Optional, Dict, Any, List
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F 

from src.core.logger_config import setup_application_logger


# Improved U-Net Architecture with dynamic configuration
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, dilation: bool = False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Ensure proper padding to maintain spatial dimensions
        if dilation:
            padding1 = 1  # For first conv
            padding2 = 2  # For dilated conv (dilation=2)
            dilation_value = 2
        else:
            padding1 = 1
            padding2 = 1
            dilation_value = 1
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding2, dilation=dilation_value, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels: int, out_channels: int, dilation: bool = False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ModifiedDown(nn.Module):
    """Downscaling with stride conv then double conv (for modified architecture)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True, dilation: bool = False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dilation=dilation)
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


class ModifiedUp(nn.Module):
    """Upscaling with explicit channel management (for modified architecture)"""
    def __init__(self, up_in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(up_in_channels, skip_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(skip_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # now x1 has skip_channels channels
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


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
                 config_path: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None,
                 architecture: str = "standard",  # options: "standard", "lightweight", "modified"
                 app_logger: Optional[logging.Logger] = None):
        super(UNet, self).__init__()
        
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('UNet')
        
        # For modified architecture, force n_channels to 1
        if architecture == "modified":
            if n_channels != 1:
                self.logger.warning(f"Modified architecture requires n_channels=1, but {n_channels} was provided. Setting to 1.")
                n_channels = 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Load configuration
        if config_path or config_dict:
            self.config = self._load_config(config_path, config_dict)
            self.logger.info(f"Using dynamic UNet architecture from config")
            self._build_dynamic_architecture()
        else:
            # Fallback to hardcoded architectures for backward compatibility
            self.architecture = architecture
            if architecture == "standard":
                self.logger.info("Using UNet - Standard architecture")
                self._build_standard_architecture()
            elif architecture == "lightweight":
                self.logger.info("Using UNet - Lightweight architecture")
                self._build_lightweight_architecture()
            elif architecture == "modified":
                self.logger.info("Using UNet - Modified (SlimUNet) architecture")
                self._build_modified_architecture()
            else:
                raise ValueError(f"Unknown architecture: {architecture}. Use 'standard', 'lightweight', 'modified', or provide config_path/config_dict for dynamic architectures.")

    def _load_config(self, config_path: Optional[str], config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from file or dictionary"""
        if config_dict:
            return config_dict
        elif config_path:
            try:
                with open(config_path, 'r') as file:
                    return yaml.safe_load(file)
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
                raise
        else:
            raise ValueError("Either config_path or config_dict must be provided")

    def _build_dynamic_architecture(self):
        """Build UNet architecture dynamically from configuration"""
        arch_config = self.config.get('architecture', {})
        
        # Get encoder (downsampling) configuration
        encoder_config = arch_config.get('encoder', [])
        decoder_config = arch_config.get('decoder', [])
        
        # Override bilinear setting from config if specified
        self.bilinear = arch_config.get('bilinear', self.bilinear)
        
        # Build initial convolution
        initial_config = arch_config.get('initial', {})
        initial_channels = initial_config.get('out_channels', 64)
        initial_dilation = initial_config.get('dilation', False)
        
        self.inc = DoubleConv(
            self.n_channels, 
            initial_channels, 
            dilation=initial_dilation
        )
        
        # Build encoder (downsampling path)
        self.encoder_layers = nn.ModuleList()
        prev_channels = initial_channels
        
        for i, layer_config in enumerate(encoder_config):
            out_channels = layer_config.get('out_channels')
            dilation = layer_config.get('dilation', False)
            
            if out_channels is None:
                raise ValueError(f"Missing 'out_channels' in encoder layer {i}")
            
            self.encoder_layers.append(
                Down(prev_channels, out_channels, dilation=dilation)
            )
            prev_channels = out_channels
            
        # Build decoder (upsampling path)
        self.decoder_layers = nn.ModuleList()
        encoder_channels = [initial_channels] + [layer['out_channels'] for layer in encoder_config]
        
        for i, layer_config in enumerate(decoder_config):
            out_channels = layer_config.get('out_channels')
            dilation = layer_config.get('dilation', False)
            
            if out_channels is None:
                raise ValueError(f"Missing 'out_channels' in decoder layer {i}")
            
            # Calculate input channels (current + skip connection)
            # Skip connections are in reverse order: last encoder layer connects to first decoder layer
            skip_idx = len(encoder_channels) - 2 - i
            if skip_idx >= 0:
                skip_channels = encoder_channels[skip_idx]
                in_channels = prev_channels + skip_channels
            else:
                in_channels = prev_channels
            
            self.logger.debug(f"Decoder layer {i}: in_channels={in_channels}, out_channels={out_channels}, skip_idx={skip_idx}")
            
            self.decoder_layers.append(
                Up(in_channels, out_channels, self.bilinear, dilation=dilation)
            )
            prev_channels = out_channels
        
        # Build output convolution
        output_config = arch_config.get('output', {})
        self.outc = OutConv(prev_channels, self.n_classes)
        
        self.logger.info(f"Built dynamic architecture with {len(self.encoder_layers)} encoder and {len(self.decoder_layers)} decoder layers")

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

    def _build_modified_architecture(self):
        """Modified (SlimUNet) architecture with stride-based downsampling"""
        # Initial layer with stride-2 convolution (reduces size by 2x immediately)
        self.initial = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=True)
        
        # Encoder path using stride-based downsampling
        self.enc1 = ModifiedDown(24, 32)
        self.enc2 = ModifiedDown(32, 40)
        self.enc3 = ModifiedDown(40, 48)
        
        # Decoder path with explicit channel management
        self.dec1 = ModifiedUp(up_in_channels=48, skip_channels=40, out_channels=40)
        self.dec2 = ModifiedUp(up_in_channels=40, skip_channels=32, out_channels=32)
        self.dec3 = ModifiedUp(up_in_channels=32, skip_channels=24, out_channels=24)
        
        # Output layer
        self.outc = OutConv(24, self.n_classes)

    def forward(self, x):
        if hasattr(self, 'config'):
            return self._forward_dynamic(x)
        elif self.architecture == "standard":
            return self._forward_standard(x)
        elif self.architecture == "lightweight":
            return self._forward_lightweight(x)
        elif self.architecture == "modified":
            return self._forward_modified(x)

    def _forward_dynamic(self, x):
        """Dynamic forward pass based on configuration"""
        input_size = x.shape[-2:]  # Store original input size
        
        # Initial convolution
        x_current = self.inc(x)
        
        # Store skip connections
        skip_connections = [x_current]
        
        # Encoder path
        for encoder_layer in self.encoder_layers:
            x_current = encoder_layer(x_current)
            skip_connections.append(x_current)
        
        # Remove the last skip connection (it's the bottleneck)
        skip_connections.pop()
        
        # Decoder path
        for i, decoder_layer in enumerate(self.decoder_layers):
            if skip_connections:
                skip_connection = skip_connections.pop()
                x_current = decoder_layer(x_current, skip_connection)
            else:
                # If no more skip connections, just upsample
                x_current = decoder_layer.up(x_current)
                x_current = decoder_layer.conv(x_current)
        
        # Output convolution
        logits = self.outc(x_current)
        
        # Ensure output matches input size
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits

    def _forward_standard(self, x):
        input_size = x.shape[-2:]  # Store original input size
        
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
        
        # Ensure output matches input size
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits

    def _forward_lightweight(self, x):
        input_size = x.shape[-2:]  # Store original input size
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        # Ensure output matches input size
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits

    def _forward_modified(self, x):
        """Modified (SlimUNet) forward pass"""
        input_size = x.shape[-2:]  # Store original input size
        
        # Initial convolution with stride-2 (reduces size by 2x)
        x0 = self.initial(x)
        
        # Encoder path
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        
        # Decoder path with skip connections
        x = self.dec1(x3, x2)
        x = self.dec2(x, x1)
        x = self.dec3(x, x0)
        
        # Output convolution
        logits = self.outc(x)
        
        # Dynamically resize back to original input size
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        return logits

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
        }
        
        if hasattr(self, 'config'):
            info['architecture_type'] = 'dynamic'
            info['encoder_layers'] = len(self.encoder_layers)
            info['decoder_layers'] = len(self.decoder_layers)
        else:
            info['architecture_type'] = self.architecture
            
        return info
import os
import random
import gc
import psutil

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import
import torch.optim as optim
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def get_device_info():
    """Get device information and set optimal settings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = "CPU"
    
    if torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        # Get CPU info
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        print(f"CPU Cores: {cpu_count}")
        print(f"System Memory: {memory_gb:.1f}GB")
    
    print(f"Using device: {device_name}")
    return device

device = get_device_info()

# U-Net Architecture Implementation
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels) #, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 24)
        self.down1 = Down(24, 32)
        self.down2 = Down(32, 40)
        self.down3 = Down(40, 48)
        self.up1 = Up(88, 32, bilinear)
        self.up2 = Up(64, 32, bilinear)
        self.up3 = Up(56, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x): # number of input/output channels
        x1 = self.inc(x) # 1/24
        x2 = self.down1(x1) # 24/32
        x3 = self.down2(x2) # 32/40
        x4 = self.down3(x3) # 40/48
        x = self.up1(x4, x3) # 88/32
        x = self.up2(x, x2) # 64/32
        x = self.up3(x, x1) # 56/32
        logits = self.outc(x) # 32/1
        return logits

class MemoryEfficientSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(1024, 1024)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.samples = self._load_samples()
        
        print(f"Found {len(self.samples)} samples")
        print(f"Target image size: {target_size}")
    
    def _load_samples(self):
        samples = []
        for photo_version in os.listdir(self.root_dir):
            photo_path = os.path.join(self.root_dir, photo_version)
            if not os.path.isdir(photo_path):
                continue
                
            patches_path = os.path.join(photo_path, 'patches')
            if not os.path.exists(patches_path):
                continue
                
            for patch in os.listdir(patches_path):
                patch_path = os.path.join(patches_path, patch)
                if not os.path.isdir(patch_path):
                    continue
                
                image_path = os.path.join(patch_path, 'synthetic_dirty_patch.png')
                mask_path = os.path.join(patch_path, 'segmentation_mask_patch.png')
                labels_path = os.path.join(patch_path, 'labels_patch.json')
                
                if all(os.path.exists(p) for p in [image_path, mask_path, labels_path]):
                    samples.append({
                        'image': image_path,
                        'mask': mask_path,
                        'labels': labels_path
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load and process image efficiently
            image = cv2.imread(sample['image'], cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {sample['image']}")
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {sample['mask']}")
            
            # Resize to target size immediately to save memory
            #image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            #mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to proper types
            image = image.astype(np.float32) / 255.0
            mask = mask.astype(np.float32) / 255.0
            
            # Apply transforms if available
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                # Convert to tensor manually
                image = torch.from_numpy(image) #.transpose(2, 0, 1) / 255.0
                mask = torch.from_numpy(mask)
            
            return image, mask.unsqueeze(0)  # Add channel dimension to mask
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data in case of error
            dummy_image = torch.zeros(1, self.target_size[0], self.target_size[1])
            dummy_mask = torch.zeros(1, self.target_size[0], self.target_size[1])
            return dummy_image, dummy_mask

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, inputs, targets):
        return self.alpha * self.bce(inputs, targets) + (1 - self.alpha) * self.dice(inputs, targets)

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
        A.Normalize(
            mean=[0.33],
            std=[0.33]
        ),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(
            mean=[0.33],
            std=[0.33]
        ),
        ToTensorV2()
    ])

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        return f"GPU: {gpu_memory:.2f}GB"
    else:
        cpu_memory = psutil.Process().memory_info().rss / 1024**3
        return f"CPU: {cpu_memory:.2f}GB"

def train_epoch(model, dataloader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, masks) in enumerate(progress_bar):
        # Move to device with error handling
        try:
            images = images.to(device, non_blocking=False)
            masks = masks.to(device, non_blocking=False)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'Mem': get_memory_usage()
            })
            
            # Clean up
            del images, masks, outputs, loss
            
            # Periodic garbage collection
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            # Skip this batch and continue
            continue
    
    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        for batch_idx, (images, masks) in enumerate(progress_bar):
            try:
                images = images.to(device, non_blocking=False)
                masks = masks.to(device, non_blocking=False)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                total_loss += loss.item()
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Mem': get_memory_usage()
                })
                
                # Clean up
                del images, masks, outputs, loss
                
                # Periodic cleanup
                if batch_idx % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')

def get_optimal_settings(device):
    """Get optimal settings based on device"""
    if torch.cuda.is_available():
        # GPU settings
        return {
            'batch_size': 4,
            'num_workers': 4,
            'pin_memory': True,
            'target_size': (1024, 1024),
            'accumulation_steps': 1
        }
    else:
        print("Inside get_optimal_settings else")
        # CPU settings - more conservative
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3

        print(f"{cpu_count = }, {memory_gb = }")
        
        # Adjust based on available memory
        if memory_gb < 8:
            settings = {
                'batch_size': 1,
                'num_workers': 1,
                'pin_memory': False,
                'target_size': (1024, 1024),
                'accumulation_steps': 4
            }
        elif memory_gb < 16:
            settings = {
                'batch_size': 2,
                'num_workers': min(2, cpu_count // 2),
                'pin_memory': False,
                'target_size': (1024, 1024),
                'accumulation_steps': 2
            }
        else:
            settings = {
                'batch_size': 8,
                'num_workers': 16,  # min(16, cpu_count // 2),
                'pin_memory': False,
                'target_size': (1024, 1024),
                'accumulation_steps': 2
            }
        
        print(f"CPU optimized settings: {settings}")
        return settings

def train_model(root_dir, num_epochs=50, learning_rate=1e-4, 
                num_classes=1, save_path='best_unet_model.pth', **kwargs):
    
    # Get optimal settings for the device
    settings = get_optimal_settings(device)
    settings.update(kwargs)  # Override with user settings
    
    print(f'Using device: {device}')
    print(f'Settings: {settings}')
    
    # Memory optimization for CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # Create dataset
    full_dataset = MemoryEfficientSegmentationDataset(
        root_dir, 
        transform=None,
        target_size=settings['target_size']
    )
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid samples found in {root_dir}")
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=0.2, 
        random_state=42
    )
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings['batch_size'], 
        shuffle=True, 
        num_workers=settings['num_workers'],
        pin_memory=settings['pin_memory'],
        persistent_workers=settings['num_workers'] > 0,
        prefetch_factor=2 if settings['num_workers'] > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=settings['batch_size'],
        shuffle=False, 
        num_workers=settings['num_workers'],
        pin_memory=settings['pin_memory'],
        persistent_workers=settings['num_workers'] > 0,
        prefetch_factor=2 if settings['num_workers'] > 0 else None
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Effective batch size: {settings["batch_size"] * settings["accumulation_steps"]}')
    
    # Initialize model
    model = UNet(n_channels=1, n_classes=num_classes, bilinear=True)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-7
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    print(f"Initial memory usage: {get_memory_usage()}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Cleanup before each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            settings['accumulation_steps']
        )
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Memory usage: {get_memory_usage()}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'settings': settings
            }, save_path)
            print(f'New best model saved with validation loss: {val_loss:.4f}')
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, train_losses, val_losses

if __name__ == '__main__':
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    # Configuration
    ROOT_DIR =  "../data/synthetic_data"   # Change this to your dataset path
    NUM_EPOCHS = 5  # Reduced for testing
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 1
    SAVE_PATH = "./models/best_unet_model_fixed.pth"

    # Clear memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Initial memory usage: {get_memory_usage()}")

    try:
        # Train the model with automatic device optimization
        model, train_losses, val_losses = train_model(
            root_dir=ROOT_DIR,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            num_classes=NUM_CLASSES,
            save_path=SAVE_PATH
        )
        
        print("Training completed successfully!")
        print(f"Best model saved at: {SAVE_PATH}")
        print(f"Final memory usage: {get_memory_usage()}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

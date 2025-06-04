import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from src.core.logger_config import setup_application_logger


class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, target_size: Tuple[int, int] = (512, 512), app_logger:Optional[logging.Logger]=None):
        if app_logger is None:
            app_logger = setup_application_logger()
        
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.samples = self._load_samples()
        
        print(f"Found {len(self.samples)} samples")
        print(f"Target image size: {target_size}")
    
    def _load_samples(self) -> List[Dict[str, str]]:
        samples = []
        if not self.root_dir.exists():
            raise ValueError(f"Root directory does not exist: {self.root_dir}")
            
        for photo_version in self.root_dir.iterdir():
            if not photo_version.is_dir():
                continue
                
            patches_path = photo_version / 'patches'
            if not patches_path.exists():
                continue
                
            for patch in patches_path.iterdir():
                if not patch.is_dir():
                    continue
                
                image_path = patch / 'synthetic_dirty_patch.png'
                mask_path = patch / 'segmentation_mask_patch.png'
                labels_path = patch / 'labels_patch.json'
                
                if all(p.exists() for p in [image_path, mask_path, labels_path]):
                    samples.append({
                        'image': str(image_path),
                        'mask': str(mask_path),
                        'labels': str(labels_path)
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # More efficient image loading
            image = cv2.imread(sample['image'])
            if image is None:
                raise ValueError(f"Could not load image: {sample['image']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load mask
            mask = cv2.imread(sample['mask'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {sample['mask']}")
            
            # Resize efficiently
            if image.shape[:2] != self.target_size:
                image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            if mask.shape != self.target_size:
                mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to proper types
            image = image.astype(np.float32)
            mask = (mask.astype(np.float32) > 127).astype(np.float32)  # Binary threshold
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                # Manual conversion to tensor
                image = torch.from_numpy(image.transpose(2, 0, 1)) / 255.0
                mask = torch.from_numpy(mask)
            
            return image, mask.unsqueeze(0)
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, *self.target_size)
            dummy_mask = torch.zeros(1, *self.target_size)
            return dummy_image, dummy_mask

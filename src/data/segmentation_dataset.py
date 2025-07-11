import os
import cv2
import logging
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import Dataset


class MultiClassSegmentationDataset(Dataset):
    """
    Multi-class segmentation dataset for synthetic dirt and scratch detection
    Supports 3 classes: 0=background, 1=dirt, 2=scratches
    """
    
    def __init__(self, root_dir: str, target_size: Tuple[int, int] = (1024, 1024), 
                 image_ops=None, num_classes: int = 3, app_logger: Optional[logging.Logger] = None,
                 transform=None):
        """
        Args:
            root_dir: Root directory containing training data
            target_size: Target image size (height, width)
            image_ops: Image operations utility
            num_classes: Number of classes (3 for background, dirt, scratches)
            app_logger: Logger instance
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.image_ops = image_ops
        self.num_classes = num_classes
        self.transform = transform
        self.logger = app_logger or logging.getLogger(__name__)
        
        # Define class information
        self.class_names = ['background', 'dirt', 'scratches']
        self.dirt_categories = {
            'dirt': {'id': 1, 'name': 'dirt'},
            'scratches': {'id': 2, 'name': 'scratches'}
        }
        
        # Scan for samples
        self.samples = self._scan_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root_dir}")
            
        self.logger.info(f"Found {len(self.samples)} samples in dataset")
        
    def _scan_dataset(self) -> List[Dict]:
        """Scan root directory for valid image-mask pairs"""
        samples = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Look for image files
            image_files = [f for f in files if f.startswith('synthetic_dirty_') and f.endswith('.png')]
            
            if not image_files:
                continue
                
            # Check if we have corresponding mask
            mask_path = self._find_mask_file(root)
            if mask_path is None:
                self.logger.warning(f"No mask found for images in {root}")
                continue
                
            # Add sample for each image found
            for image_file in image_files:
                image_path = os.path.join(root, image_file)
                sample = {
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'image_dir': root,
                    'image_file': image_file
                }
                samples.append(sample)
                
        self.logger.info(f"Scanned dataset: found {len(samples)} valid samples")
        return samples
    
    def _find_mask_file(self, image_dir: str) -> Optional[str]:
        """Find the appropriate mask file for training"""
        # Priority order: multiclass masks first, then fallback to binary
        mask_candidates = [
            'segmentation_mask_multiclass.png',           # Full image multiclass
            'segmentation_mask_patch_multiclass.png',     # Patch multiclass  
            'segmentation_mask_patch.png',                # Binary patch (fallback)
            'segmentation_mask_combined.png',             # Binary combined (fallback)
            'segmentation_mask.png'                       # Any other binary mask
        ]
        
        for mask_name in mask_candidates:
            mask_path = os.path.join(image_dir, mask_name)
            if os.path.exists(mask_path):
                if 'multiclass' in mask_name:
                    self.logger.debug(f"Using multi-class mask: {mask_name}")
                else:
                    self.logger.debug(f"Using binary mask as fallback: {mask_name}")
                return mask_path
                
        return None
    
    def _load_and_validate_mask(self, mask_path: str) -> Optional[np.ndarray]:
        """Load mask and validate/convert it for multi-class training"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self.logger.error(f"Failed to load mask: {mask_path}")
            return None
            
        # Check pixel value range
        unique_values = np.unique(mask)
        mask_filename = os.path.basename(mask_path)
        
        # Handle different mask types
        if len(unique_values) == 2 and set(unique_values) == {0, 255}:
            # Binary mask with values [0, 255] - convert to [0, 1]
            self.logger.debug(f"Converting binary mask {mask_filename} to multi-class - treating all defects as dirt (class 1)")
            mask = (mask > 0).astype(np.uint8)
            
        elif len(unique_values) == 2 and set(unique_values) <= {0, 1, 2}:
            # Already in correct format or binary with correct values
            self.logger.debug(f"Mask {mask_filename} already in correct format")
            
        elif max(unique_values) <= 2:
            # Multi-class mask detected
            self.logger.debug(f"Multi-class mask {mask_filename} detected")
            
        elif max(unique_values) == 255:
            # Mask has value 255, need to map to class range
            self.logger.warning(f"Mask {mask_filename} has value 255, converting to valid class range")
            # Map 255 to class 1 (dirt), keep 0 as background
            mask = np.where(mask == 255, 1, mask)
            mask = np.clip(mask, 0, 2)  # Ensure valid range
            
        else:
            # Unusual mask values - clip to valid range
            self.logger.warning(f"Mask {mask_filename} has unusual values: {unique_values}, clipping to valid range")
            mask = np.clip(mask, 0, 2)  # Clip to valid class range [0,1,2]
        
        # Final validation
        final_values = np.unique(mask)
        if np.max(final_values) > 2:
            self.logger.error(f"Mask {mask_filename} still has invalid values after processing: {final_values}")
            return None
            
        # Log class distribution for debugging
        unique_vals, counts = np.unique(mask, return_counts=True)
        distribution = dict(zip(unique_vals, counts))
        self.logger.debug(f"Mask {mask_filename} class distribution: {distribution}")
        
        return mask
    
    def _load_and_process_image(self, image_path: str) -> np.ndarray:
        """Load and process image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Resize if needed
        if image.shape[:2] != self.target_size:
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension for grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # (1, H, W)
            
        return image
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Process mask to correct format"""
        # Resize if needed
        if mask.shape != self.target_size:
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is long type for cross entropy loss
        mask = mask.astype(np.int64)
        
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
            
        sample = self.samples[idx]
        
        try:
            # Load image
            image = self._load_and_process_image(sample['image_path'])
            
            # Load and validate mask
            mask = self._load_and_validate_mask(sample['mask_path'])
            if mask is None:
                raise ValueError(f"Could not load mask from {sample['mask_path']}")
            
            # Process mask
            mask = self._process_mask(mask)
            
            # Apply transforms if provided
            if self.transform:
                # Note: You might need to adapt this based on your transform library
                # For albumentations, you'd do something like:
                # transformed = self.transform(image=image.transpose(1,2,0), mask=mask)
                # image = transformed['image'].transpose(2,0,1)
                # mask = transformed['mask']
                pass
            
            # Convert to torch tensors
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask).long()
            
            return image_tensor, mask_tensor
            
        except Exception as e:
            self.logger.error(f"Error loading sample {idx} ({sample['image_path']}): {e}")
            # Return a dummy sample to avoid crashing training
            image_tensor = torch.zeros((1, *self.target_size), dtype=torch.float32)
            mask_tensor = torch.zeros(self.target_size, dtype=torch.long)
            return image_tensor, mask_tensor
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate class distribution across the dataset"""
        class_counts = {i: 0 for i in range(self.num_classes)}
        
        self.logger.info("Calculating class distribution across dataset...")
        
        for idx in range(min(len(self.samples), 100)):  # Sample first 100 for speed
            sample = self.samples[idx]
            mask = self._load_and_validate_mask(sample['mask_path'])
            
            if mask is not None:
                unique_vals, counts = np.unique(mask, return_counts=True)
                for val, count in zip(unique_vals, counts):
                    if val < self.num_classes:
                        class_counts[val] += count
        
        # Convert to named classes
        named_distribution = {}
        for class_id, count in class_counts.items():
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            named_distribution[class_name] = count
            
        self.logger.info(f"Class distribution: {named_distribution}")
        return named_distribution
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
            
        sample = self.samples[idx]
        mask = self._load_and_validate_mask(sample['mask_path'])
        
        info = {
            'index': idx,
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
            'image_dir': sample['image_dir']
        }
        
        if mask is not None:
            unique_vals, counts = np.unique(mask, return_counts=True)
            info['mask_distribution'] = dict(zip(unique_vals, counts))
            info['mask_shape'] = mask.shape
            
        return info
    
    def validate_dataset(self) -> bool:
        """Validate that all samples can be loaded correctly"""
        self.logger.info("Validating dataset...")
        
        failed_samples = []
        
        for idx in range(len(self.samples)):
            try:
                image, mask = self.__getitem__(idx)
                
                # Check image format
                if not isinstance(image, torch.Tensor) or len(image.shape) != 3:
                    failed_samples.append((idx, "Invalid image format"))
                    continue
                    
                # Check mask format  
                if not isinstance(mask, torch.Tensor) or len(mask.shape) != 2:
                    failed_samples.append((idx, "Invalid mask format"))
                    continue
                    
                # Check mask values
                unique_mask_vals = torch.unique(mask)
                if torch.max(unique_mask_vals) >= self.num_classes or torch.min(unique_mask_vals) < 0:
                    failed_samples.append((idx, f"Invalid mask values: {unique_mask_vals.tolist()}"))
                    
            except Exception as e:
                failed_samples.append((idx, str(e)))
        
        if failed_samples:
            self.logger.error(f"Dataset validation failed for {len(failed_samples)} samples:")
            for idx, error in failed_samples[:5]:  # Show first 5 errors
                self.logger.error(f"  Sample {idx}: {error}")
            return False
        else:
            self.logger.info("Dataset validation passed!")
            return True
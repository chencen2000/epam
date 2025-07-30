import os
import cv2
import logging
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import Dataset

from src.target_labels import TargetLabels


class MultiClassSegmentationDataset(Dataset):
    """
    Multi-class segmentation dataset for synthetic dirt and scratch detection
    Supports patches: 0=background, 1=scratch, 2=dirt, etc.
    """
    
    def __init__(self, root_dir: str, target_size: Tuple[int, int] = (1024, 1024), 
                 image_ops=None, num_classes: int = 4, app_logger: Optional[logging.Logger] = None,
                 transform=None, use_patches: bool = True):
        """
        Args:
            root_dir: Root directory containing training data
            target_size: Target image size (height, width) 
            image_ops: Image operations utility
            num_classes: Number of classes (4 for background, scratch, dirt, condensation)
            app_logger: Logger instance
            transform: Optional transforms to apply
            use_patches: If True, scan for patch directories, if False scan for full images
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.image_ops = image_ops
        self.num_classes = num_classes
        self.transform = transform
        self.use_patches = use_patches
        self.logger = app_logger or logging.getLogger(__name__)
        
        # Define class information from TargetLabels
        self.class_names = TargetLabels.values()
        self.logger.info(f"Class names: {self.class_names}")
        
        # Scan for samples
        self.samples = self._scan_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root_dir}")
            
        self.logger.info(f"Found {len(self.samples)} samples in dataset")
        
    def _scan_dataset(self) -> List[Dict]:
        """Scan root directory for valid image-mask pairs - FIXED FOR PATCHES"""
        samples = []
        
        if self.use_patches:
            # Scan for patch directories
            self.logger.info("Scanning for patch directories...")
            samples = self._scan_patch_directories()
        else:
            # Scan for full images
            self.logger.info("Scanning for full images...")
            samples = self._scan_full_images()
        
        self.logger.info(f"Scanned dataset: found {len(samples)} valid samples")
        return samples
    
    def _scan_patch_directories(self) -> List[Dict]:
        """Scan for patch directories containing patch data"""
        samples = []
        
        # Walk through directory structure looking for patch directories
        for root, dirs, files in os.walk(self.root_dir):
            # Look for patch directories (typically named like patch_000_x123_y456)
            for dir_name in dirs:
                if dir_name.startswith('patch_'):
                    patch_dir = os.path.join(root, dir_name)
                    
                    # Check if this patch directory has required files
                    if self._validate_patch_directory(patch_dir):
                        samples.append(self._create_patch_sample(patch_dir))
        
        return samples
    
    def _validate_patch_directory(self, patch_dir: str) -> bool:
        """Check if patch directory contains required files"""
        required_files = {
            'image': 'synthetic_dirty_patch.png',
            'labels': 'labels_patch.json'
        }
        
        # Check for required files
        for file_type, filename in required_files.items():
            if not os.path.exists(os.path.join(patch_dir, filename)):
                self.logger.debug(f"Missing {file_type} file in {patch_dir}: {filename}")
                return False
        
        # Check for mask file
        mask_path = self._find_mask_file(patch_dir)
        if mask_path is None:
            self.logger.debug(f"No valid mask found in {patch_dir}")
            return False
            
        return True
    
    def _create_patch_sample(self, patch_dir: str) -> Dict:
        """Create sample dictionary for a patch directory"""
        image_path = os.path.join(patch_dir, 'synthetic_dirty_patch.png')
        mask_path = self._find_mask_file(patch_dir)
        
        return {
            'image_path': image_path,
            'mask_path': mask_path,
            'image_dir': patch_dir,
            'image_file': 'synthetic_dirty_patch.png',
            'sample_type': 'patch',
            'patch_info': self._extract_patch_info(os.path.basename(patch_dir))
        }
    
    def _extract_patch_info(self, patch_name: str) -> Dict:
        """Extract patch coordinates from patch directory name"""
        # Expected format: patch_000_x123_y456
        try:
            parts = patch_name.split('_')
            if len(parts) >= 4:
                patch_id = int(parts[1])
                x_coord = int(parts[2][1:])  # Remove 'x' prefix
                y_coord = int(parts[3][1:])  # Remove 'y' prefix
                return {
                    'patch_id': patch_id,
                    'x_offset': x_coord,
                    'y_offset': y_coord
                }
        except (ValueError, IndexError):
            self.logger.warning(f"Could not parse patch info from name: {patch_name}")
        
        return {'patch_id': 0, 'x_offset': 0, 'y_offset': 0}
    
    def _scan_full_images(self) -> List[Dict]:
        """Scan for full synthetic images (fallback method)"""
        samples = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Look for full synthetic images
            image_files = [f for f in files if f.startswith('synthetic_dirty_image') and f.endswith('.png')]
            
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
                    'image_file': image_file,
                    'sample_type': 'full_image'
                }
                samples.append(sample)
                
        return samples
    
    def _find_mask_file(self, image_dir: str) -> Optional[str]:
        """Find the appropriate mask file for training - UPDATED PRIORITY"""
        # Priority order: multiclass patch masks first, then other options
        mask_candidates = [
            'segmentation_mask_patch_multiclass.png',     # HIGHEST PRIORITY - Patch multiclass
            'segmentation_mask_multiclass.png',           # Full image multiclass
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
        """Load mask and validate/convert it for multi-class training - FIXED"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            self.logger.error(f"Failed to load mask: {mask_path}")
            return None
            
        # Check pixel value range
        unique_values = np.unique(mask)
        mask_filename = os.path.basename(mask_path)
        
        self.logger.debug(f"Mask {mask_filename} loaded with unique values: {unique_values}")
        
        # Handle different mask types - FIXED CONDITIONS
        if len(unique_values) == 2 and set(unique_values) == {0, 255}:
            # Binary mask with values [0, 255] - convert to [0, 1] 
            self.logger.debug(f"Converting binary mask {mask_filename} from [0,255] to [0,1]")
            mask = (mask > 127).astype(np.uint8)
            
        elif len(unique_values) == 2 and set(unique_values) == {0, 1}:
            # Binary mask already in [0, 1] format
            self.logger.debug(f"Binary mask {mask_filename} already in [0,1] format")
            
        elif max(unique_values) <= (self.num_classes - 1) and min(unique_values) >= 0:
            # Multi-class mask with valid values [0,1,2,3...]
            self.logger.debug(f"Multi-class mask {mask_filename} detected with valid classes: {unique_values}")
            
        elif max(unique_values) == 255:
            # Mask has value 255, need to map to class range
            self.logger.warning(f"Mask {mask_filename} has value 255, attempting to map to valid classes")
            
            # For patches, we often see values like [0, 1, 2] or [0, 255]
            if len(unique_values) <= self.num_classes:
                # Map values proportionally: normalize to valid range
                value_mapping = {}
                sorted_values = sorted(unique_values)
                for i, val in enumerate(sorted_values):
                    value_mapping[val] = i
                
                mask_mapped = np.zeros_like(mask)
                for old_val, new_val in value_mapping.items():
                    mask_mapped[mask == old_val] = new_val
                    
                mask = mask_mapped
                self.logger.debug(f"Mapped mask values using mapping: {value_mapping}")
            else:
                # Too many unique values, treat as binary
                self.logger.warning(f"Too many unique values ({len(unique_values)}), treating as binary")
                mask = (mask > 127).astype(np.uint8)
                
        else:
            # Unusual mask values - attempt recovery
            self.logger.warning(f"Mask {mask_filename} has unusual values: {unique_values}")
            
            if len(unique_values) <= self.num_classes:
                # Map to sequential class IDs
                value_mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
                mask_mapped = np.zeros_like(mask)
                for old_val, new_val in value_mapping.items():
                    mask_mapped[mask == old_val] = new_val
                mask = mask_mapped
                self.logger.debug(f"Mapped unusual values using mapping: {value_mapping}")
            else:
                # Too many values - treat as binary
                mask = (mask > np.median(unique_values)).astype(np.uint8)
                self.logger.warning(f"Too many unique values, converted to binary using median threshold")
        
        # Final validation
        final_values = np.unique(mask)
        if np.max(final_values) >= self.num_classes:
            self.logger.error(f"Mask {mask_filename} still has invalid values after processing: {final_values} (max allowed: {self.num_classes-1})")
            # Clip to valid range as last resort
            mask = np.clip(mask, 0, self.num_classes-1)
            final_values = np.unique(mask)
            self.logger.warning(f"Clipped mask to valid range: {final_values}")
            
        # Log class distribution for debugging
        unique_vals, counts = np.unique(mask, return_counts=True)
        distribution = dict(zip(unique_vals, counts))
        self.logger.debug(f"Final mask {mask_filename} class distribution: {distribution}")
        
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
                # For albumentations transforms:
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
        
        self.logger.info(f"Calculating class distribution across {len(self.samples)} samples...")
        
        # Sample subset for performance
        sample_size = min(len(self.samples), 100)
        for idx in range(sample_size):
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
            
        self.logger.info(f"Class distribution (from {sample_size} samples): {named_distribution}")
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
            'image_dir': sample['image_dir'],
            'sample_type': sample.get('sample_type', 'unknown')
        }
        
        if 'patch_info' in sample:
            info['patch_info'] = sample['patch_info']
        
        if mask is not None:
            unique_vals, counts = np.unique(mask, return_counts=True)
            info['mask_distribution'] = dict(zip(unique_vals, counts))
            info['mask_shape'] = mask.shape
            
        return info
    
    def validate_dataset(self) -> bool:
        """Validate that all samples can be loaded correctly"""
        self.logger.info("Validating dataset...")
        
        failed_samples = []
        
        # Sample a subset for validation (to avoid long validation times)
        validation_samples = min(len(self.samples), 50)
        
        for idx in range(validation_samples):
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
            self.logger.info(f"Dataset validation passed for {validation_samples} samples!")
            return True
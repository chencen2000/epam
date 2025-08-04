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
    Supports 3 classes: 0=background, 1=dirt, 2=scratches
    Now focused on patch-level training data
    """
    
    def __init__(self, root_dir: str, target_size: Tuple[int, int] = (1024, 1024), 
                 image_ops=None, num_classes: int = 3, app_logger: Optional[logging.Logger] = None,
                 transform=None):
        """
        Args:
            root_dir: Root directory containing training data with patch subdirectories
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
        self.class_names = TargetLabels.values()
        self.dirt_categories = TargetLabels.get_dataset_mapping()
        
        # Scan for patch samples
        self.samples = self._scan_patch_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid patch samples found in {root_dir}")
            
        self.logger.info(f"Found {len(self.samples)} patch samples in dataset")
        
    def _scan_patch_dataset(self) -> List[Dict]:
        """Scan root directory for valid patch image-mask pairs"""
        samples = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Only process directories that are patch folders
            if not self._is_patch_directory(root):
                continue
                
            # Look for patch image files
            patch_image_files = [f for f in files if f.startswith('synthetic_dirty_patch') and f.endswith('.png')]
            
            if not patch_image_files:
                continue
                
            # Check if we have corresponding mask
            mask_path = self._find_patch_mask_file(root)
            if mask_path is None:
                self.logger.warning(f"No patch mask found in {root}")
                continue
                
            # Add sample for each patch image found
            for image_file in patch_image_files:
                image_path = os.path.join(root, image_file)
                
                # Extract patch information from directory name
                patch_info = self._extract_patch_info(root)
                
                sample = {
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'patch_dir': root,
                    'image_file': image_file,
                    'patch_info': patch_info
                }
                samples.append(sample)
                
        self.logger.info(f"Scanned patch dataset: found {len(samples)} valid patch samples")
        return samples
    
    def _is_patch_directory(self, directory_path: str) -> bool:
        """Check if directory is a patch directory based on naming convention"""
        dir_name = os.path.basename(directory_path)
        
        # Check if directory follows patch naming pattern: patch_XXX_xYYY_zZZZ
        if dir_name.startswith('patch_') and '_x' in dir_name and '_y' in dir_name:
            return True
        
        # Alternative: check if parent directory is named 'patches'
        parent_dir = os.path.basename(os.path.dirname(directory_path))
        if parent_dir == 'patches':
            return True
            
        return False
    
    def _extract_patch_info(self, patch_dir: str) -> Dict:
        """Extract patch coordinates and info from directory name"""
        dir_name = os.path.basename(patch_dir)
        patch_info = {'patch_id': dir_name}
        
        try:
            # Parse patch_000_x0_y0 format
            if '_x' in dir_name and '_y' in dir_name:
                parts = dir_name.split('_')
                for i, part in enumerate(parts):
                    if part.startswith('x') and i + 1 < len(parts):
                        patch_info['x_coord'] = int(part[1:])
                    elif part.startswith('y'):
                        patch_info['y_coord'] = int(part[1:])
                    elif part.isdigit() and 'patch_num' not in patch_info:
                        patch_info['patch_num'] = int(part)
        except (ValueError, IndexError) as e:
            self.logger.debug(f"Could not parse patch coordinates from {dir_name}: {e}")
            
        return patch_info
    
    def _find_patch_mask_file(self, patch_dir: str) -> Optional[str]:
        """Find the appropriate patch mask file for training"""
        # Priority order for patch masks
        patch_mask_candidates = [
            'segmentation_mask_patch_multiclass.png',     # Primary: patch multiclass
            'segmentation_mask_multiclass.png',           # Fallback: full image multiclass
            'segmentation_mask_patch.png',                # Binary patch (fallback)
            'segmentation_mask.png'                       # Any other mask
        ]
        
        for mask_name in patch_mask_candidates:
            mask_path = os.path.join(patch_dir, mask_name)
            if os.path.exists(mask_path):
                if 'multiclass' in mask_name:
                    self.logger.debug(f"Using multi-class patch mask: {mask_name}")
                else:
                    self.logger.debug(f"Using binary patch mask as fallback: {mask_name}")
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
        if len(unique_values) <= 2 and set(unique_values) <= {0, 255}:
            # Binary mask with values [0, 255] - convert to [0, 1]
            self.logger.debug(f"Converting binary patch mask {mask_filename} to multi-class - treating all defects as dirt (class 1)")
            mask = (mask > 0).astype(np.uint8)
            
        elif len(unique_values) <= 3 and set(unique_values) <= {0, 1, 2}:
            # Already in correct format
            self.logger.debug(f"Patch mask {mask_filename} already in correct format")
            
        elif max(unique_values) <= 3:
            # Multi-class patch mask detected
            self.logger.debug(f"Multi-class patch mask {mask_filename} detected")
            
        elif max(unique_values) == 255:
            # Mask has value 255, need to map to class range
            self.logger.warning(f"Patch mask {mask_filename} has value 255, converting to valid class range")
            # Map 255 to class 1 (dirt), keep 0 as background
            mask = np.where(mask == 255, 1, mask)
            mask = np.clip(mask, 0, 2)  # Ensure valid range
            
        else:
            # Unusual mask values - clip to valid range
            self.logger.warning(f"Patch mask {mask_filename} has unusual values: {unique_values}, clipping to valid range")
            mask = np.clip(mask, 0, 2)  # Clip to valid class range [0,1,2]
        
        # Final validation
        final_values = np.unique(mask)

        # self.logger.info(f"{final_values = }  --- {self.num_classes = }")
        if np.max(final_values) >= self.num_classes:
            self.logger.error(f"Patch mask {mask_filename} still has invalid values after processing: {final_values}")
            return None
            
        # Log class distribution for debugging
        unique_vals, counts = np.unique(mask, return_counts=True)
        distribution = dict(zip(unique_vals, counts))
        self.logger.debug(f"Patch mask {mask_filename} class distribution: {distribution}")
        
        return mask
    
    def _load_and_process_image(self, image_path: str) -> np.ndarray:
        """Load and process patch image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load patch image: {image_path}")
            
        # Resize if needed (patches should already be correct size)
        if image.shape[:2] != self.target_size:
            self.logger.debug(f"Resizing patch image from {image.shape[:2]} to {self.target_size}")
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension for grayscale
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # (1, H, W)
            
        return image
    
    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """Process patch mask to correct format"""
        # Resize if needed (patches should already be correct size)
        if mask.shape != self.target_size:
            self.logger.debug(f"Resizing patch mask from {mask.shape} to {self.target_size}")
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Ensure mask is long type for cross entropy loss
        mask = mask.astype(np.int64)
        
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a patch sample from the dataset"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
            
        sample = self.samples[idx]
        
        try:
            # Load patch image
            image = self._load_and_process_image(sample['image_path'])
            
            # Load and validate patch mask
            mask = self._load_and_validate_mask(sample['mask_path'])
            if mask is None:
                raise ValueError(f"Could not load patch mask from {sample['mask_path']}")
            
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
            self.logger.error(f"Error loading patch sample {idx} ({sample['image_path']}): {e}")
            # Return a dummy sample to avoid crashing training
            image_tensor = torch.zeros((1, *self.target_size), dtype=torch.float32)
            mask_tensor = torch.zeros(self.target_size, dtype=torch.long)
            return image_tensor, mask_tensor
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate class distribution across the patch dataset"""
        class_counts = {i: 0 for i in range(self.num_classes)}
        
        self.logger.info(f"Calculating class distribution across patch dataset... {len(self.samples)} patches")
        
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
            
        self.logger.info(f"Patch class distribution: {named_distribution}")
        return named_distribution
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get information about a specific patch sample"""
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range")
            
        sample = self.samples[idx]
        mask = self._load_and_validate_mask(sample['mask_path'])
        
        info = {
            'index': idx,
            'image_path': sample['image_path'],
            'mask_path': sample['mask_path'],
            'patch_dir': sample['patch_dir'],
            'patch_info': sample['patch_info']
        }
        
        if mask is not None:
            unique_vals, counts = np.unique(mask, return_counts=True)
            info['mask_distribution'] = dict(zip(unique_vals, counts))
            info['mask_shape'] = mask.shape
            
        return info
    
    def get_patch_statistics(self) -> Dict:
        """Get statistics specific to patch training"""
        stats = {
            'total_patches': len(self.samples),
            'unique_image_sources': len(set([
                os.path.dirname(os.path.dirname(sample['patch_dir'])) 
                for sample in self.samples
            ])),
            'patches_per_source': {}
        }
        
        # Count patches per source image
        for sample in self.samples:
            source_image = os.path.basename(os.path.dirname(os.path.dirname(sample['patch_dir'])))
            if source_image not in stats['patches_per_source']:
                stats['patches_per_source'][source_image] = 0
            stats['patches_per_source'][source_image] += 1
        
        return stats
    
    def validate_dataset(self) -> bool:
        """Validate that all patch samples can be loaded correctly"""
        self.logger.info("Validating patch dataset...")
        
        failed_samples = []
        
        for idx in range(len(self.samples)):
            try:
                image, mask = self.__getitem__(idx)
                
                # Check image format
                if not isinstance(image, torch.Tensor) or len(image.shape) != 3:
                    failed_samples.append((idx, "Invalid patch image format"))
                    continue
                    
                # Check mask format  
                if not isinstance(mask, torch.Tensor) or len(mask.shape) != 2:
                    failed_samples.append((idx, "Invalid patch mask format"))
                    continue
                    
                # Check mask values
                unique_mask_vals = torch.unique(mask)
                if torch.max(unique_mask_vals) >= self.num_classes or torch.min(unique_mask_vals) < 0:
                    failed_samples.append((idx, f"Invalid patch mask values: {unique_mask_vals.tolist()}"))
                    
            except Exception as e:
                failed_samples.append((idx, str(e)))
        
        if failed_samples:
            self.logger.error(f"Patch dataset validation failed for {len(failed_samples)} samples:")
            for idx, error in failed_samples[:5]:  # Show first 5 errors
                self.logger.error(f"  Patch sample {idx}: {error}")
            return False
        else:
            self.logger.info("Patch dataset validation passed!")
            return True
"""File and data utilities - UPDATED FOR MULTI-CLASS."""

import json
from pathlib import Path
from logging import Logger
from typing import Tuple, Dict, Optional

import cv2
import numpy as np


def load_ground_truth_data(data_path: str, logger: Optional[Logger], 
                          multiclass: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load image, mask and labels from dataset structure
    UPDATED FOR MULTI-CLASS SUPPORT
    
    Args:
        data_path: Path to the dataset directory
        logger: Logger instance
        multiclass: If True, looks for multi-class masks first, then falls back to binary
        
    Returns:
        Tuple of (image, mask, labels)
    """
    logger.debug("Loading the ground truth data...")
    data_path = Path(data_path)

    # Define required files based on mode
    if multiclass:
        mask_candidates = [
            'segmentation_mask_patch_multiclass.png',  # Multi-class patch mask
            'segmentation_mask_multiclass.png',        # Multi-class full mask
            'segmentation_mask_patch.png',             # Binary patch mask (fallback)
            'segmentation_mask_combined.png'           # Binary combined mask (fallback)
        ]
    else:
        mask_candidates = [
            'segmentation_mask_patch.png',             # Binary patch mask
            'segmentation_mask_combined.png',          # Binary combined mask
            'segmentation_mask_patch_multiclass.png',  # Multi-class as fallback
            'segmentation_mask_multiclass.png'         # Multi-class full as fallback
        ]

    required_files = {
        'image': 'synthetic_dirty_patch.png',
        'labels': 'labels_patch.json'
    }

    # Check for image and labels files
    missing_files = []
    for file_type, filename in required_files.items():
        if not (data_path / filename).exists():
            missing_files.append(filename)
    
    # Find available mask file
    mask_file = None
    mask_type = None
    
    for mask_candidate in mask_candidates:
        if (data_path / mask_candidate).exists():
            mask_file = mask_candidate
            mask_type = 'multiclass' if 'multiclass' in mask_candidate else 'binary'
            break
    
    if mask_file is None:
        missing_files.append("No valid mask file found")
    
    if missing_files:
        err_msg = f"Missing required files in {data_path}: {missing_files}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    
    logger.debug(f"Using mask file: {mask_file} (type: {mask_type})")
    
    try:
        # Load image
        image_path = data_path / required_files['image']
        image = cv2.imread(str(image_path))
        if image is None:
            err_msg = f"Could not load image: {image_path}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask with proper handling for multi-class vs binary
        mask_path = data_path / mask_file
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            err_msg = f"Could not load mask: {mask_path}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        # Process mask based on type and desired output
        mask = _process_mask(mask, mask_type, multiclass, logger)
        
        # Load labels
        labels_path = data_path / required_files['labels']
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            
        # Add mask information to labels
        labels['mask_info'] = {
            'mask_file': mask_file,
            'mask_type': mask_type,
            'multiclass_requested': multiclass,
            'final_classes': len(np.unique(mask)),
            'class_distribution': dict(zip(*np.unique(mask, return_counts=True)))
        }
            
    except Exception as e:
        err_msg = f"Failed to load ground truth data: {e}"
        logger.error(err_msg)
        raise ValueError(err_msg)
        
    logger.debug(f"Loading gt data complete - mask shape: {mask.shape}, classes: {np.unique(mask)}")
    return image, mask, labels


def _process_mask(mask: np.ndarray, mask_type: str, multiclass_requested: bool, 
                  logger: Logger) -> np.ndarray:
    """
    Process mask based on type and requirements
    
    Args:
        mask: Raw mask from file
        mask_type: 'multiclass' or 'binary'
        multiclass_requested: Whether multi-class output is requested
        logger: Logger instance
        
    Returns:
        Processed mask
    """
    unique_values = np.unique(mask)
    logger.debug(f"Raw mask values: {unique_values}")
    
    if mask_type == 'multiclass':
        # Multi-class mask processing
        if multiclass_requested:
            # Return as-is if already multi-class and multi-class requested
            if np.max(unique_values) <= 2:
                logger.debug("Multi-class mask already in correct format")
                return mask.astype(np.uint8)
            else:
                # Clip to valid range if values are too high
                logger.warning(f"Multi-class mask has values > 2: {unique_values}, clipping to [0,1,2]")
                return np.clip(mask, 0, 2).astype(np.uint8)
        else:
            # Convert multi-class to binary (combine all defect classes)
            logger.debug("Converting multi-class mask to binary")
            binary_mask = (mask > 0).astype(np.uint8)
            return binary_mask
    
    else:  # binary mask
        # Binary mask processing
        if len(unique_values) == 2 and 255 in unique_values:
            # Standard binary mask [0, 255] -> normalize
            normalized_mask = (mask > 127).astype(np.uint8)
            logger.debug("Normalized binary mask from [0,255] to [0,1]")
        else:
            # Assume already in [0,1] format
            normalized_mask = (mask > 0).astype(np.uint8)
            logger.debug("Binary mask already in [0,1] format")
        
        if multiclass_requested:
            # Convert binary to multi-class format (treat all defects as dirt class)
            logger.debug("Converting binary mask to multi-class format (all defects as dirt)")
            return normalized_mask.astype(np.uint8)  # 0=background, 1=dirt
        else:
            # Return as binary
            return normalized_mask


def load_multiclass_mask(mask_path: str, logger: Optional[Logger] = None) -> np.ndarray:
    """
    Load and validate a multi-class mask
    
    Args:
        mask_path: Path to mask file
        logger: Optional logger
        
    Returns:
        Multi-class mask with values [0, 1, 2, ...]
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    unique_values = np.unique(mask)
    logger.debug(f"Loaded mask with values: {unique_values}")
    
    # Handle different mask formats
    if len(unique_values) == 2 and 255 in unique_values:
        # Binary mask [0, 255] -> convert to [0, 1]
        logger.debug("Converting binary mask [0,255] to multi-class [0,1]")
        mask = (mask > 127).astype(np.uint8)
    elif np.max(unique_values) > 2:
        # Values too high - clip to valid multi-class range
        logger.warning(f"Mask values {unique_values} exceed multi-class range, clipping to [0,2]")
        mask = np.clip(mask, 0, 2).astype(np.uint8)
    else:
        # Already in correct format
        mask = mask.astype(np.uint8)
    
    # Validate final mask
    final_values = np.unique(mask)
    if np.max(final_values) > 2:
        raise ValueError(f"Invalid multi-class mask values after processing: {final_values}")
    
    logger.debug(f"Final mask values: {final_values}")
    return mask


def save_multiclass_mask(mask: np.ndarray, save_path: str, logger: Optional[Logger] = None) -> None:
    """
    Save a multi-class mask with proper format
    
    Args:
        mask: Multi-class mask with values [0, 1, 2, ...]
        save_path: Path to save the mask
        logger: Optional logger
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    # Validate input
    unique_values = np.unique(mask)
    if np.max(unique_values) > 255:
        raise ValueError(f"Mask values too high for saving: {unique_values}")
    
    # Save as grayscale
    success = cv2.imwrite(save_path, mask.astype(np.uint8))
    if not success:
        raise IOError(f"Failed to save mask to: {save_path}")
    
    logger.debug(f"Saved multi-class mask to {save_path} with values: {unique_values}")


def validate_dataset_structure(data_path: str, logger: Optional[Logger] = None, 
                             require_multiclass: bool = False) -> Dict:
    """
    Validate dataset structure and return information about available files
    
    Args:
        data_path: Path to dataset directory
        logger: Optional logger
        require_multiclass: If True, requires multi-class masks
        
    Returns:
        Dictionary with validation results and file information
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    data_path = Path(data_path)
    validation_result = {
        'valid': False,
        'files_found': {},
        'mask_info': {},
        'errors': []
    }
    
    # Check required files
    required_files = {
        'image': 'synthetic_dirty_patch.png',
        'labels': 'labels_patch.json'
    }
    
    for file_type, filename in required_files.items():
        file_path = data_path / filename
        validation_result['files_found'][file_type] = file_path.exists()
        if not file_path.exists():
            validation_result['errors'].append(f"Missing {file_type} file: {filename}")
    
    # Check mask files
    mask_files = [
        'segmentation_mask_patch_multiclass.png',
        'segmentation_mask_multiclass.png', 
        'segmentation_mask_patch.png',
        'segmentation_mask_combined.png'
    ]
    
    available_masks = []
    for mask_file in mask_files:
        if (data_path / mask_file).exists():
            available_masks.append(mask_file)
            
            # Analyze mask
            try:
                mask = cv2.imread(str(data_path / mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    unique_vals = np.unique(mask)
                    is_multiclass = 'multiclass' in mask_file or (len(unique_vals) > 2 and np.max(unique_vals) <= 2)
                    
                    validation_result['mask_info'][mask_file] = {
                        'values': unique_vals.tolist(),
                        'is_multiclass': is_multiclass,
                        'shape': mask.shape
                    }
            except Exception as e:
                validation_result['errors'].append(f"Error analyzing {mask_file}: {e}")
    
    validation_result['files_found']['masks'] = available_masks
    
    # Check if requirements are met
    has_image = validation_result['files_found']['image']
    has_labels = validation_result['files_found']['labels']
    has_masks = len(available_masks) > 0
    
    if require_multiclass:
        has_multiclass = any('multiclass' in mask for mask in available_masks)
        validation_result['valid'] = has_image and has_labels and has_multiclass
        if not has_multiclass:
            validation_result['errors'].append("No multi-class masks found (required)")
    else:
        validation_result['valid'] = has_image and has_labels and has_masks
    
    if not has_masks:
        validation_result['errors'].append("No mask files found")
    
    logger.debug(f"Dataset validation for {data_path}: {'VALID' if validation_result['valid'] else 'INVALID'}")
    
    return validation_result

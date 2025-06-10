"""File and data utilities."""

import json
from pathlib import Path
from logging import Logger
from typing import Tuple, Dict, Optional

import cv2
import numpy as np


def load_ground_truth_data(data_path: str, logger:Optional[Logger]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load image, mask and labels from dataset structure"""
    logger.debug("Loading the ground truth data...")
    data_path = Path(data_path)

    # Define required files
    required_files = {
        'image': 'synthetic_dirty_patch.png',
        'mask': 'segmentation_mask_patch.png',
        'labels': 'labels_patch.json'
    }

    # Check all required files exist
    missing_files = []
    for file_type, filename in required_files.items():
        if not (data_path / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        err_msg = f"Missing required files in {data_path}: {missing_files}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
    
    
    try:
        # Load image
        image_path = data_path / required_files['image']
        image = cv2.imread(str(image_path))
        if image is None:
            err_msg = f"Could not load image: {image_path}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = data_path / required_files['mask']
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            err_msg = f"Could not load mask: {mask_path}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        mask = (mask > 127).astype(np.uint8)  # Binary threshold
        
        # Load labels
        labels_path = data_path / required_files['labels']
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            
    except Exception as e:
        err_msg = f"Failed to load ground truth data: {e}"
        logger.error(err_msg)
        raise ValueError(err_msg)
        
    logger.debug("loading gt data complete...")
    return image, mask, labels
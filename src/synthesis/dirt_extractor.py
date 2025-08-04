import logging
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rank_filter

from src.core.logger_config import setup_application_logger
from src.synthesis.image_operations import ImageOperations


class DirtExtractor:
    def __init__(self, image_operations:ImageOperations, app_logger:Optional[logging.Logger]=None):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('DirtExtractor')
        self.image_operations = image_operations

    # Define the rank filter function
    def apply_rank_filter(self, image, rank, size):
        """
        Apply a rank filter to an image.
        
        :param image: Input grayscale image (numpy array).
        :param rank: Rank to apply (e.g., 0 for min, 1 for max, 0.5 for median).
        :param size: Size of the filter kernel (e.g., 3 for a 3x3 kernel).
        :return: Filtered image.
        """
        pixel_rank = int(size * size * rank)
        return rank_filter(image, rank=pixel_rank, size=(size, size))
    
    def estimate_background(self, dirt_img, bg_estimation_filter_size=51):
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dirt_eroded = cv2.morphologyEx(dirt_img, cv2.MORPH_DILATE, kernel2)
        estimated_clean_background = cv2.medianBlur(dirt_eroded, bg_estimation_filter_size)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        estimated_clean_background = cv2.morphologyEx(estimated_clean_background, cv2.MORPH_CLOSE, kernel)
        estimated_clean_background = cv2.GaussianBlur(estimated_clean_background, (5, 5), 0)

        return estimated_clean_background
    
    def create_dirt_masks(self, dirt_img, clean_bg):

        img_float = dirt_img.astype(np.float32)
        clean_bg_float = clean_bg.astype(np.float32)

        # For mask creation, use absolute difference
        abs_diff = cv2.absdiff(img_float, clean_bg_float)
        norm_diff = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Filter noise in norm_diff
        filtered_diff = cv2.bilateralFilter(norm_diff, 25, 30, 11)
        
        # Multiple thresholding strategies
        # 1. Otsu's method for automatic threshold
        threshold, high_thresh_mask = cv2.threshold(filtered_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, low_thresh_mask = cv2.threshold(filtered_diff, threshold-10, 255, cv2.THRESH_BINARY) # -10 is optional, one can adjust this value in 0..20 interval
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        low_thresh_mask = cv2.morphologyEx(low_thresh_mask, cv2.MORPH_DILATE, kernel) # optional, just another option to make this mask more "thick"

        return low_thresh_mask, high_thresh_mask, abs_diff
    
    def create_dirt_masks_deprecated(Self, dirt_img, clean_bg):

        img_float = dirt_img.astype(np.float32)
        clean_bg_float = clean_bg.astype(np.float32)

        # For mask creation, use absolute difference
        abs_diff = cv2.absdiff(img_float, clean_bg_float)
        norm_diff = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Apply Gaussian blur to reduce noise
        blurred_diff = cv2.GaussianBlur(norm_diff, (3, 3), 0)
        
        # Multiple thresholding strategies
        # 1. Otsu's method for automatic threshold
        _, otsu_mask = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. Triangle method for bimodal distribution
        _, triangle_mask = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        
        # 3. Adaptive threshold for varying lighting
        adaptive_mask = cv2.adaptiveThreshold(blurred_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
        
        # Combine masks using logical operations
        combined_mask = cv2.bitwise_or(otsu_mask, triangle_mask)
        
        # Clean up masks with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Low threshold (more sensitive)
        low_thresh_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        low_thresh_mask = cv2.morphologyEx(low_thresh_mask, cv2.MORPH_OPEN, kernel)
        
        # High threshold (less sensitive)
        high_thresh_mask = cv2.morphologyEx(triangle_mask, cv2.MORPH_OPEN, kernel)
        high_thresh_mask = cv2.morphologyEx(high_thresh_mask, cv2.MORPH_CLOSE, kernel)

        return low_thresh_mask, high_thresh_mask, norm_diff

    def process_image(self, image_path: str, bg_estimation_filter_size=51):
        # Load image
        img = self.image_operations.load_image(image_path)

        # estimate background
        background = self.estimate_background(img, bg_estimation_filter_size)

        # Create masks
        low_mask, high_mask, diff_image = self.create_dirt_masks(img, background,)

        return {
            'original': img,
            'background': background,
            'difference': diff_image,
            'low_threshold_mask': low_mask,
            'high_threshold_mask': high_mask,
        }
    
    def visualize_results(self, results: dict, save_path: Optional[str] = None, show_results=True):
        """Visualize all processing results"""
        plt.figure(figsize=(15, 5))
        plt.suptitle('Mobile Screen Dirt Detection Results', fontsize=16)

        plt.subplot(1, 4, 1)
        plt.imshow(results['original'], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(results['difference'], cmap='gray')
        plt.title('norm_diff')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(results['low_threshold_mask'], cmap='gray')
        plt.title('low_thresh_mask')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(results["high_threshold_mask"], cmap='gray')
        plt.title('high_thresh_mask')
        plt.axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_results:
            plt.show()

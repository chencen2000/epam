import os
import random
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.core.logger_config import setup_application_logger


class ImageProcessor:

    _SUPPORTED_FORMAT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def __init__(self, app_logger:Optional[logging.Logger]=None):
        self.logger = app_logger or setup_application_logger().getChild('ImageOperations')

    # Image operations

    def _check_img_path_exists(self, path:str):
        img_path = Path(path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        if img_path.suffix.lower() not in self._SUPPORTED_FORMAT:
            raise ValueError(
                f"Unsupported image format: {img_path.suffix}. Supported formats: {self._SUPPORTED_FORMAT}"
            )
        return True

    def load_image(self, image_path:str, grey_scale:bool=True):
        # Validate the path first
        self._check_img_path_exists(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if grey_scale else cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img 
  
    def save_image(self, image, path:str):
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        return cv2.imwrite(path, image)
    
    def resize_image(self, image, downscale_factor=2):
        if downscale_factor == 1:
            return image
             
        h, w = image.shape[:2]
        new_h, new_w = h // downscale_factor , w // downscale_factor
        self.logger.debug(f"Got Image shape - {w}*{h} resizing to {new_w}*{new_h}")

        downscaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.logger.debug(f"Image downscaled to {downscaled_img.shape[1]}x{downscaled_img.shape[0]} from {w}x{h}.")
        return downscaled_img

    def apply_scale(self, image, scale, interpolation=cv2.INTER_CUBIC):
        """
        Apply scaling with cubic interpolation by default
        """
        if scale <= 0: return image, image.shape[1], image.shape[0]
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
        if new_width <= 0 or new_height <= 0: return None, 0, 0
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation), new_width, new_height
    
    def apply_rotation(self, image, angle, interpolation=cv2.INTER_CUBIC, border_value=0):
        """
        Apply rotation with cubic interpolation by default
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
        ones = np.ones(shape=(len(corners), 1))
        points_ones = np.hstack([corners, ones])
        transformed_points = M.dot(points_ones.T).T
        
        x_coords, y_coords = transformed_points[:,0], transformed_points[:,1]
        nW_robust = int(np.ceil(x_coords.max() - x_coords.min()))
        nH_robust = int(np.ceil(y_coords.max() - y_coords.min()))

        M[0,2] += (nW_robust / 2) - center[0]
        M[1,2] += (nH_robust / 2) - center[1]
        
        rotated_image = cv2.warpAffine(image, M, (nW_robust, nH_robust), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
        return rotated_image, nW_robust, nH_robust
    
    #  bounding box

    def get_margin_bbox(self, margins, img_w, img_h):
        b_xmin = int(img_w * (margins["left"] / 100.0))
        b_ymin = int(img_h * (margins["top"] / 100.0))
        b_xmax = int(img_w * (1.0 - margins["right"] / 100.0))
        b_ymax = int(img_h * (1.0 - margins["bottom"] / 100.0))
        return b_xmin, b_ymin, b_xmax, b_ymax
    
    # mobile boundary detector

    def get_device_bbox(self, image,):
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # resized = cv2.resize(gray, None, fx=scale, fy=scale)
        filtered = cv2.medianBlur(gray, 7) # to remove bright stripes from PowerON samples, glares, etc.
        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Threshold the image to binary
        # Safe range for threshold [55..79], check edge cases, they fail if threshold goes out of this range:
        _, binary_image = cv2.threshold(normalized, 64, 255, cv2.THRESH_BINARY, cv2.CV_8U)
        
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return int(x), int(y), int(w), int(h)    
        
        # Default fall-off (if binarization didn't produce anything, which is weird)
        h, w = gray.shape
        xmin = int(w * 0.12)
        ymin = int(h * 0.15)
        xmax = int(w * (1.0 - 0.12))
        ymax = int(h * (1.0 - 0.15))
        return xmin, ymin, xmax-xmin, ymax-ymin
    

    def check_overlap(self, new_bbox, placed_bboxes):
        for placed_bbox in placed_bboxes:
            if (new_bbox[0] < placed_bbox[2] and new_bbox[2] > placed_bbox[0] and
                new_bbox[1] < placed_bbox[3] and new_bbox[3] > placed_bbox[1]):
                return True
        return False
    
    # distort edges

    def distort_point(self, point, max_distortion=0.1):
        """Randomly distort a point within a given range."""
        x, y = point
        x += random.uniform(-max_distortion, max_distortion)
        y += random.uniform(-max_distortion, max_distortion)
        return (x, y)

    def add_vertices(self, square, num_vertices_per_side=3, max_distortion=0.1):
        """Add vertices to the sides of a square and distort them."""
        new_polygon = []
        for i in range(len(square)):
            start = square[i]
            end = square[(i + 1) % len(square)]
            
            # Add the starting vertex
            new_polygon.append(self.distort_point(start, max_distortion))
            
            # Add intermediate vertices
            for j in range(1, num_vertices_per_side + 1):
                t = j / (num_vertices_per_side + 1)
                intermediate_point = (
                    start[0] + t * (end[0] - start[0]),
                    start[1] + t * (end[1] - start[1])
                )
                new_polygon.append(self.distort_point(intermediate_point, max_distortion))
        
        return new_polygon

    def find_bounding_box(self, polygon):
        """Find bounding box of polygon."""
        left = min(polygon, key=lambda x: x[0])[0]
        bottom = min(polygon, key=lambda x: x[1])[1]
        right = max(polygon, key=lambda x: x[0])[0]
        top = max(polygon, key=lambda x: x[1])[1]
        return (left, bottom, right, top)

    def normalize_polygon_to_size(self, polygon, target_width, target_height):
        """Normalize polygon to fit within target dimensions."""
        left, bottom, right, top = self.find_bounding_box(polygon)
        # normalize each point in the polygon
        return [((x-left) / (right-left) * target_width, (y-bottom) / (top-bottom) * target_height) for x, y in polygon]

    def create_distorted_mask_boundary(self, mask_shape, num_vertices_per_side=3, max_distortion=0.1):
        """
        Create a distorted polygon boundary mask from the original rectangular mask.
        Returns the distorted mask.
        """
        height, width = mask_shape
        
        # Define the initial square (normalized coordinates)
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        
        # Add vertices and distort them
        distorted_polygon = self.add_vertices(square, num_vertices_per_side=num_vertices_per_side, max_distortion=max_distortion)
        
        # Normalize to mask dimensions
        normalized_polygon = self.normalize_polygon_to_size(distorted_polygon, width, height)
        
        # Create distorted boundary mask
        distorted_boundary_mask = np.zeros((height, width), dtype=np.uint8)
        polygon_vertices = np.array(normalized_polygon, dtype=np.int32)
        cv2.fillPoly(distorted_boundary_mask, [polygon_vertices], color=255)
        
        return distorted_boundary_mask

    def apply_distorted_boundary_to_mask(self, original_mask, num_vertices_per_side=3, max_distortion=0.1):
        """
        Apply distorted polygon boundary to an existing mask.
        """
        # Create distorted boundary mask
        distorted_boundary = self.create_distorted_mask_boundary(
            original_mask.shape, 
            num_vertices_per_side=num_vertices_per_side, 
            max_distortion=max_distortion
        )
        
        # Apply the distorted boundary to the original mask
        distorted_mask = cv2.bitwise_and(original_mask, distorted_boundary)
        
        return distorted_mask

    # dirt extractor

    def estimate_dirt_background(self, dirt_img, bg_estimation_filter_size=51):
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
    
    def process_dirt_mask_extraction(self, image_path: str, bg_estimation_filter_size=51, resize_factor:int =False):
        # Load image
        img = self.load_image(image_path, True)
        if resize_factor:
            img = self.resize_image(img, resize_factor)

        # estimate background
        background = self.estimate_dirt_background(img, bg_estimation_filter_size)

        # Create masks
        low_mask, high_mask, diff_image = self.create_dirt_masks(img, background,)

        return {
            'original': img,
            'background': background,
            'difference': diff_image,
            'low_threshold_mask': low_mask,
            'high_threshold_mask': high_mask,
        }
    
    def visualize_dirt_results(self, results: dict, save_path: Optional[str] = None, show_results=True):
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


    # Preprocess

    def preprocess_prediction_image(self, image: np.ndarray, device: str, 
                               target_size: Tuple[int, int] = (1024, 1024),
                               config: Dict = None) -> torch.Tensor:
        """
        Preprocess image for prediction - FIXED FOR GRAYSCALE MODEL
        
        Args:
            image: Input image (RGB or grayscale)
            device: Target device
            target_size: Target size for resizing
            config: Model configuration
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to grayscale if RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.logger.debug(f"Converted RGB image {image.shape} to grayscale {gray_image.shape}")
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Already single channel, squeeze
            gray_image = image.squeeze(-1)
        elif len(image.shape) == 2:
            # Already grayscale
            gray_image = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Resize if needed
        if gray_image.shape[:2] != target_size:
            gray_image = cv2.resize(gray_image, target_size, interpolation=cv2.INTER_CUBIC)
            self.logger.debug(f"Resized image to {target_size}")
        
        # Normalize to [0, 1]
        if gray_image.dtype == np.uint8:
            gray_image = gray_image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch + channel dimensions
        # Shape: (H, W) -> (1, 1, H, W)
        tensor = torch.from_numpy(gray_image).unsqueeze(0).unsqueeze(0).to(device)
        
        self.logger.debug(f"Final tensor shape: {tensor.shape}")
        
        return tensor
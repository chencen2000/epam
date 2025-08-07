import os
import random
import logging
from typing import Optional, Tuple, Dict
from pathlib import Path

import cv2
import torch
import numpy as np
from torch import Tensor

from src.data.transformations import get_transforms
from src.core.logger_config import setup_application_logger


class ImageOperations:

    _SUPPORTED_FORMAT = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    def __init__(self, app_logger:Optional[logging.Logger]=None):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('ImageOperations')

    def _check_img_path_exists(self, path:str):
        img_path = Path(path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        
        if img_path.suffix.lower() not in self._SUPPORTED_FORMAT:
            raise ValueError(f"Unsupported image format: {img_path.suffix}. "
                           f"Supported formats: {self._SUPPORTED_FORMAT}")
        
        return True
        

    def load_image(self, image_path:str):
        # Validate the path first
        self._check_img_path_exists(image_path)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img

    def load_image_color(self, image_path):
        # Validate the path first
        self._check_img_path_exists(image_path)
        
        return cv2.imread(image_path, cv2.IMREAD_COLOR)
  
    def save_image(self, image, path):
        output_dir = os.path.dirname(path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        return cv2.imwrite(path, image)
    
    def get_margin_bbox(self, margins, img_w, img_h):
        b_xmin = int(img_w * (margins["left"] / 100.0))
        b_ymin = int(img_h * (margins["top"] / 100.0))
        b_xmax = int(img_w * (1.0 - margins["right"] / 100.0))
        b_ymax = int(img_h * (1.0 - margins["bottom"] / 100.0))

        return b_xmin, b_ymin, b_xmax, b_ymax
    
    def resize_image(self, image, downscale_factor=2):
        h, w = image.shape[:2]

        self.logger.debug(f"{w = } -> {w // downscale_factor}\n{h = } -> {h // downscale_factor}")
        downscaled_img = cv2.resize(
            image, 
            (w // downscale_factor, h // downscale_factor), 
            interpolation=cv2.INTER_AREA
        )
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
    
    def check_overlap(self, new_bbox, placed_bboxes):
        for placed_bbox in placed_bboxes:
            if (new_bbox[0] < placed_bbox[2] and new_bbox[2] > placed_bbox[0] and
                new_bbox[1] < placed_bbox[3] and new_bbox[3] > placed_bbox[1]):
                return True
        return False
    
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

    def preprocess_prediction_image(self,  image: np.ndarray, device:str="auto", target_size: Tuple[int, int] = (1792, 1792), config={}) -> Tensor:
        """Preprocess single image for inference"""
        # if target_size is None:
        #     target_size = self.config.get('target_size', (1024, 1024))
            
        transform = get_transforms(target_size, config)
        
        # Apply transforms
        transformed = transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(device)
    

    def preprocess_prediction_image(self, image: np.ndarray, device: str, 
                               target_size: Tuple[int, int] = (1792, 1792),
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

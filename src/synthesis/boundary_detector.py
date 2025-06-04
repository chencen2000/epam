import logging
from typing import Optional

import cv2
import numpy as np

from src.core.logger_config import setup_application_logger
from src.synthesis.image_operations import ImageOperations


class BoundaryDetector:
    def __init__(self, image_operations:ImageOperations, app_logger:Optional[logging.Logger]=None):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.image_operations = image_operations
        self.logger = app_logger.getChild('BoundaryDetector')

    def detect_mobile_boundaries(self, image, draw=False):
        """
        Enhanced phone screen boundary detection optimized for phones with rounded corners
        and complex backgrounds
        """
        # in percentage
        default_margins = {
            "top": 15,
            "bottom": 15,
            "left": 12,
            "right": 12
        }
        x, y, w, h = None, None, None, None

        img_h, img_w, _ = image.shape
        if image is None:
            self.logger.error("Error: Input image is None. Cannot detect boundaries.")
            return x, y, w, h

        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
            self.logger.warning("Warning: Input image was not a standard 3-channel BGR image. Proceeding as grayscale.")
        
        self.logger.debug(f"Original image shape: {image.shape}")
        
        # Try multiple approaches in order of reliability for this type of image
        boundary_methods = [
            self._detect_via_intensity_gradient,
            self._detect_via_morphological_operations,
            self._detect_via_contour_filtering,
            self._detect_via_edge_accumulation
        ]
        
        for method in boundary_methods:
            try:
                x, y, w, h = method(gray_image)
                if x is not None and self._validate_boundary(x, y, w, h, gray_image.shape):
                    self.logger.debug(f"Boundary detected using {method.__name__}")
                    break
            except Exception as e:
                self.logger.error(f"Method {method.__name__} failed: {e}")
                continue
        
        if x is not None and draw:
            # Draw main rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 6)
            
            # Draw corner markers
            corner_size = 30
            thickness = 8
            # Top-left
            cv2.line(image, (x, y), (x + corner_size, y), (255, 0, 0), thickness)
            cv2.line(image, (x, y), (x, y + corner_size), (255, 0, 0), thickness)
            # Top-right  
            cv2.line(image, (x + w, y), (x + w - corner_size, y), (255, 0, 0), thickness)
            cv2.line(image, (x + w, y), (x + w, y + corner_size), (255, 0, 0), thickness)
            # Bottom-left
            cv2.line(image, (x, y + h), (x + corner_size, y + h), (255, 0, 0), thickness)
            cv2.line(image, (x, y + h), (x, y + h - corner_size), (255, 0, 0), thickness)
            # Bottom-right
            cv2.line(image, (x + w, y + h), (x + w - corner_size, y + h), (255, 0, 0), thickness)
            cv2.line(image, (x + w, y + h), (x + w, y + h - corner_size), (255, 0, 0), thickness)
            
            # Add center cross
            center_x, center_y = x + w//2, y + h//2
            cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 4)
            cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 4)
        
        if x is None:
            self.logger.warning("No mobile screen boundaries detected with any method.\nUsing default values - (t, b, l, r) - (15, 15, 12, 12)%")
            x, y, w, h = self.image_operations.get_margin_bbox(default_margins, img_w, img_h)
            self.logger.debug(f"Using boundary: x={x}, y={y}, w={w}, h={h}")
        else:
            self.logger.debug(f"Detected boundary: x={x}, y={y}, w={w}, h={h}")
        
        return x, y, w, h
    
    def _detect_via_intensity_gradient(self, gray_image):
        """Detect phone boundary using intensity gradients - works well for bright screens"""
        height, width = gray_image.shape
        
        # Apply strong gaussian blur to remove noise and small details
        blurred = cv2.GaussianBlur(gray_image, (21, 21), 0)
        
        # Find the brightest region (likely the phone screen)
        # Use morphological opening to remove small bright spots
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        
        # Find the maximum intensity value and create a threshold
        max_val = np.max(opened)
        threshold_val = max(max_val * 0.7, 128)  # At least 70% of max brightness
        
        # Create binary mask of bright regions
        _, bright_mask = cv2.threshold(opened, threshold_val, 255, cv2.THRESH_BINARY)
        
        # Clean up the mask
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel_clean)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Find the largest connected component
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None, None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Check if the area is reasonable (at least 20% of image)
        if area < (height * width * 0.2):
            return None, None, None, None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Expand the boundary slightly to ensure we capture the edges
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(width - x, w + 2 * padding)
        h = min(height - y, h + 2 * padding)
        
        return x, y, w, h

    def _detect_via_morphological_operations(self, gray_image):
        """Use morphological operations to find rectangular regions"""
        height, width = gray_image.shape
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_image)
        
        # Apply median filter to reduce noise
        filtered = cv2.medianBlur(enhanced, 9)
        
        # Use adaptive threshold to handle varying lighting
        binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 2)
        
        # Create rectangular kernel for morphological operations
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        
        # Close operation to fill gaps in phone screen
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_rect, iterations=2)
        
        # Open operation to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_rect)
        
        # Find contours
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None, None
        
        # Filter contours by area and shape
        valid_contours = []
        min_area = height * width * 0.15  # At least 15% of image area
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Check rectangularity using contour approximation
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Phone-like aspect ratio
                if 0.3 < aspect_ratio < 0.9:
                    extent = float(area) / (w * h)
                    if extent > 0.6:  # At least 60% filled rectangle
                        valid_contours.append((contour, area, x, y, w, h))
        
        if valid_contours:
            # Sort by area and return largest
            valid_contours.sort(key=lambda x: x[1], reverse=True)
            return valid_contours[0][2], valid_contours[0][3], valid_contours[0][4], valid_contours[0][5]
        
        return None, None, None, None

    def _detect_via_contour_filtering(self, gray_image):
        """Enhanced contour detection with better filtering"""
        height, width = gray_image.shape
        
        # Bilateral filter to preserve edges while smoothing
        bilateral = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # Multiple threshold values to catch different lighting conditions
        thresholds = [100, 128, 150, 180]
        all_contours = []
        
        for thresh_val in thresholds:
            _, binary = cv2.threshold(bilateral, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        if not all_contours:
            return None, None, None, None
        
        # Score contours based on multiple criteria
        scored_contours = []
        min_area = height * width * 0.1
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate scores
            area_score = min(1.0, area / (height * width * 0.5))
            aspect_ratio = float(w) / h
            aspect_score = 1.0 if 0.4 <= aspect_ratio <= 0.8 else max(0, 1.0 - abs(aspect_ratio - 0.6))
            
            # Position score (prefer centered)
            center_x, center_y = x + w//2, y + h//2
            img_center_x, img_center_y = width//2, height//2
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
            position_score = 1.0 - (distance / max_distance)
            
            total_score = area_score * 0.5 + aspect_score * 0.3 + position_score * 0.2
            scored_contours.append((x, y, w, h, total_score))
        
        if scored_contours:
            best_contour = max(scored_contours, key=lambda x: x[4])
            return best_contour[0], best_contour[1], best_contour[2], best_contour[3]
        
        return None, None, None, None

    def _detect_via_edge_accumulation(self, gray_image):
        """Detect boundary by accumulating edge information"""
        height, width = gray_image.shape
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Multiple Canny edge detection with different parameters
        edge_maps = []
        canny_params = [(50, 150), (30, 100), (100, 200)]
        
        for low, high in canny_params:
            edges = cv2.Canny(blurred, low, high)
            edge_maps.append(edges)
        
        # Combine edge maps
        combined_edges = np.zeros_like(edge_maps[0])
        for edge_map in edge_maps:
            combined_edges = cv2.bitwise_or(combined_edges, edge_map)
        
        # Dilate edges to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(combined_edges, kernel, iterations=2)
        
        # Find contours from edges
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, None, None
        
        # Find the contour that best represents a phone screen
        best_contour = None
        best_score = 0
        min_area = height * width * 0.1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Score based on area and aspect ratio
            area_score = min(1.0, area / (height * width * 0.4))
            aspect_score = 1.0 if 0.4 <= aspect_ratio <= 0.8 else 0.5
            
            total_score = area_score * aspect_score
            
            if total_score > best_score:
                best_score = total_score
                best_contour = (x, y, w, h)
        
        if best_contour:
            return best_contour
        
        return None, None, None, None

    def _validate_boundary(self, x, y, w, h, image_shape):
        """Validate that the detected boundary makes sense"""
        height, width = image_shape
        
        # Check if boundary is within image
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return False
        
        # Check minimum size
        if w < width * 0.3 or h < height * 0.3:
            return False
        
        # Check maximum size (shouldn't be the entire image)
        if w > width * 0.95 or h > height * 0.95:
            return False
        
        # Check aspect ratio
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.2 or aspect_ratio > 1.5:
            return False
        
        return True


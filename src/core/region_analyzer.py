"""Region analysis utilities."""

import logging
from typing import Dict, Optional

import cv2
import numpy as np

from src.core.logger_config import setup_application_logger


class RegionAnalyzer:
    """Class for analyzing dirt regions in predictions."""
    
    def __init__(self, app_logger:Optional[logging.Logger]=None):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('RegionAnalyzer')
    
    def analyze_dirt_regions(self, binary_prediction: np.ndarray) -> Dict:
        """Analyze connected components of dirt regions."""
        try:
            from scipy import ndimage
            from skimage import measure
            return self._analyze_dirt_regions_scipy(binary_prediction)
        except ImportError:
            return self._analyze_dirt_regions_opencv(binary_prediction)
    
    def _analyze_dirt_regions_scipy(self, binary_prediction: np.ndarray) -> Dict:
        """Analyze regions using scipy and skimage."""
        from scipy import ndimage
        from skimage import measure
        
        labeled_array, num_features = ndimage.label(binary_prediction)
        
        if num_features == 0:
            return self._empty_analysis()
        
        regions = measure.regionprops(labeled_array)
        areas = [region.area for region in regions]
        
        return {
            'num_regions': num_features,
            'total_area': sum(areas),
            'largest_region_area': max(areas),
            'smallest_region_area': min(areas),
            'average_region_area': np.mean(areas),
            'median_region_area': np.median(areas),
            'region_areas': areas,
            'region_centroids': [region.centroid for region in regions],
            'region_bboxes': [region.bbox for region in regions]
        }
    
    def _analyze_dirt_regions_opencv(self, binary_prediction: np.ndarray) -> Dict:
        """Fallback region analysis using OpenCV."""
        contours, _ = cv2.findContours(binary_prediction.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._empty_analysis()
        
        areas, centroids, bboxes = [], [], []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            areas.append(area)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cy, cx))
            else:
                centroids.append((0, 0))
            
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((y, x, y+h, x+w))
        
        return {
            'num_regions': len(contours),
            'total_area': sum(areas),
            'largest_region_area': max(areas) if areas else 0,
            'smallest_region_area': min(areas) if areas else 0,
            'average_region_area': np.mean(areas) if areas else 0,
            'median_region_area': np.median(areas) if areas else 0,
            'region_areas': areas,
            'region_centroids': centroids,
            'region_bboxes': bboxes
        }
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure."""
        return {
            'num_regions': 0, 'total_area': 0, 'largest_region_area': 0,
            'smallest_region_area': 0, 'average_region_area': 0, 'median_region_area': 0,
            'region_areas': [], 'region_centroids': [], 'region_bboxes': []
        }

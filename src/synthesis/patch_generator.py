import logging
from typing import Optional, List, Union, Tuple

import cv2
import numpy as np

from src.core.logger_config import setup_application_logger


class PatchGenerator:
    def __init__(self, app_logger:Optional[logging.Logger]=None):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('PatchGenerator')
        self.logger.debug("Patch generator initialized")

    def split_image_into_patches(self, image: np.ndarray, 
                            mask: np.ndarray, 
                            annotations: List[dict], 
                            patch_size: int,
                            full_img_boundary_x_min: int, 
                            full_img_boundary_y_min: int, 
                            full_img_boundary_x_max: int, 
                            full_img_boundary_y_max: int,
                            overlap: Union[int, float] = 0,
                            return_empty_patches: bool = False) -> List[Tuple]:
        """
        Split image into patches with optional overlap and optimized performance.
        
        Args:
            image: Input image array
            mask: Corresponding mask array
            annotations: List of annotation dictionaries with 'bbox', 'segmentation', 'category_id', 'area'
            patch_size: Size of each patch (assuming square patches)
            full_img_boundary_*: Boundary coordinates for patch extraction
            overlap: Overlap between patches. If int, pixel overlap. If float (0-1), percentage overlap
            return_empty_patches: Whether to return patches without annotations
            
        Returns:
            List of tuples: (patch_img, patch_mask, adjusted_annotations, x_start, y_start)
        """
        
        # Calculate step size based on overlap
        if isinstance(overlap, float):
            if not 0 <= overlap < 1:
                raise ValueError("Float overlap must be between 0 and 1")
            step_size = int(patch_size * (1 - overlap))
        else:
            step_size = patch_size - overlap
        
        if step_size <= 0:
            raise ValueError("Step size must be positive")
        
        patches_data = []
        
        # Early exit if no annotations and not returning empty patches
        if not annotations and not return_empty_patches:
            return patches_data
        
        # Pre-calculate annotation data for faster processing
        ann_data = self._precalculate_annotation_data(annotations)
        
        # Pre-allocate patch arrays to avoid repeated allocation
        patch_img_template = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
        patch_mask_template = np.zeros((patch_size, patch_size), dtype=mask.dtype)
        
        # Generate patch coordinates
        y_coords = range(full_img_boundary_y_min, full_img_boundary_y_max, step_size)
        x_coords = range(full_img_boundary_x_min, full_img_boundary_x_max, step_size)
        
        # Process patches in batches for better memory efficiency
        for y_start in y_coords:
            for x_start in x_coords:
                y_end = min(y_start + patch_size, full_img_boundary_y_max)
                x_end = min(x_start + patch_size, full_img_boundary_x_max)
                
                # Extract patch regions
                patch_h = y_end - y_start
                patch_w = x_end - x_start
                
                # Create patches efficiently
                patch_img = patch_img_template.copy()
                patch_img[:patch_h, :patch_w] = image[y_start:y_end, x_start:x_end]
                
                patch_mask = patch_mask_template.copy()
                patch_mask[:patch_h, :patch_w] = mask[y_start:y_end, x_start:x_end]
                
                # Process annotations for this patch
                adjusted_annotations = self._process_patch_annotations(
                    ann_data, x_start, y_start, x_end, y_end, patch_size
                )
                
                # Add patch if it has annotations or if returning empty patches
                if adjusted_annotations or return_empty_patches:
                    patches_data.append((patch_img, patch_mask, adjusted_annotations, x_start, y_start))
        
        return patches_data


    def _precalculate_annotation_data(self, annotations: List[dict]) -> List[Optional[dict]]:
        """Pre-calculate annotation masks and bounding boxes for faster processing."""
        ann_data = []
        
        for ann in annotations:
            try:
                seg_poly = ann['segmentation'][0]  # Assuming single polygon segmentation
                points = np.array(seg_poly).reshape(-1, 2)
                
                # Calculate tight bounding box
                min_x, min_y = np.min(points, axis=0).astype(int)
                max_x, max_y = np.max(points, axis=0).astype(int)
                
                ann_w = max_x - min_x + 1
                ann_h = max_y - min_y + 1
                
                if ann_w <= 0 or ann_h <= 0:
                    ann_data.append(None)
                    continue
                
                # Create optimized local mask
                local_mask = np.zeros((ann_h, ann_w), dtype=np.uint8)
                translated_points = (points - [min_x, min_y]).astype(np.int32)
                cv2.fillPoly(local_mask, [translated_points], 255)
                
                ann_data.append({
                    'local_mask': local_mask,
                    'bbox': ann['bbox'],
                    'bounds': (min_x, min_y, max_x, max_y),
                    'segmentation': ann['segmentation'],
                    'category_id': ann['category_id'],
                    'original_area': ann.get('area', 0),
                    "category_name": ann.get("category_name", "unknown"),
                })
                
            except (KeyError, IndexError, ValueError):
                ann_data.append(None)
        
        return ann_data


    def _process_patch_annotations(self, ann_data: List[Optional[dict]], 
                                x_start: int, y_start: int, 
                                x_end: int, y_end: int, 
                                patch_size: int) -> List[dict]:
        """Process annotations for a specific patch with optimized intersection calculation."""
        adjusted_annotations = []
        
        for ann_info in ann_data:
            if ann_info is None:
                continue
                
            bbox_x, bbox_y, bbox_w, bbox_h = ann_info['bbox']
            min_x, min_y, max_x, max_y = ann_info['bounds']
            
            # Quick bounding box intersection check
            if not self._boxes_intersect(bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h,
                                x_start, y_start, x_end, y_end):
                continue
            
            # Calculate intersection coordinates
            intersect_x1 = max(bbox_x, x_start)
            intersect_y1 = max(bbox_y, y_start)
            intersect_x2 = min(bbox_x + bbox_w, x_end)
            intersect_y2 = min(bbox_y + bbox_h, y_end)
            
            # Adjust bounding box to patch coordinates
            adj_bbox_x = intersect_x1 - x_start
            adj_bbox_y = intersect_y1 - y_start
            adj_bbox_w = intersect_x2 - intersect_x1
            adj_bbox_h = intersect_y2 - intersect_y1
            
            # Adjust segmentation coordinates
            adj_seg_polys = []
            for seg_poly in ann_info['segmentation']:
                points = np.array(seg_poly).reshape(-1, 2)
                translated_points = points - [x_start, y_start]
                adj_seg_polys.append(translated_points.flatten().tolist())
            
            # Calculate new area using pre-calculated mask
            new_area = self._calculate_intersection_area(
                ann_info['local_mask'], min_x, min_y, x_start, y_start, x_end, y_end
            )
            
            if new_area > 0:
                adjusted_annotations.append({
                    "bbox": [adj_bbox_x, adj_bbox_y, adj_bbox_w, adj_bbox_h],
                    "segmentation": adj_seg_polys,
                    "category_id": ann_info['category_id'],
                    "area": new_area,
                    "category_name": ann_info["category_name"]
                })
        
        return adjusted_annotations


    def _boxes_intersect(self, x1: int, y1: int, x2: int, y2: int,
                        x3: int, y3: int, x4: int, y4: int) -> bool:
        """Fast bounding box intersection check."""
        return not (x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1)


    def _calculate_intersection_area(self, local_mask: np.ndarray, 
                                ann_min_x: int, ann_min_y: int,
                                patch_x_start: int, patch_y_start: int,
                                patch_x_end: int, patch_y_end: int) -> int:
        """Calculate intersection area using pre-calculated local mask."""
        # Calculate intersection in local mask coordinates
        intersect_local_x1 = max(0, patch_x_start - ann_min_x)
        intersect_local_y1 = max(0, patch_y_start - ann_min_y)
        intersect_local_x2 = min(local_mask.shape[1], patch_x_end - ann_min_x)
        intersect_local_y2 = min(local_mask.shape[0], patch_y_end - ann_min_y)
        
        if intersect_local_x2 <= intersect_local_x1 or intersect_local_y2 <= intersect_local_y1:
            return 0
        
        # Extract intersecting region and count non-zero pixels
        roi = local_mask[intersect_local_y1:intersect_local_y2, 
                        intersect_local_x1:intersect_local_x2]
        return cv2.countNonZero(roi)


    def _create_prediction_patch_from_screen(self, screen_image: np.ndarray, 
                                  patch_size: int, overlap: float) -> List[Tuple]:
        """Create patches from the detected screen area"""
        # Create dummy mask and annotations for patch generator
        dummy_mask = np.ones((screen_image.shape[0], screen_image.shape[1]), dtype=np.uint8) * 255
        dummy_annotations = []  # No annotations needed for inference
        
        # Calculate boundaries (use entire screen)
        screen_h, screen_w = screen_image.shape[:2]
        
        patches_data = self.split_image_into_patches(
            image=screen_image,
            mask=dummy_mask,
            annotations=dummy_annotations,
            patch_size=patch_size,
            full_img_boundary_x_min=0,
            full_img_boundary_y_min=0, 
            full_img_boundary_x_max=screen_w,
            full_img_boundary_y_max=screen_h,
            overlap=overlap,
            return_empty_patches=True  # We want all patches for inference
        )
        
        return patches_data
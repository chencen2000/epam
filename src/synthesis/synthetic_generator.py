import os
import json
import random
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# import seaborn as sns

from src.core.logger_config import setup_application_logger
from src.synthesis.dirt_extractor import DirtExtractor
from src.synthesis.patch_generator import PatchGenerator
from src.synthesis.image_operations import ImageOperations
from src.synthesis.boundary_detector import BoundaryDetector


class SyntheticDataGenerator:
    def __init__(
            self, 
            dirt_extractor:DirtExtractor,
            image_operations:ImageOperations,
            boundary_detector:BoundaryDetector,
            patch_generator: PatchGenerator,
            clean_images_dir:str, 
            dirt_base_dir: str,  # Changed from dirt_images_dir
            dirt_categories: dict,  # New parameter 
            num_version_per_image:int=1,
            num_dirt_super_impose:int=40,
            clean_image_downscale_factor:int=2,
            scale_dirt:Tuple[int, int]=(0.9, 1.15),
            rotation_range:Tuple[int, int]=(0, 360),
            app_logger:Optional[logging.Logger]=None
            ):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('SyntheticDataGenerator')

        self.clean_dir = clean_images_dir
        self.dirt_base_dir = dirt_base_dir
        self.dirt_categories = dirt_categories
        self.category_weights = [cat['weight'] for cat in dirt_categories.values()]
        self.category_names = list(dirt_categories.keys())
        
        self.dirt_extractor = dirt_extractor
        self.image_operations = image_operations
        self.boundary_detector = boundary_detector
        self.patch_gen = patch_generator
        self.num_version_per_image = num_version_per_image
        self.num_dirt_super_impose = num_dirt_super_impose
        self.clean_image_downscale_factor = clean_image_downscale_factor
        self.scale_range = scale_dirt
        self.rotation_range = rotation_range

        # Define distinct colors for each category
        self.category_colors = self._initialize_category_colors()
        self.category_color_map = self._create_category_colormap()

    def _initialize_category_colors(self):
        """Initialize distinct colors for each dirt category"""
        # Use a colorblind-friendly palette
        base_colors = [
            [255, 255, 0], # Yellow 
            [255, 0, 0],      # Red 
            [0, 255, 0],      # Green 
            [0, 0, 255],      # Blue 
            [255, 0, 255],    # Magenta -
            [0, 255, 255],    # Cyan - additional category
            [255, 165, 0],    # Orange - additional category
            [128, 0, 128],    # Purple - additional category
            [255, 192, 203],  # Pink - additional category
            [165, 42, 42]     # Brown - additional category
        ]
        
        category_colors = {}
        color_idx = 0
        
        for category_name, category_info in self.dirt_categories.items():
            if color_idx < len(base_colors):
                category_colors[category_name] = base_colors[color_idx]
                color_idx += 1
            else:
                # Generate random colors if we run out of predefined ones
                category_colors[category_name] = [
                    np.random.randint(0, 256),
                    np.random.randint(0, 256), 
                    np.random.randint(0, 256)
                ]
        
        # Add background color (black)
        category_colors['background'] = [0, 0, 0]
        
        self.logger.debug(f"Initialized colors for categories: {list(category_colors.keys())}")
        return category_colors
    
    def _create_category_colormap(self):
        """Create a colormap for matplotlib visualization"""
        colors = []
        labels = []
        
        # Add background first (index 0)
        colors.append([0, 0, 0])  # Black background
        labels.append('background')
        
        # Add category colors
        for category_name, category_info in self.dirt_categories.items():
            color = np.array(self.category_colors[category_name]) / 255.0  # Normalize to 0-1
            colors.append(color)
            labels.append(category_name)
        
        return ListedColormap(colors), labels

    def _fetch_dir_info(self):
        """Enhanced to handle multiple dirt categories"""
        self.dirt_imgs_by_category = {}
        
        for category_name, category_info in self.dirt_categories.items():
            category_dir = os.path.join(self.dirt_base_dir, category_name)
            if os.path.exists(category_dir):
                category_images = [f for f in os.listdir(category_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                self.dirt_imgs_by_category[category_name] = category_images
                self.logger.debug(f"Found {len(category_images)} images for category '{category_name}'")
            else:
                self.logger.warning(f"Category directory not found: {category_dir}")
                self.dirt_imgs_by_category[category_name] = []
        
        # Get clean images
        self.clean_imgs = os.listdir(self.clean_dir)
        self.logger.debug(f"Found {len(self.clean_imgs)} clean images")
  
    def scale_and_rotate(self, results_threshold_mask, scale, rotation_angle):
        scaled_mask, scaled_width, scaled_height = self.image_operations.apply_scale(results_threshold_mask, scale, interpolation=cv2.INTER_CUBIC)
        if scaled_mask is None or scaled_width <= 0 or scaled_height <= 0: return None, None, None
        _, scaled_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)

        transformed_mask, transformed_width, transformed_height = self.image_operations.apply_rotation(scaled_mask, rotation_angle, interpolation=cv2.INTER_CUBIC, border_value=0)
        if transformed_mask is None or transformed_width <= 0 or transformed_height <= 0: return None, None, None
        _, transformed_mask = cv2.threshold(transformed_mask, 127, 255, cv2.THRESH_BINARY)

        return transformed_mask, transformed_width, transformed_height

    def super_impose_dirt_sample(
            self, base_image_float, results, boundary_x, boundary_y, boundary_w, boundary_h,
            segmentation_mask_accumulator, max_attempts, placed_bboxes,
            annotation_list=[], category_id=1, category_name="unknown",
            num_vertices_per_side=3, 
            max_distortion=0.1 
    ):
        if results["low_threshold_mask"] is None or results['high_threshold_mask'] is None or results["difference"] is None:
            return base_image_float, segmentation_mask_accumulator, None
    
        for _ in range(max_attempts):
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            rotation_angle = random.uniform(self.rotation_range[0], self.rotation_range[1])

            # Scale and rotate low threshold mask (for superimposing)
            transformed_mask_low, transformed_width, transformed_height = self.scale_and_rotate(results["low_threshold_mask"], scale, rotation_angle)
            if transformed_mask_low is None: continue

            transformed_mask_low_distorted = self.image_operations.apply_distorted_boundary_to_mask(
                transformed_mask_low, 
                num_vertices_per_side=num_vertices_per_side, 
                max_distortion=max_distortion
            )

            # Scale and rotate high threshold mask (for annotations)
            transformed_mask_high, _, _ = self.scale_and_rotate(results["high_threshold_mask"], scale, rotation_angle)
            if transformed_mask_high is None: continue

            # Resize high threshold mask to match low threshold mask dimensions
            if transformed_mask_high.shape != transformed_mask_low.shape:
                transformed_mask_high = cv2.resize(transformed_mask_high, (transformed_mask_low.shape[1], transformed_mask_low.shape[0]), interpolation=cv2.INTER_CUBIC)
                _, transformed_mask_high = cv2.threshold(transformed_mask_high, 127, 255, cv2.THRESH_BINARY)

            # Apply distorted boundary to the high threshold mask (for annotations)
            transformed_mask_high_distorted = self.image_operations.apply_distorted_boundary_to_mask(
                transformed_mask_high, 
                num_vertices_per_side=num_vertices_per_side, 
                max_distortion=max_distortion
            )

            # Scale and rotate difference image using cubic interpolation (check)
            scaled_difference_float, _, _ = self.image_operations.apply_scale(results['difference'], scale, interpolation=cv2.INTER_CUBIC)
            if scaled_difference_float is None: continue

            transformed_difference_float, _, _ = self.image_operations.apply_rotation(scaled_difference_float, rotation_angle, interpolation=cv2.INTER_CUBIC, border_value=0.0)
            if transformed_difference_float is None or scaled_difference_float is None: continue


            min_x_placement, max_x_placement = boundary_x, (boundary_x + boundary_w) - transformed_width
            min_y_placement, max_y_placement = boundary_y, (boundary_y + boundary_h) - transformed_height
            if max_x_placement < min_x_placement or max_y_placement < min_y_placement: continue

            place_x = random.randint(min_x_placement, max_x_placement)
            place_y = random.randint(min_y_placement, max_y_placement)
            proposed_bbox = (place_x, place_y, place_x + transformed_width, place_y + transformed_height)

            if ((proposed_bbox[0] >= boundary_x) and (proposed_bbox[1] >= boundary_y) and
                (proposed_bbox[2] <= (boundary_x + boundary_w)) and (proposed_bbox[3] <= (boundary_y + boundary_h)) and
                not self.image_operations.check_overlap(proposed_bbox, placed_bboxes)):
                
                target_roi = base_image_float[place_y : place_y + transformed_height, place_x : place_x + transformed_width]
                if not (target_roi.shape[:2] == transformed_mask_low_distorted.shape[:2] == transformed_difference_float.shape[:2]):
                    continue 

                # Use low threshold mask with distorted boundary for superimposing dirt effect
                binary_mask_float_0_1 = transformed_mask_low_distorted.astype(np.float32) / 255.0
                dirt_diff_roi_3channel = np.stack([transformed_difference_float] * 3, axis=-1)
                binary_mask_float_3channel = np.stack([binary_mask_float_0_1] * 3, axis=-1)

                result_roi = target_roi - (dirt_diff_roi_3channel * binary_mask_float_3channel)
                result_roi = np.clip(result_roi, 0, 255)
                base_image_float[place_y : place_y + transformed_height, place_x : place_x + transformed_width] = result_roi

                # FIXED: Update segmentation mask accumulator
                segmentation_roi = segmentation_mask_accumulator[place_y : place_y + transformed_height, place_x : place_x + transformed_width]
                segmentation_roi = cv2.bitwise_or(segmentation_roi, transformed_mask_high)
                segmentation_mask_accumulator[place_y : place_y + transformed_height, place_x : place_x + transformed_width] = segmentation_roi

                

                # Use high threshold mask with distorted boundary for annotations
                contours, _ = cv2.findContours(transformed_mask_high_distorted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    for contour in contours:
                        if cv2.contourArea(contour) < 20:
                            continue
                            
                        offset_contour_np = contour + (place_x, place_y)
                        segmentation_points = [p for sublist in offset_contour_np.tolist() for p in sublist[0]]
                        annotation_list.append({
                            "bbox": [place_x, place_y, transformed_width, transformed_height],
                            "segmentation": [segmentation_points], 
                            "category_id": category_id,  # Use actual category ID
                            "category_name": category_name,  # Add category name for reference
                            "area": cv2.countNonZero(transformed_mask_high_distorted)
                        })
                        
                return base_image_float, segmentation_mask_accumulator, proposed_bbox
        return base_image_float, segmentation_mask_accumulator, None

    def process_super_impose(
        self, image, num_dirt_super_impose, boundary_x, boundary_y, boundary_w, boundary_h,
        bg_estimation_filter_size=51, num_vertices_per_side=3, max_distortion=0.1 
    ):
        self.logger.debug(f"Super imposing dirt images with multiple categories")

        target_height, target_width, _ = image.shape
        image_float = image.copy().astype(np.float32)

        # Create separate segmentation masks for each category
        segmentation_masks_by_category = {}
        for category_name in self.dirt_categories.keys():
            segmentation_masks_by_category[category_name] = np.zeros((target_height, target_width), dtype=np.uint8)
        
        # Combined mask for overlap checking
        combined_segmentation_mask = np.zeros((target_height, target_width), dtype=np.uint8)
        
        placed_bboxes, annotations_for_this_image = [], []
        successfully_placed_count = 0
        category_counts = {cat: 0 for cat in self.dirt_categories.keys()}

        for dirt_idx in range(num_dirt_super_impose):
            # Select category based on weights
            selected_category = random.choices(
                self.category_names, 
                weights=self.category_weights, 
                k=1
            )[0]
            
            category_info = self.dirt_categories[selected_category]
            available_images = self.dirt_imgs_by_category.get(selected_category, [])
            
            if not available_images:
                self.logger.warning(f"No images available for category '{selected_category}'. Skipping.")
                continue
                
            random_dirty_image = random.choice(available_images)
            random_dirty_image_path = os.path.join(self.dirt_base_dir, selected_category, random_dirty_image)
            
            self.logger.debug(f"  Attempting to place {selected_category} sample {dirt_idx+1}/{num_dirt_super_impose}")
            self.logger.debug(f"    -> Using source: {random_dirty_image}")

            results = self.dirt_extractor.process_image(random_dirty_image_path, bg_estimation_filter_size)
            if results["low_threshold_mask"] is None or results["high_threshold_mask"] is None:
                continue
            
            # Place dirt sample with category information
            image_float, updated_mask, placed_bbox = self.super_impose_dirt_sample(
                image_float, results, boundary_x, boundary_y, boundary_w, boundary_h,
                segmentation_masks_by_category[selected_category], 10, placed_bboxes, 
                annotations_for_this_image, category_info['id'], selected_category,
                num_vertices_per_side, max_distortion
            )
            
            if placed_bbox:
                placed_bboxes.append(placed_bbox)
                successfully_placed_count += 1
                category_counts[selected_category] += 1
                
                # Update combined mask
                combined_segmentation_mask = cv2.bitwise_or(
                    combined_segmentation_mask, 
                    segmentation_masks_by_category[selected_category]
                )
            
        self.logger.debug(f"  Successfully placed {successfully_placed_count}/{num_dirt_super_impose} samples.")
        self.logger.debug(f"  Category distribution: {category_counts}")
        
        return (np.clip(image_float, 0, 255).astype(np.uint8), 
                segmentation_masks_by_category, 
                combined_segmentation_mask,
                annotations_for_this_image, category_counts)

    # Complete generate_synthetic_dataset method with proper patch creation
    def generate_synthetic_dataset(self, bg_estimation_filter_size: int, output_dir: str, 
                                  num_vertices_per_side: int = 3, max_distortion: float = 0.1, 
                                  patch_size=1024, generate_visualizations=True):
        
        self._fetch_dir_info()
        self.logger.info("---- Starting multi-label synthetic dataset generation with visualizations -----")
        from tqdm import tqdm
        
        for i, image_name in tqdm(enumerate(self.clean_imgs), desc="total images processed: ", total=len(self.clean_imgs)):
            # Debug limits for testing
            # if i < 1: continue
            if i > 0: break
            # if "specific_image_name" not in image_name: continue

            self.logger.debug(f"\n=== Processing {image_name} ===")
            original_image = self.image_operations.load_image_color(f"{self.clean_dir}/{image_name}")
            
            # Step 3: Downscale if enabled
            if self.clean_image_downscale_factor > 1:
                image = self.image_operations.resize_image(original_image, self.clean_image_downscale_factor)
            else:
                image = original_image
            
            self.logger.debug(f"Image shape: {image.shape}")
            img_h, img_w, _ = image.shape

            # Step 4: Detect mobile boundaries
            x, y, w, h = self.boundary_detector.detect_mobile_boundaries(image, False)
            self.logger.debug(f"Detected boundaries: x={x}, y={y}, w={w}, h={h}")
            
            # Step 5: Loop over number of versions
            for version_idx in range(self.num_version_per_image):
                self.logger.debug(f"\n--- {image_name} - Version {version_idx+1} ---")
                
                # Step 5.1: Super-impose multi-category dirt on clean image
                dirty_full, masks_by_category, combined_mask, ann_full, category_counts = self.process_super_impose(
                    image, self.num_dirt_super_impose, x, y, w, h, bg_estimation_filter_size, 
                    num_vertices_per_side, max_distortion
                )

                multiclass_mask = self.create_multiclass_mask(masks_by_category, dirty_full.shape)
                
                # Create version output directory
                ver_out_dir = os.path.join(output_dir, f"{image_name.split('.')[0]}_v{version_idx:02d}")
                os.makedirs(ver_out_dir, exist_ok=True)
                
                # Save main outputs
                self.image_operations.save_image(dirty_full, 
                    os.path.join(ver_out_dir, 'synthetic_dirty_image_full.png'))
                self.image_operations.save_image(combined_mask, 
                    os.path.join(ver_out_dir, 'segmentation_mask_combined.png'))
                self.image_operations.save_image(multiclass_mask, 
                    os.path.join(ver_out_dir, 'segmentation_mask_multiclass.png'))
                
                # Save category-specific masks
                masks_dir = os.path.join(ver_out_dir, 'category_masks')
                os.makedirs(masks_dir, exist_ok=True)
                
                for category_name, category_mask in masks_by_category.items():
                    if np.any(category_mask):  # Only save non-empty masks
                        mask_path = os.path.join(masks_dir, f'mask_{category_name}.png')
                        self.image_operations.save_image(category_mask, mask_path)
                
                # Generate comprehensive visualizations for full image
                if generate_visualizations:
                    viz_dir = os.path.join(ver_out_dir, 'visualizations')
                    visualization_paths = self.generate_comprehensive_visualization(
                        dirty_full, masks_by_category, ann_full, viz_dir, 
                        f"{image_name.split('.')[0]}_v{version_idx:02d}",
                        category_counts=category_counts,
                    )
                    self.logger.debug(f"Generated full image visualizations: {list(visualization_paths.keys())}")
                
                # Enhanced labels with category information
                enhanced_labels = {
                    "annotations": ann_full,
                    "categories": {str(info['id']): {"name": name, "supercategory": "dirt"} 
                                 for name, info in self.dirt_categories.items()},
                    "image_size": [img_w, img_h],
                    "boundary_pixels": [x, y, (x + w), (y + h)],
                    "category_statistics": self._calculate_category_stats(ann_full),
                    "category_colors": self.category_colors,
                    "category_counts_attempted": category_counts,
                    "success_rates": self._calculate_success_rates(category_counts, ann_full),
                }
                
                with open(os.path.join(ver_out_dir, 'labels_multi_category.json'), 'w') as f_json:
                    json.dump(enhanced_labels, f_json, indent=4)
                
                self.logger.debug(f"Saved full synthetic image, masks, and labels for {image_name} (Version {version_idx+1})")

                # Step 5.2: Create patches from the full synthetic image
                self.logger.debug(f"\n--- Splitting patches for {image_name} (Version {version_idx+1}) ---")
                patch_data_list = self.patch_gen.split_image_into_patches(
                    dirty_full, combined_mask, ann_full, patch_size, 
                    x, y, (x + w), (y + h),
                    overlap=0,  # You can configure this
                    return_empty_patches=True  # Set to False if you only want patches with annotations
                )
                
                self.logger.debug(f"Split into {len(patch_data_list)} patches")
                
                # Step 5.3: Save patches with multi-category support
                patch_base_dir = os.path.join(ver_out_dir, 'patches')
                os.makedirs(patch_base_dir, exist_ok=True)
                
                for p_idx, (p_img, p_mask, p_ann, gx, gy) in enumerate(patch_data_list):
                    p_name = f"patch_{p_idx:03d}_x{gx}_y{gy}"
                    p_dir = os.path.join(patch_base_dir, p_name)
                    os.makedirs(p_dir, exist_ok=True)
                    
                    # Save basic patch data
                    self.image_operations.save_image(p_img, os.path.join(p_dir, 'synthetic_dirty_patch.png'))
                    self.image_operations.save_image(p_mask, os.path.join(p_dir, 'segmentation_mask_patch.png'))
                    
                    # Create patch-level category-specific masks
                    patch_masks_by_category = {}
                    for category_name, full_category_mask in masks_by_category.items():
                        # Calculate patch boundaries
                        y_end = min(gy + patch_size, full_category_mask.shape[0])
                        x_end = min(gx + patch_size, full_category_mask.shape[1])
                        
                        # Create patch-sized mask
                        patch_category_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                        
                        # Extract the relevant region from full image mask
                        patch_h = y_end - gy
                        patch_w = x_end - gx
                        
                        if patch_h > 0 and patch_w > 0:
                            patch_category_mask[:patch_h, :patch_w] = full_category_mask[gy:y_end, gx:x_end]
                        
                        patch_masks_by_category[category_name] = patch_category_mask

                    patch_multiclass_mask = self.create_multiclass_mask(patch_masks_by_category, (patch_size, patch_size))

                    self.image_operations.save_image(patch_multiclass_mask, 
                        os.path.join(p_dir, 'segmentation_mask_patch_multiclass.png'))
                    
                    self.image_operations.save_image(p_mask, os.path.join(p_dir, 'segmentation_mask_patch.png'))
                    
                    # Save patch-level category masks
                    patch_masks_dir = os.path.join(p_dir, 'category_masks')
                    os.makedirs(patch_masks_dir, exist_ok=True)
                    
                    for category_name, patch_category_mask in patch_masks_by_category.items():
                        if np.any(patch_category_mask):  # Only save non-empty masks
                            patch_mask_path = os.path.join(patch_masks_dir, f'patch_mask_{category_name}.png')
                            self.image_operations.save_image(patch_category_mask, patch_mask_path)
                    
                    # Generate patch-level visualizations if there are annotations
                    if p_ann and generate_visualizations:
                        patch_viz_dir = os.path.join(p_dir, 'visualizations')
                        try:
                            patch_viz_paths = self.generate_comprehensive_visualization(
                                p_img, patch_masks_by_category, p_ann, patch_viz_dir, p_name
                            )
                            self.logger.debug(f"    Generated patch visualizations: {list(patch_viz_paths.keys())}")
                        except Exception as e:
                            self.logger.warning(f"Failed to generate visualizations for patch {p_name}: {e}")
                    
                    # Save enhanced patch labels
                    patch_labels = {
                        "annotations": p_ann,
                        "categories": {str(info['id']): {"name": name, "supercategory": "dirt"} 
                                    for name, info in self.dirt_categories.items()},
                        "image_size": [patch_size, patch_size],
                        "original_global_offset": [gx, gy],
                        "category_statistics": self._calculate_category_stats(p_ann),
                        "category_colors": self.category_colors,
                        "parent_image": f"{image_name.split('.')[0]}_v{version_idx:02d}",
                        "patch_boundary_in_full": [gx, gy, gx + patch_size, gy + patch_size]
                    }
                    
                    with open(os.path.join(p_dir, 'labels_patch.json'), 'w') as f_json:
                        json.dump(patch_labels, f_json, indent=4)
                    
                    self.logger.debug(f"    Saved patch {p_idx:03d} to {p_dir}")

                self.logger.debug(f"Completed patch processing for {image_name} (Version {version_idx+1})")

    def _calculate_category_stats(self, annotations):
        """Calculate statistics for each category in the current image"""
        stats = {name: 0 for name in self.dirt_categories.keys()}
        
        for ann in annotations:
            # FIXED: More robust category name extraction
            category_name = ann.get('category_name', None)
            category_id = ann.get('category_id', 0)
            
            # If category_name is missing or unknown, try to map from category_id
            if category_name is None or category_name == 'unknown':
                for cat_name, cat_info in self.dirt_categories.items():
                    if cat_info['id'] == category_id:
                        category_name = cat_name
                        break
                else:
                    category_name = 'unknown'
            
            if category_name in stats:
                stats[category_name] += 1
            else:
                # Handle unknown categories
                if 'unknown' not in stats:
                    stats['unknown'] = 0
                stats['unknown'] += 1
        
        return stats

    def _ensure_category_balance(self, target_samples_per_category):
        """Ensure minimum samples per category across the dataset"""
        # Implementation for balancing dataset if needed
        pass

    def get_category_info(self, category_id):
        """Get category information by ID"""
        for name, info in self.dirt_categories.items():
            if info['id'] == category_id:
                return name, info
        return "unknown", {"id": 0, "name": "unknown"}

    def create_colored_segmentation_mask(self, masks_by_category, image_shape):
        """
        Create a colored segmentation mask where each category has a distinct color
        
        Args:
            masks_by_category: Dictionary of category masks
            image_shape: Shape of the image (height, width)
            
        Returns:
            colored_mask: RGB image with colored categories
            legend_info: Dictionary mapping colors to categories
        """
        height, width = image_shape[:2]
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        legend_info = {}
        
        # Process each category mask
        for category_name, mask in masks_by_category.items():
            if mask is None or not np.any(mask):
                continue
                
            # Get category color
            color = self.category_colors[category_name]
            
            # Apply color to mask regions
            mask_bool = mask > 0
            colored_mask[mask_bool] = color
            
            # Store legend info
            legend_info[category_name] = {
                'color': color,
                'pixel_count': np.sum(mask_bool),
                'category_id': self.dirt_categories[category_name]['id']
            }
        
        return colored_mask, legend_info
    
    def create_indexed_segmentation_mask(self, masks_by_category, image_shape):
        """
        Create an indexed segmentation mask where each pixel value represents a category ID
        
        Args:
            masks_by_category: Dictionary of category masks
            image_shape: Shape of the image (height, width)
            
        Returns:
            indexed_mask: Single-channel image with category IDs as pixel values
        """
        height, width = image_shape[:2]
        indexed_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Process each category mask
        for category_name, mask in masks_by_category.items():
            if mask is None or not np.any(mask):
                continue
                
            category_id = self.dirt_categories[category_name]['id']
            mask_bool = mask > 0
            
            # Set pixel values to category ID
            # Handle overlaps by giving priority to higher category IDs
            indexed_mask[mask_bool] = np.maximum(indexed_mask[mask_bool], category_id)
        
        return indexed_mask
    
    def visualize_annotations_with_colors(self, image, annotations, category_counts=None, 
                                    save_path=None, show_plot=True):
        """
        Visualize bounding boxes and annotations with category-specific colors
        
        Args:
            image: Original image
            annotations: List of annotation dictionaries
            category_counts: Dictionary of actual category counts from generation process
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display image
        if len(image.shape) == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='gray')
        
        # Draw annotations
        legend_elements = []
        annotation_counts = {}  # Count from annotations
        
        # FIXED: Add debugging and better category name handling
        self.logger.debug(f"Visualizing {len(annotations)} annotations")
        
        for i, ann in enumerate(annotations):
            # FIXED: More robust category name extraction
            category_name = ann.get('category_name', None)
            category_id = ann.get('category_id', 0)
            
            # If category_name is missing or unknown, try to map from category_id
            if category_name is None or category_name == 'unknown':
                # Try to find category by ID
                for cat_name, cat_info in self.dirt_categories.items():
                    if cat_info['id'] == category_id:
                        category_name = cat_name
                        break
                else:
                    category_name = 'unknown'
                    self.logger.warning(f"Annotation {i}: Could not resolve category_name for ID {category_id}")
            
            self.logger.debug(f"Annotation {i}: category_name='{category_name}', category_id={category_id}")
            
            bbox = ann['bbox']
            
            # Get color for this category
            color = np.array(self.category_colors.get(category_name, [128, 128, 128])) / 255.0
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add category label
            ax.text(bbox[0], bbox[1] - 5, f'{category_name}_{category_id}', 
                color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Count categories from annotations
            if category_name not in annotation_counts:
                annotation_counts[category_name] = 0
            annotation_counts[category_name] += 1
        
        # Create legend using either category_counts or annotation_counts
        counts_to_use = category_counts if category_counts is not None else annotation_counts
        
        for category_name in self.dirt_categories.keys():
            color = np.array(self.category_colors.get(category_name, [128, 128, 128])) / 255.0
            count = counts_to_use.get(category_name, 0)
            category_id = self.dirt_categories[category_name]['id']
            
            if count > 0 or category_counts is not None:  # Show all categories if category_counts provided
                legend_elements.append(
                    patches.Patch(
                        color=color, 
                        label=f'{category_name} (ID: {category_id}, Count: {count})'
                    )
                )
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Enhanced title showing both attempted and successful placements
        if category_counts is not None:
            total_attempted = sum(category_counts.values())
            total_annotations = len(annotations)
            title = f'Dirt Detection - {total_annotations} annotations from {total_attempted} attempts across {len([c for c in category_counts.values() if c > 0])} categories'
        else:
            title = f'Dirt Detection - {len(annotations)} instances across {len(annotation_counts)} categories'
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Saved annotation visualization to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_category_statistics_plot(self, annotations=None, category_counts=None, 
                                    save_path=None, title="Category Distribution"):
        """
        Save a bar plot showing the distribution of dirt categories
        
        Args:
            annotations: List of annotation dictionaries (optional if category_counts provided)
            category_counts: Dictionary of category counts from generation process
            save_path: Path to save the plot
            title: Title for the plot
        """
        # Use category_counts if provided, otherwise extract from annotations
        if category_counts is not None:
            counts_to_plot = category_counts.copy()
        elif annotations is not None:
            counts_to_plot = {}
            for ann in annotations:
                category_name = ann.get('category_name', 'unknown')
                counts_to_plot[category_name] = counts_to_plot.get(category_name, 0) + 1
        else:
            self.logger.error("Either annotations or category_counts must be provided")
            return
        
        if not counts_to_plot:
            self.logger.warning("No category data found for statistics plot")
            return
        
        # Create bar plot including zero counts for comprehensive view
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: All categories (including zeros) - useful when category_counts provided
        if category_counts is not None:
            all_categories = list(self.dirt_categories.keys())
            all_counts = [counts_to_plot.get(cat, 0) for cat in all_categories]
            all_colors = [np.array(self.category_colors.get(cat, [128, 128, 128])) / 255.0 
                        for cat in all_categories]
            
            bars1 = ax1.bar(all_categories, all_counts, color=all_colors, alpha=0.8, 
                        edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, count in zip(bars1, all_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_title(f'{title} - All Categories', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Dirt Categories', fontsize=10)
            ax1.set_ylabel('Number of Instances', fontsize=10)
            ax1.grid(axis='y', alpha=0.3)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        else:
            ax1.axis('off')
            ax1.text(0.5, 0.5, 'All Categories View\n(Available with category_counts)', 
                    ha='center', va='center', transform=ax1.transAxes, 
                    fontsize=12, style='italic', alpha=0.6)
        
        # Plot 2: Only categories with instances
        active_categories = [cat for cat, count in counts_to_plot.items() if count > 0]
        active_counts = [counts_to_plot[cat] for cat in active_categories]
        active_colors = [np.array(self.category_colors.get(cat, [128, 128, 128])) / 255.0 
                        for cat in active_categories]
        
        if active_categories:
            bars2 = ax2.bar(active_categories, active_counts, color=active_colors, alpha=0.8, 
                        edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar, count in zip(bars2, active_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_title(f'{title} - Active Categories Only', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Dirt Categories', fontsize=10)
            ax2.set_ylabel('Number of Instances', fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            ax2.axis('off')
            ax2.text(0.5, 0.5, 'No Active Categories', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, style='italic', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.debug(f"Saved category statistics plot to: {save_path}")
        else:
            plt.show()
    
    # def save_category_statistics_plot(self, annotations, save_path, title="Category Distribution"):
    #     """
    #     Save a bar plot showing the distribution of dirt categories
        
    #     Args:
    #         annotations: List of annotation dictionaries
    #         save_path: Path to save the plot
    #         title: Title for the plot
    #     """
    #     # Count categories
    #     category_counts = {}
    #     for ann in annotations:
    #         category_name = ann.get('category_name', 'unknown')
    #         category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
    #     if not category_counts:
    #         self.logger.warning("No annotations found for statistics plot")
    #         return
        
    #     # Create bar plot
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
    #     categories = list(category_counts.keys())
    #     counts = list(category_counts.values())
    #     colors = [np.array(self.category_colors.get(cat, [128, 128, 128])) / 255.0 for cat in categories]
        
    #     bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
    #     # Add value labels on bars
    #     for bar, count in zip(bars, counts):
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
    #                f'{count}', ha='center', va='bottom', fontweight='bold')
        
    #     ax.set_title(title, fontsize=14, fontweight='bold')
    #     ax.set_xlabel('Dirt Categories', fontsize=12)
    #     ax.set_ylabel('Number of Instances', fontsize=12)
    #     ax.grid(axis='y', alpha=0.3)
        
    #     # Rotate x-axis labels if needed
    #     plt.xticks(rotation=45, ha='right')
    #     plt.tight_layout()
        
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close()
        
    #     self.logger.debug(f"Saved category statistics plot to: {save_path}")
    
    def _calculate_success_rates(self, category_counts, annotations):
        """Calculate success rates for dirt placement by category"""
        success_rates = {}
        annotation_counts = self._calculate_category_stats(annotations)
        
        for category_name in self.dirt_categories.keys():
            attempted = category_counts.get(category_name, 0)
            successful = annotation_counts.get(category_name, 0)
            
            if attempted > 0:
                success_rate = (successful / attempted) * 100
            else:
                success_rate = 0
                
            success_rates[category_name] = {
                'attempted': attempted,
                'successful': successful,
                'success_rate_percent': round(success_rate, 2)
            }
        
        return success_rates

    def generate_comprehensive_visualization(self, image, masks_by_category, annotations, 
                                        output_dir, base_name="visualization", 
                                        category_counts=None):
        """
        Generate a comprehensive set of visualizations for multi-category dirt detection
        
        Args:
            image: Original image
            masks_by_category: Dictionary of category-specific masks
            annotations: List of annotations
            output_dir: Directory to save visualizations
            base_name: Base name for output files
            category_counts: Dictionary of actual category counts from generation process
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Create colored segmentation mask
        colored_mask, legend_info = self.create_colored_segmentation_mask(masks_by_category, image.shape)
        colored_mask_path = os.path.join(output_dir, f"{base_name}_colored_mask.png")
        cv2.imwrite(colored_mask_path, colored_mask)
        
        # 2. Create indexed segmentation mask
        indexed_mask = self.create_indexed_segmentation_mask(masks_by_category, image.shape)
        indexed_mask_path = os.path.join(output_dir, f"{base_name}_indexed_mask.png")
        cv2.imwrite(indexed_mask_path, indexed_mask)
        
        # 3. Create overlay visualization
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        overlay_image = self.create_overlay_visualization(image, colored_mask, save_path=overlay_path)
        
        # 4. Create annotation visualization WITH category_counts
        annotation_path = os.path.join(output_dir, f"{base_name}_annotations.png")
        self.visualize_annotations_with_colors(image, annotations, category_counts=category_counts, 
                                            save_path=annotation_path, show_plot=False)
        
        # 5. Create statistics plot WITH category_counts
        stats_path = os.path.join(output_dir, f"{base_name}_statistics.png")
        self.save_category_statistics_plot(annotations=annotations, category_counts=category_counts, 
                                        save_path=stats_path)
        
        # 6. Save enhanced legend information
        enhanced_legend_info = {
            'category_colors': self.category_colors,
            'legend_info': legend_info,
            'total_categories': len(self.dirt_categories),
            'category_counts_attempted': category_counts if category_counts else {},
            'category_counts_successful': self._calculate_category_stats(annotations),
            'generation_success_rate': self._calculate_success_rates(category_counts, annotations) if category_counts else {}
        }
        
        legend_path = os.path.join(output_dir, f"{base_name}_legend.json")
        with open(legend_path, 'w') as f:
            json.dump(enhanced_legend_info, f, indent=4, default=str)
        
        self.logger.debug(f"Generated comprehensive visualizations with category counts in: {output_dir}")
        
        return {
            'colored_mask': colored_mask_path,
            'indexed_mask': indexed_mask_path,
            'overlay': overlay_path,
            'annotations': annotation_path,
            'statistics': stats_path,
            'legend': legend_path
        }
    
    def create_overlay_visualization(self, original_image, colored_mask, alpha=0.6, save_path=None):
        """
        Create an overlay of the original image with colored segmentation mask
        
        Args:
            original_image: Original image
            colored_mask: Colored segmentation mask
            alpha: Transparency for overlay
            save_path: Path to save the result
            
        Returns:
            overlay_image: Combined overlay image
        """
        # Ensure images are the same size
        if original_image.shape[:2] != colored_mask.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to same data type
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Create overlay where there's segmentation
        mask_present = np.any(colored_mask > 0, axis=2)
        overlay_image = original_image.copy()
        
        # Apply colored mask with transparency
        overlay_image[mask_present] = (
            (1 - alpha) * original_image[mask_present] + 
            alpha * colored_mask[mask_present]
        )
        
        if save_path:
            cv2.imwrite(save_path, overlay_image)
            self.logger.debug(f"Saved overlay visualization to: {save_path}")
        
        return overlay_image

    def create_multiclass_mask(self, masks_by_category, image_shape):
        """
        Create a single multi-class mask where pixel values represent class IDs
        
        Args:
            masks_by_category: Dictionary of category-specific binary masks
            image_shape: Shape of the image (height, width)
            
        Returns:
            multiclass_mask: Single-channel mask with class IDs as pixel values
        """
        height, width = image_shape[:2]
        multiclass_mask = np.zeros((height, width), dtype=np.uint8)  # Background = 0
        
        # Process each category mask in order of priority (higher ID overwrites lower ID)
        for category_name, mask in masks_by_category.items():
            if mask is None or not np.any(mask):
                continue
                
            category_id = self.dirt_categories[category_name]['id']
            mask_bool = mask > 0
            
            # Set pixel values to category ID (higher IDs will overwrite lower IDs in overlaps)
            multiclass_mask[mask_bool] = category_id
            
            self.logger.debug(f"Applied category '{category_name}' (ID: {category_id}) to {np.sum(mask_bool)} pixels")
        
        # Log class distribution in the mask
        unique_values, counts = np.unique(multiclass_mask, return_counts=True)
        class_distribution = dict(zip(unique_values, counts))
        self.logger.debug(f"Multi-class mask distribution: {class_distribution}")
        
        return multiclass_mask

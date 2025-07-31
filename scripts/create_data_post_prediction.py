import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Import existing components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.target_labels import TargetLabels
from src.image_processor import ImageProcessor
from src.synthesis.patch_generator import PatchGenerator
from src.core.logger_config import setup_application_logger
from scripts.create_data_by_superimposing import SyntheticDataGenerator


class PostPredictionSyntheticDataGenerator(SyntheticDataGenerator):
    
    
    def __init__(
         self,
        image_processor: ImageProcessor,
        patch_generator: PatchGenerator,
        scratch_inference_dir: str,
        clean_images_dir: str,
        dirt_base_dir: str,
        dirt_categories: dict,
        treat_inf_class:TargetLabels = TargetLabels.SCRATCH,
        num_version_per_image: int = 1,
        num_dirt_super_impose: int = 40,
        clean_image_downscale_factor: int = 2,
        scale_dirt: Tuple[float, float] = (0.9, 1.15),
        rotation_range: Tuple[int, int] = (0, 360),
        bg_estimation_filter_size: int = 51,
        app_logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the scratch-dirt synthetic generator.
        
        Args:
            scratch_inference_dir: Directory containing scratch inference results
            full_phone_images_dir: Directory with original full phone images
            dirt_base_dir: Base directory containing dirt category subdirectories (dirt, condensation, etc.)
            Other args inherited from parent class
        """
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('PostPredictionSyntheticDataGenerator')
        
        # Initialize parent class
        super().__init__(
            image_processor=image_processor,
            patch_generator=patch_generator,
            clean_images_dir=clean_images_dir,
            dirt_base_dir=dirt_base_dir,
            dirt_categories=dirt_categories,
            num_version_per_image=num_version_per_image,
            num_dirt_super_impose=num_dirt_super_impose,
            clean_image_downscale_factor=clean_image_downscale_factor,
            scale_dirt=scale_dirt,
            rotation_range=rotation_range,
            bg_estimation_filter_size=bg_estimation_filter_size,
            app_logger=app_logger
        )
        
        self.scratch_inference_dir = scratch_inference_dir
        self.treat_inf_class = treat_inf_class
        
        # FIXED: Use proper TargetLabels enum for category colors and weights
        self.category_colors[self.treat_inf_class.value] = [128, 0, 128]
        
        # Ensure scratch category has weight 0 (since we load from inference, not generate)
        if self.treat_inf_class.value in self.dirt_categories:
            self.dirt_categories[self.treat_inf_class.value] = 0.0
        
        # Update category weights to reflect we don't generate scratches
        for i, cat_name in enumerate(self.category_names):
            if cat_name == self.treat_inf_class.value:
                self.category_weights[i] = 0.0
                break
        
        
    def _discover_dirt_categories(self, dirt_base_dir: str) -> Dict:
        """
        Discover dirt categories from directory structure.
        Maps each category to its corresponding TargetLabels enum.
        
        Args:
            dirt_base_dir: Base directory containing category subdirectories
            
        Returns:
            Dictionary of dirt categories with IDs from TargetLabels enum
        """
        dirt_categories = {}
        
        if not os.path.exists(dirt_base_dir):
            self.logger.warning(f"Dirt base directory does not exist: {dirt_base_dir}")
            return dirt_categories
            
        # Scan for subdirectories
        for item in os.listdir(dirt_base_dir):
            category_path = os.path.join(dirt_base_dir, item)
            if os.path.isdir(category_path):
                # Check if directory contains image files
                image_files = [f for f in os.listdir(category_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if image_files:
                    # FIXED: Map directory name to TargetLabels enum
                    try:
                        # Try to find matching TargetLabels enum
                        target_label = None
                        for label in TargetLabels:
                            if label.value.lower() == item.lower():
                                target_label = label
                                break
                        
                        if target_label:
                            dirt_categories[item] = {
                                'id': target_label.index,
                                'name': target_label.value,
                                'weight': target_label.weight
                            }
                            self.logger.info(f"Discovered dirt category '{item}' -> TargetLabels.{target_label.name} (id={target_label.index}) with {len(image_files)} images")
                        else:
                            self.logger.warning(f"Directory '{item}' does not match any TargetLabels enum. Skipping.")
                            
                    except Exception as e:
                        self.logger.error(f"Error mapping category '{item}': {e}")
                        continue
        
        self.logger.info(f"Total dirt categories discovered: {list(dirt_categories.keys())}")
        return dirt_categories
           
    def load_scratch_mask_from_inference(self, image_name: str) -> Optional[np.ndarray]:
        """
        Load pre-existing scratch mask from inference results.
        Returns the mask in its original size (screen area only).
        
        Args:
            image_name: Base name of the image
            
        Returns:
            Scratch mask in screen coordinates or None
        """
        # Remove extension from image name
        base_name = Path(image_name).stem
        
        # Look for scratch mask in inference results
        # First try in subdirectory
        inference_subdir = Path(self.scratch_inference_dir) / base_name
        
        # Try different possible mask names
        possible_mask_names = [
            f"{base_name}_screen_dirt_mask.png",  # From the example in requirements
            f"{base_name}_screen_scratches_mask.png",  # Alternative name
            "screen_dirt_mask.png",  # Generic name without base
            "screen_scratches_mask.png",  # Generic name
            "dirt_mask.png",  # Simple name
            "scratches_mask.png"  # Simple name
        ]
        
        scratch_mask_path = None
        
        # Try subdirectory first
        if inference_subdir.exists():
            for mask_name in possible_mask_names:
                candidate_path = inference_subdir / mask_name
                if candidate_path.exists():
                    scratch_mask_path = candidate_path
                    self.logger.info(f"Found scratch mask: {mask_name} in subdirectory")
                    break
        
        # If not found in subdirectory, try main inference directory
        if scratch_mask_path is None:
            inference_main_dir = Path(self.scratch_inference_dir)
            for mask_name in possible_mask_names:
                candidate_path = inference_main_dir / mask_name
                if candidate_path.exists():
                    scratch_mask_path = candidate_path
                    self.logger.info(f"Found scratch mask: {mask_name} in main directory")
                    break
        
        if scratch_mask_path is None:
            # List available files for debugging
            if inference_subdir.exists():
                available_files = list(inference_subdir.glob("*.png"))
                self.logger.warning(f"Available PNG files in {inference_subdir}: {[f.name for f in available_files]}")
            else:
                self.logger.warning(f"Inference subdirectory does not exist: {inference_subdir}")
            self.logger.warning(f"Scratch mask not found. Tried: {possible_mask_names}")
            return None
            
        # Load the screen-area scratch mask
        screen_scratch_mask = cv2.imread(str(scratch_mask_path), cv2.IMREAD_GRAYSCALE)
        if screen_scratch_mask is None:
            self.logger.error(f"Failed to load scratch mask: {scratch_mask_path}")
            return None
            
        # Convert to binary (0 or 1)
        screen_scratch_mask = (screen_scratch_mask > 127).astype(np.uint8)
        
        # FIXED: Validate that mask contains scratch pixels
        scratch_pixel_count = np.sum(screen_scratch_mask > 0)
        if scratch_pixel_count == 0:
            self.logger.warning(f"Loaded scratch mask is empty (no scratch pixels found)")
            return None
        
        self.logger.info(f"Loaded scratch mask with shape: {screen_scratch_mask.shape}, scratch pixels: {scratch_pixel_count}")
        return screen_scratch_mask
    
    def place_scratch_mask_in_full_image(self, screen_scratch_mask: np.ndarray, 
                                       screen_bounds: Tuple[int, int, int, int],
                                       full_image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Place the screen scratch mask in the correct position within the full image.
        
        Args:
            screen_scratch_mask: Scratch mask for screen area only
            screen_bounds: (x, y, w, h) of screen area in full image
            full_image_shape: (height, width) of the full image
            
        Returns:
            Full image mask with scratches in correct position
        """
        x, y, w, h = screen_bounds
        full_height, full_width = full_image_shape
        
        # Create full image mask
        full_scratch_mask = np.zeros((full_height, full_width), dtype=np.uint8)
        
        # Log original scratch mask info
        scratch_pixels_original = np.sum(screen_scratch_mask > 0)
        self.logger.debug(f"Original screen scratch mask: {screen_scratch_mask.shape}, pixels: {scratch_pixels_original}")
        
        # Resize screen mask to match screen bounds if needed
        if screen_scratch_mask.shape != (h, w):
            self.logger.info(f"Resizing scratch mask from {screen_scratch_mask.shape} to ({h}, {w})")
            screen_scratch_mask = cv2.resize(screen_scratch_mask, (w, h), interpolation=cv2.INTER_AREA)
            
        # Ensure binary values
        screen_scratch_mask = (screen_scratch_mask > 0).astype(np.uint8)
        
        # Place screen mask in full image coordinates
        # Ensure we don't go out of bounds
        y_end = min(y + h, full_height)
        x_end = min(x + w, full_width)
        mask_h = y_end - y
        mask_w = x_end - x
        
        full_scratch_mask[y:y_end, x:x_end] = screen_scratch_mask[:mask_h, :mask_w]
        
        # Log final mask info
        scratch_pixels_final = np.sum(full_scratch_mask > 0)
        self.logger.debug(f"Placed scratch mask at ({x},{y}), size ({w},{h}), final pixels: {scratch_pixels_final}")
        
        return full_scratch_mask

    
    def create_scratch_annotations(self, scratch_mask: np.ndarray, 
                                category_id: int = None, 
                                category_name: str = None) -> List[Dict]:
        """
        FIXED: Create annotations from scratch mask loaded from inference.
        
        Args:
            scratch_mask: Binary scratch mask (0s and 1s)
            category_id: Category ID for scratches (uses TargetLabels.SCRATCH.index if None)
            category_name: Category name for scratches (uses TargetLabels.SCRATCH.value if None)
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        
        # FIXED: Use TargetLabels enum consistently
        if category_id is None:
            category_id = self.treat_inf_class.index
        if category_name is None:
            category_name = self.treat_inf_class.value
        
        # Convert mask to binary if needed
        if scratch_mask.dtype != np.uint8:
            scratch_mask = scratch_mask.astype(np.uint8)
        
        # Ensure binary values (0 or 255)
        scratch_mask_binary = (scratch_mask > 0).astype(np.uint8) * 255
        
        # Find contours for scratches
        contours, _ = cv2.findContours(scratch_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.logger.debug(f"Found {len(contours)} scratch contours")
        
        for contour in contours:
            # Filter out very small contours (noise)
            area = cv2.contourArea(contour)
            if area < 10:  # Minimum area threshold
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create segmentation points
            segmentation_points = []
            for point in contour:
                segmentation_points.extend([int(point[0][0]), int(point[0][1])])
            
            # Create annotation
            annotation = {
                "bbox": [int(x), int(y), int(w), int(h)],
                "segmentation": [segmentation_points],
                "category_id": category_id,
                "category_name": category_name,
                "area": int(area)
            }
            annotations.append(annotation)
            
        self.logger.info(f"Created {len(annotations)} scratch annotations from mask")
        return annotations
    
    def process_single_image(
    self,
    image_path: str,
    output_dir: str,
    bg_estimation_filter_size: int = 51,
    num_vertices_per_side: int = 3,
    max_distortion: float = 0.1,
    patch_size: int = 1792,
    generate_visualizations: bool = True,
    num_versions: int = 1) -> Dict:
        """
        Process a single full phone image with scratches and add dirt categories,
        generating multiple randomized versions.
        
        Returns:
            Dictionary with per-version processing results and statistics
        """
        from copy import deepcopy
        image_name = Path(image_path).name
        base_name = Path(image_path).stem
        self.logger.info(f"\n=== Processing {image_name} ===")

        # Load the high-resolution image once
        self.logger.info(f"Loading high-resolution image: {image_path}")
        resized_img, (x, y, w, h) = self.preprocess_image_synthesis(image_name)

        # Load scratch mask once
        screen_scratch_mask = self.load_scratch_mask_from_inference(image_name)
        if screen_scratch_mask is None:
            self.logger.warning(f"No scratch mask found for {image_name}, creating empty mask")
            scratch_mask_original = np.zeros(resized_img.shape[:2], dtype=np.uint8)
        else:
            scratch_mask_original = self.place_scratch_mask_in_full_image(
                screen_scratch_mask, (x, y, w, h), resized_img.shape[:2]
            )
        initial_scratch_pixels = np.sum(scratch_mask_original > 0)
        self.logger.info(f"Initial scratch pixels: {initial_scratch_pixels}")

        # FIXED: Create scratch annotations from the loaded mask
        scratch_annotations = self.create_scratch_annotations(
            scratch_mask_original, 
            category_id=self.treat_inf_class.index, 
            category_name=self.treat_inf_class.value
        )
        
        self.logger.info(f"Added {len(scratch_annotations)} scratch annotations")

        # Output all versions' metadata
        all_versions_metadata = {}

        for version_idx in range(num_versions):
            version_suffix = f"_v{version_idx:02d}"
            version_dir = Path(output_dir) / f"{base_name}{version_suffix}"
            version_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"\n--- Generating version {version_idx + 1}/{num_versions} ---")

            # Deep copy base image and scratch mask
            image_float = resized_img.copy().astype(np.float32)
            scratch_mask = scratch_mask_original.copy()

            # Initialize masks - FIXED: Use TargetLabels enum values consistently
            masks_by_category = {self.treat_inf_class.value: scratch_mask}
            for category_name in self.dirt_categories.keys():
                if category_name == self.treat_inf_class.value: 
                    continue
                masks_by_category[category_name] = np.zeros(resized_img.shape[:2], dtype=np.uint8)

            placed_bboxes = []
            annotations_list = []

            # FIXED: Add scratch annotations to the list
            annotations_list.extend(scratch_annotations)

            self.logger.info(f"Superimposing {self.num_dirt_super_impose} dirt samples from categories...")

            for selected_category in self.dirt_imgs_by_category:
                if selected_category == self.treat_inf_class.value: 
                    continue
                    
                # FIXED: Get category weight from TargetLabels enum
                category_weight = self.dirt_categories[selected_category]
                n_dirt = int(category_weight * self.num_dirt_super_impose)
                
                for dirt_idx in range(n_dirt):
                    random_dirty_image = random.choice(self.dirt_imgs_by_category[selected_category])
                    random_dirty_image_path = f"{self.dirt_base_dir}/{selected_category}/{random_dirty_image}"

                    results = self.image_processor.process_dirt_mask_extraction(
                        random_dirty_image_path,
                        bg_estimation_filter_size,
                        2
                    )
                    if results["low_threshold_mask"] is None or results["high_threshold_mask"] is None:
                        continue

                    # FIXED: Use TargetLabels enum index for category_id
                    category_label = None
                    for label in TargetLabels:
                        if label.value == selected_category:
                            category_label = label
                            break
                    
                    if category_label is None:
                        self.logger.warning(f"Category {selected_category} not found in TargetLabels enum")
                        continue

                    image_float, updated_category_mask, placed_bbox = self.super_impose_dirt_with_scratch_priority(
                        image_float, results, x, y, w, h,
                        masks_by_category[selected_category], scratch_mask,
                        10, placed_bboxes, annotations_list,
                        category_id=category_label.index,
                        category_name=category_label.value,
                        num_vertices_per_side=num_vertices_per_side,
                        max_distortion=max_distortion
                    )
                    masks_by_category[selected_category] = updated_category_mask
                    if placed_bbox:
                        placed_bboxes.append(placed_bbox)

            # FIXED: Create multiclass mask with consistent TargetLabels indices
            multiclass_mask = self.create_priority_multiclass_mask_fixed(masks_by_category)

            # Save synthetic image and masks
            synthetic_image = np.clip(image_float, 0, 255).astype(np.uint8)

            self.image_processor.save_image(synthetic_image, str(version_dir / 'synthetic_dirty_image_full.png'))

            # FIXED: Save multiclass mask with correct TargetLabels indices
            max_class_id = max([label.index for label in TargetLabels])
            clean_multiclass_mask = np.clip(multiclass_mask, 0, max_class_id).astype(np.uint8)
            cv2.imwrite(str(version_dir / 'segmentation_mask_multiclass.png'), clean_multiclass_mask)

            combined_mask = (multiclass_mask > 0).astype(np.uint8) * 255
            self.image_processor.save_image(combined_mask, str(version_dir / 'segmentation_mask_combined.png'))

            masks_dir = version_dir / 'category_masks'
            masks_dir.mkdir(exist_ok=True)
            for category_name, mask in masks_by_category.items():
                if np.any(mask):
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    self.image_processor.save_image(binary_mask, str(masks_dir / f'mask_{category_name}.png'))

            # FIXED: Compute statistics using correct TargetLabels indices
            total_pixels = multiclass_mask.size
            class_statistics = {
                'background_pixels': int(np.sum(multiclass_mask == TargetLabels.BACKGROUND.index)),
                'background_percentage': float(np.sum(multiclass_mask == TargetLabels.BACKGROUND.index) / total_pixels * 100),
            }
            
            # Add statistics for all TargetLabels
            for label in TargetLabels:
                if label == TargetLabels.BACKGROUND:
                    continue  # Already handled above
                class_pixels = np.sum(multiclass_mask == label.index)
                class_statistics[f'{label.value}_pixels'] = int(class_pixels)
                class_statistics[f'{label.value}_percentage'] = float((class_pixels / total_pixels) * 100)

            self.logger.critical(f"{class_statistics = }")

            # FIXED: Create categories dict using TargetLabels enum
            categories_dict = {}
            for label in TargetLabels:
                categories_dict[str(label.index)] = {
                    "name": label.value, 
                    "supercategory": "background" if label == TargetLabels.BACKGROUND else "defect", 
                    "id": label.index
                }

            enhanced_labels = {
                "annotations": annotations_list,
                "categories": categories_dict,
                "image_size": [resized_img.shape[1], resized_img.shape[0]],
                "boundary_pixels": [x, y, x + w, y + h],
                "category_statistics": self._calculate_category_stats(annotations_list),
                "category_colors": self.category_colors,
                "category_counts_attempted": self._get_category_counts_attempted(),
                "success_rates": self._calculate_success_rates(self._get_category_counts_attempted(), annotations_list),
            }

            with open(version_dir / 'labels_multi_category.json', 'w') as f:
                json.dump(enhanced_labels, f, indent=4)

            if patch_size:
                try:
                    self.logger.info(f"Generating patches of size {patch_size}...")
                    self.generate_patches_from_synthetic(
                        synthetic_image, multiclass_mask, annotations_list,
                        x, y, w, h, patch_size, version_dir / 'patches'
                    )
                except Exception as e:
                    self.logger.error(f"Failed to generate patches: {e}")

            if generate_visualizations:
                try:
                    viz_dir = version_dir / 'visualizations'
                    self.generate_comprehensive_visualization(
                        synthetic_image, masks_by_category, annotations_list,
                        viz_dir, base_name,
                        category_counts=self._get_category_counts_attempted()
                    )
                except Exception as e:
                    self.logger.error(f"Failed to generate visualization: {e}")

            metadata = {
                'source_image': image_name,
                'image_size': list(resized_img.shape[:2]),
                'screen_boundaries': [x, y, w, h],
                'class_statistics': class_statistics,
                'annotations': annotations_list,
                'categories': categories_dict,
                'processing_info': {
                    'initial_scratch_pixels': int(initial_scratch_pixels),
                    'dirt_samples_attempted': self.num_dirt_super_impose,
                    'dirt_categories_available': list(self.dirt_categories.keys())
                }
            }

            all_versions_metadata[version_suffix] = metadata

        return all_versions_metadata
    
    def _get_category_counts_attempted(self):
        """FIXED: Get the attempted counts for each category using TargetLabels"""
        counts = {}
        
        # Scratch count (from inference, always 1 attempt)
        counts[self.treat_inf_class.value] = 1
        
        # For dirt categories, use weights from TargetLabels enum
        total_weight = sum(label.weight for label in TargetLabels if label != TargetLabels.BACKGROUND)
        
        for label in TargetLabels:
            if label == TargetLabels.BACKGROUND or label == self.treat_inf_class:
                continue
            expected_count = int((label.weight / total_weight) * self.num_dirt_super_impose)
            counts[label.value] = expected_count
            
        return counts
    
    def _calculate_category_stats(self, annotations):
        """FIXED: Calculate statistics for each category using TargetLabels"""
        stats = {}
    
        # Initialize with all TargetLabels categories
        for label in TargetLabels:
            if label == TargetLabels.BACKGROUND:
                continue
            stats[label.value] = 0
        
        for ann in annotations:
            category_name = ann.get('category_name', 'unknown')
            category_id = ann.get('category_id', -1)
            
            # Handle category mapping using TargetLabels
            matched_label = None
            for label in TargetLabels:
                if label.value == category_name or label.index == category_id:
                    matched_label = label
                    break
            
            if matched_label and matched_label.value in stats:
                stats[matched_label.value] += 1
            else:
                # Handle unknown categories
                if 'unknown' not in stats:
                    stats['unknown'] = 0
                stats['unknown'] += 1
        
        return stats

    def _calculate_success_rates(self, category_counts, annotations):
        """FIXED: Calculate success rates using TargetLabels"""
        success_rates = {}
        annotation_counts = self._calculate_category_stats(annotations)
        
        for label in TargetLabels:
            if label == TargetLabels.BACKGROUND:
                continue
                
            category_name = label.value
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
    
    def super_impose_dirt_with_scratch_priority(
        self, base_image_float, results, boundary_x, boundary_y, boundary_w, boundary_h,
        dirt_mask_accumulator, scratch_mask, max_attempts, placed_bboxes,
        annotation_list=[], category_id=2, category_name="dirt",
        num_vertices_per_side=3, max_distortion=0.1
    ):
        """
        Modified superimpose function that respects existing scratch masks.
        Scratches have priority over dirt. Uses parent class scale_and_rotate method.
        Returns updated image, updated dirt mask, and placed bbox.
        """
        # Validate required inputs
        required_keys = ["low_threshold_mask", "high_threshold_mask", "difference"]
        if not all(results.get(key) is not None for key in required_keys):
            return base_image_float, dirt_mask_accumulator, None
            
        # Make a copy of the dirt mask to ensure we're updating it properly
        dirt_mask_accumulator = dirt_mask_accumulator.copy()
            
        for _ in range(max_attempts):
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            rotation_angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
            
            # Use parent class scale_and_rotate method
            transformed_mask_low, transformed_width, transformed_height = self.scale_and_rotate(
                results["low_threshold_mask"], scale, rotation_angle
            )
            if transformed_mask_low is None:
                continue
                
            transformed_mask_low_distorted = self.image_processor.apply_distorted_boundary_to_mask(
                transformed_mask_low, num_vertices_per_side, max_distortion
            )
            
            transformed_mask_high, _, _ = self.scale_and_rotate(
                results["high_threshold_mask"], scale, rotation_angle
            )
            if transformed_mask_high is None:
                continue
                
            if transformed_mask_high.shape != transformed_mask_low.shape:
                transformed_mask_high = cv2.resize(
                    transformed_mask_high,
                    (transformed_mask_low.shape[1], transformed_mask_low.shape[0]),
                    interpolation=cv2.INTER_CUBIC
                )
                _, transformed_mask_high = cv2.threshold(transformed_mask_high, 127, 255, cv2.THRESH_BINARY)
                
            transformed_mask_high_distorted = self.image_processor.apply_distorted_boundary_to_mask(
                transformed_mask_high, num_vertices_per_side, max_distortion
            )
            
            # Scale and rotate difference image
            scaled_difference_float, _, _ = self.image_processor.apply_scale(
                results['difference'], scale, interpolation=cv2.INTER_CUBIC
            )
            if scaled_difference_float is None:
                continue
                
            transformed_difference_float, _, _ = self.image_processor.apply_rotation(
                scaled_difference_float, rotation_angle, interpolation=cv2.INTER_CUBIC, border_value=0.0
            )
            if transformed_difference_float is None:
                continue
                
            # Find placement position within boundaries
            min_x_placement = boundary_x
            max_x_placement = max(boundary_x, (boundary_x + boundary_w) - transformed_width)
            min_y_placement = boundary_y
            max_y_placement = max(boundary_y, (boundary_y + boundary_h) - transformed_height)
            
            if max_x_placement <= min_x_placement or max_y_placement <= min_y_placement:
                continue
                
            place_x = random.randint(min_x_placement, max_x_placement)
            place_y = random.randint(min_y_placement, max_y_placement)
            
            # Calculate actual placement bounds
            x_end = min(place_x + transformed_width, base_image_float.shape[1])
            y_end = min(place_y + transformed_height, base_image_float.shape[0])
            actual_width = x_end - place_x
            actual_height = y_end - place_y
            
            if actual_width <= 0 or actual_height <= 0:
                continue
                
            proposed_bbox = (place_x, place_y, x_end, y_end)
            
            # Check if placement is valid
            if ((proposed_bbox[0] >= boundary_x) and (proposed_bbox[1] >= boundary_y) and
                (proposed_bbox[2] <= (boundary_x + boundary_w)) and 
                (proposed_bbox[3] <= (boundary_y + boundary_h)) and
                not self.image_processor.check_overlap(proposed_bbox, placed_bboxes)):
                
                # Extract regions
                target_roi = base_image_float[place_y:y_end, place_x:x_end]
                scratch_roi = scratch_mask[place_y:y_end, place_x:x_end]
                
                # Adjust transformed masks to match actual dimensions
                transformed_mask_low_distorted = transformed_mask_low_distorted[:actual_height, :actual_width]
                transformed_mask_high = transformed_mask_high[:actual_height, :actual_width]
                transformed_mask_high_distorted = transformed_mask_high_distorted[:actual_height, :actual_width]
                transformed_difference_float = transformed_difference_float[:actual_height, :actual_width]
                
                if not (target_roi.shape[:2] == transformed_mask_low_distorted.shape[:2] == 
                       transformed_difference_float.shape[:2]):
                    continue

                # Apply dirt effect naturally without respecting scratch boundaries
                effective_dirt_mask = transformed_mask_low_distorted.astype(np.float32) / 255.0
                    
                # Create mask that respects scratch priority
                # Only apply dirt where there are no scratches
                scratch_free_mask = (scratch_roi == 0).astype(np.float32)
                effective_dirt_mask = effective_dirt_mask * scratch_free_mask
                
                # Apply dirt effect only where there are no scratches
                dirt_diff_roi_3channel = np.stack([transformed_difference_float] * 3, axis=-1)
                binary_mask_float_3channel = np.stack([effective_dirt_mask] * 3, axis=-1)
                
                result_roi = target_roi - (dirt_diff_roi_3channel * binary_mask_float_3channel)
                result_roi = np.clip(result_roi, 0, 255)
                base_image_float[place_y:y_end, place_x:x_end] = result_roi
                
                # For dirt mask accumulator, only add where there are no scratches
                # This maintains scratch priority in the training targets
                new_dirt = cv2.bitwise_and(transformed_mask_high, 
                                        (scratch_free_mask * 255).astype(np.uint8))
                
                # Ensure new_dirt is binary
                _, new_dirt = cv2.threshold(new_dirt, 127, 1, cv2.THRESH_BINARY)
                
                # Update the accumulator mask (only where no scratches)
                dirt_mask_accumulator[place_y:y_end, place_x:x_end] = np.maximum(
                    dirt_mask_accumulator[place_y:y_end, place_x:x_end], 
                    new_dirt
                )
                
                # Create annotations only for actual dirt areas (excluding scratches)
                # This ensures training targets respect scratch priority
                effective_mask_for_annotation = cv2.bitwise_and(
                    transformed_mask_high_distorted,
                    (scratch_free_mask * 255).astype(np.uint8)
                )
                
                contours, _ = cv2.findContours(effective_mask_for_annotation, 
                                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    for contour in contours:
                        if cv2.contourArea(contour) < 20:
                            continue
                            
                        offset_contour_np = contour + (place_x, place_y)
                        segmentation_points = [p for sublist in offset_contour_np.tolist() 
                                             for p in sublist[0]]
                        # Adjust annotation bbox to be within screen bounds if needed
                        ann_x = max(boundary_x, place_x)
                        ann_y = max(boundary_y, place_y) 
                        ann_x_end = min(boundary_x + boundary_w, place_x + actual_width)
                        ann_y_end = min(boundary_y + boundary_h, place_y + actual_height)
                        ann_width = ann_x_end - ann_x
                        ann_height = ann_y_end - ann_y
                        
                        if ann_width > 0 and ann_height > 0:
                            annotation_list.append({
                                "bbox": [ann_x, ann_y, ann_width, ann_height],
                                "segmentation": [segmentation_points],
                                "category_id": category_id,
                                "category_name": category_name,
                                "area": cv2.countNonZero(effective_mask_for_annotation)
                            })
                        
                return base_image_float, dirt_mask_accumulator, proposed_bbox
                
        return base_image_float, dirt_mask_accumulator, None
    
    def validate_saved_multiclass_mask(self, mask_path: str, expected_classes: List[int]) -> bool:
        """
        Validate that a saved multiclass mask has the correct values.
        
        Args:
            mask_path: Path to the saved mask file
            expected_classes: List of expected class values
            
        Returns:
            True if mask is valid, False otherwise
        """
        try:
            # Load the saved mask
            saved_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if saved_mask is None:
                self.logger.error(f"Could not load saved mask: {mask_path}")
                return False
            
            # Check unique values
            unique_values = np.unique(saved_mask)
            self.logger.debug(f"Saved mask {mask_path} contains values: {unique_values}")
            
            # Check if all values are in expected range
            invalid_values = [v for v in unique_values if v not in expected_classes]
            if invalid_values:
                self.logger.warning(f"Mask {mask_path} contains unexpected values: {invalid_values}")
                return False
            
            # Check if we have the expected non-background classes
            non_bg_values = [v for v in unique_values if v > 0]
            if len(non_bg_values) == 0:
                self.logger.warning(f"Mask {mask_path} contains only background pixels")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating mask {mask_path}: {e}")
            return False
    
    def create_priority_multiclass_mask_fixed(self, masks_by_category: Dict[str, np.ndarray]) -> np.ndarray:
        """
        FIXED: Create multiclass mask with priority using TargetLabels enum indices consistently.
        Priority: background (0) < dirt categories < scratches (highest priority)
        """
        height, width = next(iter(masks_by_category.values())).shape
        multiclass_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Debug logging
        self.logger.debug(f"Creating multiclass mask from categories: {list(masks_by_category.keys())}")
        
        # Apply all categories using their TargetLabels enum indices
        # Process in order of priority (lower index = lower priority, applied first)
        categories_to_process = []
        
        for category_name, category_mask in masks_by_category.items():
            if category_mask is None or not np.any(category_mask):
                continue
                
            # Find corresponding TargetLabels enum
            target_label = None
            for label in TargetLabels:
                if label.value == category_name:
                    target_label = label
                    break
            
            if target_label:
                categories_to_process.append((target_label.index, category_name, category_mask))
        
        # Sort by index (priority) - lower indices applied first, higher indices (like scratches) applied last
        categories_to_process.sort(key=lambda x: x[0])
        
        for label_index, category_name, category_mask in categories_to_process:
            category_pixels = np.sum(category_mask > 0)
            self.logger.debug(f"Applying {category_name} (index={label_index}) mask: {category_mask.shape}, non-zero pixels: {category_pixels}")
            
            # Apply category mask with its TargetLabels index
            multiclass_mask[category_mask > 0] = label_index
            
        # Debug final mask
        unique_values, counts = np.unique(multiclass_mask, return_counts=True)
        self.logger.debug(f"Final multiclass mask values: {dict(zip(unique_values, counts))}")
        
        return multiclass_mask
    
    def create_indexed_segmentation_mask(self, masks_by_category: Dict[str, np.ndarray], 
                                        image_shape: Tuple[int, int]) -> np.ndarray:
        """
        FIXED: Create an indexed segmentation mask using TargetLabels enum indices
        """
        return self.create_priority_multiclass_mask_fixed(masks_by_category)
    
    def create_colored_segmentation_mask_new(self, masks_by_category, image_shape):
        """
        FIXED: Create a colored segmentation mask using TargetLabels enum
        
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
            
            # Find corresponding TargetLabels enum
            target_label = None
            for label in TargetLabels:
                if label.value == category_name:
                    target_label = label
                    break
            
            if not target_label:
                continue
                
            # Get category color
            color = self.category_colors.get(category_name, [128, 128, 128])  # Default gray
            
            # Apply color to mask regions
            mask_bool = mask > 0
            colored_mask[mask_bool] = color
            
            # Store legend info
            legend_info[category_name] = {
                'color': color,
                'pixel_count': np.sum(mask_bool),
                'category_id': target_label.index
            }
        
        return colored_mask, legend_info
    
    def generate_patches_from_synthetic(
        self, synthetic_image, multiclass_mask, annotations,
        screen_x, screen_y, screen_w, screen_h, patch_size, output_dir
    ):
        """FIXED: Generate patches using TargetLabels enum indices"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use existing patch generator
        patches_data = self.patch_gen.split_image_into_patches(
            synthetic_image, multiclass_mask, annotations, patch_size,
            screen_x, screen_y, screen_x + screen_w, screen_y + screen_h,
            overlap=0, return_empty_patches=True
        )
        
        self.logger.info(f"Generated {len(patches_data)} patches")
        
        for idx, (patch_img, patch_mask, patch_ann, x_start, y_start) in enumerate(patches_data):
            patch_name = f"patch_{idx:03d}_x{x_start}_y{y_start}"
            patch_dir = output_dir / patch_name
            patch_dir.mkdir(exist_ok=True)
            
            # Save patch image with correct naming
            self.image_processor.save_image(patch_img, str(patch_dir / 'synthetic_dirty_patch.png'))
            
            # FIXED: Save patch multiclass mask using TargetLabels indices
            patch_multiclass_path = str(patch_dir / 'segmentation_mask_patch_multiclass.png')
            max_class_id = max([label.index for label in TargetLabels])
            clean_patch_mask = np.clip(patch_mask, 0, max_class_id).astype(np.uint8)
            success = cv2.imwrite(patch_multiclass_path, clean_patch_mask)
            if not success:
                self.logger.error(f"Failed to save patch multiclass mask to {patch_multiclass_path}")
            else:
                self.logger.debug(f"Saved patch multiclass mask with values {np.unique(clean_patch_mask)} to {patch_multiclass_path}")
                # Validate the saved patch mask
                expected_classes = [label.index for label in TargetLabels]
                self.validate_saved_multiclass_mask(patch_multiclass_path, expected_classes)
            
            # Save binary combined mask for compatibility
            combined_patch_mask = (patch_mask > 0).astype(np.uint8) * 255
            self.image_processor.save_image(combined_patch_mask, str(patch_dir / 'segmentation_mask_patch.png'))
            
            # FIXED: Save individual category masks using TargetLabels
            patch_masks_dir = patch_dir / 'category_masks'
            patch_masks_dir.mkdir(exist_ok=True)
            
            # Create individual class masks from multiclass mask using TargetLabels
            for label in TargetLabels:
                if label == TargetLabels.BACKGROUND:
                    continue
                class_mask = (patch_mask == label.index).astype(np.uint8) * 255
                if np.any(class_mask):
                    self.image_processor.save_image(class_mask, str(patch_masks_dir / f'patch_mask_{label.value}.png'))
            
            # FIXED: Create categories dict for patch labels using TargetLabels
            patch_categories = {}
            for label in TargetLabels:
                patch_categories[str(label.index)] = {
                    "name": label.value, 
                    "supercategory": "background" if label == TargetLabels.BACKGROUND else "defect", 
                    "id": label.index
                }
            
            # Save patch labels with correct format
            patch_labels = {
                "annotations": patch_ann,
                "categories": patch_categories,
                "image_size": [patch_size, patch_size],
                "original_global_offset": [x_start, y_start],
                "category_statistics": self._calculate_category_stats(patch_ann),
                "category_colors": self.category_colors,
                "parent_image": f"{Path(synthetic_image).stem if hasattr(synthetic_image, 'stem') else 'unknown'}_v00",
                "patch_boundary_in_full": [x_start, y_start, x_start + patch_size, y_start + patch_size]
            }
            
            with open(patch_dir / 'labels_patch.json', 'w') as f:
                json.dump(patch_labels, f, indent=4)
                
        self.logger.info(f"Saved {len(patches_data)} patches to {output_dir}")
                
    def generate_dataset(
        self,
        output_dir: str,
        bg_estimation_filter_size: int = 51,
        num_vertices_per_side: int = 3,
        max_distortion: float = 0.1,
        patch_size: int = 1792,
        generate_visualizations: bool = True,
        max_images: Optional[int] = None
    ):
        """
        Generate synthetic dataset from all available images.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get list of full phone images
        image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']
        image_files = []
        
        images_path = Path(self.clean_dir)
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            
        if max_images:
            image_files = image_files[:max_images]
        
            
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self.process_single_image(
                    str(image_path),
                    str(output_path),
                    bg_estimation_filter_size=bg_estimation_filter_size,
                    num_vertices_per_side=num_vertices_per_side,
                    max_distortion=max_distortion,
                    patch_size=patch_size,
                    generate_visualizations=generate_visualizations,
                    num_versions=self.num_version_per_image
                )
                
                if result:
                    results.append(result)
                    self.logger.info(f"Successfully processed {image_path.name}")
                else:
                    self.logger.warning(f"No result returned for {image_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {image_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        # FIXED: Save dataset summary using TargetLabels
        summary = {
            'dataset_info': {
                'total_images': len(results),
                'classes': {
                    str(label.index): label.value for label in TargetLabels
                },
                'source_directories': {
                    'full_phone_images': self.clean_dir,
                    'scratch_inference': self.scratch_inference_dir,
                    'dirt_base_dir': self.dirt_base_dir,
                    'dirt_categories': list(self.dirt_categories.keys())
                }
            },
            'processing_parameters': {
                'num_dirt_super_impose': self.num_dirt_super_impose,
                'scale_dirt': self.scale_range,
                'rotation_range': self.rotation_range,
                'patch_size': patch_size
            },
            'results': results
        }
        
        with open(output_path / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
        self.logger.info(f"Dataset generation complete! Processed {len(results)} images.")
        self.logger.info(f"Results saved to: {output_path}")
        
        # FIXED: Print summary statistics using TargetLabels
        if results:
            total_pixels = sum(r['image_size'][0] * r['image_size'][1] for res in results for v, r in res.items())
            
            self.logger.info("\n=== DATASET SUMMARY ===")
            self.logger.info(f"Total images processed: {len(results)}")
            self.logger.info(f"Total pixels: {total_pixels:,}")
            
            # Statistics for each TargetLabels category
            for label in TargetLabels:
                if label == TargetLabels.BACKGROUND:
                    continue
                total_category = sum(r['class_statistics'].get(f'{label.value}_pixels', 0) for res in results for v, r in res.items())
                self.logger.info(f"Total {label.value} pixels: {total_category:,} ({total_category/total_pixels*100:.2f}%)")
            
            self.logger.info("======================\n")


def get_argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data with scratches and dirt categories')
    
    # input directories
    parser.add_argument('--scratch-inference-dir', type=str, default="tests/test_images/inference_result_dirt_only_model_scratch",
                       help='Directory containing scratch inference results')
    parser.add_argument('--clean-images-dir', type=str, default="tests/test_images/temp",
                       help='Directory containing full phone images')
    parser.add_argument('--dirt-categories-dir', type=str, default="tests/test_images/input/dirt_categories",
                       help='Base directory containing dirt category subdirectories (dirt, condensation, etc.)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='tests/test_images/output/synthetic_dataset_post_prediction',
                       help='Output directory for synthetic data')
    
    
    
    # Generation parameters
    parser.add_argument('--down-scale-clean-img', type=int, default=2,
                       help='Downscale factor for clean image')
    parser.add_argument('--num-dirt', type=int, default=150,
                       help='Number of dirt samples to superimpose per image')
    parser.add_argument('--num-version', type=int, default=1,
                       help='Number of versions per clean image')
    parser.add_argument('--patch-size', type=int, default=1792,
                       help='Size of patches to generate')
    parser.add_argument('--max-images', type=int, default=1,
                       help='Maximum number of images to process')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip saving visualization images')
    
    # Transformation parameters
    parser.add_argument('--num-vertices', type=int, default=3,
                       help='Number of vertices per side for distortion')
    parser.add_argument('--max-distortion', type=float, default=0.1,
                       help='Maximum distortion factor for boundaries')
    
    # Logging
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def main():
    """Main function to run the scratch-dirt synthetic data generation"""

    args = get_argparse()
    
    # Setup logger
    log_level = logging.DEBUG if args.debug else logging.INFO
    app_logger = setup_application_logger(
        app_name="data_generation_pp",
        log_file_name="logs/scratch_dirt_post_prediction_generation.log"
    )
    # app_logger.setLevel(log_level)
    
    # Also set module loggers to same level
    logging.getLogger('src.synthesis').setLevel(log_level)
    
    # Log configuration
    app_logger.info("Starting scratch-dirt synthetic data generation with configuration:")
    app_logger.info(f"  Scratch inference dir: {args.scratch_inference_dir}")
    app_logger.info(f"  Full phone images dir: {args.clean_images_dir}")
    app_logger.info(f"  Dirt base dir: {args.dirt_categories_dir}")

    app_logger.info(f"  Output dir: {args.output_dir}")
    
    app_logger.info(f"  Downscale clean image by: {args.down_scale_clean_img}")
    app_logger.info(f"  Number of dirt samples: {args.num_dirt}")
    app_logger.info(f"  Number of versions: {args.num_version}")
    app_logger.info(f"  Patch size: {args.patch_size}")
    app_logger.info(f"  Max images: {args.max_images}")
    app_logger.info(f"  Generate visualizations: {not args.no_visualizations}")
    app_logger.info(f"  Number of vertices: {not args.num_vertices}")
    app_logger.info(f"  Maximum distortion: {not args.max_distortion}")

    app_logger.info(f"  Debug level: {not args.debug}")
    
    # Initialize components
    image_processor = ImageProcessor(app_logger)
    patch_generator = PatchGenerator(app_logger)
    
    # Create generator
    generator = PostPredictionSyntheticDataGenerator(
        image_processor, patch_generator, 
        args.scratch_inference_dir, args.clean_images_dir,
        args.dirt_categories_dir, TargetLabels.value_to_weight_map(),
        TargetLabels.SCRATCH, args.num_version, args.num_dirt, args.down_scale_clean_img,
        app_logger=app_logger
    )
    
    
    # Generate dataset
    generator.generate_dataset(
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        generate_visualizations=not args.no_visualizations,
        max_images=args.max_images,
        max_distortion=args.max_distortion,
        num_vertices_per_side=args.num_vertices,

    )


if __name__ == "__main__":
    main()
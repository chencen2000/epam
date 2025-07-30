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
        self.category_colors[self.treat_inf_class.value] = [128, 0, 128]
        self.category_weights[self.treat_inf_class.index] = 0.0
        
        
    def _discover_dirt_categories(self, dirt_base_dir: str) -> Dict:
        """
        Discover dirt categories from directory structure.
        
        Args:
            dirt_base_dir: Base directory containing category subdirectories
            
        Returns:
            Dictionary of dirt categories with IDs starting from 2 (after scratches=1)
        """
        dirt_categories = {}
        category_id = 2  # Start from 2 (scratches = 1, background = 0)
        
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
                    dirt_categories[item] = {
                        'id': category_id,
                        'name': item,
                        'weight': 1.0
                    }
                    category_id += 1
                    self.logger.info(f"Discovered dirt category '{item}' with {len(image_files)} images")
                if "scratch" == item.lower():
                    dirt_categories[item]["weight"] = 0.0
        
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
        
        self.logger.info(f"Loaded scratch mask with shape: {screen_scratch_mask.shape}")
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
    
    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
        bg_estimation_filter_size: int = 51,
        num_vertices_per_side: int = 3,
        max_distortion: float = 0.1,
        patch_size: int = 1024,
        generate_visualizations: bool = True
    ) -> Dict:
        """
        Process a single full phone image with scratches and add dirt categories.
        
        Returns:
            Dictionary with processing results and statistics
        """
        image_name = Path(image_path).name
        base_name = Path(image_path).stem
        self.logger.info(f"\n=== Processing {image_name} ===")
        
        # Load the high-resolution image
        self.logger.info(f"Loading high-resolution image: {image_path}")
        resized_img, (x, y, w, h) = self.preprocess_image_synthesis(image_name)
        
        # Load pre-existing scratch mask (in screen coordinates)
        screen_scratch_mask = self.load_scratch_mask_from_inference(image_name)
        if screen_scratch_mask is None:
            self.logger.warning(f"No scratch mask found for {image_name}, creating empty mask")
            scratch_mask = np.zeros(resized_img.shape[:2], dtype=np.uint8)
        else:
            # Place scratch mask in full image with correct alignment
            scratch_mask = self.place_scratch_mask_in_full_image(
                screen_scratch_mask, (x, y, w, h), resized_img.shape[:2]
            )
            
        # Log scratch mask info
        scratch_pixels = np.sum(scratch_mask > 0)
        self.logger.info(f"Scratch mask loaded: {scratch_pixels} pixels marked as scratches")
            
        # Initialize multi-class mask with scratches (class 1)
        multiclass_mask = scratch_mask.copy()  # 0 for background, 1 for scratches
        
        # Count initial scratch pixels
        initial_scratch_pixels = np.sum(scratch_mask > 0)
        self.logger.info(f"Initial scratch pixels: {initial_scratch_pixels}")
        
        # FIXED: Initialize masks by category using discovered categories
        masks_by_category = {'scratch': scratch_mask}
        for category_name in self.dirt_categories.keys():
            if category_name == self.treat_inf_class.value: continue
            masks_by_category[category_name] = np.zeros(resized_img.shape[:2], dtype=np.uint8)
        
        # Prepare for dirt superimposition
        image_float = resized_img.copy().astype(np.float32)
        placed_bboxes = []
        annotations_list = []
        
        # FIXED: Use parent class method to superimpose all dirt categories
        self.logger.info(f"Superimposing {self.num_dirt_super_impose} dirt samples from categories: {list(self.dirt_categories.keys())}")

        for selected_category in self.dirt_imgs_by_category:
            if selected_category == self.treat_inf_class.value: continue
            n_dirt = int(self.dirt_categories[selected_category] * self.num_dirt_super_impose)
            self.logger.debug(f"for {selected_category = } needs {n_dirt} / {self.num_dirt_super_impose} superimposition...")
            for dirt_idx in range(int(n_dirt)):
                random_dirty_image = random.choice(self.dirt_imgs_by_category[selected_category])
                random_dirty_image = f"{self.dirt_base_dir}/{selected_category}/{random_dirty_image}"

                self.logger.debug(f"  Attempting to place {selected_category} sample {dirt_idx+1}/{n_dirt}")
                self.logger.debug(f"    -> Using source: {random_dirty_image}")

                results = self.image_processor.process_dirt_mask_extraction(
                    random_dirty_image, self.bg_estimation_filter_size
                )
                if results["low_threshold_mask"] is None or results["high_threshold_mask"] is None:
                    continue

                # Superimpose dirt with scratch-aware placement
                image_float, updated_category_mask, placed_bbox = self.super_impose_dirt_with_scratch_priority(
                    image_float, results, x, y, w, h,
                    masks_by_category[selected_category], scratch_mask,
                    10, placed_bboxes, annotations_list,
                    category_id=self.category_names.index(selected_category), 
                    category_name=selected_category,
                    num_vertices_per_side=num_vertices_per_side, 
                    max_distortion=max_distortion
                )
                # Update the category mask in the dictionary
                masks_by_category[selected_category] = updated_category_mask
                
                if placed_bbox:
                    placed_bboxes.append(placed_bbox)

                
        # Log mask statistics before creating multiclass mask
        for category_name, mask in masks_by_category.items():
            pixels = np.sum(mask > 0)
            self.logger.info(f"  {category_name} pixels: {pixels}")
        
        # Update multiclass mask with all categories, ensuring proper priority
        multiclass_mask = self.create_priority_multiclass_mask(masks_by_category)
        
        # Log final multiclass mask statistics
        unique_values, counts = np.unique(multiclass_mask, return_counts=True)
        class_distribution = dict(zip(unique_values, counts))
        self.logger.info(f"Final multiclass mask distribution: {class_distribution}")
        
        # Convert back to uint8
        synthetic_image = np.clip(image_float, 0, 255).astype(np.uint8)
        
        # Create output directory structure to match generate_data.py format
        version_dir = Path(output_dir) / f"{base_name}_v00"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save outputs with correct naming
        self.logger.info("Saving outputs...")
        
        try:
            # Save synthetic image with correct naming
            self.image_processor.save_image(synthetic_image, str(version_dir / 'synthetic_dirty_image_full.png'))
            
            # FIXED: Save multiclass mask with correct number of classes
            multiclass_mask_path = str(version_dir / 'segmentation_mask_multiclass.png')
            max_class_id = max([0, 1] + [index for index in range(len(self.dirt_categories))])
            clean_multiclass_mask = np.clip(multiclass_mask, 0, max_class_id).astype(np.uint8)
            success = cv2.imwrite(multiclass_mask_path, clean_multiclass_mask)
            if not success:
                self.logger.error(f"Failed to save multiclass mask to {multiclass_mask_path}")
            else:
                self.logger.debug(f"Saved multiclass mask with values {np.unique(clean_multiclass_mask)} to {multiclass_mask_path}")
                # Validate the saved mask
                expected_classes = [0, 1] + [index for index in range(len(self.dirt_categories))]
                self.validate_saved_multiclass_mask(multiclass_mask_path, expected_classes)
            
            # FIXED: Save combined mask (binary all defects for compatibility)
            combined_mask = (multiclass_mask > 0).astype(np.uint8) * 255
            self.image_processor.save_image(combined_mask, str(version_dir / 'segmentation_mask_combined.png'))
            
            # Save individual class masks
            masks_dir = version_dir / 'category_masks'
            masks_dir.mkdir(exist_ok=True)
            
            for category_name, mask in masks_by_category.items():
                if np.any(mask):
                    # Save as binary mask (0 or 255)
                    binary_mask = (mask > 0).astype(np.uint8) * 255
                    self.image_processor.save_image(binary_mask, str(masks_dir / f'mask_{category_name}.png'))
                    self.logger.debug(f"Saved {category_name} mask with {np.sum(mask > 0)} pixels")
                    
        except Exception as e:
            self.logger.error(f"Failed to save output files: {e}")
            return None
                
        # FIXED: Calculate statistics for all classes
        total_pixels = multiclass_mask.size
        class_statistics = {}
        
        # Background
        class_pixels = np.sum(multiclass_mask == 0)
        class_statistics['background_pixels'] = int(class_pixels)
        class_statistics['background_percentage'] = float((class_pixels / total_pixels) * 100)
        
        # Scratches
        class_pixels = np.sum(multiclass_mask == 1)
        class_statistics['scratches_pixels'] = int(class_pixels)
        class_statistics['scratches_percentage'] = float((class_pixels / total_pixels) * 100)
        
        # All dirt categories
        for index, category_name in enumerate(self.dirt_categories):
            class_pixels = np.sum(multiclass_mask == index)
            class_statistics[f'{category_name}_pixels'] = int(class_pixels)
            class_statistics[f'{category_name}_percentage'] = float((class_pixels / total_pixels) * 100)
            
        # FIXED: Enhanced labels with all discovered categories
        categories_dict = {
            "0": {"name": "background", "supercategory": "background", "id": 0},
            "1": {"name": "scratches", "supercategory": "defect", "id": 1}
        }
        
        # Add all dirt categories to the categories dict
        for index, category_name in enumerate(self.dirt_categories):
            categories_dict[str(index)] = {
                "name": category_name, 
                "supercategory": "defect", 
                "id": index
            }
        
        enhanced_labels = {
            "annotations": annotations_list,
            "categories": categories_dict,
            "image_size": [resized_img.shape[1], resized_img.shape[0]],  # [width, height]
            "boundary_pixels": [x, y, x + w, y + h],
            "category_statistics": self._calculate_category_stats(annotations_list),
            "category_colors": self.category_colors,
            "category_counts_attempted": self._get_category_counts_attempted(),
            "success_rates": self._calculate_success_rates(self._get_category_counts_attempted(), annotations_list),
        }
        
        with open(version_dir / 'labels_multi_category.json', 'w') as f:
            json.dump(enhanced_labels, f, indent=4)
        
        # Generate patches if needed
        if patch_size:
            try:
                self.logger.info(f"Generating patches of size {patch_size}...")
                self.generate_patches_from_synthetic(
                    synthetic_image, multiclass_mask, annotations_list,
                    x, y, w, h, patch_size, version_dir / 'patches'
                )
            except Exception as e:
                self.logger.error(f"Failed to generate patches: {e}")
                # Continue even if patch generation fails
        
        # Generate visualizations if requested
        if generate_visualizations:
            viz_dir = version_dir / 'visualizations'
            try:
                

                viz_image = synthetic_image
                viz_masks = masks_by_category
                
                # Use parent's visualization method if available
                self.generate_comprehensive_visualization(
                    viz_image, viz_masks, annotations_list,
                    viz_dir, base_name,
                    category_counts=self._get_category_counts_attempted()
                )
            except Exception as e:
                self.logger.error(f"Failed to generate visualization: {e}")
                # Continue processing even if visualization fails
            
        # Save metadata
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
        
        self.logger.info(f"Processing complete for {image_name}")
        
        return metadata
    
    def _get_category_counts_attempted(self):
        """Get the attempted counts for each category"""
        counts = {self.treat_inf_class.value: 1}  # Always attempt 1 scratch (from inference)
        
        # For dirt categories, approximate based on weights and total attempts
        total_weight = sum(self.category_weights)
        for i, category_name in enumerate(self.category_names):
            category_weight = self.category_weights[i]
            expected_count = int((category_weight / total_weight) * self.num_dirt_super_impose)
            counts[category_name] = expected_count
            
        return counts
    
    def _calculate_category_stats(self, annotations):
        """Calculate statistics for each category in the current image"""
        stats = {}
        
        # Initialize with all categories
        stats[self.treat_inf_class.value] = 0
        for category_name in self.dirt_categories.keys():
            stats[category_name] = 0
        
        for ann in annotations:
            # More robust category name extraction
            category_name = ann.get('category_name', None)
            category_id = ann.get('category_id', 0)
            
            # If category_name is missing or unknown, try to map from category_id
            if category_name is None or category_name == 'unknown':
                if category_id == self.treat_inf_class.index:
                    category_name = self.treat_inf_class.value
                else:
                    # Find category by ID
                    for index, cat_name in enumerate(self.dirt_categories):
                        if index == category_id:
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

    def _calculate_success_rates(self, category_counts, annotations):
        """Calculate success rates for dirt placement by category"""
        success_rates = {}
        annotation_counts = self._calculate_category_stats(annotations)
        
        for category_name in [self.treat_inf_class.value] + list(self.dirt_categories.keys()):
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
                    
                # Create mask that respects scratch priority
                # Only apply dirt where there are no scratches
                scratch_free_mask = (scratch_roi == 0).astype(np.float32)
                effective_dirt_mask = transformed_mask_low_distorted.astype(np.float32) / 255.0
                effective_dirt_mask = effective_dirt_mask * scratch_free_mask
                
                # Apply dirt effect only where there are no scratches
                dirt_diff_roi_3channel = np.stack([transformed_difference_float] * 3, axis=-1)
                binary_mask_float_3channel = np.stack([effective_dirt_mask] * 3, axis=-1)
                
                result_roi = target_roi - (dirt_diff_roi_3channel * binary_mask_float_3channel)
                result_roi = np.clip(result_roi, 0, 255)
                base_image_float[place_y:y_end, place_x:x_end] = result_roi
                
                # Update dirt mask (only where no scratches)
                # Create the new dirt mask respecting scratch boundaries
                new_dirt = cv2.bitwise_and(transformed_mask_high, 
                                          (scratch_free_mask * 255).astype(np.uint8))
                
                # Ensure new_dirt is binary
                _, new_dirt = cv2.threshold(new_dirt, 127, 1, cv2.THRESH_BINARY)
                
                # Update the accumulator mask
                dirt_mask_accumulator[place_y:y_end, place_x:x_end] = np.maximum(
                    dirt_mask_accumulator[place_y:y_end, place_x:x_end], 
                    new_dirt
                )
                
                # Create annotations only for actual dirt areas (excluding scratches)
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
    
    def create_priority_multiclass_mask(self, masks_by_category: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create multiclass mask with priority: scratches (1) > dirt categories (2, 3, ...) > background (0)
        """
        height, width = next(iter(masks_by_category.values())).shape
        multiclass_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Debug logging
        self.logger.debug(f"Creating multiclass mask from categories: {list(masks_by_category.keys())}")
        
        # Apply dirt categories first (in order of category ID, lowest priority)
        for category_name, category_info in self.dirt_categories.items():
            if category_name in masks_by_category:
                category_mask = masks_by_category[category_name]
                category_pixels = np.sum(category_mask > 0)
                self.logger.debug(f"{category_name} mask shape: {category_mask.shape}, non-zero pixels: {category_pixels}")
                # Set category pixels to their class ID
                multiclass_mask[category_mask > 0] = self.category_names.index(category_name)
        
        # Apply scratches last (highest priority)
        if self.treat_inf_class.value in masks_by_category:
            scratch_mask = masks_by_category[self.treat_inf_class.value]
            scratch_pixels = np.sum(scratch_mask > 0)
            self.logger.debug(f"Scratch mask shape: {scratch_mask.shape}, non-zero pixels: {scratch_pixels}")
            # Set scratch pixels to class 1 (overwrites everything else)
            multiclass_mask[scratch_mask > 0] = 1
            
        # Debug final mask
        unique_values, counts = np.unique(multiclass_mask, return_counts=True)
        self.logger.debug(f"Final multiclass mask values: {dict(zip(unique_values, counts))}")
        
        return multiclass_mask
    
    def create_indexed_segmentation_mask(self, masks_by_category: Dict[str, np.ndarray], 
                                        image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create an indexed segmentation mask where each pixel value represents a category ID
        Override parent method to ensure proper handling
        """
        height, width = image_shape[:2]
        indexed_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Apply masks in order of priority (highest priority applied last)
        for index, category_name in enumerate(self.dirt_categories):
            if category_name in masks_by_category:
                category_mask = masks_by_category[category_name] > 0
                indexed_mask[category_mask] = index
            
        if self.treat_inf_class.value in masks_by_category:
            scratch_mask = masks_by_category[self.treat_inf_class.value] > 0
            indexed_mask[scratch_mask] = 1
            
        return indexed_mask
    
    def create_colored_segmentation_mask(self, masks_by_category, image_shape):
        """
        Override parent method to handle scratches category properly.
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
            color = self.category_colors.get(category_name, [128, 128, 128])  # Default gray
            
            # Apply color to mask regions
            mask_bool = mask > 0
            colored_mask[mask_bool] = color
            
            # Store legend info
            if category_name == self.treat_inf_class.value:
                category_id = self.treat_inf_class.index
            elif category_name in self.dirt_categories:
                category_id = self.category_names.index(category_name)
            else:
                category_id = 0  # Default to background
                
            legend_info[category_name] = {
                'color': color,
                'pixel_count': np.sum(mask_bool),
                'category_id': category_id
            }
        
        return colored_mask, legend_info
    
    def generate_patches_from_synthetic(
        self, synthetic_image, multiclass_mask, annotations,
        screen_x, screen_y, screen_w, screen_h, patch_size, output_dir
    ):
        """Generate patches from the synthetic image with correct naming conventions"""
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
            
            # Save patch multiclass mask (raw values for training)
            patch_multiclass_path = str(patch_dir / 'segmentation_mask_patch_multiclass.png')
            max_class_id = max([0, 1] + [i for i, cat in enumerate(self.dirt_categories)])
            clean_patch_mask = np.clip(patch_mask, 0, max_class_id).astype(np.uint8)
            success = cv2.imwrite(patch_multiclass_path, clean_patch_mask)
            if not success:
                self.logger.error(f"Failed to save patch multiclass mask to {patch_multiclass_path}")
            else:
                self.logger.debug(f"Saved patch multiclass mask with values {np.unique(clean_patch_mask)} to {patch_multiclass_path}")
                # Validate the saved patch mask
                expected_classes = [0, 1] + [i for i, cat in enumerate(self.dirt_categories)]
                self.validate_saved_multiclass_mask(patch_multiclass_path, expected_classes)
            
            # Save binary combined mask for compatibility
            combined_patch_mask = (patch_mask > 0).astype(np.uint8) * 255
            self.image_processor.save_image(combined_patch_mask, str(patch_dir / 'segmentation_mask_patch.png'))
            
            # Save individual category masks for patches
            patch_masks_dir = patch_dir / 'category_masks'
            patch_masks_dir.mkdir(exist_ok=True)
            
            # Create individual class masks from multiclass mask
            # Scratches
            if np.any(patch_mask == 1):
                scratch_mask = (patch_mask == 1).astype(np.uint8) * 255
                self.image_processor.save_image(scratch_mask, str(patch_masks_dir / f'patch_mask_scratches.png'))
            
            # All dirt categories
            for index, category_name in enumerate(self.dirt_categories):
                class_mask = (patch_mask == index).astype(np.uint8) * 255
                if np.any(class_mask):
                    self.image_processor.save_image(class_mask, str(patch_masks_dir / f'patch_mask_{category_name}.png'))
            
            # Create categories dict for patch labels
            patch_categories = {
                "0": {"name": "background", "supercategory": "background", "id": 0},
                "1": {"name": "scratches", "supercategory": "defect", "id": 1}
            }
            for category_name, category_info in self.dirt_categories.items():
                patch_categories[str(index)] = {
                    "name": category_name, 
                    "supercategory": "defect", 
                    "id": index
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
        patch_size: int = 1024,
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
                    generate_visualizations=generate_visualizations
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
                
        # Save dataset summary
        summary = {
            'dataset_info': {
                'total_images': len(results),
                'classes': {
                    f"{index}": f"{self.category_names[index]}" for index in range(len(self.category_names))
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
        
        # Print summary statistics
        if results:
            total_scratches = sum(r['class_statistics']['scratches_pixels'] for r in results)
            total_pixels = sum(r['image_size'][0] * r['image_size'][1] for r in results)
            
            self.logger.info("\n=== DATASET SUMMARY ===")
            self.logger.info(f"Total images processed: {len(results)}")
            self.logger.info(f"Total pixels: {total_pixels:,}")
            self.logger.info(f"Total scratch pixels: {total_scratches:,} ({total_scratches/total_pixels*100:.2f}%)")
            
            # Statistics for each dirt category
            for category_name in self.dirt_categories.keys():
                total_category = sum(r['class_statistics'].get(f'{category_name}_pixels', 0) for r in results)
                self.logger.info(f"Total {category_name} pixels: {total_category:,} ({total_category/total_pixels*100:.2f}%)")
            
            self.logger.info("======================\n")

def get_argparse():
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data with scratches and dirt categories')
    # input directories
    parser.add_argument('--scratch-inference-dir', type=str, default="tests/inf_dirt_only",
                       help='Directory containing scratch inference results')
    parser.add_argument('--clean-images-dir', type=str, default="tests/crack",
                       help='Directory containing full phone images')
    parser.add_argument('--dirt-categories-dir', type=str, default="C:/Users/umang_maheshwari/Desktop/FutureDial/repository/epam/tests/input_new/dirt_categories",
                       help='Base directory containing dirt category subdirectories (dirt, condensation, etc.)')
    parser.add_argument('--output-dir', type=str, default='tests/demo',
                       help='Output directory for synthetic data')
    
    # Output settings
    
    # Generation parameters
    parser.add_argument('--down-scale-clean-img', type=int, default=2,
                       help='Downscale factor for clean image')
    parser.add_argument('--num-dirt', type=int, default=10,
                       help='Number of dirt samples to superimpose per image')
    parser.add_argument('--num-version', type=int, default=1,
                       help='Number of versions per clean image')
    parser.add_argument('--patch-size', type=int, default=1024,
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
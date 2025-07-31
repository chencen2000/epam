import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, List



import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger_config import setup_application_logger
from src.image_processor import ImageProcessor
from src.synthesis.patch_generator import PatchGenerator
from src.target_labels import TargetLabels


class SyntheticDataGenerator:
    def __init__(
            self,
            image_processor: ImageProcessor,
            patch_generator: PatchGenerator,
            clean_images_dir: str, 
            dirt_base_dir: str,
            dirt_categories: dict,
            num_version_per_image: int = 1,
            num_dirt_super_impose: int = 40,
            clean_image_downscale_factor: int = 2,
            scale_dirt: Tuple[int, int] = (0.9, 1.15),
            rotation_range: Tuple[int, int] = (0, 360),
            bg_estimation_filter_size: int = 51,  # TOD: think to add in method or here
            app_logger: Optional[logging.Logger] = None
            ):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('SyntheticDataGenerator')

        self.clean_dir = clean_images_dir
        self.dirt_base_dir = dirt_base_dir
        self.dirt_categories = dirt_categories
        self.category_weights = list(dirt_categories.values())
        self.category_names = list(dirt_categories.keys())

        self.image_processor = image_processor
        self.patch_gen = patch_generator

        self.num_version_per_image = num_version_per_image
        self.num_dirt_super_impose = num_dirt_super_impose
        self.clean_image_downscale_factor = clean_image_downscale_factor
        self.scale_range = scale_dirt
        self.rotation_range = rotation_range
        self.bg_estimation_filter_size = bg_estimation_filter_size

        # load img paths dirs
        self.clean_imgs = self._get_images_from_dir(self.clean_dir)
        self.logger.debug(f"Found {len(self.clean_imgs)} clean images")

        FOLDER_TO_IGNORE = [
            TargetLabels.BACKGROUND.value, 
            # TargetLabels.CONDENSATION.value
        ]
        self.dirt_imgs_by_category = {
            category: self._load_category_images(category)
            for category in self.dirt_categories if category not in FOLDER_TO_IGNORE
        }

        # Define distinct colors for each category
        self.category_colors = self.create_category_colors()
        self.category_color_map = self.create_colormap()        

    def _load_category_images(self, category_name: str) -> list:
        """Load images for a specific dirt category"""
        category_dir = f"{self.dirt_base_dir}/{category_name}" 
        images = self._get_images_from_dir(category_dir)
        
        if os.path.exists(category_dir):
            self.logger.debug(f"Found {len(images)} images for category '{category_name}'")
        else:
            self.logger.warning(f"Category directory not found: {category_dir}")
        
        return images

    def _get_images_from_dir(self, directory: str) -> list:
        """Get all image files from a directory"""
        if not os.path.exists(directory):
            return []
        
        return [f for f in os.listdir(directory) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    def create_category_colors(self) -> Dict[str, List[int]]:
        """Initialize distinct colors for each dirt category."""
        # Use a colorblind-friendly palette
        base_colors = [
            [255, 255, 0],    # Yellow 
            [255, 0, 0],      # Red 
            [0, 255, 0],      # Green 
            [0, 0, 255],      # Blue 
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [255, 165, 0],    # Orange
            [128, 0, 128],    # Purple
            [255, 192, 203],  # Pink
            [165, 42, 42]     # Brown
        ]
        
        colors = {'background': [0, 0, 0]}
        
        for i, category in enumerate(self.dirt_categories.keys()):
            if category == TargetLabels.BACKGROUND: continue 
            colors[category] = (
                base_colors[i] if i < len(base_colors) 
                else [np.random.randint(0, 256) for _ in range(3)]
            )
        
        self.logger.debug(f"Initialized colors for categories: {list(colors.keys())}")
        
        return colors
    
    def create_colormap(self) -> Tuple:
        """Create a colormap for matplotlib visualization."""
        colors = [np.array([0, 0, 0]) / 255.0]  # Background
        labels = ['background']
        
        for category in self.dirt_categories.keys():
            if category == TargetLabels.BACKGROUND: continue 
            colors.append(np.array(self.category_colors[category]) / 255.0)
            labels.append(category)
        
        return ListedColormap(colors), labels
    
    def super_impose_dirt_sample(
            self, 
            base_image_float: np.ndarray, 
            results: dict, 
            boundary_x: int, 
            boundary_y: int, 
            boundary_w: int, 
            boundary_h: int,
            segmentation_mask_accumulator: np.ndarray, 
            max_attempts: int, 
            placed_bboxes: list,
            annotation_list: list = None, 
            category_id: int = 1, 
            category_name: str = "unknown",
            num_vertices_per_side: int = 3, 
            max_distortion: float = 0.1 
    ) -> tuple[np.ndarray, np.ndarray, tuple | None]:
        """
        Superimpose dirt sample onto base image with proper mask handling and annotations.
        
        Returns:
            tuple: (modified_base_image, updated_segmentation_mask, placed_bbox or None)
        """
        # Validate required inputs
        required_keys = ["low_threshold_mask", "high_threshold_mask", "difference"]
        if not all(results.get(key) is not None for key in required_keys):
            return base_image_float, segmentation_mask_accumulator, None
        
        if annotation_list is None:
            annotation_list = []

        for attempt in range(max_attempts):
            # Generate random transformations
            scale = random.uniform(*self.scale_range)
            rotation_angle = random.uniform(*self.rotation_range)

            # Transform masks and difference image
            transformed_data = self._transform_dirt_components(
                results, scale, rotation_angle, num_vertices_per_side, max_distortion
            )
            if not transformed_data:
                continue

            (transformed_mask_low_distorted, transformed_mask_high_distorted, 
            transformed_difference_float, transformed_width, transformed_height) = transformed_data

            # Calculate valid placement bounds
            placement_bounds = self._calculate_placement_bounds(
                boundary_x, boundary_y, boundary_w, boundary_h, 
                transformed_width, transformed_height
            )
            if not placement_bounds:
                continue

            min_x, max_x, min_y, max_y = placement_bounds

            # Generate random placement position
            place_x = random.randint(min_x, max_x)
            place_y = random.randint(min_y, max_y)
            proposed_bbox = (place_x, place_y, place_x + transformed_width, place_y + transformed_height)

            # Validate placement
            if not self._is_valid_placement(proposed_bbox, boundary_x, boundary_y, 
                                        boundary_w, boundary_h, placed_bboxes):
                continue

            # Apply dirt effect to base image
            success = self._apply_dirt_effect(
                base_image_float, transformed_mask_low_distorted, 
                transformed_difference_float, place_x, place_y, 
                transformed_width, transformed_height
            )
            if not success:
                continue

            # Update segmentation mask
            self._update_segmentation_mask(
                segmentation_mask_accumulator, results["high_threshold_mask"], 
                scale, rotation_angle, place_x, place_y, transformed_width, transformed_height
            )

            # Generate annotations
            self._generate_annotations(
                transformed_mask_high_distorted, annotation_list, place_x, place_y,
                transformed_width, transformed_height, category_id, category_name
            )

            return base_image_float, segmentation_mask_accumulator, proposed_bbox

        return base_image_float, segmentation_mask_accumulator, None


    def _transform_dirt_components(
            self, 
            results: dict, 
            scale: float, 
            rotation_angle: float, 
            num_vertices_per_side: int, 
            max_distortion: float
    ) -> tuple | None:
        """Transform and distort dirt components (masks and difference image)."""
        
        # Transform low threshold mask (for visual effect)
        transformed_mask_low, width, height = self.scale_and_rotate(
            results["low_threshold_mask"], scale, rotation_angle
        )
        if transformed_mask_low is None:
            return None

        # Apply distortion to low threshold mask
        transformed_mask_low_distorted = self.image_processor.apply_distorted_boundary_to_mask(
            transformed_mask_low, num_vertices_per_side, max_distortion
        )

        # Transform high threshold mask (for annotations)
        transformed_mask_high, _, _ = self.scale_and_rotate(
            results["high_threshold_mask"], scale, rotation_angle
        )
        if transformed_mask_high is None:
            return None

        # Ensure masks have matching dimensions
        if transformed_mask_high.shape != transformed_mask_low.shape:
            transformed_mask_high = cv2.resize(
                transformed_mask_high, 
                (transformed_mask_low.shape[1], transformed_mask_low.shape[0]), 
                interpolation=cv2.INTER_CUBIC
            )
            _, transformed_mask_high = cv2.threshold(
                transformed_mask_high, 127, 255, cv2.THRESH_BINARY
            )

        # Apply distortion to high threshold mask
        transformed_mask_high_distorted = self.image_processor.apply_distorted_boundary_to_mask(
            transformed_mask_high, num_vertices_per_side, max_distortion
        )

        # Transform difference image
        scaled_difference_float, _, _ = self.image_processor.apply_scale(
            results['difference'], scale, interpolation=cv2.INTER_CUBIC
        )
        if scaled_difference_float is None:
            return None

        transformed_difference_float, _, _ = self.image_processor.apply_rotation(
            scaled_difference_float, rotation_angle, 
            interpolation=cv2.INTER_CUBIC, border_value=0.0
        )
        if transformed_difference_float is None:
            return None

        return (transformed_mask_low_distorted, transformed_mask_high_distorted, 
                transformed_difference_float, width, height)


    def _calculate_placement_bounds(
            self, 
            boundary_x: int, 
            boundary_y: int, 
            boundary_w: int, 
            boundary_h: int,
            transformed_width: int, 
            transformed_height: int
    ) -> tuple | None:
        """Calculate valid placement bounds for transformed dirt sample."""
        
        min_x = boundary_x
        max_x = (boundary_x + boundary_w) - transformed_width
        min_y = boundary_y
        max_y = (boundary_y + boundary_h) - transformed_height

        # Check if placement is possible
        if max_x < min_x or max_y < min_y:
            return None

        return min_x, max_x, min_y, max_y


    def _is_valid_placement(
            self, 
            proposed_bbox: tuple, 
            boundary_x: int, 
            boundary_y: int, 
            boundary_w: int, 
            boundary_h: int,
            placed_bboxes: list
    ) -> bool:
        """Check if proposed placement is valid (within bounds and no overlap)."""
        
        x1, y1, x2, y2 = proposed_bbox
        
        # Check boundary constraints
        within_bounds = (
            x1 >= boundary_x and y1 >= boundary_y and
            x2 <= (boundary_x + boundary_w) and y2 <= (boundary_y + boundary_h)
        )
        
        # Check overlap with existing placements
        no_overlap = not self.image_processor.check_overlap(proposed_bbox, placed_bboxes)
        
        return within_bounds and no_overlap


    def _apply_dirt_effect(
            self, 
            base_image_float: np.ndarray, 
            mask_distorted: np.ndarray, 
            difference_float: np.ndarray,
            place_x: int, 
            place_y: int, 
            width: int, 
            height: int
    ) -> bool:
        """Apply dirt effect to the base image using mask and difference."""
        
        # Extract ROI from base image
        target_roi = base_image_float[place_y:place_y + height, place_x:place_x + width]
        
        # Validate dimensions match
        if not (target_roi.shape[:2] == mask_distorted.shape[:2] == difference_float.shape[:2]):
            return False

        # Prepare mask and difference for blending
        binary_mask = mask_distorted.astype(np.float32) / 255.0
        dirt_diff_3channel = np.stack([difference_float] * 3, axis=-1)
        binary_mask_3channel = np.stack([binary_mask] * 3, axis=-1)

        # Apply dirt effect
        result_roi = target_roi - (dirt_diff_3channel * binary_mask_3channel)
        result_roi = np.clip(result_roi, 0, 255)
        
        # Update base image
        base_image_float[place_y:place_y + height, place_x:place_x + width] = result_roi
        
        return True


    def _update_segmentation_mask(
            self, 
            segmentation_mask_accumulator: np.ndarray, 
            high_threshold_mask: np.ndarray,
            scale: float, 
            rotation_angle: float, 
            place_x: int, 
            place_y: int, 
            width: int, 
            height: int
    ) -> None:
        """Update segmentation mask accumulator with new dirt placement."""
        
        # Transform high threshold mask for segmentation
        transformed_mask_high, _, _ = self.scale_and_rotate(
            high_threshold_mask, scale, rotation_angle
        )
        if transformed_mask_high is None:
            return

        # Update segmentation mask
        segmentation_roi = segmentation_mask_accumulator[place_y:place_y + height, place_x:place_x + width]
        updated_roi = cv2.bitwise_or(segmentation_roi, transformed_mask_high)
        segmentation_mask_accumulator[place_y:place_y + height, place_x:place_x + width] = updated_roi


    def _generate_annotations(
            self, 
            mask_high_distorted: np.ndarray, 
            annotation_list: list, 
            place_x: int, 
            place_y: int,
            width: int, 
            height: int, 
            category_id: int, 
            category_name: str,
            min_contour_area: int = 20
    ) -> None:
        """Generate annotations from distorted high threshold mask."""
        
        # Find contours in the distorted mask
        contours, _ = cv2.findContours(
            mask_high_distorted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < min_contour_area:
                continue
            
            # Offset contour points to global coordinates
            offset_contour = contour + (place_x, place_y)
            segmentation_points = [
                point for sublist in offset_contour.tolist() 
                for point in sublist[0]
            ]
            
            # Create annotation
            annotation = {
                "bbox": [place_x, place_y, width, height],
                "segmentation": [segmentation_points], 
                "category_id": category_id,
                "category_name": category_name,
                "area": cv2.countNonZero(mask_high_distorted)
            }
            annotation_list.append(annotation)

    def scale_and_rotate(self, results_threshold_mask, scale, rotation_angle):
        scaled_mask, scaled_width, scaled_height = self.image_processor.apply_scale(results_threshold_mask, scale, interpolation=cv2.INTER_CUBIC)
        if scaled_mask is None or scaled_width <= 0 or scaled_height <= 0: return None, None, None
        _, scaled_mask = cv2.threshold(scaled_mask, 127, 255, cv2.THRESH_BINARY)

        transformed_mask, transformed_width, transformed_height = self.image_processor.apply_rotation(scaled_mask, rotation_angle, interpolation=cv2.INTER_CUBIC, border_value=0)
        if transformed_mask is None or transformed_width <= 0 or transformed_height <= 0: return None, None, None
        _, transformed_mask = cv2.threshold(transformed_mask, 127, 255, cv2.THRESH_BINARY)

        return transformed_mask, transformed_width, transformed_height

    def process_single_image_version(
            self,
            image: np.ndarray, image_name: str, version_idx: int,
            boundary_x: int, boundary_y: int, boundary_w: int, boundary_h: int,
            num_vertices_per_side: int, 
            max_distortion: float, 
        ):
        self.logger.debug(f"--- Processing for {image_name} - Version {version_idx+1} ---")

        target_height, target_width, _ = image.shape
        image_float = image.copy().astype(np.float32)

        # Create separate segmentation masks for each category
        segmentation_masks_by_category = {}
        category_counts = {}
        for category in self.dirt_categories:
            segmentation_masks_by_category[category] = np.zeros((target_height, target_width), dtype=np.uint8)
            category_counts[category] = 0

        # Combined mask for overlap checking
        combined_segmentation_mask = np.zeros((target_height, target_width), dtype=np.uint8)

        placed_bboxes, annotations_for_this_image = [], []
        successfully_placed_count = 0
        

        for category in self.dirt_imgs_by_category:
            n_dirt = self.dirt_categories[category] * self.num_dirt_super_impose
            self.logger.debug(f"for {category = } needs {n_dirt} / {self.num_dirt_super_impose} superimposition...")
            for dirt_idx in range(int(n_dirt)):
                random_dirty_image = random.choice(self.dirt_imgs_by_category[category])
                random_dirty_image = f"{self.dirt_base_dir}/{category}/{random_dirty_image}"

                self.logger.debug(f"  Attempting to place {category} sample {dirt_idx+1}/{self.num_dirt_super_impose}")
                self.logger.debug(f"    -> Using source: {random_dirty_image}")

                results = self.image_processor.process_dirt_mask_extraction(
                    random_dirty_image, self.bg_estimation_filter_size,  resize_factor=2
                )

                if results["low_threshold_mask"] is None or results["high_threshold_mask"] is None:
                    continue

                image_float, updated_mask, placed_bbox = self.super_impose_dirt_sample(
                    image_float, results, boundary_x, boundary_y, boundary_w, boundary_h,
                    segmentation_masks_by_category[category], 10, placed_bboxes, 
                    annotations_for_this_image, self.category_names.index(category), category,
                    num_vertices_per_side, max_distortion
                )

                if placed_bbox:
                    placed_bboxes.append(placed_bbox)
                    successfully_placed_count += 1
                    category_counts[category] += 1
                    
                    # Update combined mask
                    combined_segmentation_mask = cv2.bitwise_or(
                        combined_segmentation_mask, 
                        segmentation_masks_by_category[category]
                    )

        self.logger.debug(f"--- Processing complete for {image_name} - Version {version_idx+1} ---")
        self.logger.debug(f"  Successfully placed {successfully_placed_count}/{self.num_dirt_super_impose} samples.")
        self.logger.debug(f"  Category distribution: {category_counts}")
        return (np.clip(image_float, 0, 255).astype(np.uint8), 
                segmentation_masks_by_category, 
                combined_segmentation_mask,
                annotations_for_this_image, category_counts)

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
                
            category_id = self.category_names.index(category_name)
            mask_bool = mask > 0
            
            # Set pixel values to category ID (higher IDs will overwrite lower IDs in overlaps)
            multiclass_mask[mask_bool] = category_id
            
            self.logger.debug(f"Applied category '{category_name}' (ID: {category_id}) to {np.sum(mask_bool)} pixels")
        
        # Log class distribution in the mask
        unique_values, counts = np.unique(multiclass_mask, return_counts=True)
        class_distribution = dict(zip(unique_values, counts))
        self.logger.debug(f"Multi-class mask distribution: {class_distribution}")
        
        return multiclass_mask
    
    
    def _calculate_category_stats(self, annotations):
        """Calculate how many annotations exist for each category"""
        # Initialize counts for all known categories
        stats = {name: 0 for name in self.dirt_categories.keys()}
        
        # Count annotations by category
        for annotation in annotations:
            category_name = annotation.get('category_name', 'unknown')
            
            if category_name in stats:
                stats[category_name] += 1
            else:
                # Handle unknown categories
                stats.setdefault('unknown', 0)
                stats['unknown'] += 1
        
        return stats
    
    def _calculate_success_rates(self, category_counts, annotations):
        """Calculate success rates for dirt placement by category"""
        success_rates = {}
        successful_counts = self._calculate_category_stats(annotations)
        
        for category in self.dirt_categories.keys():
            attempted = category_counts.get(category, 0)
            successful = successful_counts.get(category, 0)
            
            success_rate = (successful / attempted * 100) if attempted > 0 else 0
            
            success_rates[category] = {
                'attempted': attempted,
                'successful': successful,
                'success_rate_percent': round(success_rate, 2)
            }
        
        return success_rates
    
    def save_category_statistics_plot(self, annotations=None, category_counts=None, 
                                    save_path=None, title="Category Distribution"):
        """Save a simple bar plot showing category distribution"""
        # Get counts from either source
        if category_counts:
            counts = category_counts.copy()
        elif annotations:
            counts = {}
            for ann in annotations:
                category = ann.get('category_name', 'unknown')
                counts[category] = counts.get(category, 0) + 1
        else:
            self.logger.error("Need either annotations or category_counts")
            return
        
        if not counts:
            return
        
        # Create simple bar plot
        categories = list(counts.keys())
        values = list(counts.values())
        colors = [np.array(self.category_colors.get(cat, [128, 128, 128])) / 255.0 for cat in categories]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=12, fontweight='bold')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def _calculate_success_rates(self, category_counts, annotations):
        """Calculate success rates for dirt placement by category"""
        success_rates = {}
        successful_counts = self._calculate_category_stats(annotations)
        
        for category in self.dirt_categories.keys():
            attempted = category_counts.get(category, 0)
            successful = successful_counts.get(category, 0)
            
            success_rate = (successful / attempted * 100) if attempted > 0 else 0
            
            success_rates[category] = {
                'attempted': attempted,
                'successful': successful,
                'success_rate_percent': round(success_rate, 2)
            }
        
        return success_rates


    def create_colored_segmentation_mask(self, masks_by_category, image_shape):
        """Create a colored mask where each category has a distinct color"""
        height, width = image_shape[:2]
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        for category_name, mask in masks_by_category.items():
            if mask is None or not np.any(mask):
                continue
                
            color = self.category_colors[category_name]
            colored_mask[mask > 0] = color
        
        return colored_mask


    def create_indexed_segmentation_mask(self, masks_by_category, image_shape):
        """Create an indexed mask where pixel values represent category IDs"""
        height, width = image_shape[:2]
        indexed_mask = np.zeros((height, width), dtype=np.uint8)
        
        for category_name, mask in masks_by_category.items():
            if mask is None or not np.any(mask):
                continue
                
            category_id = self.category_names.index(category_name)
            indexed_mask[mask > 0] = category_id
        
        return indexed_mask


    def visualize_annotations_with_colors(self, image, annotations, save_path=None):
        """Draw bounding boxes with category-specific colors"""
        plt.figure(figsize=(12, 8))
        
        # Display image
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        
        # Draw annotations
        category_counts = {}
        
        for ann in annotations:
            category_name = ann.get('category_name', 'unknown')
            bbox = ann['bbox']
            
            # Get color and draw bounding box
            color = np.array(self.category_colors.get(category_name, [128, 128, 128])) / 255.0
            
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor=color, facecolor='none'
            )
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(bbox[0], bbox[1] - 5, category_name, 
                    color=color, fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Count categories
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
        
        # Create legend
        legend_elements = []
        for category_name in self.dirt_categories.keys():
            color = np.array(self.category_colors.get(category_name, [128, 128, 128])) / 255.0
            count = category_counts.get(category_name, 0)
            
            if count > 0:
                legend_elements.append(
                    patches.Patch(color=color, label=f'{category_name} ({count})')
                )
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title(f'Annotations - {len(annotations)} instances')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


    def create_overlay_visualization(self, original_image, colored_mask, alpha=0.6, save_path=None):
        """Create an overlay of original image with colored segmentation mask"""
        # Ensure same size
        if original_image.shape[:2] != colored_mask.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (original_image.shape[1], original_image.shape[0]))
        
        # Convert to BGR if grayscale
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Create overlay where mask exists
        overlay_image = original_image.copy()
        mask_present = np.any(colored_mask > 0, axis=2)
        
        overlay_image[mask_present] = (
            (1 - alpha) * original_image[mask_present] + 
            alpha * colored_mask[mask_present]
        )
        
        if save_path:
            cv2.imwrite(save_path, overlay_image)
        
        return overlay_image


    def generate_comprehensive_visualization(self, image, masks_by_category, annotations, 
                                        output_dir, base_name="visualization", 
                                        category_counts=None):
        """Generate all visualizations in one go"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create all visualizations
        colored_mask = self.create_colored_segmentation_mask(masks_by_category, image.shape)
        indexed_mask = self.create_indexed_segmentation_mask(masks_by_category, image.shape)
        
        # Save files
        results = {}
        
        # Save colored mask
        colored_path = os.path.join(output_dir, f"{base_name}_colored_mask.png")
        cv2.imwrite(colored_path, colored_mask)
        results['colored_mask'] = colored_path
        
        # Save indexed mask
        indexed_path = os.path.join(output_dir, f"{base_name}_indexed_mask.png")
        cv2.imwrite(indexed_path, indexed_mask)
        results['indexed_mask'] = indexed_path
        
        # Save overlay
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        self.create_overlay_visualization(image, colored_mask, save_path=overlay_path)
        results['overlay'] = overlay_path
        
        # Save annotation visualization
        annotation_path = os.path.join(output_dir, f"{base_name}_annotations.png")
        self.visualize_annotations_with_colors(image, annotations, save_path=annotation_path)
        results['annotations'] = annotation_path
        
        # Save statistics plot
        stats_path = os.path.join(output_dir, f"{base_name}_statistics.png")
        self.save_category_statistics_plot(annotations=annotations, category_counts=category_counts, 
                                        save_path=stats_path)
        results['statistics'] = stats_path
        
        return results
    
    def count_category_ids(self, annotations):
        """Count occurrences of each category_id in annotations"""
        category_counts = {}
        
        for ann in annotations:
            category_id = ann.get('category_id', 0)
            category_counts[self.category_names[category_id]] = category_counts.get(self.category_names[category_id], 0) + 1
        
        return category_counts
    
    def preprocess_image_synthesis(self, image_name:str):
        # 1. Load images
        original_img = self.image_processor.load_image(f"{self.clean_dir}/{image_name}", False)

        # 2. Resize image
        resized_img = self.image_processor.resize_image(original_img, self.clean_image_downscale_factor)
        self.logger.debug(f"Image shape: {resized_img.shape}")

        # 3. Detect mobile boundaries
        x, y, w, h = self.image_processor.get_device_bbox(resized_img)
        self.logger.debug(f"Detected boundaries: x={x}, y={y}, w={w}, h={h}")

        return resized_img, (x, y, w, h)
    
    def save_dirty_and_mask_image(self, dirty_img, combined_mask, multiclass_mask, output_dir:Path, is_patch=False):
        dirty_img_name = "synthetic_dirty_image_full" if not is_patch else "synthetic_dirty_patch"
        self.image_processor.save_image(dirty_img, str(output_dir / f"{dirty_img_name}.png"))

        comb_name = "patch" if is_patch else "combined"
        self.image_processor.save_image(combined_mask, str(output_dir / f"segmentation_mask_{comb_name}.png") )
        
        multi_name = "_patch" if is_patch else ""
        self.image_processor.save_image(multiclass_mask, str(output_dir / f"segmentation_mask{multi_name}_multiclass.png"))

    def save_category_specific_masks(self, masks_by_category, output_dir:Path):
        
        masks_dir = os.path.join(output_dir, 'category_masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        for category_name, category_mask in masks_by_category.items():
            if np.any(category_mask):  # Only save non-empty masks
                mask_path = os.path.join(masks_dir, f'mask_{category_name}.png')
                self.image_processor.save_image(category_mask, mask_path)


    def generate_synthetic_data_using_clean_images(
            self, 
            output_dir: str, num_vertices_per_side: int = 3, max_distortion: float = 0.1, 
            patch_size: int = 1792, generate_visualizations: bool = True, max_images: int = 1
        ):
        """This method helps in generating the synthetic data using the clean images

        Args:
            output_dir (str): _description_
            num_vertices_per_side (int, optional): _description_. Defaults to 3.
            max_distortion (float, optional): _description_. Defaults to 0.1.
            patch_size (int, optional): _description_. Defaults to 1024.
            generate_visualizations (bool, optional): _description_. Defaults to True.
            max_images (int, optional): _description_. Defaults to 1.
        """

        self.logger.info(f"--- Starting synthetic data generation ---")
        output_dir = Path(output_dir)

        clean_images = os.listdir(self.clean_dir)
        self.logger.debug(f"Found {len(clean_images)} images under clean images")
        max_images = max_images if max_images else len(clean_images)
        clean_images = clean_images[:max_images]

        for i, image_name in tqdm(enumerate(clean_images), desc="total images processed: ", total=len(clean_images)):

            self.logger.debug(f"=== Processing {image_name} {max_images = }===")

            resized_img, (x, y, w, h) = self.preprocess_image_synthesis(image_name)

            img_h, img_w, _ = resized_img.shape

            # 4. create n different version of same image 
            for version_idx in range(self.num_version_per_image):
                dirty_full, masks_by_category, combined_mask, ann_full, category_counts = self.process_single_image_version(
                    resized_img, image_name, version_idx,
                    x, y, w, h, num_vertices_per_side, max_distortion 
                )

                multiclass_mask = self.create_multiclass_mask(masks_by_category, dirty_full.shape)
                
                # Create version output directory
                ver_out_dir = os.path.join(output_dir, f"{image_name.split('.')[0]}_v{version_idx:02d}")

                ver_out_dir = output_dir / f"{image_name.split('.')[0]}_v{version_idx:02d}"
                ver_out_dir.mkdir(parents=True, exist_ok=True)
                
                # Save main outputs
                self.save_dirty_and_mask_image(dirty_full, combined_mask, multiclass_mask, ver_out_dir, )

                # Save category-specific masks
                self.save_category_specific_masks(masks_by_category, ver_out_dir)

                category_label = {
                    str(self.category_names.index(category)) :{
                        "name": category, "supercategory": "dirt"
                    } for category in self.dirt_categories
                    
                }

                enhanced_labels = {
                    "annotations": ann_full,
                    "categories": category_label,
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

                # Generate comprehensive visualizations for full image
                if generate_visualizations:
                    viz_dir = os.path.join(ver_out_dir, 'visualizations')
                    visualization_paths = self.generate_comprehensive_visualization(
                        dirty_full, masks_by_category, ann_full, viz_dir, 
                        f"{image_name.split('.')[0]}_v{version_idx:02d}",
                        category_counts=category_counts,
                    )
                    self.logger.debug(f"Generated full image visualizations: {list(visualization_paths.keys())}")

                # Step : Create patches from the full synthetic image
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
                    self.image_processor.save_image(p_img, os.path.join(p_dir, 'synthetic_dirty_patch.png'))
                    self.image_processor.save_image(p_mask, os.path.join(p_dir, 'segmentation_mask_patch.png'))

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
                    self.image_processor.save_image(patch_multiclass_mask, os.path.join(p_dir, 'segmentation_mask_patch_multiclass.png'))
                    self.image_processor.save_image(p_mask, os.path.join(p_dir, 'segmentation_mask_patch.png'))

                    # Save patch-level category masks
                    patch_masks_dir = os.path.join(p_dir, 'category_masks')
                    os.makedirs(patch_masks_dir, exist_ok=True)

                    for category_name, patch_category_mask in patch_masks_by_category.items():
                        if np.any(patch_category_mask):  # Only save non-empty masks
                            patch_mask_path = os.path.join(patch_masks_dir, f'patch_mask_{category_name}.png')
                            self.image_processor.save_image(patch_category_mask, patch_mask_path)

                    # Save enhanced patch labels
                    patch_labels = {
                        "annotations": p_ann,
                        "categories": category_label,
                        "image_size": [patch_size, patch_size],
                        "original_global_offset": [gx, gy],
                        "category_statistics": self._calculate_category_stats(p_ann),
                        "category_colors": self.category_colors,
                        "parent_image": f"{image_name.split('.')[0]}_v{version_idx:02d}",
                        "patch_boundary_in_full": [gx, gy, gx + patch_size, gy + patch_size]
                    }
                    
                    with open(os.path.join(p_dir, 'labels_patch.json'), 'w') as f_json:
                        json.dump(patch_labels, f_json, indent=4)

                    # Generate patch-level visualizations if there are annotations
                    if p_ann and generate_visualizations:
                        patch_viz_dir = os.path.join(p_dir, 'visualizations')
                        p_category_counts = self.count_category_ids(annotations=p_ann)
                        try:
                            patch_viz_paths = self.generate_comprehensive_visualization(
                                p_img, patch_masks_by_category, p_ann, patch_viz_dir, p_name, p_category_counts
                            )
                            self.logger.debug(f"    Generated patch visualizations: {list(patch_viz_paths.keys())}")
                        except Exception as e:
                            self.logger.warning(f"Failed to generate visualizations for patch {p_name}: {e}")

                    self.logger.debug(f"Completed patch processing for {image_name} (Version {version_idx+1})")
                
                

            self.logger.debug(f"=== Processing complete for {image_name} ===")
            
        self.logger.info("Dataset generation completed successfully!")



def get_argparse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate synthetic dirt detection dataset')
    
    # Input directories
    parser.add_argument('--clean-images-dir', type=str, default="tests/test_images/input/clean_images",
                       help='Directory containing clean phone images')
    parser.add_argument('--dirt-categories-dir', type=str, default="tests/test_images/input/dirt_categories",
                       help='Directory containing clean phone images')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='tests/test_images/output/synthetic_dataset_by_superimposing',
                       help='Output directory for generated dataset')
    
    # Generation parameters
    parser.add_argument('--down_scale_clean_img', type=int, default=2,
                       help='Downscale factor for clean image')
    parser.add_argument('--num-dirt', type=int, default=150,
                       help='Number of dirt samples per image')
    parser.add_argument('--num-version', type=int, default=1,
                       help='Number of version of clean imags')
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
    
    # Config file option
    # parser.add_argument('--config', type=str,
    #                    help='Path to configuration file (overrides CLI args)')

    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    
    return parser.parse_args()


def main():
    args = get_argparse()

    log_level = logging.DEBUG if not args.debug else logging.INFO
    app_logger = setup_application_logger(
        app_name="scratch_dirt_generator",
        log_file_name="logs/synthetic_dataset_by_superimposing.log"
    )
    app_logger.setLevel(log_level)

    app_logger.info("Starting scratch-dirt synthetic data generation with configuration:")
    app_logger.info(f"  Clean images dir: {args.clean_images_dir}")
    app_logger.info(f"  Dirt categories dir: {args.dirt_categories_dir}")
    app_logger.info(f"  Output dir: {args.output_dir}")
    app_logger.info(f"  Downscale clean image by: {args.down_scale_clean_img}")
    app_logger.info(f"  Number of dirt samples: {args.num_dirt}")
    app_logger.info(f"  Patch size: {args.patch_size}")
    app_logger.info(f"  Max images: {args.max_images}")
    app_logger.info(f"  Generate visualizations: {not args.no_visualizations}")
    app_logger.info(f"  Number of vertices: {not args.num_vertices}")
    app_logger.info(f"  Maximum distortion: {not args.max_distortion}")

    # Intialize component
    image_processor = ImageProcessor(app_logger)
    patch_generator = PatchGenerator(app_logger)
    generator = SyntheticDataGenerator(
        image_processor, patch_generator, 
        args.clean_images_dir, args.dirt_categories_dir,
        TargetLabels.value_to_weight_map(),
        args.num_version, args.num_dirt, args.down_scale_clean_img,
        app_logger=app_logger
    )

    generator.generate_synthetic_data_using_clean_images(
        args.output_dir, 
        max_images=args.max_images,
        generate_visualizations=not args.no_visualizations,
        max_distortion=args.max_distortion,
        num_vertices_per_side=args.num_vertices,
    )



if __name__ == "__main__":

    main()

    
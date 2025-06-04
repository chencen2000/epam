import os
import json
import random
import logging
from typing import Tuple, Optional

import cv2
import numpy as np

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
            dirt_images_dir:str,  
            num_version_per_image:int=1,
            num_dirt_super_impose:int=40,
            clean_image_downscale_factor:int=2,
            scale_dirt:Tuple[int, int]=(0.75, 1.5),
            rotation_range:Tuple[int, int]=(0, 360),
            app_logger:Optional[logging.Logger]=None
            ):
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('SyntheticDataGenerator')

        self.clean_dir = clean_images_dir
        self.dirt_dir = dirt_images_dir
        self.dirt_extractor = dirt_extractor
        self.image_operations = image_operations
        self.boundary_detector = boundary_detector
        self.patch_gen = patch_generator
        self.num_version_per_image = num_version_per_image
        self.num_dirt_super_impose = num_dirt_super_impose
        self.clean_image_downscale_factor = clean_image_downscale_factor
        self.scale_range = scale_dirt
        self.rotation_range = rotation_range

    def _fetch_dir_info(self,):
        # step 1: Get all dirt images name
        self.dirt_imgs = os.listdir(self.dirt_dir)
        self.logger.debug(f"---- We have found {len(self.dirt_imgs)} dirt images ----")

        # Step 2: Get all clean images name
        self.clean_imgs = os.listdir(self.clean_dir)
        self.logger.debug(f"---- We have found {len(self.clean_imgs)} clean images ----")
  
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
            annotation_list=[], num_vertices_per_side=3, 
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
                        # Filter out very small contours
                        if cv2.contourArea(contour) < 20:
                            continue
                            
                        offset_contour_np = contour + (place_x, place_y)
                        segmentation_points = [p for sublist in offset_contour_np.tolist() for p in sublist[0]]
                        annotation_list.append({
                            "bbox": [place_x, place_y, transformed_width, transformed_height],
                            "segmentation": [segmentation_points], "category_id": 1,
                            "area": cv2.countNonZero(transformed_mask_high_distorted)})
                        
                return base_image_float, segmentation_mask_accumulator, proposed_bbox
        return base_image_float, segmentation_mask_accumulator, None

    def process_super_impose(
            self, image, num_dirt_super_impose, boundary_x, boundary_y, boundary_w, boundary_h,
            bg_estimation_filter_size=51, num_vertices_per_side=3, 
            max_distortion=0.1 
        ):
        self.logger.debug(f"Super imposing dirt images: ")

        target_height, target_width, _ = image.shape
        image_float = image.copy().astype(np.float32)

        segmentation_mask_accumulator = np.zeros((target_height, target_width), dtype=np.uint8)
        placed_bboxes, annotations_for_this_image = [], []
        successfully_placed_count = 0

        for dirt_idx in range(num_dirt_super_impose):
            self.logger.debug(f"  Attempting to place dirt sample {dirt_idx+1}/{num_dirt_super_impose}.")
            random_dirty_image_path = f"{self.dirt_dir}/{random.choice(self.dirt_imgs)}"
            self.logger.debug(f"    -> Using source: {os.path.basename(random_dirty_image_path)}")

            results = self.dirt_extractor.process_image(random_dirty_image_path, bg_estimation_filter_size)
            if results["low_threshold_mask"] is None or results["high_threshold_mask"] is None or results["difference"] is None:
                self.logger.error(f"    -> Failed to create masks/difference. Skipping sample.")
                continue
            
            # FIXED: Pass the correct variables and get updated segmentation mask
            image_float, segmentation_mask_accumulator, placed_bbox = self.super_impose_dirt_sample(
                image_float, results, boundary_x, boundary_y, boundary_w, boundary_h,
                segmentation_mask_accumulator, 10, placed_bboxes, annotations_for_this_image,
                num_vertices_per_side, max_distortion
            )
            
            if placed_bbox:
                placed_bboxes.append(placed_bbox)
                successfully_placed_count += 1
            else:
                self.logger.warning(f"    -> Could not place sample {dirt_idx+1}. Skipping.")
            
        self.logger.debug(f"  Successfully placed {successfully_placed_count}/{self.num_dirt_super_impose} samples.")
        return np.clip(image_float, 0, 255).astype(np.uint8), segmentation_mask_accumulator, annotations_for_this_image


    def generate_synthetic_dataset(self, bg_estimation_filter_size:int, output_dir:str, num_vertices_per_side:int=3, max_distortion:float=0.1, patch_size=1024):
        
        self._fetch_dir_info()

        self.logger.info("---- Starting synthetic dataset generation -----")
        
        for i, image_name in enumerate(self.clean_imgs):
            # if i < 1: continue
            if i > 0: break
            # if "354183139993956_177_0_0_122529443" not in image_name: continue

            self.logger.debug(f"{image_name = }")
            original_image = self.image_operations.load_image_color(f"{self.clean_dir}/{image_name}")
            
            # Step 3: if downscale (resize) enabled then downscale
            if self.clean_image_downscale_factor > 1:
                image = self.image_operations.resize_image(original_image, self.clean_image_downscale_factor)
            else:
                image = original_image
            self.logger.debug(image.shape)
            img_h, img_w, _ = image.shape

            # Step 4: detect boundaries of mobile in image
            x, y, w, h = self.boundary_detector.detect_mobile_boundaries(image, False)
            
            
            # Step 5: Loop over num of version
            for version_idx in range(self.num_version_per_image):
                self.logger.debug(f"\n--- {image_name} - Version {version_idx+1} ---")
                #   Step 5.1: super-impose dirts on clean image
                #   Step 5.1: create mask and labels
                dirty_full, mask_full, ann_full = self.process_super_impose(
                    image, self.num_dirt_super_impose, x, y, w, h, bg_estimation_filter_size, 
                    num_vertices_per_side, max_distortion
                )
                
                ver_out_dir = os.path.join(output_dir, f"{image_name.split('.')[0]}_v{version_idx:02d}")
                os.makedirs(ver_out_dir, exist_ok=True)
                self.image_operations.save_image(dirty_full, os.path.join(ver_out_dir, 'synthetic_dirty_image_full.png'))
                self.image_operations.save_image(mask_full, os.path.join(ver_out_dir, 'segmentation_mask_full.png'))
                with open(os.path.join(ver_out_dir, 'labels_full.json'), 'w') as f_json:
                    json.dump({"annotations": ann_full, "image_size": [img_w, img_h], 
                            "boundary_pixels": [x, y, (x + w), (y + h)]}, f_json, indent=4)
                self.logger.debug(f"  Saved full synthetic image, mask, labels for {image_name} (Version {version_idx+1}).")

                #   Step 5.2: create patches
                self.logger.debug(f"\n--- Splitting patches for {image_name} (Version {version_idx+1}) ---")
                patch_data_list = self.patch_gen.split_image_into_patches(dirty_full, mask_full, ann_full, patch_size, 
                                 x, y, (x + w), (y + h))
                # patch_data_list = self.patch_gen.split_image_into_patches(dirty_full, mask_full, ann_full, patch_size, 
                #                  x, y, (x + w), (y + h), overlap=0.2)
                self.logger.debug(f"  Split into {len(patch_data_list)} patches with annotations.")
                #   Step 5.3: save patches
                patch_base_dir = os.path.join(ver_out_dir, 'patches')
                os.makedirs(patch_base_dir, exist_ok=True)
                for p_idx, (p_img, p_mask, p_ann, gx, gy) in enumerate(patch_data_list):
                    p_name = f"patch_{p_idx:03d}_x{gx}_y{gy}"
                    p_dir = os.path.join(patch_base_dir, p_name)
                    os.makedirs(p_dir, exist_ok=True)
                    self.image_operations.save_image(p_img, os.path.join(p_dir, 'synthetic_dirty_patch.png'))
                    self.image_operations.save_image(p_mask, os.path.join(p_dir, 'segmentation_mask_patch.png'))
                    with open(os.path.join(p_dir, 'labels_patch.json'), 'w') as f_json:
                        json.dump({"annotations": p_ann, "image_size": [patch_size, patch_size], 
                                "original_global_offset": [gx, gy]}, f_json, indent=4)
                    self.logger.debug(f"    Saved patch {p_idx:03d} to {p_dir}")
        self.logger.info("--- Dataset Generation Complete ---")


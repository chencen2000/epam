import os
import random

import cv2
import numpy as np

from visualization import visualize_results, visualize_dual_threshold_results
from image_operations import load_image_grayscale, load_image_color, apply_rotation, apply_scale, check_overlap, get_margin_bbox, apply_distorted_boundary_to_mask


def estimate_background_luminance(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Flatten the histogram and calculate the cumulative sum
    hist_flat = hist.flatten()
    cumulative_sum = np.cumsum(hist_flat)

    # Normalize the cumulative sum to get percentiles
    cumulative_sum_normalized = cumulative_sum / cumulative_sum[-1]

    # Find the 60th percentile
    percentile_60 = np.where(cumulative_sum_normalized >= 0.60)[0][0]

    # Find the most probable value above the 60th percentile
    most_probable_value = np.argmax(hist[percentile_60:]) + percentile_60

    return most_probable_value

def create_dirt_mask_from_dirty_image(read_path, threshold_low, threshold_high, BACKGROUND_ESTIMATION_FILTER_SIZE, visualize=False):
    """
    Create dirt mask with two different thresholds and improved background estimation
    """
    dirty_img_gray = load_image_grayscale(read_path)
    if dirty_img_gray is None:
        print(f"Failed to load grayscale image {read_path}")
        return None, None, None
    original_color = load_image_color(read_path)

    # Improved background estimation using multiple filters
    # Use median blur for primary background estimation
    # estimated_clean_background = cv2.medianBlur(dirty_img_gray, BACKGROUND_ESTIMATION_FILTER_SIZE)
    background_luminance = estimate_background_luminance(dirty_img_gray)
    estimated_clean_background = np.full(dirty_img_gray.shape, background_luminance, dtype=np.uint8)
    
    # Additional morphological operations for better background estimation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    estimated_clean_background = cv2.morphologyEx(estimated_clean_background, cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur for smoother background
    estimated_clean_background = cv2.GaussianBlur(estimated_clean_background, (5, 5), 0)
    
    dirty_img_float = dirty_img_gray.astype(np.float32)
    estimated_clean_background_float = estimated_clean_background.astype(np.float32)
    
    # Calculate raw difference (removing the capping modification as requested)
    estimated_dirt_difference_raw = dirty_img_float - estimated_clean_background_float

    # For mask creation, use absolute difference
    abs_estimated_dirt_difference = np.abs(estimated_dirt_difference_raw)
    abs_estimated_dirt_difference_normalized = cv2.normalize(abs_estimated_dirt_difference, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create two different masks with different thresholds
    _, dirt_mask_low = cv2.threshold(abs_estimated_dirt_difference_normalized, threshold_low, 255, cv2.THRESH_BINARY)
    _, dirt_mask_high = cv2.threshold(abs_estimated_dirt_difference_normalized, threshold_high, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to make masks more realistic
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Clean up the masks
    dirt_mask_low = cv2.morphologyEx(dirt_mask_low, cv2.MORPH_CLOSE, kernel_small)
    dirt_mask_low = cv2.morphologyEx(dirt_mask_low, cv2.MORPH_OPEN, kernel_small)
    
    dirt_mask_high = cv2.morphologyEx(dirt_mask_high, cv2.MORPH_CLOSE, kernel_medium)
    dirt_mask_high = cv2.morphologyEx(dirt_mask_high, cv2.MORPH_OPEN, kernel_small)
    
    # Remove small noise components
    dirt_mask_low = remove_small_components(dirt_mask_low, min_area=50)
    dirt_mask_high = remove_small_components(dirt_mask_high, min_area=100)

    if visualize and original_color is not None:
        visualize_dual_threshold_results(original_color, estimated_clean_background, 
                                       abs_estimated_dirt_difference_normalized, 
                                       dirt_mask_low, dirt_mask_high, 
                                       f"Dual Threshold Mask Creation for {os.path.basename(read_path)}")

    return dirt_mask_low, dirt_mask_high, estimated_dirt_difference_raw

def remove_small_components(binary_mask, min_area=50):
    """
    Remove small connected components from binary mask
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Create output mask
    output_mask = np.zeros_like(binary_mask)
    
    # Keep components larger than min_area (skip background label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output_mask[labels == i] = 255
    
    return output_mask

def place_dirt_sample(target_img_float, 
                      dirt_mask_low,  # For superimposing
                      dirt_mask_high, # For annotations
                      dirt_difference_for_this_sample,
                      placed_bboxes, 
                      mask_color_for_saving,
                      scale_range, rotation_range, max_attempts, 
                      segmentation_mask_accumulator, annotations_list,
                      boundary_x_min, boundary_y_min, boundary_x_max, boundary_y_max,
                      use_distorted_boundary=True,
                      distortion_params=None):
    """
    Place dirt sample using low threshold mask for superimposing and high threshold mask for annotations
    with optional distorted polygon boundaries.
    """
    if dirt_mask_low is None or dirt_mask_high is None or dirt_difference_for_this_sample is None:
        return target_img_float, segmentation_mask_accumulator, None

    # Default distortion parameters
    if distortion_params is None:
        distortion_params = {
            'num_vertices_per_side': random.randint(2, 5),
            'max_distortion': random.uniform(0.05, 0.15)
        }

    for _ in range(max_attempts):
        scale = random.uniform(scale_range[0], scale_range[1])
        rotation_angle = random.uniform(rotation_range[0], rotation_range[1])

        # Scale and rotate low threshold mask (for superimposing)
        scaled_mask_low, scaled_width, scaled_height = apply_scale(dirt_mask_low, scale, interpolation=cv2.INTER_NEAREST)
        if scaled_mask_low is None or scaled_width <= 0 or scaled_height <= 0: continue
        _, scaled_mask_low = cv2.threshold(scaled_mask_low, 127, 255, cv2.THRESH_BINARY)

        transformed_mask_low, transformed_width, transformed_height = apply_rotation(scaled_mask_low, rotation_angle, interpolation=cv2.INTER_NEAREST, border_value=0)
        if transformed_mask_low is None or transformed_width <= 0 or transformed_height <= 0: continue
        _, transformed_mask_low = cv2.threshold(transformed_mask_low, 127, 255, cv2.THRESH_BINARY)

        # Apply distorted boundary to low threshold mask if enabled
        if use_distorted_boundary:
            transformed_mask_low = apply_distorted_boundary_to_mask(
                transformed_mask_low, 
                num_vertices_per_side=distortion_params['num_vertices_per_side'],
                max_distortion=distortion_params['max_distortion']
            )

        # Scale and rotate high threshold mask (for annotations)
        scaled_mask_high, _, _ = apply_scale(dirt_mask_high, scale, interpolation=cv2.INTER_NEAREST)
        if scaled_mask_high is None: continue
        _, scaled_mask_high = cv2.threshold(scaled_mask_high, 127, 255, cv2.THRESH_BINARY)

        transformed_mask_high, _, _ = apply_rotation(scaled_mask_high, rotation_angle, interpolation=cv2.INTER_NEAREST, border_value=0)
        if transformed_mask_high is None: continue
        _, transformed_mask_high = cv2.threshold(transformed_mask_high, 127, 255, cv2.THRESH_BINARY)

        # Resize high threshold mask to match low threshold mask dimensions
        if transformed_mask_high.shape != transformed_mask_low.shape:
            transformed_mask_high = cv2.resize(transformed_mask_high, (transformed_mask_low.shape[1], transformed_mask_low.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, transformed_mask_high = cv2.threshold(transformed_mask_high, 127, 255, cv2.THRESH_BINARY)

        # Apply distorted boundary to high threshold mask if enabled
        if use_distorted_boundary:
            transformed_mask_high = apply_distorted_boundary_to_mask(
                transformed_mask_high, 
                num_vertices_per_side=distortion_params['num_vertices_per_side'],
                max_distortion=distortion_params['max_distortion']
            )

        # Scale and rotate difference image using cubic interpolation
        scaled_difference_float, _, _ = apply_scale(dirt_difference_for_this_sample, scale, interpolation=cv2.INTER_CUBIC)
        if scaled_difference_float is None: continue
        transformed_difference_float, _, _ = apply_rotation(scaled_difference_float, rotation_angle, interpolation=cv2.INTER_CUBIC, border_value=0.0)
        if transformed_difference_float is None: continue
        
        if transformed_mask_low.shape != transformed_difference_float.shape:
            transformed_difference_float = cv2.resize(transformed_difference_float, (transformed_mask_low.shape[1], transformed_mask_low.shape[0]), interpolation=cv2.INTER_CUBIC)

        min_x_placement, max_x_placement = boundary_x_min, boundary_x_max - transformed_width
        min_y_placement, max_y_placement = boundary_y_min, boundary_y_max - transformed_height
        if max_x_placement < min_x_placement or max_y_placement < min_y_placement: continue 

        place_x = random.randint(min_x_placement, max_x_placement)
        place_y = random.randint(min_y_placement, max_y_placement)
        proposed_bbox = (place_x, place_y, place_x + transformed_width, place_y + transformed_height)

        if (proposed_bbox[0] >= boundary_x_min and proposed_bbox[1] >= boundary_y_min and
            proposed_bbox[2] <= boundary_x_max and proposed_bbox[3] <= boundary_y_max and
            not check_overlap(proposed_bbox, placed_bboxes)):
            
            target_roi = target_img_float[place_y : place_y + transformed_height, place_x : place_x + transformed_width]
            if not (target_roi.shape[:2] == transformed_mask_low.shape[:2] == transformed_difference_float.shape[:2]):
                continue 

            # Use low threshold mask with distorted boundary for superimposing dirt effect
            binary_mask_float_0_1 = transformed_mask_low.astype(np.float32) / 255.0
            dirt_diff_roi_3channel = np.stack([transformed_difference_float] * 3, axis=-1)
            binary_mask_float_3channel = np.stack([binary_mask_float_0_1] * 3, axis=-1)

            result_roi = target_roi + (dirt_diff_roi_3channel * binary_mask_float_3channel)
            result_roi = np.clip(result_roi, 0, 255)
            target_img_float[place_y : place_y + transformed_height, place_x : place_x + transformed_width] = result_roi

            # Use high threshold mask with distorted boundary for annotations
            contours, _ = cv2.findContours(transformed_mask_high, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    # Filter out very small contours
                    if cv2.contourArea(contour) < 20:
                        continue
                        
                    offset_contour_np = contour + (place_x, place_y)
                    segmentation_points = [p for sublist in offset_contour_np.tolist() for p in sublist[0]]
                    annotations_list.append({
                        "bbox": [place_x, place_y, transformed_width, transformed_height],
                        "segmentation": [segmentation_points], "category_id": 1,
                        "area": cv2.countNonZero(transformed_mask_high)})
                    if mask_color_for_saving is not None:
                        cv2.drawContours(segmentation_mask_accumulator, [offset_contour_np], -1, mask_color_for_saving, cv2.FILLED)
            return target_img_float, segmentation_mask_accumulator, proposed_bbox
    return target_img_float, segmentation_mask_accumulator, None


def process_image_for_dirt_placement(base_image_color, base_image_name, dirt_source_image_paths,
                                    args, margins,
                                    visualize_first_mask_current_overall=False,
                                    use_distorted_boundary=True
):
    if base_image_color is None: return None, None, []
    print(f"\n--- Generating full synthetic image for: {base_image_name} ---")
    target_height, target_width, _ = base_image_color.shape
    base_image_float = base_image_color.copy().astype(np.float32)

    # boundary_x_min = int(target_width * (margins["left"] / 100.0))
    # boundary_y_min = int(target_height * (margins["top"] / 100.0))
    # boundary_x_max = int(target_width * (1.0 - margins["right"] / 100.0))
    # boundary_y_max = int(target_height * (1.0 - margins["bottom"] / 100.0))
    
    boundary_x_min, boundary_y_min, boundary_x_max, boundary_y_max = get_margin_bbox(margins, target_width, target_height)
    if boundary_x_min >= boundary_x_max: boundary_x_min, boundary_x_max = 0, target_width
    if boundary_y_min >= boundary_y_max: boundary_y_min, boundary_y_max = 0, target_height

    segmentation_mask_accumulator = np.zeros((target_height, target_width), dtype=np.uint8)
    placed_bboxes, annotations_for_this_image = [], []
    successfully_placed_count = 0

    for i in range(args.num_dirt_sample):
        print(f"  Attempting to place dirt sample {i+1}/{args.num_dirt_sample} for {base_image_name}.")
        random_dirty_image_path = random.choice(dirt_source_image_paths)
        print(f"    -> Using source: {os.path.basename(random_dirty_image_path)}")
        
        visualize_call = visualize_first_mask_current_overall and (i == 0)
        dirt_mask_low, dirt_mask_high, dirt_diff_float = create_dirt_mask_from_dirty_image(
            random_dirty_image_path, args.binary_threshold_value_low,
            args.binary_threshold_value_high, args.background_estimation_filter_size, visualize=visualize_call)

        if dirt_mask_low is None or dirt_mask_high is None or dirt_diff_float is None:
            print(f"    -> Failed to create masks/difference. Skipping sample.")
            continue

        # Generate random distortion parameters for each sample
        distortion_params = {
            'num_vertices_per_side': random.randint(2, 5),
            'max_distortion': random.uniform(0.05, 0.15)
        } if use_distorted_boundary else None

        # base_image_float, segmentation_mask_accumulator, placed_bbox = place_dirt_sample(
        #     base_image_float, args.binary_threshold_value_low, args.binary_threshold_value_high, dirt_mask_binary, dirt_diff_float, placed_bboxes, args.mask_color,
        #     args.scale_range, args.rotation_range, args.max_placement_attempts,
        #     segmentation_mask_accumulator, annotations_for_this_image,
        #     boundary_x_min, boundary_y_min, boundary_x_max, boundary_y_max)
        
        base_image_float, segmentation_mask_accumulator, placed_bbox = place_dirt_sample(
            base_image_float, dirt_mask_low, dirt_mask_high, dirt_diff_float, placed_bboxes, args.mask_color,
            args.scale_range, args.rotation_range, args.max_placement_attempts,
            segmentation_mask_accumulator, annotations_for_this_image,
            boundary_x_min, boundary_y_min, boundary_x_max, boundary_y_max,
            use_distorted_boundary=use_distorted_boundary,
            distortion_params=distortion_params)

        if placed_bbox:
            placed_bboxes.append(placed_bbox)
            successfully_placed_count += 1
        else:
            print(f"    -> Could not place sample {i+1}. Skipping.")
            
    print(f"  Successfully placed {successfully_placed_count}/{args.num_dirt_sample} samples for {base_image_name}.")
    return np.clip(base_image_float, 0, 255).astype(np.uint8), segmentation_mask_accumulator, annotations_for_this_image

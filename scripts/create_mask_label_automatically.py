import os
import re
import sys
import shutil
import logging
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.image_processor import ImageProcessor
from src.core.logger_config import setup_application_logger


def copy_rename_simple(source_path, dest_dir, new_name):
    """
    Copy image file and rename it using shutil
    
    Args:
        source_path: Path to source image
        dest_dir: Destination directory
        new_name: New filename (with extension)
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Create full destination path
    dest_path = os.path.join(dest_dir, new_name)
    
    # Copy and rename
    shutil.copy2(source_path, dest_path)
    app_logger.debug(f"\t✓ Copied: {source_path} → {dest_path}")
    
    return dest_path

# Your existing helper functions remain the same
def get_device_bbox(image, scale):
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    filtered = cv2.medianBlur(resized, 13)
    normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    max_val = cumulative_hist.max()
    cumulative_hist_normalized = cumulative_hist / max_val

    threshold = 64
    if cumulative_hist_normalized[threshold] > 0.62:
        while cumulative_hist_normalized[threshold] > 0.62:
            threshold -= 1
        if threshold > 40:
            threshold = max(40, threshold-4)

    _, binary_image = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        
    h, w = image.shape
    xmin = int(w * 0.12)
    ymin = int(h * 0.15)
    xmax = int(w * (1.0 - 0.12))
    ymax = int(h * (1.0 - 0.15))
    return xmin, ymin, xmax-xmin, ymax-ymin

def process_image(image,):
    x, y, w, h = get_device_bbox(image, 0.04)
    cropped_img = image[y:y+h, x:x+w]
    img_h, img_w = image.shape            
    rgb_image = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    result_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    result_mask[y:y+h, x:x+w] = create_mask(cropped_img, 301)
    return np.dstack((rgb_image, result_mask))


def create_mask(img, filter_size=51):
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dirt_eroded = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel2)
    clean_bg = cv2.medianBlur(dirt_eroded, filter_size)

    img_float = img.astype(np.float32)
    clean_bg_float = clean_bg.astype(np.float32)
    abs_diff = cv2.absdiff(img_float, clean_bg_float)
    norm_diff = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    filtered_diff = cv2.GaussianBlur(norm_diff, (5, 5), 0)
        
    threshold, _ = cv2.threshold(filtered_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, low_thresh_mask = cv2.threshold(filtered_diff, max(15, threshold-40), 255, cv2.THRESH_BINARY)
    return low_thresh_mask


def create_indexed_ground_truth(image, mask_image, labels):
    """
    Creates indexed ground truth with class mapping:
    0 - background
    1 - condensation
    2 - dirt (includes smudges, dirt, glue residue, fingerprint, unknown)
    3 - scratch
    """
    # Resize labels if needed
    if labels.shape[:2] != image.shape[:2]:
        (img_h, img_w) = image.shape[:2]
        labels = cv2.resize(labels, (img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT)
    
    # Original color labels indexes and their colors
    colors = [(255, 255, 255), (0, 255, 255), (66, 0, 96), (0, 173, 105), 
              (255, 91, 0), (255, 240, 0), (0, 0, 255), (255, 0, 0)]
    
    # Create individual masks for each defect type
    smudges_mask = cv2.inRange(labels, colors[1], colors[1])
    dirt_mask = cv2.inRange(labels, colors[2], colors[2])
    glue_residue_mask = cv2.inRange(labels, colors[3], colors[3])
    fingerprint_mask = cv2.inRange(labels, colors[4], colors[4])
    condensation_mask = cv2.inRange(labels, colors[5], colors[5])
    scratches_mask = cv2.inRange(labels, colors[6], colors[6])
    unknown_mask = cv2.inRange(labels, colors[7], colors[7])
    
    # Combine all dirt-related defects into single dirt mask
    combined_dirt_mask = np.maximum.reduce([
        smudges_mask, dirt_mask, glue_residue_mask, 
        fingerprint_mask, unknown_mask
    ])
    
    # Get pixel mask from alpha channel (device region)
    pixel_mask = mask_image[:, :, 3]
    
    # Intersect all defect masks with pixel mask (only within device region)
    condensation_mask = np.minimum(condensation_mask, pixel_mask)
    combined_dirt_mask = np.minimum(combined_dirt_mask, pixel_mask)
    scratches_mask = np.minimum(scratches_mask, pixel_mask)
    
    # Initialize ground truth with all zeros (background)
    ground_truth = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Assign class indices with priority (higher priority overwrites lower)
    # Priority order: scratch (3) > dirt (2) > condensation (1) > background (0)
    
    # Apply masks in priority order (lowest to highest priority)
    ground_truth[condensation_mask > 0] = 1  # condensation
    ground_truth[combined_dirt_mask > 0] = 2  # dirt
    ground_truth[scratches_mask > 0] = 3     # scratch
    
    return ground_truth


def overlay_mask_and_labels(image, mask_image, labels):
    if labels.shape[:2] != image.shape[:2]:
        (img_h, img_w) = image.shape[:2]
        labels = cv2.resize(labels, (img_w, img_h), interpolation=cv2.INTER_NEAREST_EXACT)
    
    colors = [(255, 255, 255), (0, 255, 255), (66, 0, 96), (0, 173, 105), 
              (255, 91, 0), (255, 240, 0), (0, 0, 255), (255, 0, 0)]
    
    smudges_mask = cv2.inRange(labels, colors[1], colors[1])
    dirt_mask = cv2.inRange(labels, colors[2], colors[2])
    glue_residue_mask = cv2.inRange(labels, colors[3], colors[3])
    fingerprint_mask = cv2.inRange(labels, colors[4], colors[4])
    unknown_mask = cv2.inRange(labels, colors[7], colors[7])
    condensation_mask = cv2.inRange(labels, colors[5], colors[5])
    scratches_mask = cv2.inRange(labels, colors[6], colors[6])
    
    dirt_mask_1 = np.maximum(smudges_mask, glue_residue_mask)
    dirt_mask_2 = np.maximum(dirt_mask, fingerprint_mask)
    dirt_mask_3 = np.maximum(dirt_mask_1, unknown_mask)
    dirt_mask = np.maximum(dirt_mask_2, dirt_mask_3)
    
    pixel_mask = mask_image[:, :, 3]
    dirt_mask = np.minimum(dirt_mask, pixel_mask)
    condensation_mask = np.minimum(condensation_mask, pixel_mask)
    scratches_mask = np.minimum(scratches_mask, pixel_mask)
    
    stacked_masks = np.dstack((condensation_mask, scratches_mask, dirt_mask))
    overlay_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.maximum(overlay_image, stacked_masks)



def process_batch_with_indexed_gt(base_path, label_base_path,  save_path:str):
    save_path = Path(str(save_path))
    

    list_folders = os.listdir(base_path)

    for idx, dirt_type_name in enumerate(list_folders):

        if ".DS_Store" in dirt_type_name:
            continue

        # path = os.path.join(base_path, dirt_type_name)
        path = Path(base_path) / dirt_type_name
        label_path = Path(label_base_path) / dirt_type_name

        app_logger.info(f"Processing {idx+1}/{len(list_folders)} folder: {str(path)}")

        all_subfolders = os.listdir(path)
        for sub_idx, device_id in enumerate (all_subfolders):
            if ".DS_Store" in dirt_type_name:
                continue

            folder_path = path / device_id
            label_folder_path = label_path / device_id
            app_logger.debug(f"  Processing subfolder: {str(folder_path)}")

            all_files = os.listdir(folder_path)
            #  Process both frontal and rear surfaces
            patterns = [
                (r".*_.*_0_0_.*", "frontal"),
                (r".*_.*_3_1_.*", "rear")
            ]
            for pattern_str, surface_type in patterns:
                pattern = re.compile(pattern_str)
                matching_files = [file for file in all_files if pattern.match(file)]
                
                if not matching_files:
                    app_logger.warning(f"    No {surface_type} surface files found")
                    continue

                image_name = matching_files[0]
                app_logger.debug(f"    Processing {surface_type}: {image_name}")

                image_basename = image_name.split('.')[0]
                write_path = save_path / dirt_type_name / image_basename 

                # Load image and labels
                check_img_path = folder_path / image_name
                if not os.path.exists(str(check_img_path)):
                    app_logger.error(f"    Skipping: Image file not found - {check_img_path}")
                    continue
                
                check_label_path = label_folder_path / f"{image_basename}_map.png"
                if not os.path.exists(str(check_label_path)):
                    app_logger.error(f"    Skipping: Labels file not found - {check_label_path}")
                    continue

                check_coco_path = label_folder_path / "coco.json"
                if not os.path.exists(str(check_coco_path)):
                    app_logger.error(f"    Skipping: coco json file not found - {check_coco_path}")
                    continue

                image = image_processor.load_image(folder_path / image_name, True)
                labels = image_processor.load_image(label_folder_path / f"{image_basename}_map.png", False)

                if image is not None and labels is not None:
                    # Convert BGR to RGB for labels
                    labels_rgb = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)

                    # Process image to get mask
                    mask = process_image(image)

                    # Create indexed ground truth
                    indexed_gt = create_indexed_ground_truth(image, mask, labels_rgb)

                    # Save indexed ground truth
                    
                    # indexed_gt_path = os.path.join(folder_path, f"{base_name}_indexed_gt.png")
                    # indexed_gt_path = os.path.join(folder_path, f"segmentation_mask_multiclass.png")
                    indexed_gt_path = write_path / "segmentation_mask_multiclass.png"
                    app_logger.debug(f"    saving indexed_gt_path = {str(indexed_gt_path)}")
                    cv2.imwrite(str(indexed_gt_path), indexed_gt)

                    # Optional: Create and save overlay for visualization
                    overlay_image = overlay_mask_and_labels(image, mask, labels_rgb)
                    overlay_image = cv2.resize(overlay_image, None, fx=0.2, fy=0.2)
                    overlay_path = write_path /  f"{str(image_basename)}_overlay.png"
                    app_logger.debug(f"    saving overlay_path = {str(overlay_path)}")
                    cv2.imwrite(str(overlay_path), overlay_image)

                    #  save original image 
                    app_logger.debug(f"    coping original from = {str(folder_path)}/{image_name} to = {str(write_path)}/original_image.bmp")
                    copy_rename_simple(folder_path / image_name, write_path, f"original_image.bmp")

                    #  save labels.json
                    app_logger.debug(f"    coping label from = {str(label_folder_path)}/coco.json to {str(write_path)}/labels_patch.json")
                    copy_rename_simple(label_folder_path / "coco.json", write_path, f"labels_patch.json")
                    
                else:
                    app_logger.error(f"    Failed to load image or labels for {image_name}")
                
    app_logger.info("Data is now ready for inference - process completed")



if __name__ == "__main__":

    log_level = logging.DEBUG if True else logging.INFO
    app_logger = setup_application_logger(
        app_name="mask_label",
        log_file_name="logs/create_mask_label_automatically.log"
    )


    image_processor = ImageProcessor()

    BASE_PATH = "tests/b3"
    BASE_LABEL_PATH = "tests/batch3/Batch3-labels"
    WRITE_PATH = "tests/b3o"

    process_batch_with_indexed_gt(BASE_PATH, BASE_LABEL_PATH, WRITE_PATH)

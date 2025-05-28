import os
import glob

import numpy as np
import cv2


def list_image_files_in_folder(folder_path):
    """Lists all common image files (jpg, png, bmp, jpeg) in a folder."""
    if not os.path.isdir(folder_path):
        return []
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        print("No image files found in the folder. Exiting.")
        exit()

    print(f"we have total {len(image_files)} images.")
    return image_files

def split_image_into_patches(image, mask, annotations, patch_size, 
                             full_img_boundary_x_min, full_img_boundary_y_min, 
                             full_img_boundary_x_max, full_img_boundary_y_max):
    patches_data = []
    
    # Pre-calculate masks for each annotation once
    precalculated_ann_masks = []
    for ann in annotations:
        seg_poly = ann['segmentation'][0] # Assuming single polygon segmentation
        points = np.array(seg_poly).reshape(-1, 2)
        
        # Create a minimal bounding box for the annotation to optimize mask creation size
        min_x = int(np.min(points[:, 0]))
        max_x = int(np.max(points[:, 0]))
        min_y = int(np.min(points[:, 1]))
        max_y = int(np.max(points[:, 1]))

        ann_w = max_x - min_x + 1
        ann_h = max_y - min_y + 1

        if ann_w <= 0 or ann_h <= 0:
            precalculated_ann_masks.append(None)
            continue

        local_mask = np.zeros((ann_h, ann_w), dtype=np.uint8)
        # Translate points to local coordinates for drawing
        translated_points = points - [min_x, min_y]
        cv2.drawContours(local_mask, [translated_points.reshape(-1,1,2).astype(np.int32)], -1, 255, cv2.FILLED)
        precalculated_ann_masks.append((local_mask, min_x, min_y, max_x, max_y))

    for y_start in range(full_img_boundary_y_min, full_img_boundary_y_max, patch_size):
        for x_start in range(full_img_boundary_x_min, full_img_boundary_x_max, patch_size):
            y_end = min(y_start + patch_size, full_img_boundary_y_max)
            x_end = min(x_start + patch_size, full_img_boundary_x_max)

            patch_img_roi = image[y_start:y_end, x_start:x_end]
            patch_mask_roi = mask[y_start:y_end, x_start:x_end]
            
            patch_img = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
            patch_img[0:patch_img_roi.shape[0], 0:patch_img_roi.shape[1]] = patch_img_roi
            
            patch_mask_padded = np.zeros((patch_size, patch_size), dtype=mask.dtype)
            patch_mask_padded[0:patch_mask_roi.shape[0], 0:patch_mask_roi.shape[1]] = patch_mask_roi

            adjusted_annotations = []
            for ann_idx, ann in enumerate(annotations):
                bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
                
                # Check for bounding box intersection
                intersect_x1 = max(bbox_x, x_start)
                intersect_y1 = max(bbox_y, y_start)
                intersect_x2 = min(bbox_x + bbox_w, x_start + patch_size)
                intersect_y2 = min(bbox_y + bbox_h, y_start + patch_size)

                if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
                    adj_bbox_x = intersect_x1 - x_start
                    adj_bbox_y = intersect_y1 - y_start
                    adj_bbox_w = intersect_x2 - intersect_x1
                    adj_bbox_h = intersect_y2 - intersect_y1

                    adj_seg_polys = []
                    for seg_poly in ann['segmentation']:
                        points = np.array(seg_poly).reshape(-1, 2)
                        translated_points = points - [x_start, y_start]
                        adj_seg_polys.append(translated_points.flatten().tolist())
                    
                    # Optimized Area Recalculation
                    precalc_data = precalculated_ann_masks[ann_idx]
                    if precalc_data is not None:
                        local_mask, ann_min_x, ann_min_y, ann_max_x, ann_max_y = precalc_data
                        
                        # Calculate the intersection coordinates in the local annotation mask's frame
                        intersect_local_x1 = max(0, x_start - ann_min_x)
                        intersect_local_y1 = max(0, y_start - ann_min_y)
                        intersect_local_x2 = min(local_mask.shape[1], x_end - ann_min_x)
                        intersect_local_y2 = min(local_mask.shape[0], y_end - ann_min_y)
                        
                        new_area = 0
                        if intersect_local_x2 > intersect_local_x1 and intersect_local_y2 > intersect_local_y1:
                            # Extract the intersecting part from the pre-calculated local mask
                            intersecting_mask_roi = local_mask[intersect_local_y1:intersect_local_y2, 
                                                               intersect_local_x1:intersect_local_x2]
                            new_area = cv2.countNonZero(intersecting_mask_roi)

                        if new_area > 0:
                            adjusted_annotations.append({
                                "bbox": [adj_bbox_x, adj_bbox_y, adj_bbox_w, adj_bbox_h],
                                "segmentation": adj_seg_polys, "category_id": ann['category_id'], "area": new_area})
            
            if adjusted_annotations: # Only add patch if it has annotations
                patches_data.append((patch_img, patch_mask_padded, adjusted_annotations, x_start, y_start))
    return patches_data
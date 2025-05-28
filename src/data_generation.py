import os
import random
import json
import cv2
import numpy as np
import argparse

from arguments import parse_arguments
from data_generation_config import DOWNSCALE_CLEAN_IMAGES_FACTOR
from image_operations import load_image_color, save_image, get_margin_bbox
from utils import list_image_files_in_folder, split_image_into_patches
from dirt_placement import process_image_for_dirt_placement



if __name__ == "__main__":

    # Read all default values or pass then using input arguments
    args = parse_arguments()
    print(args)

    # create margin variables
    sides = ['top', 'bottom', 'left', 'right']  
    margins = {k:v for k, v in zip(sides, args.boundary_margins)}

    # fetch dirt images path
    print("\n--- Listing available dirt source images ---")
    dirt_source_image_paths = list_image_files_in_folder(args.dirt_folder)
    
    # fetch clean images path
    print("\n--- Listing available clean images ---")
    clean_image_paths = list_image_files_in_folder(args.clean_folder)

    # for key, value in vars(args).items():
    #     print(f"{key}: {value}\n")

    first_mask_viz_done = False
    clean_img_idx = 0

    # loop over all the available clean paths
    for clean_path in clean_image_paths:
        clean_name = os.path.basename(clean_path)
        if "_0_0_" not in clean_name: continue

        clean_img_color = load_image_color(clean_path)
        if clean_img_color is None: print(f"Failed to load {clean_name}. Skipping."); continue

        # downscale the image if passed
        if args.downscale_clean_images_factor > 1:
            h, w = clean_img_color.shape[:2]
            clean_img_color = cv2.resize(
                clean_img_color, 
                (w // args.downscale_clean_images_factor, h // args.downscale_clean_images_factor), 
                interpolation=cv2.INTER_AREA
            )
            print(f"{clean_name} downscaled to {clean_img_color.shape[1]}x{clean_img_color.shape[0]} from {w}x{h}.")

        img_h, img_w, _ = clean_img_color.shape
        b_xmin, b_ymin, b_xmax, b_ymax = get_margin_bbox(margins, img_w, img_h)
        

        for ver_idx in range(args.num_versions):
            print(f"\n--- {clean_name} - Version {ver_idx+1} ---")
            viz_this_run = args.visualize_first_mask_creation and not first_mask_viz_done and clean_img_idx == 0 and ver_idx == 0
            
            dirty_full, mask_full, ann_full = process_image_for_dirt_placement(
                clean_img_color, clean_name, dirt_source_image_paths, args, margins,
                viz_this_run)
            
            if viz_this_run and dirty_full is not None: first_mask_viz_done = True
            if dirty_full is None: print(f"Skipping patches for {clean_name} ver {ver_idx+1}."); continue

            ver_out_dir = os.path.join(args.output_dir, f"{os.path.splitext(clean_name)[0]}_v{ver_idx:02d}")
            os.makedirs(ver_out_dir, exist_ok=True)
            save_image(dirty_full, os.path.join(ver_out_dir, 'synthetic_dirty_image_full.png'))
            save_image(mask_full, os.path.join(ver_out_dir, 'segmentation_mask_full.png'))
            with open(os.path.join(ver_out_dir, 'labels_full.json'), 'w') as f_json:
                json.dump({"annotations": ann_full, "image_size": [img_w, img_h], 
                           "boundary_pixels": [b_xmin, b_ymin, b_xmax, b_ymax]}, f_json, indent=4)
            print(f"  Saved full synthetic image, mask, labels for {clean_name} (Version {ver_idx+1}).")

            print(f"\n--- Splitting patches for {clean_name} (Version {ver_idx+1}) ---")
            patch_data_list = split_image_into_patches(dirty_full, mask_full, ann_full, args.patch_size, b_xmin, b_ymin, b_xmax, b_ymax)
            print(f"  Split into {len(patch_data_list)} patches with annotations.")
            
            patch_base_dir = os.path.join(ver_out_dir, 'patches')
            os.makedirs(patch_base_dir, exist_ok=True)
            for p_idx, (p_img, p_mask, p_ann, gx, gy) in enumerate(patch_data_list):
                p_name = f"patch_{p_idx:03d}_x{gx}_y{gy}"
                p_dir = os.path.join(patch_base_dir, p_name)
                os.makedirs(p_dir, exist_ok=True)
                save_image(p_img, os.path.join(p_dir, 'synthetic_dirty_patch.png'))
                save_image(p_mask, os.path.join(p_dir, 'segmentation_mask_patch.png'))
                with open(os.path.join(p_dir, 'labels_patch.json'), 'w') as f_json:
                    json.dump({"annotations": p_ann, "image_size": [args.patch_size, args.patch_size], 
                               "original_global_offset": [gx, gy]}, f_json, indent=4)
                print(f"    Saved patch {p_idx:03d} to {p_dir}")
        clean_img_idx += 1
    print("\n--- Dataset Generation Complete ---")






    
    
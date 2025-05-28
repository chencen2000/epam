import argparse
import ast

from data_generation_config import *


def parse_arguments():
    """
    Defines and parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description='DataGeneration Script')
    # --- Directory Paths ---
    parser.add_argument('--dirt_folder', help='Path to dirt image folder.', default=DIRT_IMAGE_DIR)
    parser.add_argument('--clean_folder', help='Path to clean image folder.', default=CLEAN_IMAGE_DIR)
    parser.add_argument('--output_dir', help='Path to output image(s) folder.', default=OUTPUT_IMAGE_DIR)
    
    # --- Numerical Parameters ---
    parser.add_argument(
        '--num_dirt_sample',
        type=validate_int_range(INT_MIN_RANGE, INT_MAX_RANGE),
        help='number of dirt samples to add to EACH clean image/patch (e.g., 5)', 
        default=NUM_DIRT_SAMPLE
    )
    parser.add_argument(
        '--num_versions', 
        type=validate_int_range(INT_MIN_RANGE, INT_MAX_RANGE),
        help='Num versions per clean image (e.g., 5)', 
        default=NUM_VERSIONS
    )
    parser.add_argument(
        '--background_estimation_filter_size',
        type=int,
        default=BACKGROUND_ESTIMATION_FILTER_SIZE,
        help="Size of the filter used for background estimation (e.g., Gaussian filter size)."
    )
    parser.add_argument(
        '--binary_threshold_value',
        type=int,
        default=BINARY_THRESHOLD_VALUE,
        help="Threshold value for binarizing images."
    )
    parser.add_argument(
        '--binary_threshold_value_low',
        type=int,
        default=BINARY_THRESHOLD_VALUE_LOW,
        help="Threshold value for binarizing images (Lower threshold for dirt superimposing)."
    )
    parser.add_argument(
        '--binary_threshold_value_high',
        type=int,
        default=BINARY_THRESHOLD_VALUE_HIGH,
        help="Threshold value for binarizing images (Higher threshold for annotations)."
    )
    parser.add_argument(
        '--max_placement_attempts',
        type=int,
        default=MAX_PLACEMENT_ATTEMPTS,
        help="Maximum attempts to place dirt patches without overlap."
    )
    parser.add_argument(
        '--downscale_clean_images_factor',
        type=int,
        default=DOWNSCALE_CLEAN_IMAGES_FACTOR,
        help="Factor by which to downscale clean images before processing."
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=PATCH_SIZE,
        help="Size of the image patches to process (e.g., 1024 for 1024x1024)."
    )
    
    # --- Range Parameters (Tuples/Lists) ---
    parser.add_argument(
        '--scale_range',
        type=lambda s: ast.literal_eval(s), # Allows passing tuples like "(0.5, 2)"
        default=SCALE_RANGE,
        help="A tuple (min, max) defining the scale range for dirt patches."
    )
    parser.add_argument(
        '--rotation_range',
        type=lambda s: ast.literal_eval(s), # Allows passing tuples like "(0, 360)"
        default=ROTATION_RANGE,
        help="A tuple (min, max) defining the rotation range in degrees for dirt patches."
    )
    parser.add_argument(
        '--mask_color',
        type=lambda s: ast.literal_eval(s), # Allows passing tuples like "(0, 360)"
        default=MASK_COLOR,
        help="A tuple (channel1, channel2, channel3) definingcolor for mask (for saving masks)."
    )

    # --- Boolean Flag ---
    parser.add_argument(
        '--visualize_first_mask_creation',
        default=VISUALIZE_FIRST_MASK_CREATION, # Default is False if not present
        help="Enable visualization during the first mask creation step."
    )

    # --- Special Boundary Margins ---
    parser.add_argument(
        '--boundary_margins',
        type=validate_boundaries_string,
        default=BOUNDARY_MARGINS, # Default string will be validated by custom type
        help="A comma-separated string of four integer boundary values (top,right,bottom,left). "
             "Each value must be between 0 and 100 inclusive."
    )

    return parser.parse_args()


def validate_int_range(min_val, max_val):
    """
    Returns a custom type validation function for argparse.
    This function checks if an integer is within the specified min and max range.
    """
    def int_range_checker(arg_value):
        try:
            value = int(arg_value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{arg_value}' is not a valid integer.")
        if not (min_val <= value <= max_val):
            raise argparse.ArgumentTypeError(f"Value must be between {min_val} and {max_val} (inclusive).")
        return value
    return int_range_checker

def validate_boundaries_string(arg_value):
    """
    Custom type validator for a comma-separated string of four integer boundaries.
    Each boundary must be between 0 and 100 (inclusive).
    """
    parts = arg_value.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Expected exactly 4 comma-separated values, but got {len(parts)}."
        )

    validated_boundaries = []
    for i, part in enumerate(parts):
        try:
            value = int(part.strip()) # .strip() to handle potential whitespace
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Boundary value '{part}' at position {i+1} is not a valid integer."
            )

        MIN_ALLOWED_VALUE = 0
        MAX_ALLOWED_VALUE = 100
        if not (MIN_ALLOWED_VALUE <= value <= MAX_ALLOWED_VALUE):
            raise argparse.ArgumentTypeError(
                f"Boundary value '{value}' at position {i+1} is out of range. "
                f"Must be between {MIN_ALLOWED_VALUE} and {MAX_ALLOWED_VALUE} (inclusive)."
            )
        validated_boundaries.append(value)

    return validated_boundaries
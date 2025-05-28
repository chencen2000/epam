
# ---------- Configuration --------------------
DIRT_IMAGE_DIR = "./data/selected/dirt/"
# CLEAN_IMAGE_DIR = "./data/new_data/Samsung-GalaxyS21-Violet/351852150247905_15/"
CLEAN_IMAGE_DIR = "./data/selected/clean_downloaded_new_images/front/"
OUTPUT_IMAGE_DIR = "./data/synthetic_data/"
NUM_DIRT_SAMPLE = 40
NUM_VERSIONS = 15
VISUALIZE_FIRST_MASK_CREATION = True    

BACKGROUND_ESTIMATION_FILTER_SIZE = 51
BINARY_THRESHOLD_VALUE = 30
MASK_COLOR = (255, 255, 255) 
SCALE_RANGE = (0.5, 2)
ROTATION_RANGE = (0, 360)
MAX_PLACEMENT_ATTEMPTS = 10
DOWNSCALE_CLEAN_IMAGES_FACTOR = 2
PATCH_SIZE = 1024
BOUNDARY_MARGINS = "15,15,12,12"

INT_MIN_RANGE = 0
INT_MAX_RANGE = 100

BINARY_THRESHOLD_VALUE_LOW = 20   # Lower threshold for dirt superimposing
BINARY_THRESHOLD_VALUE_HIGH = 40  # Higher threshold for annotations

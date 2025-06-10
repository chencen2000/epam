import os
import sys
import random
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger_config import setup_application_logger
from src.synthesis.dirt_extractor import DirtExtractor
from src.synthesis.patch_generator import PatchGenerator
from src.synthesis.image_operations import ImageOperations
from src.synthesis.boundary_detector import BoundaryDetector
from src.synthesis.synthetic_generator import SyntheticDataGenerator



def main():
    
    app_logger = setup_application_logger(
        app_name="data_generator", 
        log_file_name="logs/data_generation.log"
    )
    logger = app_logger.getChild('generate_data')
    logger.info("Data generation started")

    BASE_PATH = '../../data/training-dataset/Dirt'
    DIRT_DIR = '../../data/selected/new_dirt'
    CLEAN_IMAGE_DIR = "../../data/selected/clean_downloaded_new_images/front"
    OUTPUT_BASE_PATH = "../../data/del_test/"

    # Initialize detector
    image_operations = ImageOperations(logger)
    detector = DirtExtractor(image_operations, logger)
    patch_generator = PatchGenerator(logger)
    boundary_detector = BoundaryDetector(image_operations, logger)
    data_generator = SyntheticDataGenerator(
        detector, image_operations,
        boundary_detector, patch_generator,
        CLEAN_IMAGE_DIR, DIRT_DIR,
        num_version_per_image=4,
        num_dirt_super_impose=100,
        clean_image_downscale_factor=2,
        scale_dirt=(0.75, 1.5),
        rotation_range=(0, 360),
        app_logger=logger
    )
    data_generator.generate_synthetic_dataset(
        bg_estimation_filter_size=51,
        output_dir=OUTPUT_BASE_PATH
    )

    logger.info("complete!")



if __name__ == "__main__":
    os.environ["DEBUG"] = "false"
    random.seed(42)

    main()

    

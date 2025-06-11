import os
import sys
import random
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logger_config import setup_application_logger
from src.core.config_parser import ConfigParser
from src.synthesis.dirt_extractor import DirtExtractor
from src.synthesis.patch_generator import PatchGenerator
from src.synthesis.image_operations import ImageOperations
from src.synthesis.boundary_detector import BoundaryDetector
from src.synthesis.synthetic_generator import SyntheticDataGenerator


def main():

    """Enhanced main function with config-based inference"""
    parser = argparse.ArgumentParser(description='Enhanced Dirt Detection Model Inference with Config Support')
    parser.add_argument('--config', type=str, 
                       default='config/data_generation/synthetic_data.yaml',
                       help='Path to the configuration YAML file')
    
    args = parser.parse_args()
    
    # Load configuration
    config_parser = ConfigParser()

    try:
        config = config_parser.load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        config_parser.print_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Set environment variables from config
    os.environ["DEBUG"] = str(config['app']['debug']).lower()
    random.seed(config['app']['random_seed'])
    
    # Setup logger
    app_logger = setup_application_logger(
        app_name=config['app']['name'], 
        log_file_name=config['logging']['log_file']
    )
    logger = app_logger.getChild('generate_data')
    logger.info("Data generation started")
    
    # Extract paths from config
    paths = config['paths']
    BASE_PATH = paths['base_path']
    DIRT_DIR = paths['dirt_dir']
    CLEAN_IMAGE_DIR = paths['clean_image_dir']
    OUTPUT_BASE_PATH = paths['output_base_path']
    
    # Extract generation parameters
    gen_params = config['generation']
    dirt_cat = config["dirt_categories"]
    
    # Initialize components
    image_operations = ImageOperations(logger)
    detector = DirtExtractor(image_operations, logger)
    patch_generator = PatchGenerator(logger)
    boundary_detector = BoundaryDetector(image_operations, logger)
    
    # Create data generator with config parameters
    data_generator = SyntheticDataGenerator(
        detector, image_operations,
        boundary_detector, patch_generator,
        CLEAN_IMAGE_DIR, 
        DIRT_DIR,
        dirt_cat,
        num_version_per_image=gen_params['num_version_per_image'],
        num_dirt_super_impose=gen_params['num_dirt_super_impose'],
        clean_image_downscale_factor=gen_params['clean_image_downscale_factor'],
        scale_dirt=(gen_params['scale_dirt']['min'], gen_params['scale_dirt']['max']),
        rotation_range=(gen_params['rotation_range']['min'], gen_params['rotation_range']['max']),
        app_logger=logger
    )
    
    # Generate synthetic dataset
    data_generator.generate_synthetic_dataset(
        bg_estimation_filter_size=gen_params['bg_estimation_filter_size'],
        output_dir=OUTPUT_BASE_PATH
    )
    
    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()
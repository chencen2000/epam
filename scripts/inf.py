import os
import sys
import logging
from pathlib import Path
import argparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.core.config_parser import ConfigParser
from src.inference.predictor import ModelInference
from src.core.logger_config import setup_application_logger



import os
import sys
import logging
from pathlib import Path
import argparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.core.config_parser import ConfigParser
from src.inference.predictor import ModelInference
from src.core.region_analyzer import RegionAnalyzer
from src.synthesis.patch_generator import PatchGenerator
from src.synthesis.image_operations import ImageOperations
from src.synthesis.boundary_detector import BoundaryDetector
from src.core.logger_config import setup_application_logger
from src.inference.batch_predictor import batch_prediction
from src.inference.single_predictor import SingleImagePredictor
from src.inference.full_phone_prediction import FullScreenPredictor


def main():
    """Enhanced main function with config-based inference"""
    parser = argparse.ArgumentParser(description='Enhanced Dirt Detection Model Inference with Config Support')
    parser.add_argument('--config', type=str, 
                       default='config/default_predictor.yaml',
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
    
    # Setup logger
    log_file = config.get('logging', {}).get('file', 'logs/inference.log')
    app_logger = setup_application_logger(
        app_name="predictor", 
        log_file_name=log_file
    )
    
    app_logger.info(f"Starting inference with config: {args.config}")
    app_logger.info(f"Configuration: {config}")

    # Extract configuration values
    model_config = config.get('model', {})
    input_config = config.get('input', {})
    output_config = config.get('output', {})
    processing_config = config.get('processing', {})
    
    # Model parameters
    model_path = model_config.get('path')
    device = model_config.get('device', 'auto')
    threshold = model_config.get('threshold', 0.5)
    
    # Input parameters
    input_path = input_config.get('path')
    image_extensions = input_config.get('image_extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'])
    max_samples = input_config.get('max_samples')
    
    # Output parameters
    output_dir = output_config.get('dir', './output')
    
    # Processing parameters
    batch_mode = processing_config.get('batch_mode', False)
    new_images = processing_config.get('new_images', False)
    
    # Full phone processing parameters (with defaults)
    full_phone_config = processing_config.get('full_phone', {})
    patch_size = full_phone_config.get('patch_size', 1024)
    overlap = full_phone_config.get('overlap', 0.2)
    min_dirt_threshold = full_phone_config.get('min_dirt_threshold', 0.01)
    
    # Validate required parameters
    if not model_path:
        app_logger.error("Error: Model path not specified in configuration")
        sys.exit(1)
    
    if not input_path:
        app_logger.error("Error: Input path not specified in configuration")
        sys.exit(1)
    
    # Initialize inference
    try:
        image_operations = ImageOperations(app_logger)
        patch_generator = PatchGenerator(app_logger)
        boundary_detector = BoundaryDetector(image_operations, app_logger)
        region_analyzer = RegionAnalyzer(app_logger)

        single_predictor = SingleImagePredictor(
            region_analyzer=region_analyzer, 
            image_operations=image_operations, 
            model_path=model_path, 
            app_logger=app_logger,
            confidence_threshold=threshold,
        )

        full_ph_predictor = FullScreenPredictor(
            patch_generator=patch_generator, 
            boundary_detector=boundary_detector,
            region_analyzer=region_analyzer, 
            image_operations=image_operations,
            single_image_predictor=single_predictor,
            model_path=model_path, 
            app_logger=app_logger,
        )

        app_logger.info(f"Model loaded successfully from: {model_path}")
    except Exception as e:
        app_logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    app_logger.info(f"Output directory: {output_path}")
    
    # Determine processing mode based on configuration
    full_phone_enabled = full_phone_config.get('enabled', False)
    app_logger.info(full_phone_enabled)

    input_path = Path(input_path)


    if full_phone_enabled and batch_mode:

        # Batch processing of full phone images
        app_logger.info("Running batch inference on full phone images...")
        try:
            results = batch_prediction(
                prediction_service = full_ph_predictor,
                image_dir=str(input_path),
                patch_size=patch_size,
                overlap=overlap,
                output_dir=str(output_path),
                image_extensions=image_extensions,
                max_images=max_samples,
                threshold = threshold,
                min_dirt_threshold = min_dirt_threshold,
            )
            app_logger.info(f"Batch full phone inference completed. Results saved to: {output_path}")
        except Exception as e:
            app_logger.error(f"Error during batch full phone inference: {e}")
            sys.exit(1)

    elif full_phone_enabled and not batch_mode:

        # Single full phone image
        app_logger.info(f"Running full phone inference on single image on: {input_path}")
        try:
            result = full_ph_predictor.single_prediction_pipeline(
                image_path = str(input_path),
                patch_size = patch_size,
                overlap = overlap,
                save_results = True,
                output_dir = str(output_path),
                show_plot = True,
                min_dirt_threshold = min_dirt_threshold,
                threshold = threshold,
            )
            app_logger.info("Full phone inference completed!")
        except Exception as e:
            app_logger.error(f"Error during full phone inference: {e}")
            sys.exit(1)
    
    elif new_images and batch_mode:
        # Batch processing of full phone images
        app_logger.info("Running batch inference on new images (patches)...")
        try:
            results = batch_prediction(
                prediction_service = single_predictor,
                image_dir=str(input_path),
                patch_size=patch_size,
                overlap=overlap,
                output_dir=str(output_path),
                image_extensions=image_extensions,
                max_images=max_samples,
                app_logger=app_logger
            )
            app_logger.info(f"Batch inference completed. Results saved to: {output_path}")
        except Exception as e:
            app_logger.error(f"Error during batch inference on new images: {e}")
            sys.exit(1)

    elif new_images and not batch_mode:
        # Single full phone image
        app_logger.info(f"Running full phone inference on single image on: {input_path}")
        try:
            result = single_predictor.single_prediction_pipeline(
                image_path = str(input_path),
                patch_size = patch_size,
                overlap = overlap,
                save_results = True,
                output_dir = str(output_path),
                show_plot = True,
                min_dirt_threshold = min_dirt_threshold,
            )
            app_logger.info("Full phone inference completed!")
        except Exception as e:
            app_logger.error(f"Error during full phone inference: {e}")
            sys.exit(1)

    elif not new_images and not batch_mode:
        # run with mask
        single_predictor.predict_and_compare_mask(input_path, output_path)

    elif not new_images and batch_mode:
        # run with mask in batch mode
        pass 


if __name__ == "__main__":
    os.environ["DEBUG"] = "true"
    main()








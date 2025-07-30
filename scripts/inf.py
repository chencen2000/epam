import os
import sys
import logging
from pathlib import Path
import argparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.core.config_parser import ConfigParser
# from src.inference.predictor import ModelInference
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
# from src.inference.predictor import ModelInference
from src.core.region_analyzer import RegionAnalyzer
from src.synthesis.patch_generator import PatchGenerator
from src.synthesis.image_operations import ImageOperations
from src.synthesis.boundary_detector import BoundaryDetector
from src.core.logger_config import setup_application_logger
from src.inference.batch_predictor import batch_prediction
from src.inference.single_predictor import SingleImagePredictor
from src.inference.full_phone_prediction import FullScreenPredictor


def validate_multiclass_setup(single_predictor, app_logger):
    """Validate that the model is properly configured for multi-class"""
    try:
        # Check model configuration
        num_classes = single_predictor.num_classes
        class_names = single_predictor.class_names
        
        if num_classes != len(class_names):
            app_logger.warning(f"Mismatch between num_classes ({num_classes}) and class_names ({len(class_names)})")
        
        if num_classes < 3:
            app_logger.warning(f"Model appears to be binary (num_classes={num_classes}), but multi-class inference is expected")
        
        app_logger.info(f"Multi-class validation passed: {num_classes} classes - {class_names}")
        return True
        
    except Exception as e:
        app_logger.error(f"Multi-class validation failed: {e}")
        return False
    


def main():
    """Enhanced main function with config-based inference"""
    parser = argparse.ArgumentParser(description='Enhanced Dirt Detection Model Inference with Config Support')
    parser.add_argument('--config', type=str, 
                       default='config/inference/mask_predictor.yaml',
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
    log_file = config.get('logging', {}).get('file', 'logs/inference1.log')
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
    
    # Initialize inference components
    try:
        image_operations = ImageOperations(app_logger)
        patch_generator = PatchGenerator(app_logger)
        boundary_detector = BoundaryDetector(image_operations, app_logger)
        region_analyzer = RegionAnalyzer(app_logger)

        # UPDATED: Initialize single predictor with explicit multi-class support
        single_predictor = SingleImagePredictor(
            region_analyzer=region_analyzer, 
            image_operations=image_operations, 
            model_path=model_path, 
            app_logger=app_logger,
            confidence_threshold=threshold,
        )
        
        # Log model information for verification
        app_logger.info(f"Loaded multi-class model with {single_predictor.num_classes} classes: {single_predictor.class_names}")

        # UPDATED: Initialize full screen predictor 
        full_ph_predictor = FullScreenPredictor(
            patch_generator=patch_generator, 
            boundary_detector=boundary_detector,
            single_image_predictor=single_predictor,  # Pass the single predictor
            app_logger=app_logger,
        )

        app_logger.info(f"Multi-class model loaded successfully from: {model_path}")
        app_logger.info(f"Model architecture: {single_predictor.config.get('model_architecture', 'standard')}")
        app_logger.info(f"Device: {single_predictor.device}")
        
    except Exception as e:
        app_logger.error(f"Error loading multi-class model: {e}")
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
        app_logger.info("Running batch multi-class inference on full phone images...")
        try:
            # Validate setup
            if not validate_multiclass_setup(single_predictor, app_logger):
                app_logger.warning("Multi-class setup validation failed, but continuing...")
            
            results = batch_prediction(
                prediction_service=full_ph_predictor,
                image_dir=str(input_path),
                output_dir=str(output_path),
                image_extensions=image_extensions,
                max_images=max_samples,
                logger=app_logger,
                # Pass multi-class specific parameters
                patch_size=patch_size,
                overlap=overlap,
                min_dirt_threshold=min_dirt_threshold,
            )
            
            # Log multi-class batch results summary
            if results:
                total_images = len(results)
                avg_dirt = np.mean([r['overall_statistics']['screen_analysis']['class_statistics'].get('dirt_percentage', 0) for r in results])
                avg_scratches = np.mean([r['overall_statistics']['screen_analysis']['class_statistics'].get('scratches_percentage', 0) for r in results])
                
                app_logger.info(f"Multi-class batch inference completed!")
                app_logger.info(f"Processed {total_images} images")
                app_logger.info(f"Average dirt coverage: {avg_dirt:.2f}%")
                app_logger.info(f"Average scratch coverage: {avg_scratches:.2f}%")
                app_logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            app_logger.error(f"Error during multi-class batch full phone inference: {e}")
            sys.exit(1)

    elif full_phone_enabled and not batch_mode:

         # Single full phone image
        app_logger.info(f"Running multi-class full phone inference on: {input_path}")
        try:
            # Validate setup
            if not validate_multiclass_setup(single_predictor, app_logger):
                app_logger.warning("Multi-class setup validation failed, but continuing...")
            
            result = full_ph_predictor.single_prediction_pipeline(
                image_path=str(input_path),
                patch_size=patch_size,
                overlap=overlap,
                save_results=True,
                output_dir=str(output_path),
                show_plot=True,
                min_dirt_threshold=min_dirt_threshold,
            )
            
            # Log multi-class results
            if result:
                class_stats = result['overall_statistics']['screen_analysis']['class_statistics']
                app_logger.info("Multi-class full phone inference completed!")
                
                for cls in range(single_predictor.num_classes):
                    class_name = single_predictor.class_names[cls]
                    percentage_key = f'{class_name}_percentage'
                    if percentage_key in class_stats:
                        app_logger.info(f"{class_name.title()} coverage: {class_stats[percentage_key]:.2f}%")
            
        except Exception as e:
            app_logger.error(f"Error during multi-class full phone inference: {e}")
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
        # Single new image inference
        app_logger.info(f"Running multi-class inference on single image: {input_path}")
        try:
            # Validate setup
            if not validate_multiclass_setup(single_predictor, app_logger):
                app_logger.warning("Multi-class setup validation failed, but continuing...")
            
            result = single_predictor.single_prediction_pipeline(
                image_path=str(input_path),
                save_results=True,
                output_dir=str(output_path),
                show_plot=True,
            )
            
            # Log multi-class results
            if result and 'class_statistics' in result['prediction']:
                class_stats = result['prediction']['class_statistics']
                app_logger.info("Multi-class inference completed!")
                
                for cls in range(single_predictor.num_classes):
                    class_name = single_predictor.class_names[cls]
                    percentage_key = f'{class_name}_percentage'
                    if percentage_key in class_stats:
                        app_logger.info(f"{class_name.title()} coverage: {class_stats[percentage_key]:.2f}%")
            
        except Exception as e:
            app_logger.error(f"Error during multi-class inference: {e}")
            sys.exit(1)

    elif not new_images and not batch_mode:
        # run with mask
        # Run with ground truth comparison
        app_logger.info("Running multi-class inference with ground truth comparison...")
        try:
            # Validate setup
            if not validate_multiclass_setup(single_predictor, app_logger):
                app_logger.warning("Multi-class setup validation failed, but continuing...")
            
            single_predictor.predict_and_compare_mask(input_path, output_path)
            
        except Exception as e:
            app_logger.error(f"Error during multi-class ground truth comparison: {e}")
            sys.exit(1)

    elif not new_images and batch_mode:
        # run with mask in batch mode
        # Run batch ground truth comparison on dataset samples
        app_logger.info("Running batch multi-class inference with ground truth comparison...")
        try:
            # Validate setup
            if not validate_multiclass_setup(single_predictor, app_logger):
                app_logger.warning("Multi-class setup validation failed, but continuing...")
            
            # Use existing functions through batch wrapper
            results = single_predictor.batch_predict_and_compare_mask(
                
                dataset_dir=str(input_path),
                output_dir=str(output_path),
                max_samples=max_samples,
                # logger=app_logger
            )
            
            # Log batch comparison results summary
            if results:
                total_samples = len(results)
                successful_samples = len([r for r in results if r.get('status') == 'completed'])
                multiclass_samples = len([r for r in results if r.get('sample_type') == 'multiclass'])
                binary_samples = len([r for r in results if r.get('sample_type') == 'binary'])
                
                app_logger.info(f"Batch ground truth comparison completed!")
                app_logger.info(f"Processed {total_samples} samples total")
                app_logger.info(f"  Successful: {successful_samples}")
                app_logger.info(f"  Multi-class samples: {multiclass_samples}")
                app_logger.info(f"  Binary samples: {binary_samples}")
                app_logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            app_logger.error(f"Error during batch ground truth comparison: {e}")
            sys.exit(1) 


if __name__ == "__main__":
    os.environ["DEBUG"] = "false"
    main()








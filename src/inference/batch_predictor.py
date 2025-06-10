import json
from pathlib import Path
from typing import Optional, Dict, List, Protocol
from logging import Logger

from tqdm import tqdm
import numpy as np

from src.core.logger_config import setup_application_logger


class PredictionService(Protocol):
    """Protocol defining the interface for prediction services"""
    
    def single_prediction_pipeline(self, image_path: str, save_results: bool = True, 
                                 output_dir: str = None, show_plot: bool = False, **kwargs) -> Dict:
        """Single prediction pipeline method"""
        ...
    
    def save_batch_summary(self, results: List[Dict], save_path: str) -> None:
        """Save batch summary method"""
        ...


def batch_prediction(prediction_service: PredictionService, image_dir: str, output_dir: str = None,
                    image_extensions: List[str] = None, 
                    max_images: int = None, logger: Optional[Logger] = None, **kwargs) -> List[Dict]:
    """
    Run inference on multiple new images in a directory
    
    Args:
        prediction_service: Service that implements PredictionService protocol
        image_dir: Directory containing images
        output_dir: Directory to save results
        image_extensions: List of image extensions to process
        max_images: Maximum number of images to process
        logger: Logger instance
        
    Returns:
        List of prediction results
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if logger is None:
        
        logger = setup_application_logger().getChild('BatchPredictor')
    
    image_path = Path(image_dir)
    if not image_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        # image_files.extend(image_path.glob(f"*{ext.upper()}"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"Found {len(image_files)} images for processing")
    
    if output_dir is None:
        output_dir = image_path / 'batch_inference_results'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            logger.debug(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
            
            # Run prediction
            result = prediction_service.single_prediction_pipeline(
                str(image_file),
                save_results=True,
                output_dir=str(output_dir / image_file.stem),
                show_plot=False,
                **kwargs
            )
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {image_file.name}: {e}")
            continue
    
    # Save batch summary
    if results:
        save_path = str(output_dir / "batch_summary.json")
        prediction_service.save_batch_summary(results, save_path)
        logger.info(f"Batch summary saved to: {save_path}")
        logger.info(f"\nBatch processing completed!")
        logger.info(f"Processed {len(results)} images successfully")
        logger.info(f"Results saved to: {output_dir}")
    
    return results
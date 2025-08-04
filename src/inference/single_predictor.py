import time
import json
from pathlib import Path
from typing import Union, Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import no_grad, softmax
import matplotlib.patches as mpatches

from src.target_labels import TargetLabels
from src.core.region_analyzer import RegionAnalyzer
from src.inference.base_predictor import BasePredictor
from src.synthesis.image_operations import ImageOperations
from src.training.metrics import calculate_multiclass_iou, calculate_multiclass_dice, calculate_metrics, _find_objects
from src.core.logger_config import setup_application_logger
from src.utils.file_utils import load_ground_truth_data


class SingleImagePredictor(BasePredictor):

    def __init__(self, region_analyzer:RegionAnalyzer, image_operations:ImageOperations, app_logger=None, *args, **kwargs):
        super().__init__(app_logger=app_logger, *args, **kwargs)
        self.region_analyzer = region_analyzer
        self.image_operations = image_operations
        self.num_classes = self.config.get('num_classes', 3)
        self.multiclass_gt_metrics = {
            "iou_metrics" : [],
            "dice_metrics": []
        }

        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('SingleImagePredictor')


    def predict(self, image: np.ndarray, target_size=(1024, 1024), return_raw: bool = False) -> Dict:
        """Predict on a single image - FIXED FOR GRAYSCALE MODEL."""
        original_size = image.shape[:2]
        
        # QUICK FIX: Convert to grayscale before preprocessing
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.logger.debug(f"Converted RGB patch {image.shape} to grayscale {gray_image.shape}")
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Already single channel, squeeze
            gray_image = image.squeeze(-1)
        elif len(image.shape) == 2:
            # Already grayscale
            gray_image = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Preprocess the grayscale image
        image_tensor = self.image_operations.preprocess_prediction_image(gray_image, self.device, target_size, self.config)
        
        with no_grad():
            start_time = time.time()
            logits = self.model(image_tensor)
            inference_time = time.time() - start_time

            # self.logger.info(f"Model inference time = {inference_time}")
            
            # FIXED: Use softmax for multi-class instead of sigmoid
            probabilities = softmax(logits, dim=1).cpu().numpy()[0]  # Shape: (num_classes, H, W)
            
            # Get class predictions (argmax across class dimension)
            class_prediction = np.argmax(probabilities, axis=0).astype(np.uint8)  # Shape: (H, W)
            
            # Resize back to original size
            class_prediction = cv2.resize(class_prediction, (original_size[1], original_size[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            
            # Resize probability maps for each class
            class_probabilities = {}
            for cls in range(self.num_classes):
                prob_map = cv2.resize(probabilities[cls], (original_size[1], original_size[0]), 
                                    interpolation=cv2.INTER_CUBIC)
                class_probabilities[f'class_{cls}'] = prob_map
                class_probabilities[self.class_names[cls]] = prob_map
            
            # Create binary masks for each class
            binary_masks = {}
            for cls in range(self.num_classes):
                binary_masks[f'class_{cls}'] = (class_prediction == cls).astype(np.uint8)
                binary_masks[self.class_names[cls]] = binary_masks[f'class_{cls}']
            
            # Calculate per-class statistics
            class_stats = {}
            total_pixels = class_prediction.size
            
            for cls in range(self.num_classes):
                class_pixels = np.sum(class_prediction == cls)
                class_percentage = (class_pixels / total_pixels) * 100
                class_stats[f'class_{cls}_pixels'] = int(class_pixels)
                class_stats[f'class_{cls}_percentage'] = float(class_percentage)
                class_stats[f'{self.class_names[cls]}_pixels'] = int(class_pixels)
                class_stats[f'{self.class_names[cls]}_percentage'] = float(class_percentage)
        
        result = {
            'class_prediction': class_prediction,  # Multi-class prediction map
            'class_probabilities': class_probabilities,  # Probability maps for each class
            'binary_masks': binary_masks,  # Binary mask for each class
            'class_statistics': class_stats,  # Per-class pixel counts and percentages
            'confidence_threshold': self.confidence_threshold,
            'inference_time': inference_time,
            'original_size': original_size,
            'num_classes': self.num_classes,
            
            # Legacy compatibility - dirt as primary defect class
            'binary_prediction': binary_masks['dirt'],  # For backward compatibility
            'probabilities': class_probabilities['dirt'],  # For backward compatibility  
            'dirt_pixels': class_stats['dirt_pixels'],
            'dirt_percentage': class_stats['dirt_percentage']
        }
        
        if return_raw:
            result['raw_logits'] = logits.cpu().numpy()[0]
            
        return result

    def single_prediction_pipeline(self,
                                    image_path: str, save_results: bool = True, 
                                    output_dir: str = None, show_plot: bool = False, **kwargs) -> Dict:
        """
        Complete pipeline for predicting on a new image without ground truth
        UPDATED FOR MULTI-CLASS
        """
        self.logger.debug(f"Running multi-class single prediction pipeline ....")
        image_path = Path(image_path)

        image = self.image_operations.load_image_color(image_path)

        prediction_result = self.predict(image, return_raw=True)

        # Analyze regions for each defect class (excluding background)
        region_analysis = {}
        for cls in range(1, self.num_classes):  # Skip background (class 0)
            class_name = self.class_names[cls]
            binary_mask = prediction_result['binary_masks'][class_name]
            region_analysis[class_name] = self.region_analyzer.analyze_dirt_regions(binary_mask)

        # Overall defect analysis (combining all defect classes)
        combined_defects = np.logical_or(
            prediction_result['binary_masks']['dirt'],
            prediction_result['binary_masks']['scratches']
        ).astype(np.uint8)
        region_analysis['combined_defects'] = self.region_analyzer.analyze_dirt_regions(combined_defects)

        # Prepare output directory
        if output_dir is None:
            output_dir = image_path.parent / 'inference_results'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive results
        results = {
            'image_info': {
                'filename': image_path.name,
                'original_size': prediction_result['original_size'],
                'total_pixels': prediction_result['original_size'][0] * prediction_result['original_size'][1]
            },
            'prediction': prediction_result,
            'region_analysis': region_analysis,
            'model_info': {
                'architecture': self.config.get('model_architecture', 'standard'),
                'confidence_threshold': self.confidence_threshold,
                'device': str(self.device),
                'num_classes': self.num_classes,
                'class_names': self.class_names
            }
        }

        # Save results
        if save_results:
            try:
                # Save visualization
                vis_path = output_dir / f"{image_path.stem}_multiclass_detection.png"
                self.visualize_multiclass_prediction(
                    image, prediction_result, region_analysis,
                    image_name=image_path.name,
                    save_path=str(vis_path), 
                    show_plot=show_plot
                )
                
                # Save detailed results as JSON
                results_path = output_dir / f"{image_path.stem}_results.json"
                self.save_prediction_results(results, str(results_path))
                
                # Save class prediction map
                class_pred_path = output_dir / f"{image_path.stem}_class_prediction.png"
                cv2.imwrite(str(class_pred_path), prediction_result['class_prediction'])
                
                # Save individual class masks
                for cls in range(self.num_classes):
                    class_name = self.class_names[cls]
                    mask_path = output_dir / f"{image_path.stem}_{class_name}_mask.png"
                    cv2.imwrite(str(mask_path), prediction_result['binary_masks'][class_name] * 255)
                
                # Save probability maps for each class
                for cls in range(self.num_classes):
                    class_name = self.class_names[cls]
                    prob_path = output_dir / f"{image_path.stem}_{class_name}_probability.png"
                    prob_vis = (prediction_result['class_probabilities'][class_name] * 255).astype(np.uint8)
                    cv2.imwrite(str(prob_path), prob_vis)
                
                self.logger.debug(f"\nResults saved to: {output_dir}")
                self.logger.debug(f"- Visualization: {vis_path.name}")
                self.logger.debug(f"- Results JSON: {results_path.name}")
                self.logger.debug(f"- Class prediction: {class_pred_path.name}")
                for cls in range(self.num_classes):
                    class_name = self.class_names[cls]
                    self.logger.debug(f"- {class_name} mask and probability maps saved")

                suction_cup_path = output_dir / f"{image_path.stem}_suction_cup.png"
                self.predict_and_save_suction_cup(suction_cup_path, image)

                if "_0_0_" in image_path.stem:
                    self.check_power_on_device_algo(
                        output_dir, image, image_path.stem
                    )
                    
            except Exception as e:
                self.logger.error(f"Failed to save results: {e}")
        
        # Print summary
        self.print_multiclass_summary(results)

        self.logger.debug(f"Multi-class single_prediction_pipeline complete! ")
        
        return results
    
    def print_multiclass_summary(self, results: Dict) -> None:
        """Print a summary of multi-class prediction results"""
        pred = results['prediction']
        region = results['region_analysis']
        image_info = results['image_info']
        
        self.logger.debug(f"\n" + "="*70)
        self.logger.debug(f"MULTI-CLASS DIRT DETECTION SUMMARY")
        self.logger.debug(f"="*70)
        self.logger.debug(f"Image: {image_info['filename']}")
        self.logger.debug(f"Size: {image_info['original_size'][1]} × {image_info['original_size'][0]} pixels")
        self.logger.debug(f"Model Classes: {results['model_info']['num_classes']} ({', '.join(self.class_names)})")
        self.logger.debug(f"")
        
        self.logger.debug(f"CLASS-WISE DETECTION RESULTS:")
        for cls in range(self.num_classes):
            class_name = self.class_names[cls]
            pixels = pred['class_statistics'][f'{class_name}_pixels']
            percentage = pred['class_statistics'][f'{class_name}_percentage']
            self.logger.debug(f"  {class_name.title()}: {percentage:.2f}% ({pixels:,} pixels)")
        
        self.logger.debug(f"")
        self.logger.debug(f"DEFECT REGION ANALYSIS:")
        
        for class_name in ['dirt', 'scratches']:
            if class_name in region:
                class_regions = region[class_name]
                if class_regions['num_regions'] > 0:
                    self.logger.debug(f"  {class_name.title()} Regions:")
                    self.logger.debug(f"    Number: {class_regions['num_regions']}")
                    self.logger.debug(f"    Largest: {class_regions['largest_region_area']:,} pixels")
                    self.logger.debug(f"    Average: {class_regions['average_region_area']:.1f} pixels")
                else:
                    self.logger.debug(f"  {class_name.title()}: No regions detected")
        
        # Combined defects summary
        if 'combined_defects' in region and region['combined_defects']['num_regions'] > 0:
            combined = region['combined_defects']
            total_defect_percentage = pred['class_statistics']['dirt_percentage'] + pred['class_statistics']['scratches_percentage']
            self.logger.debug(f"  Total Defects: {total_defect_percentage:.2f}% coverage")
            self.logger.debug(f"  Combined Regions: {combined['num_regions']}")
        
        self.logger.debug(f"")
        self.logger.debug(f"PERFORMANCE:")
        self.logger.debug(f"  Inference Time: {pred['inference_time']:.3f} seconds")
        self.logger.debug(f"  Device: {results['model_info']['device']}")
        self.logger.debug(f"="*70)

    def visualize_multiclass_prediction(self, image: np.ndarray, prediction_result: Dict,
                                     region_analysis: Dict = None, image_name: str = "Unknown",
                                     save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """
        Create visualization for multi-class prediction results
        """
        try:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            
            # Row 1: Original and class probability maps
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Individual class probability maps
            for cls in range(min(3, self.num_classes)):  # Show first 3 classes
                class_name = self.class_names[cls]
                prob_map = prediction_result['class_probabilities'][class_name]
                
                if cls == 0:  # Background
                    im = axes[0, 1].imshow(prob_map, cmap='gray', vmin=0, vmax=1)
                elif cls == 1:  # Dirt
                    im = axes[0, 2].imshow(prob_map, cmap='Greens', vmin=0, vmax=1)
                else:  # Scratches
                    im = axes[0, 3].imshow(prob_map, cmap='Reds', vmin=0, vmax=1)
                
                axes[0, cls+1].set_title(f'{class_name.title()} Probability', fontsize=12, fontweight='bold')
                axes[0, cls+1].axis('off')
                plt.colorbar(im, ax=axes[0, cls+1], fraction=0.046, pad=0.04)
            
            # Row 2: Class predictions and combined visualization
            # Multi-class prediction visualization
            class_pred_colored = self.create_multiclass_visualization(prediction_result['class_prediction'])
            axes[1, 0].imshow(class_pred_colored)
            axes[1, 0].set_title('Multi-Class Prediction\n(Black=BG, Green=Scratches, Red=Dirt)', 
                               fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Individual class binary masks
            for cls in range(min(3, self.num_classes)):
                class_name = self.class_names[cls]
                binary_mask = prediction_result['binary_masks'][class_name]
                
                if cls == 0:  # Background - show inverted
                    axes[1, cls+1].imshow(1 - binary_mask, cmap='gray', vmin=0, vmax=1)
                elif cls == 1:  # Dirt
                    axes[1, cls+1].imshow(binary_mask, cmap='Greens', vmin=0, vmax=1)
                else:  # Scratches
                    axes[1, cls+1].imshow(binary_mask, cmap='Reds', vmin=0, vmax=1)
                
                class_pixels = prediction_result['class_statistics'][f'{class_name}_pixels']
                class_pct = prediction_result['class_statistics'][f'{class_name}_percentage']
                axes[1, cls+1].set_title(f'{class_name.title()} Mask\n({class_pct:.1f}%, {class_pixels:,} px)', 
                                       fontsize=12, fontweight='bold')
                axes[1, cls+1].axis('off')
            
            # Row 3: Overlays and information
            # Prediction overlay on original image
            overlay = self.create_multiclass_overlay(image, prediction_result['class_prediction'])
            axes[2, 0].imshow(overlay)
            axes[2, 0].set_title('Multi-Class Overlay\n(Green=Scratches, Red=Dirt)', 
                               fontsize=12, fontweight='bold')
            axes[2, 0].axis('off')
            
            # Individual class overlays
            for cls in range(1, min(3, self.num_classes)):  # Skip background
                class_name = self.class_names[cls]
                class_overlay = image.copy()
                class_mask = prediction_result['binary_masks'][class_name] == 1
                
                if cls == 1:  # Dirt - green overlay
                    class_overlay[class_mask] = class_overlay[class_mask] * 0.3 + np.array([0, 255, 0]) * 0.7
                else:  # Scratches - red overlay
                    class_overlay[class_mask] = class_overlay[class_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
                
                axes[2, cls].imshow(class_overlay.astype(np.uint8))
                axes[2, cls].set_title(f'{class_name.title()} Overlay', fontsize=12, fontweight='bold')
                axes[2, cls].axis('off')
            
            # Information display
            info_text = f"""Multi-Class Prediction Summary:

Image: {image_name}
Size: {prediction_result['original_size'][1]} × {prediction_result['original_size'][0]}
Classes: {self.num_classes} ({', '.join(self.class_names)})

Class Distribution:"""

            for cls in range(self.num_classes):
                class_name = self.class_names[cls]
                pixels = prediction_result['class_statistics'][f'{class_name}_pixels']
                percentage = prediction_result['class_statistics'][f'{class_name}_percentage']
                info_text += f"\n• {class_name.title()}: {percentage:.2f}% ({pixels:,} px)"

            if region_analysis:
                info_text += f"\n\nRegion Analysis:"
                for class_name in ['dirt', 'scratches']:
                    if class_name in region_analysis:
                        regions = region_analysis[class_name]
                        if regions['num_regions'] > 0:
                            info_text += f"\n• {class_name.title()}: {regions['num_regions']} regions"
                            info_text += f"\n  Largest: {regions['largest_region_area']:,} px"

            info_text += f"\n\nPerformance:"
            info_text += f"\n• Inference: {prediction_result['inference_time']:.3f}s"
            info_text += f"\n• Device: {str(self.device)}"
            
            axes[2, 3].text(0.05, 0.95, info_text, transform=axes[2, 3].transAxes,
                           verticalalignment='top', fontsize=9, fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[2, 3].set_title('Prediction Information', fontsize=12, fontweight='bold')
            axes[2, 3].axis('off')
            
            # Add main title
            fig.suptitle(f'Multi-Class Dirt Detection Results - {image_name}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.debug(f"Multi-class visualization saved to: {save_path}")
            
            if show_plot:
                plt.show()
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Failed to create multi-class visualization: {e}")
            return None

    def create_multiclass_visualization(self, class_prediction: np.ndarray) -> np.ndarray:
        """Create colored visualization of multi-class prediction"""
        h, w = class_prediction.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply colors for each class
        for num_class in range(self.num_classes):
            mask = class_prediction == num_class
            if num_class == 0:  # Background - black
                colored[mask] = [0, 0, 0]
            elif num_class == 1:  # condensation - green
                colored[mask] = [0, 255, 0]
            elif num_class == 2:  # dirt - red
                colored[mask] = [255, 0, 0]
            elif num_class == 3:  # scratch - blue
                colored[mask] = [0, 0, 255]
        
        return colored

    def create_multiclass_overlay(self, image: np.ndarray, class_prediction: np.ndarray) -> np.ndarray:
        """Create overlay of multi-class prediction on original image"""
        overlay = image.copy()

        # condensation overlay (green)
        condensation_mask = class_prediction == 1
        if np.any(condensation_mask):
            overlay[condensation_mask] = overlay[condensation_mask] * 0.3 + np.array([0, 255, 0]) * 0.7
        
        # Dirt overlay (red)
        dirt_mask = class_prediction == 2
        if np.any(dirt_mask):
            overlay[dirt_mask] = overlay[dirt_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
        
        # Scratches overlay (blue)
        scratch_mask = class_prediction == 3
        if np.any(scratch_mask):
            overlay[scratch_mask] = overlay[scratch_mask] * 0.3 + np.array([0, 0, 255]) * 0.7
        
        return overlay.astype(np.uint8)

    def save_prediction_results(self, results: Dict, save_path: str) -> None:
        """Save prediction results to JSON file - handles multi-class data"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_for_json(results)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Detailed results saved to: {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results to {save_path}: {e}")
    
    def _convert_numpy_for_json(self, obj) -> Union[Dict, List, int, float, str, None]:
        """Recursively convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def save_batch_summary(self, results: List[Dict], save_path: str):
        """Save summary of batch processing results - updated for multi-class"""
        # Extract multi-class statistics
        dirt_percentages = [r['prediction']['class_statistics']['dirt_percentage'] for r in results]
        scratch_percentages = [r['prediction']['class_statistics']['scratches_percentage'] for r in results]
        inference_times = [r['prediction']['inference_time'] for r in results]
        
        summary = {
            'total_images': len(results),
            'model_info': {
                'num_classes': results[0]['model_info']['num_classes'] if results else 3,
                'class_names': results[0]['model_info']['class_names'] if results else self.class_names
            },
            'summary_stats': {
                'dirt': {
                    'avg_percentage': float(np.mean(dirt_percentages)),
                    'std_percentage': float(np.std(dirt_percentages)),
                    'min_percentage': float(np.min(dirt_percentages)),
                    'max_percentage': float(np.max(dirt_percentages))
                },
                'scratches': {
                    'avg_percentage': float(np.mean(scratch_percentages)),
                    'std_percentage': float(np.std(scratch_percentages)),
                    'min_percentage': float(np.min(scratch_percentages)),
                    'max_percentage': float(np.max(scratch_percentages))
                },
                'performance': {
                    'avg_inference_time': float(np.mean(inference_times)),
                    'total_inference_time': float(np.sum(inference_times))
                }
            },
            'individual_results': [
                {
                    'filename': r['image_info']['filename'],
                    'dirt_percentage': r['prediction']['class_statistics']['dirt_percentage'],
                    'scratches_percentage': r['prediction']['class_statistics']['scratches_percentage'],
                    'background_percentage': r['prediction']['class_statistics']['background_percentage'],
                    'dirt_regions': r['region_analysis']['dirt']['num_regions'] if 'dirt' in r['region_analysis'] else 0,
                    'scratch_regions': r['region_analysis']['scratches']['num_regions'] if 'scratches' in r['region_analysis'] else 0,
                    'inference_time': r['prediction']['inference_time']
                } for r in results
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.debug(f"Multi-class batch summary saved to: {save_path}")

    def predict_and_save_suction_cup(self, save_path, image):
        # image = cv2.imread(f'{path}/{image_name}', cv2.IMREAD_GRAYSCALE)
        self.logger.info("inside predict_and_save_suction_cup")
        if image is None:
            return
        (h, w) = image.shape[:2]
        # scale_factor = 25 # 25 times smaller to fit 640 height
        # small_image = cv2.resize(image, (int(w/scale_factor), int(h/scale_factor)), interpolation = cv2.INTER_AREA)
        rgb_small_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Perform object detection on an image using the model
        results = self.suction_cup_model(
            rgb_small_image,
            show=False,
            save=False, #True,
            save_txt=False, #True,
            save_conf=False, #True,
            save_crop=False, #True,
            show_conf=False, #True,
            show_labels=False, #True,
            # line_width=1,
            conf=0.636,
            rect=True, # allow padding to rectangular input images
            imgsz=[640, 320], # indicating exact image dimensions reduces inference time significantly!!!
            max_det=15 # max number of detections per image
            )
        
        # Process results
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            for item in boxes.data:
                x0, y0, x1, y1, conf, _ = item
                cv2.rectangle(rgb_small_image, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 0, 0), thickness=1)
        self.logger.info(f"Saving suction cup result at {save_path = }")
        cv2.imwrite(save_path, rgb_small_image)

    # Legacy methods for backward compatibility
    def create_comparison_mask(self, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """Legacy method - for multi-class, use dirt class for comparison"""
        return self.create_multiclass_comparison_mask(prediction, ground_truth)

    def create_multiclass_comparison_mask(self, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Create visualization mask for multi-class comparison
        """
        h, w = prediction.shape
        comparison = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Correct predictions - use class colors
        correct_mask = (prediction == ground_truth)
        for cls in range(self.num_classes):
            cls_correct = np.logical_and(correct_mask, prediction == cls)
            if cls == 0:  # Background - black
                comparison[cls_correct] = [0, 0, 0]
            elif cls == 1:  # Dirt - green
                comparison[cls_correct] = [0, 255, 0]
            elif cls == 2:  # Scratches - red
                comparison[cls_correct] = [255, 0, 0]
        
        # Incorrect predictions - yellow
        incorrect_mask = (prediction != ground_truth)
        comparison[incorrect_mask] = [255, 255, 0]
        
        return comparison

    def visualize_prediction(self, image: np.ndarray, prediction_result: Dict, 
                        ground_truth: np.ndarray = None, labels: Dict = None,
                        save_path: str = None, show_plot: bool = True,
                        iou_threshold: float = 0.5) -> plt.Figure:
        """
        Legacy visualization method - updated for multi-class compatibility
        """
        if ground_truth is not None:
            return self.visualize_multiclass_comparison(image, prediction_result, ground_truth, 
                                                      labels, save_path, show_plot, iou_threshold)
        else:
            return self.visualize_multiclass_prediction(image, prediction_result, None, 
                                                      labels.get('source_image', 'Unknown') if labels else 'Unknown',
                                                      save_path, show_plot)

    def visualize_multiclass_comparison(self, image: np.ndarray, prediction_result: Dict,
                                      ground_truth: np.ndarray, labels: Dict = None,
                                      save_path: str = None, show_plot: bool = True,
                                      iou_threshold: float = 0.5) -> plt.Figure:
        """
        Create comparison visualization for multi-class predictions with ground truth
        """
        try:
            fig, axes = plt.subplots(2, 5, figsize=(20, 10))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Multi-class prediction
            class_pred_colored = self.create_multiclass_visualization(prediction_result['class_prediction'])
            axes[0, 1].imshow(class_pred_colored)
            axes[0, 1].set_title('Multi-Class Prediction\n(Black=BG, Green=Condensation, \nRed=Dirt, Blue=Scratch)', 
                               fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Ground truth (assuming it's multi-class too)
            if len(np.unique(ground_truth)) > 2:  # Multi-class ground truth
                gt_colored = self.create_multiclass_visualization(np.squeeze(ground_truth))
                axes[0, 2].imshow(gt_colored)
                axes[0, 2].set_title('Ground Truth\n(Multi-Class)', fontsize=12, fontweight='bold')
            else:  # Binary ground truth - treat as dirt class
                gt_display = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
                gt_display[ground_truth == 1] = [0, 255, 0]  # Green for dirt
                axes[0, 2].imshow(gt_display)
                axes[0, 2].set_title('Ground Truth\n(Binary - Dirt)', fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')

            # Class-wise probability visualization
            # Show dirt and scratch probabilities combined
            # dirt_prob = prediction_result['class_probabilities']['dirt']
            # scratch_prob = prediction_result['class_probabilities']['scratch']
            # combined_prob = np.maximum(dirt_prob, scratch_prob)
            probs = [prediction_result['class_probabilities'][label.value] 
                    for label in TargetLabels]
            combined_prob = np.maximum.reduce(probs)
            
            prob_display = axes[0, 3].imshow(combined_prob, cmap='hot', vmin=0, vmax=1)
            axes[0, 3].set_title('Max Defect Probability\n(All dirt category)', fontsize=12, fontweight='bold')
            axes[0, 3].axis('off')
            plt.colorbar(prob_display, ax=axes[0, 3], fraction=0.046, pad=0.04)
            
            # Comparison visualization
            if len(np.unique(ground_truth)) > 2:
                comparison = self.create_multiclass_comparison_mask(prediction_result['class_prediction'], np.squeeze(ground_truth))
            else:
                # For binary GT, compare with dirt class prediction
                dirt_pred = (prediction_result['class_prediction'] == 1).astype(np.uint8)
                comparison = self.create_comparison_mask(dirt_pred, ground_truth)
            
            axes[0, 4].imshow(comparison)
            axes[0, 4].set_title('Prediction vs Ground Truth\n(Yellow=Error)', fontsize=12, fontweight='bold')
            axes[0, 4].axis('off')

            # Bottom row - overlays and metrics

            # Prediction overlay
            overlay = self.create_multiclass_overlay(image, prediction_result['class_prediction'])
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')

            # dirt prob
            dirt_prob = probs[TargetLabels.get_by_value("dirt").index]
            dirt_display = axes[1, 1].imshow(dirt_prob, cmap='Reds', vmin=0, vmax=1)
            axes[1, 1].set_title(
                f'Dirt Probability\n({prediction_result["class_statistics"]["dirt_percentage"]:.1f}%)', 
                fontsize=12, 
                fontweight='bold'
            )
            axes[1, 1].axis('off')
            plt.colorbar(dirt_display, ax=axes[1, 1], fraction=0.046, pad=0.04)

            # scratch prob
            dirt_prob = probs[TargetLabels.get_by_value("scratch").index]
            dirt_display = axes[1, 2].imshow(dirt_prob, cmap='Blues', vmin=0, vmax=1)
            axes[1, 2].set_title(
                f'Scratch Probability\n({prediction_result["class_statistics"]["scratch_percentage"]:.1f}%)', 
                fontsize=12, 
                fontweight='bold'
            )
            axes[1, 2].axis('off')
            plt.colorbar(dirt_display, ax=axes[1, 2], fraction=0.046, pad=0.04)


            # Condensation prob
            dirt_prob = probs[TargetLabels.get_by_value("condensation").index]
            dirt_display = axes[1, 3].imshow(dirt_prob, cmap='Greens', vmin=0, vmax=1)
            axes[1, 3].set_title(
                f'Condensation Probability\n({prediction_result["class_statistics"]["condensation_percentage"]:.1f}%)', 
                fontsize=12, 
                fontweight='bold'
            )
            axes[1, 3].axis('off')
            plt.colorbar(dirt_display, ax=axes[1, 3], fraction=0.046, pad=0.04)

                        
            # # Error visualization
            # axes[1, 2].imshow(comparison)
            # axes[1, 2].set_title('Error Analysis\n(Yellow=Misclassified)', fontsize=12, fontweight='bold')
            # axes[1, 2].axis('off')
            
            # Calculate and display metrics
            if len(np.unique(ground_truth)) > 2:
                # Multi-class metrics
                metrics_text= self._calculate_multiclass_metrics_text(prediction_result['class_prediction'], np.squeeze(ground_truth))
                
            else:
                # Binary metrics for dirt class
                dirt_pred = (prediction_result['class_prediction'] == 1).astype(np.uint8)
                metrics = calculate_metrics(dirt_pred, ground_truth, iou_threshold)
                metrics_text = f"""BINARY DIRT METRICS:
IoU: {metrics.get('pixel_iou', 0):.3f}
Dice: {metrics.get('pixel_dice', 0):.3f}
Precision: {metrics.get('object_precision', 0):.3f}
Recall: {metrics.get('object_recall', 0):.3f}
F1-Score: {metrics.get('object_f1', 0):.3f}

CLASS STATISTICS:"""
                for cls in range(self.num_classes):
                    class_name = self.class_names[cls]
                    percentage = prediction_result['class_statistics'][f'{class_name}_percentage']
                    metrics_text += f"\n{class_name.title()}: {percentage:.2f}%"
                
                metrics_text += f"\n\nInference: {prediction_result['inference_time']:.3f}s"
            
            axes[1, 4].text(0.05, 0.95, metrics_text, transform=axes[1, 4].transAxes,
                           verticalalignment='top', fontsize=10, fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 4].set_title('Performance Metrics', fontsize=12, fontweight='bold', loc="left")
            axes[1, 4].axis('off')
            
            # Add main title
            if labels:
                title = f"Multi-Class Comparison - {labels.get('source_image', 'Unknown')}"
            else:
                title = "Multi-Class Dirt Detection Comparison"
            fig.suptitle(title, fontsize=16, fontweight='bold',)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.debug(f"Multi-class comparison saved to: {save_path}")
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-class comparison visualization: {e}")
            return None

    def _calculate_multiclass_metrics_text(self, prediction: np.ndarray, ground_truth: np.ndarray) -> str:
        """Calculate and format multi-class metrics"""
        try:
            # Convert to torch tensors for metric calculation
            import torch
            # FIXED: prediction is already class indices (H, W), not logits
            pred_tensor = torch.from_numpy(prediction).unsqueeze(0)  # Add batch dim: (1, H, W)
            gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0)   # Add batch dim: (1, H, W)
            
            # Calculate multi-class IoU and Dice
            iou_metrics = calculate_multiclass_iou(pred_tensor, gt_tensor, self.num_classes)
            dice_metrics = calculate_multiclass_dice(pred_tensor, gt_tensor, self.num_classes)
            
            metrics_text = "MULTI-CLASS METRICS:\n"
            metrics_text += f"Mean IoU: {iou_metrics.get('mean_iou', 0):.3f}\n"
            metrics_text += f"Mean Dice: {dice_metrics.get('mean_dice', 0):.3f}\n\n"
            
            # Per-class metrics
            for cls in range(self.num_classes):
                class_name = self.class_names[cls]
                iou_key = f'iou_class_{cls}'
                dice_key = f'dice_class_{cls}'
                
                if iou_key in iou_metrics and dice_key in dice_metrics:
                    metrics_text += f"{class_name.title()}:\n"
                    metrics_text += f"  IoU: {iou_metrics[iou_key]:.3f}\n"
                    metrics_text += f"  Dice: {dice_metrics[dice_key]:.3f}\n"
            
            return metrics_text
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate multi-class metrics: {e}")
            return "METRICS: Calculation failed"

    def batch_predict_and_compare_mask(self, dataset_dir: str, output_dir: str = None, 
                                     max_samples: int = None) -> List[Dict]:
        """Efficient batch processing with essential metrics only"""
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        if output_dir is None:
            output_dir = dataset_path / 'batch_comparison_results'
        else:
            output_dir = Path(output_dir)

        # Find dataset sample directories
        required_files_multi = ['synthetic_dirty_patch.png', 'segmentation_mask_patch_multiclass.png', 'labels_patch.json']
        fallback_files_binary = ['synthetic_dirty_patch.png', 'segmentation_mask_patch.png', 'labels_patch.json']
        
        sample_dirs = []
        for item in dataset_path.iterdir():
            if item.is_dir():
                if all((item / file).exists() for file in required_files_multi):
                    sample_dirs.append(item)
                elif all((item / file).exists() for file in fallback_files_binary):
                    sample_dirs.append(item)
                    
        if max_samples:
            sample_dirs = sample_dirs[:max_samples]
        self.logger.info(f"Found {len(sample_dirs)} dataset samples for batch comparison")

        results = []
        from tqdm import tqdm
        
        for i, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing samples")):
            try:
                # Create sample-specific output directory
                sample_output_dir = output_dir / sample_dir.name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Process sample efficiently
                result = self.predict_and_compare_mask(str(sample_dir), str(sample_output_dir))
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {sample_dir.name}: {e}")
                results.append({
                    'sample_name': sample_dir.name,
                    'status': 'error',
                    'error': str(e)
                })
                
        # Save streamlined summary
        if results:
            summary_path = output_dir / "batch_summary.json"
            self._save_efficient_batch_summary(results, str(summary_path))
            
            successful = [r for r in results if r.get('status') == 'completed']
            multiclass = [r for r in successful if r.get('sample_type') == 'multiclass']
            binary = [r for r in successful if r.get('sample_type') == 'binary']
            
            self.logger.info(f"\nBatch completed! Processed {len(successful)}/{len(results)} samples")
            self.logger.info(f"Multi-class: {len(multiclass)}, Binary: {len(binary)}")
            
            # Log only key aggregate metrics
            if multiclass:
                avg_iou = np.mean([r['metrics']['mean_iou'] for r in multiclass if 'metrics' in r])
                self.logger.info(f"Multi-class avg IoU: {avg_iou:.3f}")
            
            if binary:
                avg_iou = np.mean([r['metrics']['iou'] for r in binary if 'metrics' in r])
                self.logger.info(f"Binary avg IoU: {avg_iou:.3f}")
        
        return results

    def _save_efficient_batch_summary(self, results: List[Dict], save_path: str):
        """Save streamlined batch summary focusing on key metrics"""
        import json
        import time
        
        successful = [r for r in results if r.get('status') == 'completed']
        multiclass = [r for r in successful if r.get('sample_type') == 'multiclass']
        binary = [r for r in successful if r.get('sample_type') == 'binary']
        
        # Calculate only essential aggregates
        multiclass_summary = {}
        if multiclass:
            iou_values = [r['metrics']['mean_iou'] for r in multiclass if 'metrics' in r and 'mean_iou' in r['metrics']]
            dice_values = [r['metrics']['mean_dice'] for r in multiclass if 'metrics' in r and 'mean_dice' in r['metrics']]
            
            if iou_values:
                multiclass_summary = {
                    'count': len(multiclass),
                    'avg_iou': float(np.mean(iou_values)),
                    'avg_dice': float(np.mean(dice_values)) if dice_values else 0.0,
                    'std_iou': float(np.std(iou_values)),
                    'min_iou': float(np.min(iou_values)),
                    'max_iou': float(np.max(iou_values))
                }

        binary_summary = {}
        if binary:
            iou_values = [r['metrics']['iou'] for r in binary if 'metrics' in r and 'iou' in r['metrics']]
            precision_values = [r['metrics']['precision'] for r in binary if 'metrics' in r and 'precision' in r['metrics']]
            
            if iou_values:
                binary_summary = {
                    'count': len(binary),
                    'avg_iou': float(np.mean(iou_values)),
                    'avg_precision': float(np.mean(precision_values)) if precision_values else 0.0,
                    'std_iou': float(np.std(iou_values)),
                    'min_iou': float(np.min(iou_values)),
                    'max_iou': float(np.max(iou_values))
                }

        summary = {
            'batch_info': {
                'total_samples': len(results),
                'successful_samples': len(successful),
                'multiclass_samples': len(multiclass),
                'binary_samples': len(binary),
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'aggregate_metrics': {
                'multiclass': multiclass_summary,
                'binary': binary_summary
            },
            'individual_results': [
                {
                    'sample_name': r['sample_name'],
                    'sample_type': r.get('sample_type', 'unknown'),
                    'status': r['status'],
                    'metrics': r.get('metrics', {}),
                    'inference_time': r.get('inference_time', 0),
                    'error': r.get('error')
                } for r in results
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Efficient batch summary saved to: {save_path}")
    
    def predict_and_compare_mask(self, input_path: str, output_dir: str) -> Dict:
        """Streamlined version for multi-class dataset comparison"""
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if it's a dataset sample directory
        required_files = ['synthetic_dirty_patch.png', 'segmentation_mask_patch_multiclass.png', 'labels_patch.json']
        fallback_files = ['synthetic_dirty_patch.png', 'segmentation_mask_patch.png', 'labels_patch.json']

        if all((input_path / file).exists() for file in required_files):
            # Multi-class dataset sample
            self.logger.info(f"Running multi-class inference on dataset sample: {input_path}")
            try:
                image, ground_truth, labels = load_ground_truth_data(str(input_path), self.logger, multiclass=True)
                prediction_result = self.predict(image, target_size=(1792, 1792), return_raw=False)  # Don't return raw for speed

                # Save visualization
                output_path = output_dir / f"{input_path.name}_multiclass_inference.png"
                self.visualize_multiclass_comparison(
                    image, prediction_result, ground_truth, labels,
                    save_path=str(output_path), show_plot=False
                )
                
                # Calculate ONLY essential metrics efficiently
                metrics = self._calculate_essential_metrics(prediction_result['class_prediction'], ground_truth)
                
                self.logger.info(f"Multi-class inference completed! IoU: {metrics.get('mean_iou', 0):.3f}")

                return {
                    'sample_name': input_path.name,
                    'sample_type': 'multiclass',
                    'status': 'completed',
                    'metrics': metrics,
                    'inference_time': prediction_result['inference_time']
                }
                
            except Exception as e:
                self.logger.error(f"Error processing {input_path.name}: {e}")
                return {
                    'sample_name': input_path.name,
                    'sample_type': 'multiclass',
                    'status': 'error',
                    'error': str(e)
                }
                
        elif all((input_path / file).exists() for file in fallback_files):
            # Binary dataset sample
            self.logger.info(f"Running inference on binary dataset sample: {input_path}")
            try:
                image, ground_truth, labels = load_ground_truth_data(str(input_path), self.logger, multiclass=False)
                prediction_result = self.predict(image, return_raw=False)  # Don't return raw for speed
                
                # Save visualization
                output_path = output_dir / f"{input_path.name}_inference.png"
                self.visualize_multiclass_comparison(
                    image, prediction_result, ground_truth, labels,
                    save_path=str(output_path), show_plot=False
                )
                
                # Calculate essential binary metrics only
                dirt_pred = (prediction_result['class_prediction'] == 1).astype(np.uint8)
                metrics = self._calculate_essential_binary_metrics(dirt_pred, ground_truth)
                
                self.logger.info(f"Binary inference completed! IoU: {metrics.get('iou', 0):.3f}")
                
                return {
                    'sample_name': input_path.name,
                    'sample_type': 'binary',
                    'status': 'completed',
                    'metrics': metrics,
                    'inference_time': prediction_result['inference_time']
                }
                
            except Exception as e:
                self.logger.error(f"Error processing {input_path.name}: {e}")
                return {
                    'sample_name': input_path.name,
                    'sample_type': 'binary',
                    'status': 'error',
                    'error': str(e)
                }
        else:
            return {
                'sample_name': input_path.name,
                'sample_type': 'unknown',
                'status': 'error',
                'error': 'Missing required files'
            }

    def _calculate_essential_metrics(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Calculate only essential metrics efficiently"""
        try:
            # Flatten for efficiency
            pred_flat = prediction.flatten()
            gt_flat = ground_truth.flatten()
            
            # Calculate per-class IoU and Dice efficiently
            metrics = {}
            ious = []
            dices = []
            
            for cls in range(self.num_classes):
                pred_cls = (pred_flat == cls)
                true_cls = (gt_flat == cls)
                
                intersection = np.sum(pred_cls & true_cls)
                union = np.sum(pred_cls | true_cls)
                pred_sum = np.sum(pred_cls)
                true_sum = np.sum(true_cls)
                
                # IoU
                iou = intersection / union if union > 0 else 1.0
                # Dice  
                dice = (2 * intersection) / (pred_sum + true_sum) if (pred_sum + true_sum) > 0 else 1.0
                
                class_name = self.class_names[cls]
                metrics[f'{class_name}_iou'] = float(iou)
                metrics[f'{class_name}_dice'] = float(dice)
                
                ious.append(iou)
                dices.append(dice)
            
            # Mean metrics
            metrics['mean_iou'] = float(np.mean(ious))
            metrics['mean_dice'] = float(np.mean(dices))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to calculate essential metrics: {e}")
            return {'error': str(e)}

    def _calculate_essential_binary_metrics(self, prediction: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """Calculate essential binary metrics efficiently"""
        try:
            # Flatten for efficiency
            pred_flat = prediction.flatten()
            gt_flat = ground_truth.flatten()
            
            # Calculate basic confusion matrix
            tp = np.sum((pred_flat == 1) & (gt_flat == 1))
            fp = np.sum((pred_flat == 1) & (gt_flat == 0))
            fn = np.sum((pred_flat == 0) & (gt_flat == 1))
            
            # Essential metrics only
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            
            return {
                'iou': float(iou),
                'dice': float(dice),
                'precision': float(precision),
                'recall': float(recall),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate binary metrics: {e}")
            return {'error': str(e)}
    
    
    def _print_multiclass_comparison_metrics(self, prediction_result: Dict, ground_truth: np.ndarray):
        """Print detailed multi-class comparison metrics"""
        try:
            import torch
            # FIXED: Remove the extra .unsqueeze(0) - class_prediction is already class indices (H, W)
            pred_tensor = torch.from_numpy(prediction_result['class_prediction']).unsqueeze(0)  # Add batch dim only: (1, H, W)
            gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0)  # Add batch dim only: (1, H, W)
            
            iou_metrics = calculate_multiclass_iou(pred_tensor, gt_tensor, self.num_classes)
            dice_metrics = calculate_multiclass_dice(pred_tensor, gt_tensor, self.num_classes)
            
            self.logger.info(f"\nMulti-Class Metrics:")
            self.logger.info(f"  Mean IoU: {iou_metrics.get('mean_iou', 0):.3f}")
            self.logger.info(f"  Mean Dice: {dice_metrics.get('mean_dice', 0):.3f}")
            
            for cls in range(self.num_classes):
                class_name = self.class_names[cls]
                iou_val = iou_metrics.get(f'iou_class_{cls}', 0)
                dice_val = dice_metrics.get(f'dice_class_{cls}', 0)
                class_pct = prediction_result['class_statistics'][f'{class_name}_percentage']
                
                self.logger.info(f"  {class_name.title()}: IoU={iou_val:.3f}, Dice={dice_val:.3f}, Coverage={class_pct:.2f}%")
                
        except Exception as e:
            self.logger.error(f"Failed to calculate detailed metrics: {e}")
            e.with_traceback()


import time
import json
from pathlib import Path
from typing import Union, Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch import no_grad, sigmoid
import matplotlib.patches as mpatches

from src.models.unet import UNet
from src.core.region_analyzer import RegionAnalyzer
from src.inference.base_predictor import BasePredictor
from src.synthesis.image_operations import ImageOperations
from src.training.metrics import calculate_metrics, _find_objects
from src.core.logger_config import setup_application_logger
from src.utils.file_utils import load_ground_truth_data


class SingleImagePredictor(BasePredictor):

    def __init__(self, region_analyzer:RegionAnalyzer, image_operations:ImageOperations, app_logger=None, *args, **kwargs):
        super().__init__(app_logger=app_logger, *args, **kwargs)
        self.region_analyzer = region_analyzer
        self.image_operations = image_operations

        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('SingleImagePredictor')

    
    def predict(self, image: np.ndarray, target_size=(1024, 1024), return_raw: bool = False) -> Dict:
        """Predict on a single image."""
        original_size = image.shape[:2]
        # target_size = self.config.get('target_size', (1024, 1024))

        image_tensor = self.image_operations.preprocess_prediction_image(image,self.device, target_size, self.config)
        
        
        with no_grad():
            start_time = time.time()
            logits = self.model(image_tensor)
            inference_time = time.time() - start_time
            
            probabilities = sigmoid(logits).cpu().numpy()[0, 0]
            binary_pred = (probabilities > self.confidence_threshold).astype(np.uint8)
            
            # Resize back to original size - TODO: use image operations
            import cv2
            probabilities = cv2.resize(probabilities, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
            binary_pred = cv2.resize(binary_pred, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)
        
        result = {
            'binary_prediction': binary_pred,
            'probabilities': probabilities,
            'confidence_threshold': self.confidence_threshold,
            'inference_time': inference_time,
            'original_size': original_size,
            'dirt_pixels': np.sum(binary_pred),
            'dirt_percentage': (np.sum(binary_pred) / binary_pred.size) * 100
        }
        
        if return_raw:
            result['raw_logits'] = logits.cpu().numpy()[0, 0]
            
        return result

    def single_prediction_pipeline(self,
                                    image_path: str, save_results: bool = True, 
                                    output_dir: str = None, show_plot: bool = True, **kwargs) -> Dict:
        """
        Complete pipeline for predicting on a new image without ground truth
        
        Args:
            image_path: Path to the new image
            save_results: Whether to save visualization and results
            output_dir: Directory to save results (default: same as image directory)
            show_plot: Whether to display the plot
            **kwargs: Additional keyword arguments (for protocol compatibility)
            
        Returns:
            Complete prediction results
        """
        self.logger.debug(f"Running single prediction pipeline ....")
        image_path = Path(image_path)

        image = self.image_operations.load_image_color(image_path)

        prediction_result = self.predict(image, return_raw=True)

        # Analyze regions
        region_analysis = self.region_analyzer.analyze_dirt_regions(prediction_result['binary_prediction'])

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
                'device': str(self.device)
            }
        }

        # Save results
        if save_results:
            try:
                # Save visualization
                vis_path = output_dir / f"{image_path.stem}_dirt_detection.png"
                self.visualize_single_image_prediction(
                    image, prediction_result, region_analysis,
                    image_name=image_path.name,
                    save_path=str(vis_path), 
                    show_plot=show_plot
                )
                
                # Save detailed results as JSON
                results_path = output_dir / f"{image_path.stem}_results.json"
                self.save_prediction_results(results, str(results_path))
                
                # Save binary mask
                mask_path = output_dir / f"{image_path.stem}_dirt_mask.png"
                cv2.imwrite(str(mask_path), prediction_result['binary_prediction'] * 255)
                
                # Save probability map
                prob_path = output_dir / f"{image_path.stem}_probability_map.png"
                prob_vis = (prediction_result['probabilities'] * 255).astype(np.uint8)
                cv2.imwrite(str(prob_path), prob_vis)
                
                self.logger.debug(f"\nResults saved to: {output_dir}")
                self.logger.debug(f"- Visualization: {vis_path.name}")
                self.logger.debug(f"- Results JSON: {results_path.name}")
                self.logger.debug(f"- Binary mask: {mask_path.name}")
                self.logger.debug(f"- Probability map: {prob_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to save results: {e}")
        
        # Print summary
        self.print_prediction_summary(results)

        self.logger.debug(f"single_prediction_pipeline complete! ")
        
        return results
    
    def print_prediction_summary(self, results: Dict) -> None:
        """Print a summary of prediction results"""
        pred = results['prediction']
        region = results['region_analysis']
        image_info = results['image_info']
        
        self.logger.debug(f"\n" + "="*60)
        self.logger.debug(f"DIRT DETECTION SUMMARY")
        self.logger.debug(f"="*60)
        self.logger.debug(f"Image: {image_info['filename']}")
        self.logger.debug(f"Size: {image_info['original_size'][1]} × {image_info['original_size'][0]} pixels")
        self.logger.debug(f"")
        self.logger.debug(f"DETECTION RESULTS:")
        self.logger.debug(f"  Dirt Percentage: {pred['dirt_percentage']:.2f}%")
        self.logger.debug(f"  Dirt Pixels: {pred['dirt_pixels']:,} / {image_info['total_pixels']:,}")
        self.logger.debug(f"  Confidence Threshold: {pred['confidence_threshold']}")
        self.logger.debug(f"")
        self.logger.debug(f"REGION ANALYSIS:")
        if region['num_regions'] > 0:
            self.logger.debug(f"  Number of Dirt Regions: {region['num_regions']}")
            self.logger.debug(f"  Largest Region: {region['largest_region_area']:,} pixels")
            self.logger.debug(f"  Smallest Region: {region['smallest_region_area']:,} pixels")
            self.logger.debug(f"  Average Region Size: {region['average_region_area']:.1f} pixels")
        else:
            self.logger.debug(f"  No dirt regions detected")
        self.logger.debug(f"")
        self.logger.debug(f"PERFORMANCE:")
        self.logger.debug(f"  Inference Time: {pred['inference_time']:.3f} seconds")
        self.logger.debug(f"  Device: {results['model_info']['device']}")
        self.logger.debug(f"="*60)

    def visualize_single_image_prediction(self, image: np.ndarray, prediction_result: Dict,
                                     region_analysis: Dict = None, image_name: str = "Unknown",
                                     save_path: str = None, show_plot: bool = True) -> plt.Figure:
        """
        Create visualization for new image prediction (without ground truth)
        
        Args:
            image: Original input image
            prediction_result: Result from predict_single()
            region_analysis: Result from analyze_dirt_regions()
            image_name: Name of the image file
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Figure object if created successfully, None otherwise
        """
        try:

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(image)
            axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Prediction probability heatmap
            prob_display = axes[0, 1].imshow(prediction_result['probabilities'], 
                                        cmap='hot', vmin=0, vmax=1)
            axes[0, 1].set_title(f'Prediction Probabilities\n(Threshold: {self.confidence_threshold})', 
                            fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            plt.colorbar(prob_display, ax=axes[0, 1], fraction=0.046, pad=0.04)
            
            # Binary prediction
            binary_display = axes[0, 2].imshow(prediction_result['binary_prediction'], 
                                            cmap='Reds', vmin=0, vmax=1)
            axes[0, 2].set_title(f'Binary Prediction\n({prediction_result["dirt_percentage"]:.1f}% dirt)', 
                            fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')
            
            # Prediction overlay on original image
            overlay1 = image.copy()
            dirt_mask = prediction_result['binary_prediction'] == 1
            overlay1[dirt_mask] = overlay1[dirt_mask] * 0.3 + np.array(self.colors['dirt']) * 0.7
            axes[1, 0].imshow(overlay1.astype(np.uint8))
            axes[1, 0].set_title('Prediction Overlay\n(Red = Predicted Dirt)', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
            
            # Probability overlay
            overlay2 = image.copy().astype(np.float32)
            prob_colored = plt.cm.hot(prediction_result['probabilities'])[:, :, :3] * 255
            alpha = prediction_result['probabilities'][:, :, np.newaxis]
            overlay2 = overlay2 * (1 - alpha * 0.7) + prob_colored * alpha * 0.7
            axes[1, 1].imshow(overlay2.astype(np.uint8))
            axes[1, 1].set_title('Probability Overlay\n(Hot colormap)', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            
            # Information display
            info_text = f"""Prediction Summary:
            Image: {image_name}
            Size: {prediction_result['original_size'][1]} × {prediction_result['original_size'][0]}

            Dirt Detection:
            • Total Pixels: {prediction_result['original_size'][0] * prediction_result['original_size'][1]:,}
            • Dirt Pixels: {prediction_result['dirt_pixels']:,}
            • Dirt Percentage: {prediction_result['dirt_percentage']:.2f}%

            Performance:
            • Inference Time: {prediction_result['inference_time']:.3f}s
            • Confidence Threshold: {self.confidence_threshold}
            • Device: {str(self.device)}"""

            if region_analysis and region_analysis['num_regions'] > 0:
                info_text += f"""

    Region Analysis:
    • Number of Dirt Regions: {region_analysis['num_regions']}
    • Largest Region: {region_analysis['largest_region_area']:,} pixels
    • Average Region Size: {region_analysis['average_region_area']:.1f} pixels"""
            
            axes[1, 2].text(0.05, 0.95, info_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 2].set_title('Prediction Information', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
            
            # Add main title
            fig.suptitle(f'Dirt Detection Results - {image_name}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.debug(f"Visualization saved to: {save_path}")
            
            if show_plot:
                plt.show()
            
            return fig
        
        except Exception as e:
            self.logger.error(f"Failed to create visualization: {e}")
            return None
    

    def save_prediction_results(self, results: Dict, save_path: str) -> None:
        """Save prediction results to JSON file
        
        Args:
            results: Results dictionary to save
            save_path: Path to save JSON file
        """
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
        """Save summary of batch processing results"""
        summary = {
            'total_images': len(results),
            'summary_stats': {
                'avg_dirt_percentage': np.mean([r['prediction']['dirt_percentage'] for r in results]),
                'std_dirt_percentage': np.std([r['prediction']['dirt_percentage'] for r in results]),
                'min_dirt_percentage': np.min([r['prediction']['dirt_percentage'] for r in results]),
                'max_dirt_percentage': np.max([r['prediction']['dirt_percentage'] for r in results]),
                'avg_inference_time': np.mean([r['prediction']['inference_time'] for r in results]),
                'total_dirt_regions': sum([r['region_analysis']['num_regions'] for r in results])
            },
            'individual_results': [
                {
                    'filename': r['image_info']['filename'],
                    'dirt_percentage': r['prediction']['dirt_percentage'],
                    'dirt_pixels': r['prediction']['dirt_pixels'],
                    'num_regions': r['region_analysis']['num_regions'],
                    'inference_time': r['prediction']['inference_time']
                } for r in results
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2,  default=str)
        
        self.logger.debug(f"Batch summary saved to: {save_path}")

    def create_comparison_mask(self, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
        """
        Create visualization mask showing pixel-level agreement (simplified for IoU focus)
        
        Args:
            prediction: Binary prediction mask
            ground_truth: Ground truth binary mask
            
        Returns:
            RGB visualization mask
        """
        h, w = prediction.shape
        comparison = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Intersection (both predict dirt) - Green
        intersection = np.logical_and(prediction == 1, ground_truth == 1)
        comparison[intersection] = [0, 255, 0]  # Green for intersection
        
        # False positives (predicted but not in ground truth) - Red  
        false_positive = np.logical_and(prediction == 1, ground_truth == 0)
        comparison[false_positive] = [255, 0, 0]  # Red for FP
        
        # False negatives (in ground truth but not predicted) - Blue
        false_negative = np.logical_and(prediction == 0, ground_truth == 1) 
        comparison[false_negative] = [0, 0, 255]  # Blue for FN
        
        # True negatives (both predict clean) - Black (background)
        # No need to set as it's already initialized to black
        
        return comparison

    def visualize_prediction(self, image: np.ndarray, prediction_result: Dict, 
                        ground_truth: np.ndarray = None, labels: Dict = None,
                        save_path: str = None, show_plot: bool = True,
                        iou_threshold: float = 0.5) -> plt.Figure:
        """
        Create comprehensive visualization of prediction results with updated metrics
        """
        # Determine number of subplots
        n_cols = 4 if ground_truth is not None else 3
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        
        if n_cols == 3:
            axes = axes.reshape(2, 3)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction probability heatmap
        prob_display = axes[0, 1].imshow(prediction_result['probabilities'], 
                                    cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Prediction Probabilities\n(Threshold: {self.confidence_threshold})', 
                        fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        plt.colorbar(prob_display, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Binary prediction with object bounding boxes
        binary_display = axes[0, 2].imshow(prediction_result['binary_prediction'], 
                                        cmap='Reds', vmin=0, vmax=1)
        
        # Add bounding boxes for predicted objects
        if ground_truth is not None:  # Only draw if we have ground truth for comparison
            pred_objects = _find_objects(prediction_result['binary_prediction'])
            for obj in pred_objects:
                x, y, w, h = obj['bbox']
                rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='yellow', 
                                facecolor='none', linestyle='-')
                axes[0, 2].add_patch(rect)
        
        axes[0, 2].set_title(f'Binary Prediction\n({prediction_result["dirt_percentage"]:.1f}% dirt)', 
                        fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Ground truth with object bounding boxes (if available)
        if ground_truth is not None:
            gt_display = axes[0, 3].imshow(ground_truth, cmap='Greens', vmin=0, vmax=1)
            
            # Add bounding boxes for ground truth objects
            gt_objects = _find_objects(ground_truth)
            for obj in gt_objects:
                x, y, w, h = obj['bbox']
                rect = plt.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='cyan', 
                                facecolor='none', linestyle='--')
                axes[0, 3].add_patch(rect)
            
            axes[0, 3].set_title(f'Ground Truth Mask\n({len(gt_objects)} objects)', fontsize=14, fontweight='bold')
            axes[0, 3].axis('off')
        
        # Overlay visualizations
        # Prediction overlay on original image
        overlay1 = image.copy()
        dirt_mask = prediction_result['binary_prediction'] == 1
        overlay1[dirt_mask] = overlay1[dirt_mask] * 0.3 + np.array(self.colors['dirt']) * 0.7
        axes[1, 0].imshow(overlay1.astype(np.uint8))
        axes[1, 0].set_title('Prediction Overlay\n(Red = Predicted Dirt)', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Probability overlay
        overlay2 = image.copy().astype(np.float32)
        prob_colored = plt.cm.hot(prediction_result['probabilities'])[:, :, :3] * 255
        alpha = prediction_result['probabilities'][:, :, np.newaxis]
        overlay2 = overlay2 * (1 - alpha * 0.7) + prob_colored * alpha * 0.7
        axes[1, 1].imshow(overlay2.astype(np.uint8))
        axes[1, 1].set_title('Probability Overlay\n(Hot colormap)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        if ground_truth is not None:
            # Comparison visualization with object bounding boxes
            comparison = self.create_comparison_mask(prediction_result['binary_prediction'], ground_truth)
            axes[1, 2].imshow(comparison)
            
            # Draw bounding boxes for detected objects
            pred_objects = _find_objects(prediction_result['binary_prediction'])
            gt_objects = _find_objects(ground_truth)
            
            # Draw predicted object bounding boxes in yellow
            for i, obj in enumerate(pred_objects):
                x, y, w, h = obj['bbox']
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='yellow', 
                                facecolor='none', linestyle='-')
                axes[1, 2].add_patch(rect)
            
            # Draw ground truth object bounding boxes in cyan
            for i, obj in enumerate(gt_objects):
                x, y, w, h = obj['bbox']
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='cyan', 
                                facecolor='none', linestyle='--')
                axes[1, 2].add_patch(rect)
            
            axes[1, 2].set_title(f'Pixel + Object Comparison\n(Pred: {len(pred_objects)}, GT: {len(gt_objects)} objects)', 
                            fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
            
            # Create custom legend for pixel-level comparison and object boxes
            legend_elements = [
                mpatches.Patch(color=[0, 1, 0], label='Intersection (Both Positive)'),
                mpatches.Patch(color=[1, 0, 0], label='False Positive'),
                mpatches.Patch(color=[0, 0, 1], label='False Negative'),
                mpatches.Patch(color='black', label='True Negative'),
                plt.Line2D([0], [0], color='yellow', linewidth=2, label='Predicted Objects'),
                plt.Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='GT Objects')
            ]
            axes[1, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
            
            # Updated metrics display
            metrics = calculate_metrics(prediction_result['binary_prediction'], ground_truth, iou_threshold)
            
            metrics_text = f"""PIXEL-LEVEL METRICS:
    IoU: {metrics['pixel_iou']:.3f}
    Dice: {metrics['pixel_dice']:.3f}
    Intersection: {metrics['pixel_intersection']:,}
    Union: {metrics['pixel_union']:,}

    OBJECT-LEVEL METRICS:
    True Positives: {metrics['object_tp']}
    False Positives: {metrics['object_fp']}  
    False Negatives: {metrics['object_fn']}
    Precision: {metrics['object_precision']:.3f}
    Recall: {metrics['object_recall']:.3f}
    F1-Score: {metrics['object_f1']:.3f}

    DETECTION INFO:
    Detected Objects: {metrics['detected_objects']}
    GT Objects: {metrics['ground_truth_objects']}
    IoU Threshold: {iou_threshold}
    Avg Matched IoU: {metrics.get('avg_matched_iou', 0):.3f}

    Inference Time: {prediction_result['inference_time']:.3f}s"""
            
            axes[1, 3].text(0.05, 0.95, metrics_text, transform=axes[1, 3].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 3].set_title('Performance Metrics', fontsize=14, fontweight='bold')
            axes[1, 3].axis('off')
        else:
            # Information display without ground truth
            info_text = f"""Prediction Info:
    Dirt Pixels: {prediction_result['dirt_pixels']:,}
    Dirt Percentage: {prediction_result['dirt_percentage']:.2f}%
    Confidence Threshold: {self.confidence_threshold}
    Inference Time: {prediction_result['inference_time']:.3f}s
    Image Size: {prediction_result['original_size']}"""
            
            axes[1, 2].text(0.05, 0.95, info_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=12, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 2].set_title('Prediction Information', fontsize=14, fontweight='bold')
            axes[1, 2].axis('off')
        
        # Add labels information if available
        if labels:
            label_text = f"""Labels Info:
    Total Objects: {labels.get('total_objects', 'N/A')}
    Dirt Objects: {labels.get('dirt_objects', 'N/A')}
    Clean Objects: {labels.get('clean_objects', 'N/A')}"""
            
            title_text = f"Sample from: {labels.get('source_image', 'Unknown')}"
            fig.suptitle(title_text, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.debug(f"Visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    
    def predict_and_compare_mask(self, input_path:str, output_dir:str):
        # Check if it's a dataset sample directory
        required_files = ['synthetic_dirty_patch.png', 'segmentation_mask_patch.png', 'labels_patch.json']

        if all((input_path / file).exists() for file in required_files):
            # Single dataset sample
            print(f"Running inference on dataset sample: {input_path}")
            try:
                image, ground_truth, labels = load_ground_truth_data(str(input_path), self.logger)
                prediction_result = self.predict(image)
                
                # Save visualization with ground truth comparison
                output_path = output_dir / f"{input_path.name}_inference.png"
                self.visualize_prediction(
                    image, prediction_result, ground_truth, labels,
                    save_path=str(output_path), show_plot=False
                )
                
                # Calculate and display metrics
                metrics = calculate_metrics(prediction_result['binary_prediction'], ground_truth)
                
                print(f"Inference completed!")
                print(f"Metrics:")
                print(f"  Pixel IoU: {metrics['pixel_iou']:.3f}")
                print(f"  Pixel Dice: {metrics['pixel_dice']:.3f}")
                print(f"  Object Precision: {metrics['object_precision']:.3f}")
                print(f"  Object Recall: {metrics['object_recall']:.3f}")
                print(f"  Object F1-Score: {metrics['object_f1']:.3f}")
                print(f"  Dirt percentage: {prediction_result['dirt_percentage']:.2f}%")
                print(f"  Inference time: {prediction_result['inference_time']:.3f}s")
                print(f"Results saved to: {output_path}")
                
            except Exception as e:
                print(f"Error processing dataset sample: {e}")
        else:
            print("Directory does not contain required dataset files.")
            print("Use --batch_mode for dataset directory with multiple samples.")

    def batch_predict_and_compare_mask(self, ):
        pass    

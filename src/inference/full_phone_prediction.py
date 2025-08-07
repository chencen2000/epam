import json
import time
from pathlib import Path
from logging import Logger
from typing import Optional, Dict, List, Tuple

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.target_labels import TargetLabels
from src.core.logger_config import setup_application_logger
from src.inference.single_predictor import SingleImagePredictor
from src.synthesis.patch_generator import PatchGenerator
from src.synthesis.boundary_detector import BoundaryDetector


class FullScreenPredictor:
    def __init__(self,
                 patch_generator: PatchGenerator, 
                 boundary_detector: BoundaryDetector, 
                 single_image_predictor: SingleImagePredictor,
                 app_logger:Optional[Logger]=None, *args, **kwargs):

        self.patch_gen = patch_generator
        self.boundary_detector = boundary_detector
        self.single_image_predictor = single_image_predictor

        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('FullScreenPredictor')

    def predict(self, image, return_raw = False):
        return  self.single_image_predictor.predict(image, return_raw=return_raw)


    def batch_predict_full_phone_images(self, image_dir: str, 
                                      patch_size: int = 512,
                                      overlap: float = 0.2,
                                      output_dir: str = None,
                                      image_extensions: List[str] = None,
                                      max_images: int = None) -> List[Dict]:
        """
        Run batch inference on multiple full phone images - UPDATED FOR MULTI-CLASS
        
        Args:
            image_dir: Directory containing full phone images
            patch_size: Size of patches for inference
            overlap: Overlap ratio between patches
            output_dir: Directory to save results
            image_extensions: List of image extensions to process
            max_images: Maximum number of images to process
            
        Returns:
            List of prediction results for all processed images
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
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
        self.logger.info(f"{image_files}")
        
        self.logger.info(f"Found {len(image_files)} full phone images for multi-class processing")
        
        if output_dir is None:
            output_dir = image_path / 'batch_full_phone_results'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"{image_files = }")
        
        results = []
        
        for i, image_file in enumerate(tqdm(image_files, desc="Processing full phone images")):
            # if i > 0: continue
            try:
                self.logger.info(f"\nProcessing {i+1}/{len(image_files)}: {image_file.name}")
                
                # Run full phone prediction
                result = self.single_prediction_pipeline(
                    str(image_file),
                    patch_size=patch_size,
                    overlap=overlap,
                    save_results=True,
                    output_dir=str(output_dir / image_file.stem),
                    show_plot=False
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error processing {image_file.name}: {e}")
                continue
        
        # Save batch summary
        if results:
            self.save_batch_summary(results, str(output_dir / "batch_full_phone_summary.json"))
            self.logger.info(f"\nMulti-class batch processing completed!")
            self.logger.info(f"Processed {len(results)} full phone images successfully")
            self.logger.info(f"Results saved to: {output_dir}")
        
        return results
    

    def single_prediction_pipeline(self, image_path: str, 
                                  patch_size: int = 1024,
                                  overlap: float = 0.2,
                                  save_results: bool = True,
                                  output_dir: str = None,
                                  show_plot: bool = True,
                                  min_dirt_threshold: float = 0.01,
                                  **kwargs) -> Dict:
        """
        Complete pipeline for predicting dirt on full phone images - UPDATED FOR MULTI-CLASS
        
        Args:
            image_path: Path to the full phone image
            patch_size: Size of patches for inference
            overlap: Overlap ratio between patches (0-1)
            save_results: Whether to save visualization and results
            output_dir: Directory to save results
            show_plot: Whether to display the plot
            min_dirt_threshold: Minimum dirt percentage to consider a patch as having dirt
            **kwargs: Additional keyword arguments (for protocol compatibility)
            
        Returns:
            Complete prediction results with aggregated multi-class data
        """
        # Load the full phone image
        self.logger.info(f"Loading full phone image: {image_path}")
        full_image = self.single_image_predictor.image_operations.load_image_color(image_path)

        # todo: remove the hardcoded values
        if full_image.shape[0] > 1792: 
            full_image = self.single_image_predictor.image_operations.resize_image(full_image, 2)

        image_path = Path(image_path)
        
        # Detect mobile screen boundaries
        self.logger.info("Detecting mobile screen boundaries...")
        screen_x, screen_y, screen_w, screen_h = self.boundary_detector.detect_mobile_boundaries(
            full_image.copy(), draw=False
        )
        
        self.logger.info(f"Detected screen area: x={screen_x}, y={screen_y}, w={screen_w}, h={screen_h}")
        
        # Extract screen region
        screen_image = full_image[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w]
        
        # Create patches from screen area
        self.logger.info(f"Creating patches from screen area (patch_size={patch_size}, overlap={overlap})...")
        patches_data = self.patch_gen._create_prediction_patch_from_screen(screen_image, patch_size, overlap)
        
        self.logger.info(f"Created {len(patches_data)} patches for multi-class inference")
        
        # Run inference on all patches
        self.logger.info("Running multi-class inference on patches...")
        patch_results = self.prediction_full_screen_patch(patches_data, (patch_size, patch_size))
        
        # Aggregate results back to full screen coordinates
        self.logger.info("Aggregating multi-class patch results...")
        aggregated_results = self.aggregate_patch_results(
            patch_results, screen_image.shape, patch_size, overlap
        )
        
        # Calculate overall statistics
        overall_stats = self.calculate_overall_stats(
            aggregated_results, screen_image.shape, patch_results, min_dirt_threshold
        )
        
        # Prepare comprehensive results
        results = {
            'image_info': {
                'filename': image_path.name,
                'full_image_size': full_image.shape[:2],
                'screen_coordinates': (screen_x, screen_y, screen_w, screen_h),
                'screen_size': screen_image.shape[:2],
                'total_patches': len(patches_data)
            },
            'boundary_detection': {
                'screen_area': {
                    'x': screen_x, 'y': screen_y, 'width': screen_w, 'height': screen_h
                },
                'screen_percentage': (screen_w * screen_h) / (full_image.shape[0] * full_image.shape[1]) * 100
            },
            'patch_analysis': {
                'patch_size': patch_size,
                'overlap': overlap,
                'total_patches': len(patches_data),
                'patches_with_defects': len([p for p in patch_results if self._get_patch_total_defects(p) > min_dirt_threshold]),
                'patch_results': patch_results
            },
            'aggregated_prediction': aggregated_results,
            'overall_statistics': overall_stats,
            'model_info': {
                'architecture': self.single_image_predictor.config.get('model_architecture', 'standard'),
                'confidence_threshold': self.single_image_predictor.confidence_threshold,
                'device': str(self.single_image_predictor.device),
                'num_classes': self.single_image_predictor.num_classes,
                'class_names': self.single_image_predictor.class_names
            }
        }
        
        # Prepare output directory
        if output_dir is None:
            output_dir = image_path.parent / 'full_phone_inference_results'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        if save_results:
            suction_cup_path = output_dir / f"{image_path.stem}_suction_cup.png"
            new_image = self.single_image_predictor.image_operations.load_image(str(image_path))
            new_img_resized = self.single_image_predictor.image_operations.resize_image(new_image, 25)
            self.single_image_predictor.predict_and_save_suction_cup(
                suction_cup_path, 
                new_img_resized
            )

            self.single_image_predictor.check_power_on_device_algo(
                    output_dir, new_image, image_path.stem
            )
            
            # Save comprehensive visualization
            vis_path = output_dir / f"{image_path.stem}_full_phone_multiclass_analysis.png"
            self.visualize_full_phone_prediction(
                full_image, screen_image, results,
                save_path=str(vis_path), 
                show_plot=show_plot
            )
            
            # Save detailed results as JSON
            results_path = output_dir / f"{image_path.stem}_full_analysis.json"
            self.logger.debug(f"Saving full analysis json at {results_path}")
            # self.single_image_predictor.save_prediction_results(results, str(results_path))
            
            # Save class prediction map
            class_pred_path = output_dir / f"{image_path.stem}_screen_class_prediction.png"
            self.logger.debug(f"Saving screen class predictions at {class_pred_path}")
            cv2.imwrite(str(class_pred_path), aggregated_results['class_prediction'])
            
            # Save individual class masks
            self.logger.debug(f"Saving individual class mask")
            for cls in range(self.single_image_predictor.num_classes):
                class_name = self.single_image_predictor.class_names[cls]
                mask_path = output_dir / f"{image_path.stem}_screen_{class_name}_mask.png"
                cv2.imwrite(str(mask_path), aggregated_results['binary_masks'][class_name] * 255)
            
            # Save full image with overlay
            full_overlay_path = output_dir / f"{image_path.stem}_full_phone_multiclass_overlay.png"
            self.logger.debug(f"Saving full phone multiclass overlay at {full_overlay_path}")
            self._save_full_phone_overlay(full_image, results, str(full_overlay_path))
            
            self.logger.info(f"\nResults saved to: {output_dir}")
            self.logger.info(f"- Full analysis: {vis_path.name}")
            self.logger.info(f"- Results JSON: {results_path.name}")
            self.logger.info(f"- Class prediction: {class_pred_path.name}")
            for cls in range(self.single_image_predictor.num_classes):
                class_name = self.single_image_predictor.class_names[cls]
                self.logger.info(f"- {class_name} mask saved")
            self.logger.info(f"- Full overlay: {full_overlay_path.name}")

            
        
        # Print comprehensive summary
        self.print_full_phone_summary(results)
        
        return results
    
    def _get_patch_total_defects(self, patch_result: Dict) -> float:
        """Calculate total defect percentage for a patch (all non-background classes)"""
        if 'error' in patch_result:
            return 0.0
        
        if 'class_statistics' in patch_result:
            total_defects = 0.0
            for cls in range(1, self.single_image_predictor.num_classes):  # Skip background
                class_name = self.single_image_predictor.class_names[cls]
                percentage_key = f'{class_name}_percentage'
                if percentage_key in patch_result['class_statistics']:
                    total_defects += patch_result['class_statistics'][percentage_key]
            return total_defects
        elif 'dirt_percentage' in patch_result:
            # Legacy binary result
            return patch_result['dirt_percentage']
        
        return 0.0
    
    def aggregate_patch_results(self, patch_results: List[Dict], 
                               screen_shape: Tuple[int, int], 
                               patch_size: int, overlap: float) -> Dict:
        """Aggregate patch predictions back to full screen coordinates - UPDATED FOR MULTI-CLASS"""
        screen_h, screen_w = screen_shape[:2]
        num_classes = self.single_image_predictor.num_classes
        
        # Initialize aggregation arrays for multi-class
        class_prediction_sum = np.zeros((num_classes, screen_h, screen_w), dtype=np.float32)
        class_probability_sum = np.zeros((num_classes, screen_h, screen_w), dtype=np.float32)
        count_matrix = np.zeros((screen_h, screen_w), dtype=np.float32)
        
        # Aggregate predictions from all patches
        for patch_result in patch_results:
            if 'error' in patch_result:
                continue
                
            x_start, y_start = patch_result['patch_coordinates']
            patch_h, patch_w = patch_result['patch_size']
            
            # Calculate end coordinates
            x_end = min(x_start + patch_w, screen_w)
            y_end = min(y_start + patch_h, screen_h)
            
            # Get patch results
            if 'class_prediction' in patch_result:
                # Multi-class prediction
                patch_class_pred = patch_result['class_prediction'][:y_end-y_start, :x_end-x_start]
                
                # Convert class prediction to one-hot for averaging
                for cls in range(num_classes):
                    class_mask = (patch_class_pred == cls).astype(np.float32)
                    class_prediction_sum[cls, y_start:y_end, x_start:x_end] += class_mask
                
                # Add class probabilities
                if 'class_probabilities' in patch_result:
                    for cls in range(num_classes):
                        class_name = self.single_image_predictor.class_names[cls]
                        if class_name in patch_result['class_probabilities']:
                            patch_probs = patch_result['class_probabilities'][class_name][:y_end-y_start, :x_end-x_start]
                            class_probability_sum[cls, y_start:y_end, x_start:x_end] += patch_probs
            else:
                # Legacy binary prediction - treat as dirt class
                patch_binary = patch_result['binary_prediction'][:y_end-y_start, :x_end-x_start]
                patch_probs = patch_result['probabilities'][:y_end-y_start, :x_end-x_start]
                
                # Background class (inverse of dirt)
                class_prediction_sum[0, y_start:y_end, x_start:x_end] += (1 - patch_binary.astype(np.float32))
                class_probability_sum[0, y_start:y_end, x_start:x_end] += (1 - patch_probs)
                
                # Dirt class
                class_prediction_sum[1, y_start:y_end, x_start:x_end] += patch_binary.astype(np.float32)
                class_probability_sum[1, y_start:y_end, x_start:x_end] += patch_probs
            
            # Update count matrix
            count_matrix[y_start:y_end, x_start:x_end] += 1
        
        # Avoid division by zero
        count_matrix = np.maximum(count_matrix, 1)
        
        # Calculate averaged results
        avg_class_probabilities = {}
        avg_class_predictions = {}
        
        for cls in range(num_classes):
            class_name = self.single_image_predictor.class_names[cls]
            
            # Average probabilities and predictions
            avg_prob = class_probability_sum[cls] / count_matrix
            avg_pred = class_prediction_sum[cls] / count_matrix
            
            avg_class_probabilities[f'class_{cls}'] = avg_prob
            avg_class_probabilities[class_name] = avg_prob
            avg_class_predictions[f'class_{cls}'] = avg_pred
            avg_class_predictions[class_name] = avg_pred
        
        # Create final class prediction using argmax
        # Stack all class probabilities and take argmax
        prob_stack = np.stack([avg_class_probabilities[f'class_{cls}'] for cls in range(num_classes)], axis=0)
        final_class_prediction = np.argmax(prob_stack, axis=0).astype(np.uint8)
        
        # Create binary masks for each class
        binary_masks = {}
        class_statistics = {}
        total_pixels = final_class_prediction.size
        
        for cls in range(num_classes):
            class_name = self.single_image_predictor.class_names[cls]
            binary_mask = (final_class_prediction == cls).astype(np.uint8)
            class_pixels = np.sum(binary_mask)
            class_percentage = (class_pixels / total_pixels) * 100
            
            binary_masks[f'class_{cls}'] = binary_mask
            binary_masks[class_name] = binary_mask
            
            class_statistics[f'class_{cls}_pixels'] = int(class_pixels)
            class_statistics[f'class_{cls}_percentage'] = float(class_percentage)
            class_statistics[f'{class_name}_pixels'] = int(class_pixels)
            class_statistics[f'{class_name}_percentage'] = float(class_percentage)
        
        return {
            # Multi-class results
            'class_prediction': final_class_prediction,
            'class_probabilities': avg_class_probabilities,
            'binary_masks': binary_masks,
            'class_statistics': class_statistics,
            'coverage_count': count_matrix,
            'screen_size': screen_shape[:2],
            'num_classes': num_classes,
            
            # Legacy compatibility
            'binary_prediction': binary_masks['dirt'],  # For backward compatibility
            'probabilities': avg_class_probabilities['dirt'],  # For backward compatibility
            'dirt_pixels': class_statistics['dirt_pixels'],
            'dirt_percentage': class_statistics['dirt_percentage']
        }

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if needed"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            self.logger.debug(f"Converted RGB patch {image.shape} to grayscale {gray_image.shape}")
            return gray_image
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Already single channel, squeeze
            return image.squeeze(-1)
        elif len(image.shape) == 2:
            # Already grayscale
            return image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    
    def prediction_full_screen_patch(self, patches_data: List[Tuple], target_size:Tuple[int, int] = (1792, 1792)) -> List[Dict]:
        """Run multi-class inference on all patches - FIXED FOR GRAYSCALE"""
        patch_results = []
        
        for i, (patch_img, patch_mask, patch_annotations, x_start, y_start) in enumerate(tqdm(patches_data, desc="Processing patches")):
            try:
                # QUICK FIX: Convert patch to grayscale before inference
                gray_patch = self._convert_to_grayscale(patch_img)
                
                # Run prediction on grayscale patch
                patch_result = self.single_image_predictor.predict(gray_patch, target_size, return_raw=False)
                
                # Add patch metadata
                patch_result.update({
                    'patch_id': i,
                    'patch_coordinates': (x_start, y_start),
                    'patch_size': gray_patch.shape[:2]  # Use grayscale shape
                })
                
                patch_results.append(patch_result)

                
                
            except Exception as e:
                self.logger.warning(f"Error processing patch {i}: {e}")
                # Create dummy result for failed patch
                dummy_shape = patch_img.shape[:2] if len(patch_img.shape) >= 2 else (512, 512)
                patch_results.append({
                    'patch_id': i,
                    'patch_coordinates': (x_start, y_start),
                    'patch_size': dummy_shape,
                    'class_prediction': np.zeros(dummy_shape, dtype=np.uint8),
                    'class_probabilities': {name: np.zeros(dummy_shape, dtype=np.float32) 
                                        for name in self.single_image_predictor.class_names},
                    'binary_masks': {name: np.zeros(dummy_shape, dtype=np.uint8) 
                                for name in self.single_image_predictor.class_names},
                    'class_statistics': {f'{name}_pixels': 0 for name in self.single_image_predictor.class_names},
                    'inference_time': 0.0,
                    'error': str(e)
                })
        
        return patch_results

    def calculate_overall_stats(self, aggregated_results: Dict, 
                               screen_shape: Tuple[int, int],
                               patch_results: List[Dict],
                               min_dirt_threshold: float) -> Dict:
        """Calculate overall statistics for the full phone analysis - UPDATED FOR MULTI-CLASS"""
        screen_h, screen_w = screen_shape[:2]
        total_screen_pixels = screen_h * screen_w
        num_classes = aggregated_results.get('num_classes', 3)
        
        # Calculate patch statistics
        valid_patches = [p for p in patch_results if 'error' not in p]
        
        # Multi-class patch statistics
        patch_class_stats = {f'class_{cls}': [] for cls in range(num_classes)}
        patch_class_stats.update({name: [] for name in self.single_image_predictor.class_names})
        
        for patch in valid_patches:
            if 'class_statistics' in patch:
                for cls in range(num_classes):
                    class_name = self.single_image_predictor.class_names[cls]
                    key = f'{class_name}_percentage'
                    if key in patch['class_statistics']:
                        patch_class_stats[f'class_{cls}'].append(patch['class_statistics'][key])
                        patch_class_stats[class_name].append(patch['class_statistics'][key])
            elif 'dirt_percentage' in patch:
                # Legacy binary patch - treat as dirt
                patch_class_stats['class_1'].append(patch['dirt_percentage'])
                patch_class_stats['dirt'].append(patch['dirt_percentage'])
                patch_class_stats['class_0'].append(100 - patch['dirt_percentage'])  # Background
                patch_class_stats['background'].append(100 - patch['dirt_percentage'])
        
        # Defect threshold patches (patches with any defects above threshold)
        defect_threshold_patches = []
        for patch in valid_patches:
            total_defects = self._get_patch_total_defects(patch)
            if total_defects > min_dirt_threshold:
                defect_threshold_patches.append(patch)
        
        patch_inference_times = [p.get('inference_time', 0) for p in valid_patches]
        
        # Calculate region analysis for each defect class
        region_analysis = {}
        for cls in range(1, num_classes):  # Skip background
            class_name = self.single_image_predictor.class_names[cls]
            if class_name in aggregated_results['binary_masks']:
                class_mask = aggregated_results['binary_masks'][class_name]
                region_analysis[class_name] = self.single_image_predictor.region_analyzer.analyze_dirt_regions(class_mask)
        
        # Combined defects analysis
        combined_defects = np.zeros_like(aggregated_results['class_prediction'], dtype=np.uint8)
        for cls in range(1, num_classes):  # Combine all defect classes
            combined_defects = np.logical_or(combined_defects, aggregated_results['class_prediction'] == cls)
        region_analysis['combined_defects'] = self.single_image_predictor.region_analyzer.analyze_dirt_regions(combined_defects.astype(np.uint8))
        
        # Build comprehensive stats
        stats = {
            'screen_analysis': {
                'total_pixels': total_screen_pixels,
                'class_statistics': aggregated_results['class_statistics'].copy()
            },
            'patch_statistics': {
                'total_patches': len(patch_results),
                'valid_patches': len(valid_patches),
                'patches_with_defects': len(defect_threshold_patches),
                'class_patch_stats': {}
            },
            'performance_metrics': {
                'total_inference_time': sum(patch_inference_times),
                'avg_patch_inference_time': np.mean(patch_inference_times) if patch_inference_times else 0,
                'patches_per_second': len(valid_patches) / sum(patch_inference_times) if sum(patch_inference_times) > 0 else 0
            },
            'region_analysis': region_analysis,
            'confidence_analysis': {
                'coverage_stats': {
                    'max_coverage': float(np.max(aggregated_results['coverage_count'])),
                    'min_coverage': float(np.min(aggregated_results['coverage_count'])),
                    'avg_coverage': float(np.mean(aggregated_results['coverage_count']))
                }
            }
        }
        
        # Add per-class patch statistics
        for cls in range(num_classes):
            class_name = self.single_image_predictor.class_names[cls]
            class_patch_data = patch_class_stats.get(class_name, [])
            
            if class_patch_data:
                stats['patch_statistics']['class_patch_stats'][class_name] = {
                    'avg_percentage': float(np.mean(class_patch_data)),
                    'max_percentage': float(np.max(class_patch_data)),
                    'min_percentage': float(np.min(class_patch_data)),
                    'std_percentage': float(np.std(class_patch_data))
                }
        
        # Add confidence analysis for each class
        for cls in range(num_classes):
            class_name = self.single_image_predictor.class_names[cls]
            if class_name in aggregated_results['class_probabilities']:
                class_probs = aggregated_results['class_probabilities'][class_name]
                stats['confidence_analysis'][f'{class_name}_confidence'] = {
                    'avg_confidence': float(np.mean(class_probs)),
                    'max_confidence': float(np.max(class_probs)),
                    'high_confidence_pixels': int(np.sum(class_probs > 0.8)),
                    'medium_confidence_pixels': int(np.sum((class_probs > 0.5) & (class_probs <= 0.8))),
                    'low_confidence_pixels': int(np.sum(class_probs <= 0.5))
                }
        
        return stats
    
    def print_full_phone_summary(self, results: Dict):
        """Print comprehensive summary of full phone analysis - UPDATED FOR MULTI-CLASS"""
        stats = results['overall_statistics']
        
        self.logger.info(f"\n" + "="*80)
        self.logger.info(f"MULTI-CLASS FULL PHONE DIRT DETECTION ANALYSIS")
        self.logger.info(f"="*80)
        self.logger.info(f"Image: {results['image_info']['filename']}")
        self.logger.info(f"Full Image Size: {results['image_info']['full_image_size'][1]}×{results['image_info']['full_image_size'][0]}")
        self.logger.info(f"Screen Size: {results['image_info']['screen_size'][1]}×{results['image_info']['screen_size'][0]}")
        self.logger.info(f"Screen Coverage: {results['boundary_detection']['screen_percentage']:.1f}% of full image")
        self.logger.info(f"Model Classes: {results['aggregated_prediction'].get('num_classes', 3)}")
        self.logger.info(f"")
        
        self.logger.info(f"CLASS-WISE DETECTION RESULTS:")
        class_stats = stats['screen_analysis']['class_statistics']
        for cls in range(results['aggregated_prediction'].get('num_classes', 3)):
            class_name = self.single_image_predictor.class_names[cls]
            pixels_key = f'{class_name}_pixels'
            percentage_key = f'{class_name}_percentage'
            
            if pixels_key in class_stats and percentage_key in class_stats:
                pixels = class_stats[pixels_key]
                percentage = class_stats[percentage_key]
                self.logger.info(f"  {class_name.title()}: {percentage:.2f}% ({pixels:,} pixels)")
        
        # Calculate total defect coverage
        total_defects = 0
        for cls in range(1, results['aggregated_prediction'].get('num_classes', 3)):  # Skip background
            class_name = self.single_image_predictor.class_names[cls]
            percentage_key = f'{class_name}_percentage'
            if percentage_key in class_stats:
                total_defects += class_stats[percentage_key]
        
        self.logger.info(f"  Total Defects: {total_defects:.2f}%")
        self.logger.info(f"")
        
        self.logger.info(f"PATCH ANALYSIS:")
        self.logger.info(f"  Total Patches Processed: {stats['patch_statistics']['total_patches']}")
        self.logger.info(f"  Patches with Defects: {stats['patch_statistics']['patches_with_defects']}")
        
        # Per-class patch statistics
        for cls in range(1, results['aggregated_prediction'].get('num_classes', 3)):  # Skip background
            class_name = self.single_image_predictor.class_names[cls]
            if class_name in stats['patch_statistics']['class_patch_stats']:
                class_patch_stats = stats['patch_statistics']['class_patch_stats'][class_name]
                self.logger.info(f"  {class_name.title()} Patches - Avg: {class_patch_stats['avg_percentage']:.2f}%, Max: {class_patch_stats['max_percentage']:.2f}%")
        
        self.logger.info(f"")
        self.logger.info(f"DEFECT REGIONS:")
        region_analysis = stats['region_analysis']
        
        for cls in range(1, results['aggregated_prediction'].get('num_classes', 3)):  # Skip background
            class_name = self.single_image_predictor.class_names[cls]
            if class_name in region_analysis:
                class_regions = region_analysis[class_name]
                if class_regions['num_regions'] > 0:
                    self.logger.info(f"  {class_name.title()} Regions: {class_regions['num_regions']}")
                    self.logger.info(f"    Largest: {class_regions['largest_region_area']:,} pixels")
                    self.logger.info(f"    Average: {class_regions['average_region_area']:.1f} pixels")
                else:
                    self.logger.info(f"  {class_name.title()}: No distinct regions detected")
        
        # Combined regions
        if 'combined_defects' in region_analysis and region_analysis['combined_defects']['num_regions'] > 0:
            combined = region_analysis['combined_defects']
            self.logger.info(f"  Combined Defect Regions: {combined['num_regions']}")
        
        self.logger.info(f"")
        self.logger.info(f"PERFORMANCE METRICS:")
        self.logger.info(f"  Total Processing Time: {stats['performance_metrics']['total_inference_time']:.2f} seconds")
        self.logger.info(f"  Average Per Patch: {stats['performance_metrics']['avg_patch_inference_time']:.3f} seconds")
        self.logger.info(f"  Processing Speed: {stats['performance_metrics']['patches_per_second']:.1f} patches/second")
        self.logger.info(f"  Device Used: {results['model_info']['device']}")
        self.logger.info(f"="*80)

    def _save_full_phone_overlay(self, full_image: np.ndarray, results: Dict, save_path: str):
        """Save full phone image with multi-class dirt overlay"""
        overlay_image = self._create_full_phone_dirt_overlay(full_image, results)
        cv2.imwrite(save_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Full phone multi-class overlay saved to: {save_path}")
    
    def _create_full_phone_dirt_overlay(self, full_image: np.ndarray, results: Dict) -> np.ndarray:
        """Create full phone image with multi-class dirt regions mapped back from screen coordinates"""
        full_overlay = full_image.copy()
        
        # Get screen coordinates and class prediction
        screen_coords = results['boundary_detection']['screen_area']
        class_prediction = results['aggregated_prediction']['class_prediction']
        
        # Create overlay on screen region
        screen_x, screen_y = screen_coords['x'], screen_coords['y']
        screen_w, screen_h = screen_coords['width'], screen_coords['height']
        
        # Resize class prediction to match actual screen size if needed
        if class_prediction.shape != (screen_h, screen_w):
            class_prediction = cv2.resize(class_prediction, (screen_w, screen_h), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Apply multi-class overlay to screen region
        screen_region = full_overlay[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w]

        # Dirt overlay (green)
        condensation_mask = class_prediction == 1
        if np.any(condensation_mask):
            screen_region[condensation_mask] = screen_region[condensation_mask] * 0.3 + np.array([0, 255, 0]) * 0.7
        
        # Dirt overlay (red)
        dirt_mask = class_prediction == 2
        if np.any(dirt_mask):
            screen_region[dirt_mask] = screen_region[dirt_mask] * 0.3 + np.array([255, 0, 0]) * 0.7
        
        # Scratches overlay (blue)
        scratch_mask = class_prediction == 3
        if np.any(scratch_mask):
            screen_region[scratch_mask] = screen_region[scratch_mask] * 0.3 + np.array([0, 0, 255]) * 0.7
        
        full_overlay[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w] = screen_region
        
        # Draw screen boundary
        cv2.rectangle(full_overlay, (screen_x, screen_y), 
                     (screen_x + screen_w, screen_y + screen_h),
                     self.single_image_predictor.colors['screen_boundary'], 2)
        
        return full_overlay
    
    def visualize_full_phone_prediction(self, full_image: np.ndarray, 
                                      screen_image: np.ndarray,
                                      results: Dict,
                                      save_path: str = None, 
                                      show_plot: bool = True) -> plt.Figure:
        """Create comprehensive visualization for full phone multi-class prediction"""
        fig = plt.figure(figsize=(24, 16))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(4, 5, height_ratios=[2, 2, 2, 1], width_ratios=[1, 1, 1, 1, 1], hspace=0.2)

        # ================== ROW 1 =============================
        
        # Row 1: Basic images and class predictions
        # Full phone image with detected boundary
        ax1 = fig.add_subplot(gs[0, 0])
        full_with_boundary = full_image.copy()
        screen_coords = results['boundary_detection']['screen_area']
        cv2.rectangle(full_with_boundary, 
                     (screen_coords['x'], screen_coords['y']),
                     (screen_coords['x'] + screen_coords['width'], 
                      screen_coords['y'] + screen_coords['height']),
                     self.single_image_predictor.colors['screen_boundary'], 3)
        ax1.imshow(full_with_boundary)
        ax1.set_title(f'Full Phone Image\n(Screen: {screen_coords["width"]}×{screen_coords["height"]})', 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Extracted screen region
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(screen_image)
        ax2.set_title('Extracted Screen Region', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Multi-class prediction visualization
        ax3 = fig.add_subplot(gs[0, 2])
        class_pred_colored = self.single_image_predictor.create_multiclass_visualization(
            results['aggregated_prediction']['class_prediction']
        )
        ax3.imshow(class_pred_colored)
        ax3.set_title('Multi-Class Prediction\n(Black=BG, Green=Condensation, \nRed=Dirt, Blue=Scratch)', 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Combined defect probability
        ax4 = fig.add_subplot(gs[0, 3])
        # dirt_prob = results['aggregated_prediction']['class_probabilities']['dirt']
        # scratch_prob = results['aggregated_prediction']['class_probabilities']['scratches']
        # combined_prob = np.maximum(dirt_prob, scratch_prob)
        probs = [results['aggregated_prediction']['class_probabilities'][label.value] 
                for label in TargetLabels]
        combined_prob = np.maximum.reduce(probs)
        prob_display = ax4.imshow(combined_prob, cmap='hot', vmin=0, vmax=1)
        ax4.set_title(
            'Max Defect Probability\n(Condensation or Dirt \nor Scratches)',
            fontsize=12, 
            fontweight='bold'
        )
        ax4.axis('off')
        plt.colorbar(prob_display, ax=ax4,)
        
        # Screen with multi-class overlay
        ax5 = fig.add_subplot(gs[0, 4])
        overlay_screen = self.single_image_predictor.create_multiclass_overlay(
            screen_image, results['aggregated_prediction']['class_prediction']
        )
        ax5.imshow(overlay_screen)
        ax5.set_title('Screen Multi-Class Overlay', fontsize=12, fontweight='bold')
        ax5.axis('off')

        # ================== ROW 2 =============================

        # def create
        
        # Row 2: Individual class visualizations
        # Dirt probability map
        ax6 = fig.add_subplot(gs[1, 0])
        dirt_prob = probs[TargetLabels.get_by_value("dirt").index]
        dirt_display = ax6.imshow(dirt_prob, cmap='Reds', vmin=0, vmax=1)
        ax6.set_title(
            f'Dirt Probability\n({results["aggregated_prediction"]["class_statistics"]["dirt_percentage"]:.1f}%)', 
            fontsize=12, 
            fontweight='bold'
        )
        ax6.axis('off')
        plt.colorbar(dirt_display, ax=ax6,)

        # Dirt binary mask
        ax8 = fig.add_subplot(gs[1, 1])
        ax8.imshow(results['aggregated_prediction']['binary_masks']['dirt'], cmap='Reds', vmin=0, vmax=1)
        ax8.set_title('Dirt Binary Mask', fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # Scratches probability map
        ax7 = fig.add_subplot(gs[1, 2])
        scratch_prob = probs[TargetLabels.get_by_value("scratch").index]
        scratch_display = ax7.imshow(scratch_prob, cmap='Blues', vmin=0, vmax=1)
        ax7.set_title(
            f'Scratch Probability\n({results["aggregated_prediction"]["class_statistics"]["scratch_percentage"]:.1f}%)', 
            fontsize=12,
            fontweight='bold'
        )
        ax7.axis('off')
        plt.colorbar(scratch_display, ax=ax7, )
        
        # Scratches binary mask
        ax9 = fig.add_subplot(gs[1, 3])
        ax9.imshow(results['aggregated_prediction']['binary_masks']['scratch'], cmap='Blues', vmin=0, vmax=1)
        ax9.set_title('Scratch Binary Mask', fontsize=12, fontweight='bold')
        ax9.axis('off')
        
        # Patch coverage visualization
        ax10 = fig.add_subplot(gs[1, 4])
        coverage_display = ax10.imshow(results['aggregated_prediction']['coverage_count'], cmap='viridis')
        ax10.set_title(f'Patch Coverage\n(Max: {int(np.max(results["aggregated_prediction"]["coverage_count"]))} overlaps)', 
                      fontsize=12, fontweight='bold')
        ax10.axis('off')
        plt.colorbar(coverage_display, ax=ax10, )

        # ================== ROW 3 =============================

        # Condensation probability map
        ax17 = fig.add_subplot(gs[2, 0])
        condensation_prob = probs[TargetLabels.get_by_value("condensation").index]
        scratch_display = ax17.imshow(condensation_prob, cmap='Greens', vmin=0, vmax=1)
        ax17.set_title(
            f'Condensation Probability\n({results["aggregated_prediction"]["class_statistics"]["condensation_percentage"]:.1f}%)', 
            fontsize=12,
            fontweight='bold'
        )
        ax17.axis('off')
        plt.colorbar(scratch_display, ax=ax17, )
        
        # Scratches binary mask
        ax18 = fig.add_subplot(gs[2, 1])
        ax18.imshow(results['aggregated_prediction']['binary_masks']['condensation'], cmap='Greens', vmin=0, vmax=1)
        ax18.set_title('Condensation Binary Mask', fontsize=12, fontweight='bold')
        ax18.axis('off')
        
        # Row 3: Analysis and statistics


        # Patch statistics visualization
        ax11 = fig.add_subplot(gs[2, 2])
        patch_results = results['patch_analysis']['patch_results']
        valid_patches = [p for p in patch_results if 'error' not in p]
        
        if valid_patches and 'class_statistics' in valid_patches[0]:
            dirt_percentages = [p['class_statistics'].get('dirt_percentage', 0) for p in valid_patches]
            scratch_percentages = [p['class_statistics'].get('scratch_percentage', 0) for p in valid_patches]
            
            ax11.hist([dirt_percentages, scratch_percentages], bins=15, alpha=0.7, 
                     label=['Dirt', 'Scratches'], color=['green', 'red'])
            ax11.set_xlabel('Percentage per Patch')
            ax11.set_ylabel('Number of Patches')
            ax11.set_title('Patch Defect Distribution', fontsize=12, fontweight='bold')
            ax11.legend()
            ax11.grid(True, alpha=0.3)
        else:
            ax11.text(0.5, 0.5, 'No patch statistics available', ha='center', va='center', 
                     transform=ax11.transAxes, fontsize=12)
            ax11.set_title('Patch Statistics', fontsize=12, fontweight='bold')
        
        # Class distribution pie chart
        ax12 = fig.add_subplot(gs[2, 3])
        class_stats = results['aggregated_prediction']['class_statistics']
        class_percentages = [
            class_stats.get('background_percentage', 0),
            class_stats.get('dirt_percentage', 0),
            class_stats.get('scratches_percentage', 0)
        ]
        colors = ['lightgray',  'red', 'green',]
        labels = ['Background', 'Dirt', 'Scratches']
        
        # Only show non-zero percentages
        non_zero_idx = [i for i, pct in enumerate(class_percentages) if pct > 0.01]
        if non_zero_idx:
            filtered_percentages = [class_percentages[i] for i in non_zero_idx]
            filtered_colors = [colors[i] for i in non_zero_idx]
            filtered_labels = [f'{labels[i]}\n{class_percentages[i]:.1f}%' for i in non_zero_idx]
            
            ax12.pie(filtered_percentages, labels=filtered_labels, colors=filtered_colors, 
                    autopct='%1.1f%%', startangle=90)
        
        ax12.set_title('Screen Class Distribution', fontsize=12, fontweight='bold')

        # Full phone with multi-class overlay
        ax15 = fig.add_subplot(gs[2, 4])
        full_with_overlay = self._create_full_phone_dirt_overlay(full_image, results)
        ax15.imshow(full_with_overlay)
        ax15.set_title('Full Phone Multi-Class Overlay', fontsize=12, fontweight='bold')
        ax15.axis('off')

        # ================== ROW 4 =============================
        
        # Region analysis visualization
        ax13 = fig.add_subplot(gs[3, 0])
        region_analysis = results['overall_statistics']['region_analysis']
        
        region_text = "REGION ANALYSIS:\n\n"
        for class_name in ['dirt', 'scratches']:
            if class_name in region_analysis:
                regions = region_analysis[class_name]
                region_text += f"{class_name.title()}:\n"
                region_text += f"  Regions: {regions['num_regions']}\n"
                if regions['num_regions'] > 0:
                    region_text += f"  Largest: {regions['largest_region_area']:,}px\n"
                    region_text += f"  Average: {regions['average_region_area']:.1f}px\n"
                region_text += "\n"
        
        if 'combined_defects' in region_analysis:
            combined = region_analysis['combined_defects']
            region_text += f"Combined Defects:\n"
            region_text += f"  Total Regions: {combined['num_regions']}\n"
        
        ax13.text(0.05, 0.95, region_text, transform=ax13.transAxes,
                 verticalalignment='top', fontsize=10, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        ax13.set_title('Region Analysis', fontsize=12, fontweight='bold', loc='left')
        ax13.axis('off')
        
        # Performance metrics
        ax14 = fig.add_subplot(gs[3, 1])
        perf_stats = results['overall_statistics']['performance_metrics']
        
        perf_text = f"""PERFORMANCE METRICS:

Total Time: {perf_stats['total_inference_time']:.2f}s
Avg/Patch: {perf_stats['avg_patch_inference_time']:.3f}s
Speed: {perf_stats['patches_per_second']:.1f} patches/s

COVERAGE ANALYSIS:
Max Overlap: {results['overall_statistics']['confidence_analysis']['coverage_stats']['max_coverage']:.0f}
Avg Overlap: {results['overall_statistics']['confidence_analysis']['coverage_stats']['avg_coverage']:.1f}

DEVICE: {results['model_info']['device']}
CLASSES: {results['model_info']['num_classes']}"""
        
        ax14.text(0.05, 0.95, perf_text, transform=ax14.transAxes,
                 verticalalignment='top', fontsize=10, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax14.set_title('Performance Metrics', fontsize=12, fontweight='bold', loc='left')
        ax14.axis('off')
        
        
        
        # Bottom row: Comprehensive summary
        ax16 = fig.add_subplot(gs[3, 2:])
        stats = results['overall_statistics']
        
        summary_text = f"""MULTI-CLASS FULL PHONE DIRT DETECTION ANALYSIS - {results['image_info']['filename']}
        
IMAGE INFO: Full Size: {results['image_info']['full_image_size'][1]}×{results['image_info']['full_image_size'][0]} | Screen Size: {results['image_info']['screen_size'][1]}×{results['image_info']['screen_size'][0]} | Screen Coverage: {results['boundary_detection']['screen_percentage']:.1f}%

CLASS DISTRIBUTION:"""
        
        for cls in range(results['aggregated_prediction'].get('num_classes', 3)):
            class_name = self.single_image_predictor.class_names[cls]
            pixels_key = f'{class_name}_pixels'
            percentage_key = f'{class_name}_percentage'
            
            if pixels_key in class_stats and percentage_key in class_stats:
                pixels = class_stats[pixels_key]
                percentage = class_stats[percentage_key]
                summary_text += f" | {class_name.title()}: {percentage:.2f}% ({pixels:,}px)"
        
        summary_text += f"""

PATCH ANALYSIS: Total: {stats['patch_statistics']['total_patches']} | Valid: {stats['patch_statistics']['valid_patches']} | With Defects: {stats['patch_statistics']['patches_with_defects']}"""
        
        # Add per-class patch statistics
        for cls in range(1, results['aggregated_prediction'].get('num_classes', 3)):  # Skip background
            class_name = self.single_image_predictor.class_names[cls]
            if class_name in stats['patch_statistics']['class_patch_stats']:
                class_patch_stats = stats['patch_statistics']['class_patch_stats'][class_name]
                summary_text += f" | {class_name.title()} Avg: {class_patch_stats['avg_percentage']:.2f}%"
        
        summary_text += f"""

REGION ANALYSIS:"""
        
        for cls in range(1, results['aggregated_prediction'].get('num_classes', 3)):  # Skip background
            class_name = self.single_image_predictor.class_names[cls]
            if class_name in region_analysis:
                class_regions = region_analysis[class_name]
                summary_text += f" | {class_name.title()}: {class_regions['num_regions']} regions"
                if class_regions['num_regions'] > 0:
                    summary_text += f" (largest: {class_regions['largest_region_area']:,}px)"
        
        summary_text += f"""

PERFORMANCE: Total Time: {stats['performance_metrics']['total_inference_time']:.2f}s | Avg/Patch: {stats['performance_metrics']['avg_patch_inference_time']:.3f}s | Speed: {stats['performance_metrics']['patches_per_second']:.1f} patches/s | Device: {results['model_info']['device']}"""
        
        ax16.text(0.02, 0.98, summary_text, transform=ax16.transAxes,
                 verticalalignment='top', fontsize=9, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax16.set_title('Comprehensive Multi-Class Analysis Summary', fontsize=14, fontweight='bold')
        ax16.axis('off')
        
        # Add main title
        fig.suptitle(f'Multi-Class Full Phone Dirt Detection Analysis - {results["image_info"]["filename"]}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Multi-class full phone visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    

    def save_batch_summary(self, results: List[Dict], save_path: str) -> None:
        """Save summary of batch full phone processing results - UPDATED FOR MULTI-CLASS"""
        if not results:
            return
        
        num_classes = results[0]['aggregated_prediction'].get('num_classes', 3)
        class_names = self.single_image_predictor.class_names
        
        # Extract multi-class statistics
        class_percentages = {class_name: [] for class_name in class_names}
        processing_times = [r['overall_statistics']['performance_metrics']['total_inference_time'] for r in results]
        patch_counts = [r['image_info']['total_patches'] for r in results]
        
        for result in results:
            class_stats = result['overall_statistics']['screen_analysis']['class_statistics']
            for class_name in class_names:
                percentage_key = f'{class_name}_percentage'
                if percentage_key in class_stats:
                    class_percentages[class_name].append(class_stats[percentage_key])
                else:
                    class_percentages[class_name].append(0.0)
        
        summary = {
            'batch_info': {
                'total_images': len(results),
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model_info': {
                    'num_classes': num_classes,
                    'class_names': class_names
                }
            },
            'class_analysis_summary': {},
            'processing_summary': {
                'total_processing_time': sum(processing_times),
                'avg_processing_time': np.mean(processing_times),
                'total_patches_processed': sum(patch_counts),
                'avg_patches_per_image': np.mean(patch_counts)
            },
            'individual_results': []
        }
        
        # Per-class summary statistics
        for class_name in class_names:
            class_data = class_percentages[class_name]
            if class_data:
                summary['class_analysis_summary'][class_name] = {
                    'avg_percentage': float(np.mean(class_data)),
                    'std_percentage': float(np.std(class_data)),
                    'min_percentage': float(np.min(class_data)),
                    'max_percentage': float(np.max(class_data)),
                    'median_percentage': float(np.median(class_data))
                }
        
        # Individual results
        for r in results:
            class_stats = r['overall_statistics']['screen_analysis']['class_statistics']
            region_stats = r['overall_statistics']['region_analysis']
            
            individual_result = {
                'filename': r['image_info']['filename'],
                'total_patches': r['image_info']['total_patches'],
                'processing_time': r['overall_statistics']['performance_metrics']['total_inference_time'],
                'class_statistics': {},
                'region_counts': {}
            }
            
            # Add per-class statistics
            for class_name in class_names:
                percentage_key = f'{class_name}_percentage'
                pixels_key = f'{class_name}_pixels'
                
                individual_result['class_statistics'][class_name] = {
                    'percentage': class_stats.get(percentage_key, 0.0),
                    'pixels': class_stats.get(pixels_key, 0)
                }
                
                # Add region counts for defect classes
                if class_name != 'background' and class_name in region_stats:
                    individual_result['region_counts'][class_name] = region_stats[class_name]['num_regions']
            
            summary['individual_results'].append(individual_result)
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Multi-class batch full phone summary saved to: {save_path}")
    
    def save_batch_full_phone_summary(self, results: List[Dict], save_path: str):
        """Legacy method name for backward compatibility"""
        self.save_batch_summary(results, save_path)


    def full_phone_prediction_with_ground_truth(self, image_path: str, 
                                               patch_size: int = 1024,
                                               overlap: float = 0.2,
                                               save_results: bool = True,
                                               output_dir: str = None,
                                               show_plot: bool = True,
                                               min_dirt_threshold: float = 0.01) -> Dict:
        """
        Complete pipeline for full phone prediction WITH ground truth comparison
        Reuses existing infrastructure efficiently
        """
        # Load full phone image and ground truth
        self.logger.info(f"Loading full phone image with ground truth: {image_path}")
        image, ground_truth, labels = self._load_full_phone_ground_truth_data(image_path)
        
        image_path = Path(image_path)
        
        # Run standard full phone prediction pipeline (reuse existing method)
        prediction_results = self.single_prediction_pipeline(
            str(image_path / "original_image.bmp"),  # Adjust based on your file naming
            patch_size=patch_size,
            overlap=overlap,
            save_results=True,  # We'll handle saving ourselves
            output_dir=output_dir,
            show_plot=False,
            min_dirt_threshold=min_dirt_threshold
        )
        
        # Calculate ground truth comparison metrics efficiently
        screen_coords = prediction_results['boundary_detection']['screen_area']
        predicted_screen = prediction_results['aggregated_prediction']['class_prediction']
        
        # Prepare ground truth for comparison (resize to match predicted screen area)
        gt_screen = self._prepare_ground_truth_for_comparison(
            ground_truth, screen_coords, image.shape[:2], predicted_screen.shape
        )
        
        # Calculate essential metrics using existing efficient methods
        comparison_metrics = self.single_image_predictor._calculate_essential_metrics(
            predicted_screen, gt_screen
        )
        
        # Add ground truth information to results
        prediction_results['ground_truth_comparison'] = {
            'metrics': comparison_metrics,
            'ground_truth_shape': gt_screen.shape,
            'ground_truth_file': labels.get('mask_info', {}).get('mask_file', 'unknown'),
            'comparison_area': 'screen_region_only'
        }
        
        # Prepare output directory
        if output_dir is None:
            output_dir = image_path.parent / 'full_phone_gt_results'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results with ground truth comparison
        self.logger.info(f"saving {save_results = }")
        if save_results:
            # Save enhanced visualization with ground truth
            vis_path = output_dir / f"{image_path.stem}_full_phone_gt_comparison.png"
            self.logger.info(f"{str(vis_path)}")
            self._visualize_full_phone_with_ground_truth(
                image, prediction_results, gt_screen, labels,
                save_path=str(vis_path), show_plot=show_plot
            )
            
            # Save detailed results
            results_path = output_dir / f"{image_path.stem}_gt_results.json"
            # self.single_image_predictor.save_prediction_results(prediction_results, str(results_path))
            
            self.logger.info(f"Ground truth comparison results saved to: {output_dir}")
        
        # Print ground truth comparison summary
        self._print_full_phone_gt_summary(prediction_results)

        response = {
           "ground_truth_comparison" : prediction_results['ground_truth_comparison'] 
        }
        
        return response

    def _load_full_phone_ground_truth_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        FIXED: Load full phone image with ground truth - with better error handling
        """
        data_path = Path(data_path)
        
        # More flexible file detection
        image_candidates = ['original_image.bmp', 'full_phone_image.bmp', 'synthetic_dirty_patch.bmp',
                        'original_image.png', 'full_phone_image.png']
        mask_candidates = [
            'segmentation_mask_multiclass.png',        # Full image multi-class mask
            'segmentation_mask_combined.png',          # Full image binary mask
            'segmentation_mask_patch_multiclass.png',  # Patch multi-class (fallback)
            'segmentation_mask_patch.png'              # Patch binary (fallback)
        ]
        labels_candidates = ['labels.json', 'labels_patch.json']
        
        # Find available files
        image_file = None
        mask_file = None
        labels_file = None
        
        for candidate in image_candidates:
            if (data_path / candidate).exists():
                image_file = candidate
                break
                
        for candidate in mask_candidates:
            if (data_path / candidate).exists():
                mask_file = candidate
                break
                
        for candidate in labels_candidates:
            if (data_path / candidate).exists():
                labels_file = candidate
                break
        
        if not all([image_file, mask_file, labels_file]):
            missing = []
            if not image_file: missing.append(f"image file from {image_candidates}")
            if not mask_file: missing.append(f"mask file from {mask_candidates}") 
            if not labels_file: missing.append(f"labels file from {labels_candidates}")
            raise FileNotFoundError(f"Missing files in {data_path}: {missing}")
        
        self.logger.debug(f"Loading: {image_file}, {mask_file}, {labels_file}")
        
        # Load files with better error handling
        try:
            image = cv2.imread(str(data_path / image_file))
            if image is None:
                raise ValueError(f"Could not load image: {data_path / image_file}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(data_path / mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {data_path / mask_file}")
            
            # FIXED: Only resize if image is larger than expected
            # TODO: update hardcoded values
            if mask.shape[0] > 1792 or mask.shape[1] > 1792:
                m_h, m_w = image.shape[:2]
                mask = cv2.resize(
                    mask, 
                    (m_w // 2, m_h // 2), 
                    interpolation=cv2.INTER_NEAREST
                )
                # mask = self.single_image_predictor.image_operations.resize_image(mask, 2)
            
            with open(data_path / labels_file, 'r') as f:
                labels = json.load(f)
            
        except Exception as e:
            raise ValueError(f"Error loading files from {data_path}: {e}")
        
        # Process mask
        mask_type = 'multiclass' if 'multiclass' in mask_file else 'binary'
        mask = self._process_full_phone_mask(mask, mask_type)
        
        # Log GT distribution for debugging
        unique_classes = np.unique(mask)
        total_pixels = mask.size
        self.logger.debug(f"GT mask {data_path.name} classes: {unique_classes}")
        for cls in unique_classes:
            pixel_count = np.sum(mask == cls)
            percentage = (pixel_count / total_pixels) * 100
            self.logger.debug(f"  Class {cls}: {percentage:.2f}% ({pixel_count} pixels)")
        
        labels['mask_info'] = {
            'mask_file': mask_file,
            'mask_type': mask_type,
            'image_file': image_file,
            'labels_file': labels_file,
            'loaded_from': str(data_path)
        }
        
        return image, mask, labels

    def _process_full_phone_mask(self, mask: np.ndarray, mask_type: str) -> np.ndarray:
        """
        FIXED: Process full phone mask with better class validation
        """
        if mask_type == 'multiclass':
            # Ensure values are in correct range [0, 1, 2, 3]
            unique_vals = np.unique(mask)
            self.logger.debug(f"Original mask values: {unique_vals}")
            
            # Check for unexpected values
            if np.max(mask) > 3:
                self.logger.warning(f"Mask contains values > 3: {unique_vals}. Clipping to [0, 3]")
                mask = np.clip(mask, 0, 3)
            
            return mask.astype(np.uint8)
        else:
            # Binary mask - normalize to [0, 1] and expand to multiclass if needed
            if np.max(mask) > 1:
                mask = (mask > 127).astype(np.uint8)
            
            # For compatibility with multiclass pipeline, map to dirt class (2)
            # Background (0) stays 0, foreground (1) becomes dirt (2)
            multiclass_mask = np.zeros_like(mask, dtype=np.uint8)
            multiclass_mask[mask == 1] = 2  # Map dirt to class 2
            
            return multiclass_mask

    def _prepare_ground_truth_for_comparison(self, ground_truth: np.ndarray, 
                                           screen_coords: Dict, 
                                           full_image_shape: Tuple,
                                           predicted_shape: Tuple) -> np.ndarray:
        """Prepare ground truth mask for comparison with prediction"""
        # Extract screen region from ground truth
        x, y = screen_coords['x'], screen_coords['y'] 
        w, h = screen_coords['width'], screen_coords['height']
        
        # Clip coordinates to image bounds
        x = max(0, min(x, full_image_shape[1]))
        y = max(0, min(y, full_image_shape[0]))
        w = min(w, full_image_shape[1] - x)
        h = min(h, full_image_shape[0] - y)
        
        gt_screen = ground_truth[y:y+h, x:x+w]
        
        # Resize to match prediction if needed
        if gt_screen.shape != predicted_shape:
            gt_screen = cv2.resize(gt_screen, (predicted_shape[1], predicted_shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        return gt_screen

    def _visualize_full_phone_with_ground_truth(self, full_image: np.ndarray, 
                                              results: Dict,
                                              gt_screen: np.ndarray,
                                              labels: Dict,
                                              save_path: str = None,
                                              show_plot: bool = True) -> plt.Figure:
        """
        Standalone visualization for full phone ground truth comparison
        Similar to patch GT comparison but designed for full phone context
        """
        try:
            fig, axes = plt.subplots(3, 5, figsize=(25, 15))
            
            # Get screen coordinates and predictions
            screen_coords = results['boundary_detection']['screen_area']
            predicted_screen = results['aggregated_prediction']['class_prediction']
            screen_x, screen_y = screen_coords['x'] * 2, screen_coords['y'] * 2
            screen_w, screen_h = screen_coords['width'] * 2, screen_coords['height'] * 2
            
            # Extract actual screen region from full image
            screen_image = full_image[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w]
            
            # ================= ROW 1: Basic Images and Boundaries =================
            
            # Full phone image with detected boundary
            full_with_boundary = full_image.copy()
            cv2.rectangle(full_with_boundary, (screen_x, screen_y), 
                         (screen_x + screen_w, screen_y + screen_h),
                         self.single_image_predictor.colors['screen_boundary'], 3)
            axes[0, 0].imshow(full_with_boundary)
            axes[0, 0].set_title(f'Full Phone Image\n(Size: {full_image.shape[1]}×{full_image.shape[0]})', 
                                fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')
            
            # Extracted screen region
            axes[0, 1].imshow(screen_image)
            axes[0, 1].set_title(f'Screen Region\n(Size: {screen_w}×{screen_h})', 
                                fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')
            
            # Ground truth screen area
            gt_colored = self.single_image_predictor.create_multiclass_visualization(gt_screen)
            axes[0, 2].imshow(gt_colored)
            axes[0, 2].set_title('Ground Truth\n(Screen Area Only)', 
                                fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')
            
            # Predicted screen area
            pred_colored = self.single_image_predictor.create_multiclass_visualization(predicted_screen)
            axes[0, 3].imshow(pred_colored)
            axes[0, 3].set_title('Prediction\n(Screen Area Only)', 
                                fontsize=12, fontweight='bold')
            axes[0, 3].axis('off')
            
            # Prediction vs Ground Truth comparison
            comparison = self.single_image_predictor.create_multiclass_comparison_mask(predicted_screen, gt_screen)
            axes[0, 4].imshow(comparison)
            axes[0, 4].set_title('Prediction vs GT\n(Yellow=Error)', 
                                fontsize=12, fontweight='bold')
            axes[0, 4].axis('off')
            
            # ================= ROW 2: Class-wise Analysis =================
            
            # Get class probabilities
            class_probs = results['aggregated_prediction']['class_probabilities']
            
            # Background probability
            bg_prob = class_probs['background']
            bg_display = axes[1, 0].imshow(bg_prob, cmap='gray', vmin=0, vmax=1)
            bg_percentage = results['aggregated_prediction']['class_statistics']['background_percentage']
            axes[1, 0].set_title(f'Background Probability\n({bg_percentage:.1f}%)', 
                                fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(bg_display, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Condensation probability and GT
            condensation_prob = class_probs['condensation']
            condensation_display = axes[1, 1].imshow(condensation_prob, cmap='Greens', vmin=0, vmax=1)
            condensation_percentage = results['aggregated_prediction']['class_statistics']['condensation_percentage']
            axes[1, 1].set_title(f'Condensation Probability\n({condensation_percentage:.1f}%)', 
                                fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            plt.colorbar(condensation_display, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Dirt probability
            dirt_prob = class_probs['dirt']
            dirt_display = axes[1, 2].imshow(dirt_prob, cmap='Reds', vmin=0, vmax=1)
            dirt_percentage = results['aggregated_prediction']['class_statistics']['dirt_percentage']
            axes[1, 2].set_title(f'Dirt Probability\n({dirt_percentage:.1f}%)', 
                                fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
            plt.colorbar(dirt_display, ax=axes[1, 2], fraction=0.046, pad=0.04)
            
            # Scratches probability
            scratch_prob = class_probs['scratch']
            scratch_display = axes[1, 3].imshow(scratch_prob, cmap='Blues', vmin=0, vmax=1)
            scratch_percentage = results['aggregated_prediction']['class_statistics']['scratch_percentage']
            axes[1, 3].set_title(f'Scratch Probability\n({scratch_percentage:.1f}%)', 
                                fontsize=12, fontweight='bold')
            axes[1, 3].axis('off')
            plt.colorbar(scratch_display, ax=axes[1, 3], fraction=0.046, pad=0.04)
            
            # Combined defect probability
            combined_defect_prob = np.maximum.reduce([
                class_probs['condensation'],
                class_probs['dirt'], 
                class_probs['scratch']
            ])
            combined_display = axes[1, 4].imshow(combined_defect_prob, cmap='hot', vmin=0, vmax=1)
            total_defects = condensation_percentage + dirt_percentage + scratch_percentage
            axes[1, 4].set_title(f'Combined Defects\n({total_defects:.1f}%)', 
                                fontsize=12, fontweight='bold')
            axes[1, 4].axis('off')
            plt.colorbar(combined_display, ax=axes[1, 4], fraction=0.046, pad=0.04)
            
            # ================= ROW 3: Overlays and Metrics =================
            
            # Screen with prediction overlay
            screen_overlay = self.single_image_predictor.create_multiclass_overlay(
                self.single_image_predictor.image_operations.resize_image(screen_image, 2), 
                predicted_screen
            )
            axes[2, 0].imshow(screen_overlay)
            axes[2, 0].set_title('Screen Prediction Overlay', fontsize=12, fontweight='bold')
            axes[2, 0].axis('off')
            
            # Full phone with prediction overlay
            full_overlay = self._create_full_phone_dirt_overlay(
                self.single_image_predictor.image_operations.resize_image(full_image, 2),
                results
            )
            axes[2, 1].imshow(full_overlay)
            axes[2, 1].set_title('Full Phone Overlay', fontsize=12, fontweight='bold')
            axes[2, 1].axis('off')
            
            # Ground truth binary masks for key classes
            # Dirt GT vs Pred
            gt_dirt_mask = (gt_screen == 2).astype(np.uint8)
            pred_dirt_mask = (predicted_screen == 2).astype(np.uint8)
            dirt_comparison = np.zeros((*gt_dirt_mask.shape, 3), dtype=np.uint8)
            # TP = white, FP = red, FN = blue, TN = black
            tp_dirt = (pred_dirt_mask == 1) & (gt_dirt_mask == 1)
            fp_dirt = (pred_dirt_mask == 1) & (gt_dirt_mask == 0)
            fn_dirt = (pred_dirt_mask == 0) & (gt_dirt_mask == 1)
            dirt_comparison[tp_dirt] = [255, 255, 255]  # White - TP
            dirt_comparison[fp_dirt] = [255, 0, 0]      # Red - FP
            dirt_comparison[fn_dirt] = [0, 0, 255]      # Blue - FN
            
            axes[2, 2].imshow(dirt_comparison)
            axes[2, 2].set_title('Dirt: TP/FP/FN\n(White/Red/Blue)', fontsize=12, fontweight='bold')
            axes[2, 2].axis('off')
            
            # Scratch GT vs Pred
            gt_scratch_mask = (gt_screen == 3).astype(np.uint8)
            pred_scratch_mask = (predicted_screen == 3).astype(np.uint8)
            scratch_comparison = np.zeros((*gt_scratch_mask.shape, 3), dtype=np.uint8)
            tp_scratch = (pred_scratch_mask == 1) & (gt_scratch_mask == 1)
            fp_scratch = (pred_scratch_mask == 1) & (gt_scratch_mask == 0)
            fn_scratch = (pred_scratch_mask == 0) & (gt_scratch_mask == 1)
            scratch_comparison[tp_scratch] = [255, 255, 255]  # White - TP
            scratch_comparison[fp_scratch] = [255, 0, 0]      # Red - FP
            scratch_comparison[fn_scratch] = [0, 0, 255]      # Blue - FN
            
            axes[2, 3].imshow(scratch_comparison)
            axes[2, 3].set_title('Scratch: TP/FP/FN\n(White/Red/Blue)', fontsize=12, fontweight='bold')
            axes[2, 3].axis('off')
            
            # Detailed metrics display
            gt_metrics = results['ground_truth_comparison']['metrics']
            patch_stats = results['patch_analysis']
            perf_stats = results['overall_statistics']['performance_metrics']
            
            metrics_text = f"""FULL PHONE GROUND TRUTH COMPARISON
            
IMAGE INFO:
• Full Size: {results['image_info']['full_image_size'][1]}×{results['image_info']['full_image_size'][0]}
• Screen Size: {results['image_info']['screen_size'][1]}×{results['image_info']['screen_size'][0]}
• Screen Coverage: {results['boundary_detection']['screen_percentage']:.1f}%

GROUND TRUTH METRICS:
• Mean IoU: {gt_metrics.get('mean_iou', 0):.3f}
• Mean Dice: {gt_metrics.get('mean_dice', 0):.3f}

PER-CLASS METRICS:"""
            
            for cls in range(self.single_image_predictor.num_classes):
                class_name = self.single_image_predictor.class_names[cls]
                iou_val = gt_metrics.get(f'{class_name}_iou', 0)
                dice_val = gt_metrics.get(f'{class_name}_dice', 0)
                metrics_text += f"\n• {class_name.title()}: IoU={iou_val:.3f}, Dice={dice_val:.3f}"
            
            metrics_text += f"""

PATCH ANALYSIS:
• Total Patches: {patch_stats['total_patches']}
• Patches with Defects: {patch_stats['patches_with_defects']}

PERFORMANCE:
• Total Time: {perf_stats['total_inference_time']:.2f}s
• Avg/Patch: {perf_stats['avg_patch_inference_time']:.3f}s
• Device: {results['model_info']['device']}

DATASET INFO:
• GT File: {labels.get('mask_info', {}).get('mask_file', 'unknown')}
• GT Type: {labels.get('mask_info', {}).get('mask_type', 'unknown')}"""
            
            axes[2, 4].text(0.05, 0.95, metrics_text, transform=axes[2, 4].transAxes,
                           verticalalignment='top', fontsize=9, fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[2, 4].set_title('GT Comparison Metrics', fontsize=12, fontweight='bold')
            axes[2, 4].axis('off')
            
            # Add main title
            fig.suptitle(f'Full Phone Ground Truth Comparison - {results["image_info"]["filename"]}', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Full phone GT visualization saved to: {save_path}")
            
            if show_plot:
                plt.show()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create full phone GT visualization: {e}")
            return None

    def _create_class_comparison_visualization(self, predicted: np.ndarray, ground_truth: np.ndarray, 
                                             class_id: int) -> np.ndarray:
        """Helper method to create TP/FP/FN visualization for a specific class"""
        pred_mask = (predicted == class_id).astype(np.uint8)
        gt_mask = (ground_truth == class_id).astype(np.uint8)
        
        comparison = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
        # Calculate confusion matrix components
        tp = (pred_mask == 1) & (gt_mask == 1)      # True Positives - White
        fp = (pred_mask == 1) & (gt_mask == 0)      # False Positives - Red  
        fn = (pred_mask == 0) & (gt_mask == 1)      # False Negatives - Blue
        tn = (pred_mask == 0) & (gt_mask == 0)      # True Negatives - Black
        
        comparison[tp] = [255, 255, 255]  # White
        comparison[fp] = [255, 0, 0]      # Red
        comparison[fn] = [0, 0, 255]      # Blue
        comparison[tn] = [0, 0, 0]        # Black
        
        return comparison
    

    def _print_full_phone_gt_summary(self, results: Dict):
        """Print summary including ground truth comparison"""
        # Print standard summary first (reuse existing method)
        self.print_full_phone_summary(results)
        
        # Add ground truth comparison summary
        if 'ground_truth_comparison' in results:
            gt_metrics = results['ground_truth_comparison']['metrics']
            
            self.logger.info(f"\nGROUND TRUTH COMPARISON:")
            self.logger.info(f"  Mean IoU: {gt_metrics.get('mean_iou', 0):.3f}")
            self.logger.info(f"  Mean Dice: {gt_metrics.get('mean_dice', 0):.3f}")
            
            for cls in range(self.single_image_predictor.num_classes):
                class_name = self.single_image_predictor.class_names[cls]
                iou_val = gt_metrics.get(f'{class_name}_iou', 0)
                dice_val = gt_metrics.get(f'{class_name}_dice', 0)
                self.logger.info(f"  {class_name.title()}: IoU={iou_val:.3f}, Dice={dice_val:.3f}")
            
            self.logger.info(f"="*80)

    def batch_full_phone_prediction_with_ground_truth(self, dataset_dir: str,
                                                     patch_size: int = 1024,
                                                     overlap: float = 0.2,
                                                     output_dir: str = None,
                                                     max_samples: int = None,
                                                     min_dirt_threshold: float = 0.01) -> List[Dict]:
        """
        Efficient batch processing of full phone images with ground truth
        Reuses existing infrastructure
        """
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Find dataset samples with full phone ground truth
        sample_dirs = []
        for item in dataset_path.iterdir():
            if item.is_dir() and self._is_full_phone_gt_sample(item):
                sample_dirs.append(item)
        
        if max_samples:
            sample_dirs = sample_dirs[:max_samples]
        self.logger.info(f"Found {len(sample_dirs)} full phone ground truth samples")
        
        if output_dir is None:
            output_dir = dataset_path / 'full_phone_gt_results'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        from tqdm import tqdm
        
        for i, sample_dir in enumerate(tqdm(sample_dirs, desc="Processing full phone GT samples")):
            try:
                # Create sample output directory
                sample_output_dir = output_dir / sample_dir.name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Process efficiently (reuse method)
                result = self.full_phone_prediction_with_ground_truth(
                    str(sample_dir),
                    patch_size=patch_size,
                    overlap=overlap,
                    save_results=True,
                    output_dir=str(sample_output_dir),
                    show_plot=False,
                    min_dirt_threshold=min_dirt_threshold
                )
                
                # Add batch metadata
                result['batch_info'] = {
                    'sample_index': i,
                    'sample_name': sample_dir.name,
                    'status': 'completed'
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {sample_dir.name}: {e}")
                results.append({
                    'batch_info': {
                        'sample_index': i,
                        'sample_name': sample_dir.name,
                        'status': 'error',
                        'error': str(e)
                    }
                })
        
        # Save batch summary efficiently
        if results:
            summary_path = output_dir / "batch_full_phone_gt_summary.json"
            self._save_full_phone_gt_batch_summary(results, str(summary_path))
            
            successful = [r for r in results if r.get('batch_info', {}).get('status') == 'completed']
            
            self.logger.info(f"\nFull phone GT batch completed!")
            self.logger.info(f"Processed {len(successful)}/{len(results)} samples successfully")
            
            # Log key aggregate metrics
            if successful:
                avg_iou = np.mean([r['ground_truth_comparison']['metrics']['mean_iou'] 
                                 for r in successful if 'ground_truth_comparison' in r])
                avg_dice = np.mean([r['ground_truth_comparison']['metrics']['mean_dice'] 
                                  for r in successful if 'ground_truth_comparison' in r])
                
                self.logger.info(f"Average Mean IoU: {avg_iou:.3f}")
                self.logger.info(f"Average Mean Dice: {avg_dice:.3f}")
            
            self.logger.info(f"Results saved to: {output_dir}")
        
        return results

    def _is_full_phone_gt_sample(self, sample_dir: Path) -> bool:
        """
        FIXED: More flexible detection of full phone ground truth samples
        """
        # More flexible file detection
        image_files = ['full_phone_image.bmp', 'synthetic_dirty_patch.bmp', 'original_image.bmp', 
                    'full_phone_image.png', 'original_image.png']
        mask_files = ['segmentation_mask_multiclass.png', 'segmentation_mask_combined.png',
                    'segmentation_mask_patch_multiclass.png', 'segmentation_mask_patch.png']
        labels_files = ['labels.json', 'labels_patch.json']
        
        has_image = any((sample_dir / f).exists() for f in image_files)
        has_mask = any((sample_dir / f).exists() for f in mask_files)
        has_labels = any((sample_dir / f).exists() for f in labels_files)
        
        # Debug logging for missing files
        if not has_image:
            self.logger.debug(f"{sample_dir.name}: No valid image file found. Checked: {image_files}")
        if not has_mask:
            self.logger.debug(f"{sample_dir.name}: No valid mask file found. Checked: {mask_files}")
        if not has_labels:
            self.logger.debug(f"{sample_dir.name}: No valid labels file found. Checked: {labels_files}")
        
        return has_image and has_mask and has_labels

    def _save_full_phone_gt_batch_summary(self, results: List[Dict], save_path: str):
        """Save efficient batch summary for full phone ground truth processing"""
        import json
        import time
        
        successful = [r for r in results if r.get('batch_info', {}).get('status') == 'completed']
        errors = [r for r in results if r.get('batch_info', {}).get('status') == 'error']
        
        # Calculate aggregate metrics efficiently
        aggregate_metrics = {}
        if successful:
            # Extract ground truth metrics
            gt_metrics = [r['ground_truth_comparison']['metrics'] for r in successful 
                         if 'ground_truth_comparison' in r]
            
            if gt_metrics:
                iou_values = [m['mean_iou'] for m in gt_metrics if 'mean_iou' in m]
                dice_values = [m['mean_dice'] for m in gt_metrics if 'mean_dice' in m]
                
                aggregate_metrics = {
                    'ground_truth_comparison': {
                        'count': len(gt_metrics),
                        'avg_mean_iou': float(np.mean(iou_values)) if iou_values else 0,
                        'avg_mean_dice': float(np.mean(dice_values)) if dice_values else 0,
                        'std_mean_iou': float(np.std(iou_values)) if iou_values else 0,
                        'min_mean_iou': float(np.min(iou_values)) if iou_values else 0,
                        'max_mean_iou': float(np.max(iou_values)) if iou_values else 0
                    }
                }
                
                # Per-class aggregates
                for cls in range(self.single_image_predictor.num_classes):
                    class_name = self.single_image_predictor.class_names[cls]
                    class_ious = [m.get(f'{class_name}_iou', 0) for m in gt_metrics if m.get(f'{class_name}_iou', 0) > 0]
                    if class_ious:
                        aggregate_metrics['ground_truth_comparison'][f'{class_name}_avg_iou'] = float(np.mean(class_ious))
        
        summary = {
            'batch_info': {
                'total_samples': len(results),
                'successful_samples': len(successful),
                'error_samples': len(errors),
                'processing_type': 'full_phone_ground_truth_comparison',
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'model_info': {
                'num_classes': self.single_image_predictor.num_classes,
                'class_names': self.single_image_predictor.class_names,
                'architecture': self.single_image_predictor.config.get('model_architecture', 'standard'),
                'confidence_threshold': self.single_image_predictor.confidence_threshold,
                'device': str(self.single_image_predictor.device),
            },
            'aggregate_metrics': aggregate_metrics,
            'individual_results': [
                {
                    'sample_name': r.get('batch_info', {}).get('sample_name', 'unknown'),
                    'status': r.get('batch_info', {}).get('status', 'unknown'),
                    'ground_truth_metrics': r.get('ground_truth_comparison', {}).get('metrics', {}),
                    'inference_time': r.get('overall_statistics', {}).get('performance_metrics', {}).get('total_inference_time', 0),
                    'error': r.get('batch_info', {}).get('error')
                } for r in results
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Full phone GT batch summary saved to: {save_path}")
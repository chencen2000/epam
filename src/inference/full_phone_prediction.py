import json
import time
from pathlib import Path
from logging import Logger
from typing import Optional, Dict, List, Tuple

import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.core.logger_config import setup_application_logger
from src.inference.single_predictor import SingleImagePredictor
from src.synthesis.patch_generator import PatchGenerator
from src.inference.base_predictor import BasePredictor
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
        Run batch inference on multiple full phone images
        
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
        
        self.logger.info(f"Found {len(image_files)} full phone images for processing")
        
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
            self.logger.info(f"\nBatch processing completed!")
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
        Complete pipeline for predicting dirt on full phone images
        
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
            Complete prediction results with aggregated data
        """
        # Load the full phone image
        self.logger.info(f"Loading full phone image: {image_path}")
        full_image = self.single_image_predictor.image_operations.load_image_color(image_path)

        if full_image.shape[0] > 1024: 
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
        
        self.logger.info(f"Created {len(patches_data)} patches for inference")
        
        # Run inference on all patches
        self.logger.info("Running inference on patches...")
        patch_results = self.prediction_full_screen_patch(patches_data)
        
        # Aggregate results back to full screen coordinates
        self.logger.info("Aggregating patch results...")
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
                'patches_with_dirt': len([p for p in patch_results if p['dirt_percentage'] > min_dirt_threshold]),
                'patch_results': patch_results
            },
            'aggregated_prediction': aggregated_results,
            'overall_statistics': overall_stats,
            'model_info': {
                'architecture': self.single_image_predictor.config.get('model_architecture', 'standard'),
                'confidence_threshold': self.single_image_predictor.confidence_threshold,
                'device': str(self.single_image_predictor.device)
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
            # Save comprehensive visualization
            vis_path = output_dir / f"{image_path.stem}_full_phone_analysis.png"
            self.visualize_full_phone_prediction(
                full_image, screen_image, results,
                save_path=str(vis_path), 
                show_plot=show_plot
            )
            
            # Save detailed results as JSON
            results_path = output_dir / f"{image_path.stem}_full_analysis.json"
            self.single_image_predictor.save_prediction_results(results, str(results_path))
            
            # Save screen dirt mask
            screen_mask_path = output_dir / f"{image_path.stem}_screen_dirt_mask.png"
            cv2.imwrite(str(screen_mask_path), aggregated_results['binary_prediction'] * 255)
            
            # Save full image with overlay
            full_overlay_path = output_dir / f"{image_path.stem}_full_phone_overlay.png"
            self._save_full_phone_overlay(full_image, results, str(full_overlay_path))
            
            self.logger.info(f"\nResults saved to: {output_dir}")
            self.logger.info(f"- Full analysis: {vis_path.name}")
            self.logger.info(f"- Results JSON: {results_path.name}")
            self.logger.info(f"- Screen mask: {screen_mask_path.name}")
            self.logger.info(f"- Full overlay: {full_overlay_path.name}")
        
        # Print comprehensive summary
        self.print_full_phone_summary(results)
        
        return results
    
    def aggregate_patch_results(self, patch_results: List[Dict], 
                               screen_shape: Tuple[int, int], 
                               patch_size: int, overlap: float) -> Dict:
        """Aggregate patch predictions back to full screen coordinates"""
        screen_h, screen_w = screen_shape[:2]
        
        # Initialize aggregation arrays
        prediction_sum = np.zeros((screen_h, screen_w), dtype=np.float32)
        probability_sum = np.zeros((screen_h, screen_w), dtype=np.float32)
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
            
            # Get patch predictions
            patch_binary = patch_result['binary_prediction'][:y_end-y_start, :x_end-x_start]
            patch_probs = patch_result['probabilities'][:y_end-y_start, :x_end-x_start]
            
            # Add to aggregation arrays
            prediction_sum[y_start:y_end, x_start:x_end] += patch_binary.astype(np.float32)
            probability_sum[y_start:y_end, x_start:x_end] += patch_probs
            count_matrix[y_start:y_end, x_start:x_end] += 1
        
        # Avoid division by zero
        count_matrix = np.maximum(count_matrix, 1)
        
        # Calculate averaged results
        avg_probabilities = probability_sum / count_matrix
        avg_predictions = prediction_sum / count_matrix
        
        # Create final binary prediction using threshold
        final_binary = (avg_probabilities > self.single_image_predictor.confidence_threshold).astype(np.uint8)
        
        return {
            'binary_prediction': final_binary,
            'probabilities': avg_probabilities,
            'prediction_confidence': avg_predictions,  # Shows how many patches agreed
            'coverage_count': count_matrix,  # Shows how many patches covered each pixel
            'dirt_pixels': np.sum(final_binary),
            'dirt_percentage': (np.sum(final_binary) / final_binary.size) * 100,
            'screen_size': screen_shape[:2]
        }

    
    def prediction_full_screen_patch(self, patches_data: List[Tuple]) -> List[Dict]:
        """Run inference on all patches"""
        patch_results = []
        
        for i, (patch_img, patch_mask, patch_annotations, x_start, y_start) in enumerate(tqdm(patches_data, desc="Processing patches")):
            try:
                # Run prediction on patch
                patch_result = self.predict(patch_img, return_raw=False)
                
                # Add patch metadata
                patch_result.update({
                    'patch_id': i,
                    'patch_coordinates': (x_start, y_start),
                    'patch_size': patch_img.shape[:2]
                })
                
                patch_results.append(patch_result)
                
            except Exception as e:
                self.logger.warning(f"Error processing patch {i}: {e}")
                # Create dummy result for failed patch
                patch_results.append({
                    'patch_id': i,
                    'patch_coordinates': (x_start, y_start),
                    'patch_size': patch_img.shape[:2],
                    'binary_prediction': np.zeros(patch_img.shape[:2], dtype=np.uint8),
                    'probabilities': np.zeros(patch_img.shape[:2], dtype=np.float32),
                    'dirt_pixels': 0,
                    'dirt_percentage': 0.0,
                    'inference_time': 0.0,
                    'error': str(e)
                })
        
        return patch_results
    

    def calculate_overall_stats(self, aggregated_results: Dict, 
                               screen_shape: Tuple[int, int],
                               patch_results: List[Dict],
                               min_dirt_threshold: float) -> Dict:
        """Calculate overall statistics for the full phone analysis"""
        screen_h, screen_w = screen_shape[:2]
        total_screen_pixels = screen_h * screen_w
        
        # Calculate patch statistics
        valid_patches = [p for p in patch_results if 'error' not in p]
        dirty_patches = [p for p in valid_patches if p['dirt_percentage'] > min_dirt_threshold]
        
        patch_dirt_percentages = [p['dirt_percentage'] for p in valid_patches]
        patch_inference_times = [p['inference_time'] for p in valid_patches]
        
        # Calculate region analysis
        region_analysis = self.single_image_predictor.region_analyzer.analyze_dirt_regions(aggregated_results['binary_prediction'])
        
        stats = {
            'screen_analysis': {
                'total_pixels': total_screen_pixels,
                'dirt_pixels': aggregated_results['dirt_pixels'],
                'dirt_percentage': aggregated_results['dirt_percentage'],
                'clean_percentage': 100 - aggregated_results['dirt_percentage']
            },
            'patch_statistics': {
                'total_patches': len(patch_results),
                'valid_patches': len(valid_patches),
                'patches_with_dirt': len(dirty_patches),
                'avg_patch_dirt_percentage': np.mean(patch_dirt_percentages) if patch_dirt_percentages else 0,
                'max_patch_dirt_percentage': np.max(patch_dirt_percentages) if patch_dirt_percentages else 0,
                'min_patch_dirt_percentage': np.min(patch_dirt_percentages) if patch_dirt_percentages else 0,
                'std_patch_dirt_percentage': np.std(patch_dirt_percentages) if patch_dirt_percentages else 0
            },
            'performance_metrics': {
                'total_inference_time': sum(patch_inference_times),
                'avg_patch_inference_time': np.mean(patch_inference_times) if patch_inference_times else 0,
                'patches_per_second': len(valid_patches) / sum(patch_inference_times) if sum(patch_inference_times) > 0 else 0
            },
            'region_analysis': region_analysis,
            'confidence_analysis': {
                'high_confidence_pixels': np.sum(aggregated_results['probabilities'] > 0.8),
                'medium_confidence_pixels': np.sum((aggregated_results['probabilities'] > 0.5) & 
                                                 (aggregated_results['probabilities'] <= 0.8)),
                'low_confidence_pixels': np.sum(aggregated_results['probabilities'] <= 0.5),
                'avg_confidence': np.mean(aggregated_results['probabilities']),
                'max_coverage': np.max(aggregated_results['coverage_count']),
                'min_coverage': np.min(aggregated_results['coverage_count'])
            }
        }
        
        return stats
    
    def print_full_phone_summary(self, results: Dict):
        """Print comprehensive summary of full phone analysis"""
        stats = results['overall_statistics']
        
        self.logger.info(f"\n" + "="*80)
        self.logger.info(f"FULL PHONE DIRT DETECTION ANALYSIS")
        self.logger.info(f"="*80)
        self.logger.info(f"Image: {results['image_info']['filename']}")
        self.logger.info(f"Full Image Size: {results['image_info']['full_image_size'][1]}×{results['image_info']['full_image_size'][0]}")
        self.logger.info(f"Screen Size: {results['image_info']['screen_size'][1]}×{results['image_info']['screen_size'][0]}")
        self.logger.info(f"Screen Coverage: {results['boundary_detection']['screen_percentage']:.1f}% of full image")
        self.logger.info(f"")
        self.logger.info(f"DIRT DETECTION RESULTS:")
        self.logger.info(f"  Overall Dirt Coverage: {stats['screen_analysis']['dirt_percentage']:.2f}%")
        self.logger.info(f"  Dirt Pixels: {stats['screen_analysis']['dirt_pixels']:,} / {stats['screen_analysis']['total_pixels']:,}")
        self.logger.info(f"  Clean Coverage: {stats['screen_analysis']['clean_percentage']:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"PATCH ANALYSIS:")
        self.logger.info(f"  Total Patches Processed: {stats['patch_statistics']['total_patches']}")
        self.logger.info(f"  Patches with Dirt: {stats['patch_statistics']['patches_with_dirt']}")
        self.logger.info(f"  Average Patch Dirt: {stats['patch_statistics']['avg_patch_dirt_percentage']:.2f}%")
        self.logger.info(f"  Maximum Patch Dirt: {stats['patch_statistics']['max_patch_dirt_percentage']:.2f}%")
        self.logger.info(f"")
        self.logger.info(f"DIRT REGIONS:")
        if stats['region_analysis']['num_regions'] > 0:
            self.logger.info(f"  Number of Dirt Regions: {stats['region_analysis']['num_regions']}")
            self.logger.info(f"  Largest Region: {stats['region_analysis']['largest_region_area']:,} pixels")
            self.logger.info(f"  Average Region Size: {stats['region_analysis']['average_region_area']:.1f} pixels")
        else:
            self.logger.info(f"  No distinct dirt regions detected")
        self.logger.info(f"")
        self.logger.info(f"PERFORMANCE METRICS:")
        self.logger.info(f"  Total Processing Time: {stats['performance_metrics']['total_inference_time']:.2f} seconds")
        self.logger.info(f"  Average Per Patch: {stats['performance_metrics']['avg_patch_inference_time']:.3f} seconds")
        self.logger.info(f"  Processing Speed: {stats['performance_metrics']['patches_per_second']:.1f} patches/second")
        self.logger.info(f"  Device Used: {results['model_info']['device']}")
        self.logger.info(f"="*80)

    def _save_full_phone_overlay(self, full_image: np.ndarray, results: Dict, save_path: str):
        """Save full phone image with dirt overlay"""
        overlay_image = self._create_full_phone_dirt_overlay(full_image, results)
        cv2.imwrite(save_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        self.logger.info(f"Full phone overlay saved to: {save_path}")
    
    def _create_full_phone_dirt_overlay(self, full_image: np.ndarray, results: Dict) -> np.ndarray:
        """Create full phone image with dirt regions mapped back from screen coordinates"""
        full_overlay = full_image.copy()
        
        # Get screen coordinates and dirt prediction
        screen_coords = results['boundary_detection']['screen_area']
        dirt_prediction = results['aggregated_prediction']['binary_prediction']
        
        # Create dirt overlay on screen region
        screen_x, screen_y = screen_coords['x'], screen_coords['y']
        screen_w, screen_h = screen_coords['width'], screen_coords['height']
        
        # Resize dirt prediction to match actual screen size if needed
        if dirt_prediction.shape != (screen_h, screen_w):
            dirt_prediction = cv2.resize(dirt_prediction, (screen_w, screen_h), 
                                       interpolation=cv2.INTER_NEAREST)
        
        # Apply dirt overlay to screen region
        dirt_mask = dirt_prediction == 1
        if np.any(dirt_mask):
            screen_region = full_overlay[screen_y:screen_y+screen_h, screen_x:screen_x+screen_w]
            screen_region[dirt_mask] = screen_region[dirt_mask] * 0.3 + np.array(self.single_image_predictor.colors['dirt']) * 0.7
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
        """Create comprehensive visualization for full phone prediction"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a complex subplot layout
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], width_ratios=[1, 1, 1, 1])
        
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
        
        # Aggregated prediction heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        prob_display = ax3.imshow(results['aggregated_prediction']['probabilities'], 
                                 cmap='hot', vmin=0, vmax=1)
        ax3.set_title(f'Dirt Probability Map\n({results["overall_statistics"]["screen_analysis"]["dirt_percentage"]:.1f}% dirt)', 
                     fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(prob_display, ax=ax3, fraction=0.046, pad=0.04)
        
        # Binary prediction
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(results['aggregated_prediction']['binary_prediction'], cmap='Reds', vmin=0, vmax=1)
        ax4.set_title('Binary Dirt Prediction', fontsize=12, fontweight='bold')
        ax4.axis('off')
        
        # Screen with dirt overlay
        ax5 = fig.add_subplot(gs[1, 0])
        overlay_screen = screen_image.copy()
        dirt_mask = results['aggregated_prediction']['binary_prediction'] == 1
        overlay_screen[dirt_mask] = overlay_screen[dirt_mask] * 0.3 + np.array(self.single_image_predictor.colors['dirt']) * 0.7
        ax5.imshow(overlay_screen.astype(np.uint8))
        ax5.set_title('Screen with Dirt Overlay', fontsize=12, fontweight='bold')
        ax5.axis('off')
        
        # Patch coverage visualization
        ax6 = fig.add_subplot(gs[1, 1])
        coverage_display = ax6.imshow(results['aggregated_prediction']['coverage_count'], cmap='viridis')
        ax6.set_title(f'Patch Coverage\n(Max: {int(np.max(results["aggregated_prediction"]["coverage_count"]))} overlaps)', 
                     fontsize=12, fontweight='bold')
        ax6.axis('off')
        plt.colorbar(coverage_display, ax=ax6, fraction=0.046, pad=0.04)
        
        # Patch statistics visualization
        ax7 = fig.add_subplot(gs[1, 2])
        patch_dirt_percentages = [p['dirt_percentage'] for p in results['patch_analysis']['patch_results'] 
                                if 'error' not in p]
        if patch_dirt_percentages:
            ax7.hist(patch_dirt_percentages, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax7.axvline(np.mean(patch_dirt_percentages), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(patch_dirt_percentages):.1f}%')
            ax7.set_xlabel('Dirt Percentage per Patch')
            ax7.set_ylabel('Number of Patches')
            ax7.set_title('Patch Dirt Distribution', fontsize=12, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No valid patches', ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('Patch Dirt Distribution', fontsize=12, fontweight='bold')
        
        # Full phone with dirt regions mapped back
        ax8 = fig.add_subplot(gs[1, 3])
        full_with_dirt = self._create_full_phone_dirt_overlay(full_image, results)
        ax8.imshow(full_with_dirt)
        ax8.set_title('Full Phone with Dirt Mapping', fontsize=12, fontweight='bold')
        ax8.axis('off')
        
        # Statistics summary (spanning bottom row)
        ax9 = fig.add_subplot(gs[2, :])
        stats = results['overall_statistics']
        
        summary_text = f"""FULL PHONE DIRT DETECTION ANALYSIS
        
        IMAGE INFO:
        • File: {results['image_info']['filename']}
        • Full Image Size: {results['image_info']['full_image_size'][1]}×{results['image_info']['full_image_size'][0]}
        • Screen Size: {results['image_info']['screen_size'][1]}×{results['image_info']['screen_size'][0]}
        • Screen Coverage: {results['boundary_detection']['screen_percentage']:.1f}% of full image
        
        DIRT ANALYSIS:
        • Total Screen Pixels: {stats['screen_analysis']['total_pixels']:,}
        • Dirt Pixels: {stats['screen_analysis']['dirt_pixels']:,}
        • Dirt Coverage: {stats['screen_analysis']['dirt_percentage']:.2f}%
        • Clean Coverage: {stats['screen_analysis']['clean_percentage']:.2f}%
        
        PATCH PROCESSING:
        • Total Patches: {stats['patch_statistics']['total_patches']}
        • Patches with Dirt: {stats['patch_statistics']['patches_with_dirt']}
        • Avg Patch Dirt: {stats['patch_statistics']['avg_patch_dirt_percentage']:.2f}%
        • Max Patch Dirt: {stats['patch_statistics']['max_patch_dirt_percentage']:.2f}%
        
        DIRT REGIONS:
        • Number of Regions: {stats['region_analysis']['num_regions']}
        • Largest Region: {stats['region_analysis']['largest_region_area']:,} pixels
        • Average Region Size: {stats['region_analysis']['average_region_area']:.1f} pixels
        
        PERFORMANCE:
        • Total Processing Time: {stats['performance_metrics']['total_inference_time']:.2f}s
        • Average Patch Time: {stats['performance_metrics']['avg_patch_inference_time']:.3f}s
        • Processing Speed: {stats['performance_metrics']['patches_per_second']:.1f} patches/second
        • Device: {results['model_info']['device']}"""
        
        ax9.text(0.02, 0.98, summary_text, transform=ax9.transAxes,
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax9.set_title('Analysis Summary', fontsize=14, fontweight='bold')
        ax9.axis('off')
        
        # Add main title
        fig.suptitle(f'Full Phone Dirt Detection Analysis - {results["image_info"]["filename"]}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Full phone visualization saved to: {save_path}")
        
        if show_plot:
            plt.show()
        
        return fig
    

    def save_batch_summary(self, results: List[Dict], save_path: str) -> None:
        """Save summary of batch full phone processing results"""
        screen_dirt_percentages = [r['overall_statistics']['screen_analysis']['dirt_percentage'] for r in results]
        processing_times = [r['overall_statistics']['performance_metrics']['total_inference_time'] for r in results]
        patch_counts = [r['image_info']['total_patches'] for r in results]
        
        summary = {
            'batch_info': {
                'total_images': len(results),
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'dirt_analysis_summary': {
                'avg_dirt_percentage': np.mean(screen_dirt_percentages),
                'std_dirt_percentage': np.std(screen_dirt_percentages),
                'min_dirt_percentage': np.min(screen_dirt_percentages),
                'max_dirt_percentage': np.max(screen_dirt_percentages),
                'median_dirt_percentage': np.median(screen_dirt_percentages)
            },
            'processing_summary': {
                'total_processing_time': sum(processing_times),
                'avg_processing_time': np.mean(processing_times),
                'total_patches_processed': sum(patch_counts),
                'avg_patches_per_image': np.mean(patch_counts)
            },
            'individual_results': [
                {
                    'filename': r['image_info']['filename'],
                    'screen_dirt_percentage': r['overall_statistics']['screen_analysis']['dirt_percentage'],
                    'total_patches': r['image_info']['total_patches'],
                    'patches_with_dirt': r['overall_statistics']['patch_statistics']['patches_with_dirt'],
                    'processing_time': r['overall_statistics']['performance_metrics']['total_inference_time'],
                    'dirt_regions': r['overall_statistics']['region_analysis']['num_regions']
                } for r in results
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Batch full phone summary saved to: {save_path}")
    
    def save_batch_full_phone_summary(self, results: List[Dict], save_path: str):
        """Legacy method name for backward compatibility"""
        self.save_batch_summary(results, save_path)
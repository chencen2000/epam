import logging
from typing import Optional, Dict, List, Tuple

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from src.core.logger_config import setup_application_logger


def calculate_multiclass_iou(pred_logits: torch.Tensor, true_mask: torch.Tensor, 
                             num_classes: int = 3, ignore_index: int = -1) -> Dict[str, float]:
    """Calculate IoU for multi-class segmentation"""
    # pred_logits shape: (B, C, H, W), true_mask shape: (B, H, W) or (B, 1, H, W)
    
    # Get predictions by taking argmax
    pred_mask = torch.argmax(pred_logits, dim=1)  # (B, H, W)
    
    # Handle true_mask dimensions
    if true_mask.dim() == 4:
        true_mask = true_mask.squeeze(1)
    
    # Flatten tensors
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1).long()
    
    ious = {}
    total_iou = 0
    valid_classes = 0
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        # Binary masks for current class
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)
        
        # Calculate intersection and union
        intersection = (pred_cls & true_cls).sum().float()
        union = (pred_cls | true_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious[f'iou_class_{cls}'] = iou.item()
            total_iou += iou.item()
            valid_classes += 1
        else:
            ious[f'iou_class_{cls}'] = 1.0  # Perfect score if class not present
            total_iou += 1.0
            valid_classes += 1
    
    # Mean IoU across all classes
    ious['mean_iou'] = total_iou / valid_classes if valid_classes > 0 else 0.0
    
    return ious


def calculate_multiclass_dice(pred_logits: torch.Tensor, true_mask: torch.Tensor, 
                              num_classes: int = 3, ignore_index: int = -1) -> Dict[str, float]:
    """Calculate Dice coefficient for multi-class segmentation"""
    # Get predictions by taking argmax
    pred_mask = torch.argmax(pred_logits, dim=1)  # (B, H, W)
    
    # Handle true_mask dimensions
    if true_mask.dim() == 4:
        true_mask = true_mask.squeeze(1)
    
    # Flatten tensors
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1).long()
    
    dice_scores = {}
    total_dice = 0
    valid_classes = 0
    
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
            
        # Binary masks for current class
        pred_cls = (pred_mask == cls).float()
        true_cls = (true_mask == cls).float()
        
        # Calculate intersection
        intersection = (pred_cls * true_cls).sum()
        
        # Calculate dice
        dice = (2 * intersection) / (pred_cls.sum() + true_cls.sum() + 1e-8)
        
        dice_scores[f'dice_class_{cls}'] = dice.item()
        total_dice += dice.item()
        valid_classes += 1
    
    # Mean Dice across all classes
    dice_scores['mean_dice'] = total_dice / valid_classes if valid_classes > 0 else 0.0
    
    return dice_scores


def calculate_metrics(prediction: np.ndarray, ground_truth: np.ndarray, 
                     num_classes: int = 3, class_names: List[str] = None,
                     iou_threshold: float = 0.5, logger: Optional[logging.Logger] = None) -> Dict:
    """Calculate comprehensive metrics for multi-class segmentation"""
    if class_names is None:
        class_names = ['background', 'dirt', 'scratches']
    
    metrics = {}
    
    # Per-class pixel metrics
    for cls in range(num_classes):
        cls_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
        
        # Binary masks for current class
        pred_cls = (prediction == cls)
        true_cls = (ground_truth == cls)
        
        # Calculate IoU
        intersection = np.logical_and(pred_cls, true_cls)
        union = np.logical_or(pred_cls, true_cls)
        
        if np.sum(union) == 0:
            pixel_iou = 1.0
        else:
            pixel_iou = np.sum(intersection) / np.sum(union)
        
        # Calculate Dice
        if np.sum(pred_cls) + np.sum(true_cls) == 0:
            pixel_dice = 1.0
        else:
            pixel_dice = 2 * np.sum(intersection) / (np.sum(pred_cls) + np.sum(true_cls))
        
        metrics.update({
            f'{cls_name}_iou': pixel_iou,
            f'{cls_name}_dice': pixel_dice,
            f'{cls_name}_predicted_pixels': int(np.sum(pred_cls)),
            f'{cls_name}_ground_truth_pixels': int(np.sum(true_cls)),
            f'{cls_name}_intersection': int(np.sum(intersection))
        })
    
    # Calculate mean metrics
    mean_iou = np.mean([metrics[f'{name}_iou'] for name in class_names])
    mean_dice = np.mean([metrics[f'{name}_dice'] for name in class_names])
    
    metrics.update({
        'mean_iou': mean_iou,
        'mean_dice': mean_dice
    })
    
    # Object-level metrics for each class (excluding background)
    for cls in range(1, num_classes):  # Skip background (class 0)
        cls_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
        
        try:
            # Binary masks for current class
            pred_cls = (prediction == cls).astype(np.uint8)
            true_cls = (ground_truth == cls).astype(np.uint8)
            
            object_metrics = _calculate_object_level_metrics(
                pred_cls, true_cls, iou_threshold
            )
            
            # Add class name prefix to metrics
            for key, value in object_metrics.items():
                metrics[f'{cls_name}_{key}'] = value
                
        except Exception as e:
            if logger:
                logger.warning(f"Could not calculate object-level metrics for {cls_name}: {e}")
    
    return metrics


def _calculate_object_level_metrics(prediction: np.ndarray, ground_truth: np.ndarray, 
                                    iou_threshold: float = 0.5) -> Dict:
    """Calculate object-level TP, FP, FN based on IoU threshold."""
    pred_objects = _find_objects(prediction)
    gt_objects = _find_objects(ground_truth)
    
    iou_matrix = _calculate_object_iou_matrix(pred_objects, gt_objects, prediction.shape)
    tp, fp, fn, matches = _match_objects(iou_matrix, iou_threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'object_tp': tp, 'object_fp': fp, 'object_fn': fn,
        'object_precision': precision, 'object_recall': recall, 'object_f1': f1,
        'detected_objects': len(pred_objects), 'ground_truth_objects': len(gt_objects),
        'matched_pairs': len(matches),
        'avg_matched_iou': np.mean([iou for _, _, iou in matches]) if matches else 0.0
    }


def _find_objects(binary_mask: np.ndarray) -> List[Dict]:
    """Find connected components (objects) in binary mask."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=4
    )
    
    objects = []
    for i in range(1, num_labels):
        mask = (labels == i)
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < 10:  # Filter small objects
            continue
            
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                    stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        
        objects.append({
            'mask': mask, 'area': area, 'bbox': (x, y, w, h), 'centroid': centroids[i]
        })
    
    return objects


def _calculate_object_iou_matrix(pred_objects: List[Dict], gt_objects: List[Dict],
                                mask_shape: Tuple[int, int]) -> np.ndarray:
    """Calculate IoU matrix between predicted and ground truth objects."""
    n_pred, n_gt = len(pred_objects), len(gt_objects)
    
    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt))
    
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred_obj in enumerate(pred_objects):
        for j, gt_obj in enumerate(gt_objects):
            intersection = np.logical_and(pred_obj['mask'], gt_obj['mask'])
            union = np.logical_or(pred_obj['mask'], gt_obj['mask'])
            
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
            iou_matrix[i, j] = iou
    
    return iou_matrix


def _match_objects(iou_matrix: np.ndarray, iou_threshold: float) -> Tuple[int, int, int, List]:
    """Match objects using greedy matching based on IoU."""
    n_pred, n_gt = iou_matrix.shape
    
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0, []
    elif n_pred == 0:
        return 0, 0, n_gt, []
    elif n_gt == 0:
        return 0, n_pred, 0, []
    
    matches = []
    used_pred, used_gt = set(), set()
    
    all_matches = [(i, j, iou_matrix[i, j]) for i in range(n_pred) for j in range(n_gt) 
                    if iou_matrix[i, j] >= iou_threshold]
    all_matches.sort(key=lambda x: x[2], reverse=True)
    
    for pred_idx, gt_idx, iou in all_matches:
        if pred_idx not in used_pred and gt_idx not in used_gt:
            matches.append((pred_idx, gt_idx, iou))
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
    
    tp = len(matches)
    fp = n_pred - tp
    fn = n_gt - tp
    
    return tp, fp, fn, matches


def create_multiclass_comparison_mask(prediction: np.ndarray, ground_truth: np.ndarray, 
                                      num_classes: int = 3) -> np.ndarray:
    """Create visualization mask showing multi-class predictions and errors"""
    h, w = prediction.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Define colors for each class
    class_colors = {
        0: [0, 0, 0],       # Background - Black
        1: [0, 255, 0],     # Dirt - Green  
        2: [255, 0, 0],     # Scratches - Red
    }
    
    # Color correct predictions
    correct_mask = (prediction == ground_truth)
    for cls in range(num_classes):
        cls_mask = np.logical_and(correct_mask, prediction == cls)
        if cls in class_colors:
            comparison[cls_mask] = class_colors[cls]
    
    # Color incorrect predictions (misclassifications) - Yellow
    incorrect_mask = (prediction != ground_truth)
    comparison[incorrect_mask] = [255, 255, 0]  # Yellow for errors
    
    # Optionally, create error detail mask
    error_detail = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(1, num_classes):  # Skip background
        # False positives for this class - Lighter shade
        fp_mask = np.logical_and(prediction == cls, ground_truth != cls)
        if cls == 1:  # Dirt FP - Light green
            error_detail[fp_mask] = [150, 255, 150]
        elif cls == 2:  # Scratches FP - Light red
            error_detail[fp_mask] = [255, 150, 150]
        
        # False negatives for this class - Darker shade
        fn_mask = np.logical_and(prediction != cls, ground_truth == cls)
        if cls == 1:  # Dirt FN - Dark green
            error_detail[fn_mask] = [0, 100, 0]
        elif cls == 2:  # Scratches FN - Dark red
            error_detail[fn_mask] = [100, 0, 0]
    
    return comparison, error_detail

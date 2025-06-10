import logging
from typing import Optional, Dict, List, Tuple

import cv2
import torch
import numpy as np

from src.core.logger_config import setup_application_logger


def calculate_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    pred_mask = (torch.sigmoid(pred_mask) > threshold).float()
    
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    true_mask = true_mask.view(true_mask.size(0), -1)
    
    intersection = (pred_mask * true_mask).sum(dim=1)
    union = pred_mask.sum(dim=1) + true_mask.sum(dim=1) - intersection
    
    iou = intersection / (union + 1e-8)
    return iou.mean().item()


def calculate_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, threshold: float = 0.5) -> float:
    pred_mask = (torch.sigmoid(pred_mask) > threshold).float()
    
    pred_mask = pred_mask.view(pred_mask.size(0), -1)
    true_mask = true_mask.view(true_mask.size(0), -1)
    
    intersection = (pred_mask * true_mask).sum(dim=1)
    dice = (2 * intersection) / (pred_mask.sum(dim=1) + true_mask.sum(dim=1) + 1e-8)
    
    return dice.mean().item()


def calculate_metrics(prediction: np.ndarray, ground_truth: np.ndarray, 
                        iou_threshold: float = 0.5, logger:Optional[logging.Logger]=None) -> Dict:
    """Calculate comprehensive metrics with pixel-level IoU and object-level TP/TN/FP/FN."""
    metrics = {}
    
    # Pixel-level IoU calculation
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    
    if np.sum(union) == 0:
        pixel_iou = 1.0
    else:
        pixel_iou = np.sum(intersection) / np.sum(union)
    
    # Pixel-level Dice coefficient
    if np.sum(prediction) + np.sum(ground_truth) == 0:
        pixel_dice = 1.0
    else:
        pixel_dice = 2 * np.sum(intersection) / (np.sum(prediction) + np.sum(ground_truth))
    
    metrics.update({
        'pixel_iou': pixel_iou,
        'pixel_dice': pixel_dice,
        'pixel_intersection': int(np.sum(intersection)),
        'pixel_union': int(np.sum(union)),
        'predicted_pixels': int(np.sum(prediction)),
        'ground_truth_pixels': int(np.sum(ground_truth))
    })
    
    # Object-level metrics
    try:
        object_metrics = _calculate_object_level_metrics(
            prediction, ground_truth, iou_threshold
        )
        metrics.update(object_metrics)
    except Exception as e:
        if logger:
            logger.warning(f"Could not calculate object-level metrics: {e}")
        metrics.update({
            'object_tp': 0, 'object_fp': 0, 'object_fn': 0,
            'object_precision': 0.0, 'object_recall': 0.0, 'object_f1': 0.0,
            'detected_objects': 0, 'ground_truth_objects': 0
        })
    
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

def create_comparison_mask(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Create visualization mask showing pixel-level agreement."""
    h, w = prediction.shape
    comparison = np.zeros((h, w, 3), dtype=np.uint8)
    
    intersection = np.logical_and(prediction == 1, ground_truth == 1)
    comparison[intersection] = [0, 255, 0]  # Green for intersection
    
    false_positive = np.logical_and(prediction == 1, ground_truth == 0)
    comparison[false_positive] = [255, 0, 0]  # Red for FP
    
    false_negative = np.logical_and(prediction == 0, ground_truth == 1) 
    comparison[false_negative] = [0, 0, 255]  # Blue for FN
    
    return comparison


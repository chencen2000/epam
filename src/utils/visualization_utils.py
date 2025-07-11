from typing import List, Dict, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_training_history(train_losses: List[float], val_losses: List[float],
                         train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]],
                         epoch_times: List[float], save_path: str,
                         class_names: List[str] = None):
    """Enhanced plotting for multi-class segmentation training history"""
    
    if class_names is None:
        class_names = ['background', 'dirt', 'scratches']
    
    # Determine number of subplots needed
    num_classes = len(class_names)
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Loss curves
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mean IoU
    ax2 = plt.subplot(3, 3, 2)
    if 'mean_iou' in train_metrics:
        ax2.plot(train_metrics['mean_iou'], label='Train mIoU', linewidth=2)
        ax2.plot(val_metrics['mean_iou'], label='Val mIoU', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean IoU')
    ax2.set_title('Mean IoU Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Mean Dice
    ax3 = plt.subplot(3, 3, 3)
    if 'mean_dice' in train_metrics:
        ax3.plot(train_metrics['mean_dice'], label='Train mDice', linewidth=2)
        ax3.plot(val_metrics['mean_dice'], label='Val mDice', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean Dice')
    ax3.set_title('Mean Dice Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4-6: Per-class IoU
    for i, class_name in enumerate(class_names):
        ax = plt.subplot(3, 3, 4 + i)
        key = f'iou_{class_name}'
        if key in train_metrics:
            ax.plot(train_metrics[key], label=f'Train {class_name}', linewidth=2)
            ax.plot(val_metrics[key], label=f'Val {class_name}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title(f'{class_name.capitalize()} IoU')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    # Plot 7-9: Per-class Dice
    for i, class_name in enumerate(class_names):
        ax = plt.subplot(3, 3, 7 + i)
        key = f'dice_{class_name}'
        if key in train_metrics:
            ax.plot(train_metrics[key], label=f'Train {class_name}', linewidth=2)
            ax.plot(val_metrics[key], label=f'Val {class_name}', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice')
        ax.set_title(f'{class_name.capitalize()} Dice')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create additional summary plot
    create_final_metrics_summary(train_metrics, val_metrics, class_names, 
                               save_path.replace('.png', '_summary.png'))


def create_final_metrics_summary(train_metrics: Dict[str, List[float]], 
                               val_metrics: Dict[str, List[float]],
                               class_names: List[str], save_path: str):
    """Create a summary visualization of final metrics"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get final epoch metrics
    final_metrics = {}
    for class_name in class_names:
        iou_key = f'iou_{class_name}'
        dice_key = f'dice_{class_name}'
        
        if iou_key in val_metrics and val_metrics[iou_key]:
            final_metrics[f'{class_name}_iou'] = val_metrics[iou_key][-1]
        if dice_key in val_metrics and val_metrics[dice_key]:
            final_metrics[f'{class_name}_dice'] = val_metrics[dice_key][-1]
    
    # Bar plot for IoU
    iou_data = [(name, final_metrics.get(f'{name}_iou', 0)) for name in class_names]
    names, ious = zip(*iou_data)
    
    bars1 = ax1.bar(names, ious, color=['gray', 'green', 'red'])
    ax1.set_ylabel('IoU Score')
    ax1.set_title('Final Validation IoU by Class')
    ax1.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars1, ious):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Bar plot for Dice
    dice_data = [(name, final_metrics.get(f'{name}_dice', 0)) for name in class_names]
    names, dices = zip(*dice_data)
    
    bars2 = ax2.bar(names, dices, color=['gray', 'green', 'red'])
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Final Validation Dice by Class')
    ax2.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, value in zip(bars2, dices):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_multiclass_predictions(image: np.ndarray, ground_truth: np.ndarray, 
                                   prediction: np.ndarray, class_names: List[str] = None,
                                   save_path: Optional[str] = None):
    """Visualize multi-class segmentation predictions"""
    
    if class_names is None:
        class_names = ['background', 'dirt', 'scratches']
    
    # Define color map
    colors = {
        0: [0, 0, 0],       # Background - Black
        1: [0, 255, 0],     # Dirt - Green
        2: [255, 0, 0],     # Scratches - Red
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Ground truth colored
    gt_colored = np.zeros((*ground_truth.shape, 3), dtype=np.uint8)
    for cls, color in colors.items():
        gt_colored[ground_truth == cls] = color
    axes[0, 1].imshow(gt_colored)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Prediction colored
    pred_colored = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for cls, color in colors.items():
        pred_colored[prediction == cls] = color
    axes[0, 2].imshow(pred_colored)
    axes[0, 2].set_title('Prediction')
    axes[0, 2].axis('off')
    
    # Overlay on original
    overlay = image.copy()
    mask = prediction > 0  # Non-background
    overlay[mask] = overlay[mask] * 0.5 + pred_colored[mask] * 0.5
    axes[1, 0].imshow(overlay.astype(np.uint8))
    axes[1, 0].set_title('Prediction Overlay')
    axes[1, 0].axis('off')
    
    # Error visualization
    error_map = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    correct = (prediction == ground_truth)
    incorrect = ~correct
    
    # Show correct predictions in their class colors (dimmed)
    for cls, color in colors.items():
        mask = np.logical_and(correct, prediction == cls)
        error_map[mask] = [c // 2 for c in color]  # Dimmed colors
    
    # Show errors in bright yellow
    error_map[incorrect] = [255, 255, 0]
    axes[1, 1].imshow(error_map)
    axes[1, 2].set_title('Error Map (Yellow = Incorrect)')
    axes[1, 2].axis('off')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(colors[0])/255, label='Background'),
        Patch(facecolor=np.array(colors[1])/255, label='Dirt'),
        Patch(facecolor=np.array(colors[2])/255, label='Scratches'),
        Patch(facecolor=[1, 1, 0], label='Errors')
    ]
    axes[1, 2].legend(handles=legend_elements, loc='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_confusion_matrix(predictions: np.ndarray, ground_truths: np.ndarray,
                          class_names: List[str] = None, save_path: Optional[str] = None):
    """Create and visualize confusion matrix for multi-class segmentation"""
    
    if class_names is None:
        class_names = ['background', 'dirt', 'scratches']
    
    num_classes = len(class_names)
    
    # Flatten arrays
    y_true = ground_truths.flatten()
    y_pred = predictions.flatten()
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    ax2.set_title('Normalized Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
    
    return cm, cm_normalized

from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(train_losses: List[float], val_losses: List[float], 
                         train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]], 
                         epoch_times: List[float], save_path: str = 'training_history.png'):
    """Plot comprehensive training history with timing information"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', marker='s', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU plot
    axes[0, 1].plot(train_metrics['iou'], label='Train IoU', marker='o', color='green', linewidth=2)
    axes[0, 1].plot(val_metrics['iou'], label='Validation IoU', marker='s', color='orange', linewidth=2)
    axes[0, 1].set_title('Training and Validation IoU', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Dice plot
    axes[1, 0].plot(train_metrics['dice'], label='Train Dice', marker='o', color='purple', linewidth=2)
    axes[1, 0].plot(val_metrics['dice'], label='Validation Dice', marker='s', color='red', linewidth=2)
    axes[1, 0].set_title('Training and Validation Dice', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Epoch timing plot
    axes[1, 1].plot(epoch_times, label='Epoch Time', marker='o', color='brown', linewidth=2)
    axes[1, 1].set_title('Epoch Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    avg_time = np.mean(epoch_times[1:]) if epoch_times else 0
    total_time = sum(epoch_times) if epoch_times else 0
    axes[1, 1].text(0.02, 0.98, f'Avg: {avg_time:.1f}s\nTotal: {total_time/60:.1f}min', 
                   transform=axes[1, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

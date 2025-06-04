
import json
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from config.training import TrainingConfig
from src.core.logger_config import setup_application_logger
from src.utils.visualization_utils import plot_training_history


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_last_n: int = 3):
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.checkpoints_dir = save_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: optim.Optimizer, 
                       scheduler, val_loss: float, val_metrics: Dict, train_loss: float, 
                       train_metrics: Dict, config: TrainingConfig, history: Dict, 
                       epoch_time: float, logger: logging.Logger):
        """Save comprehensive checkpoint with epoch-specific folder"""
        
        # Create epoch-specific directory
        epoch_dir = self.checkpoints_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        
        # Save model checkpoint
        checkpoint_path = epoch_dir / "model_checkpoint.pth"
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'config': config.__dict__,
            'history': history,
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save epoch summary as JSON
        summary_path = epoch_dir / "epoch_summary.json"
        summary_data = {
            'epoch': epoch,
            'epoch_time_seconds': epoch_time,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save model architecture info
        model_info_path = epoch_dir / "model_info.txt"
        with open(model_info_path, 'w') as f:
            f.write(f"Model Architecture: {config.model_architecture}\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
            f.write(f"Model Size (MB): {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f}\n")
            f.write(f"\nModel Structure:\n{str(model)}\n")

        plot_path = epoch_dir / 'training_history.png'
        plot_training_history(
            history['train_losses'], 
            history['val_losses'],
            history['train_metrics'],
            history['val_metrics'],
            history['epoch_times'],
            str(plot_path)
        )
        
        logger.info(f"Checkpoint saved for epoch {epoch} in {epoch_dir}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(logger)
        
        return epoch_dir
    
    def _cleanup_old_checkpoints(self, logger: logging.Logger):
        """Remove old checkpoints, keeping only the last N"""
        if self.keep_last_n <= 0:
            return
            
        # Get all epoch directories
        epoch_dirs = [d for d in self.checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')]
        
        # Sort by epoch number
        epoch_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
        
        # Remove old checkpoints
        while len(epoch_dirs) > self.keep_last_n:
            old_checkpoint = epoch_dirs.pop(0)
            try:
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        self.best_weights = model.state_dict().copy()


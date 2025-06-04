import gc
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split

from config.training import TrainingConfig
from src.models.unet import UNet
from src.utils.memory_utils import MemoryTracker
from src.training.metrics import MetricsCalculator
from src.training.losses import CombinedLoss
from src.core.device_utils import get_device_info
from src.data.segmentation_dataset import SegmentationDataset
from src.data.transformations import get_train_transforms, get_val_transforms
from src.utils.visualization_utils import plot_training_history
from src.training.helper import CheckpointManager, EarlyStopping


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Enhanced training functions with comprehensive logging
def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device, scaler: GradScaler,
                accumulation_steps: int = 1, use_amp: bool = True, 
                logger: logging.Logger = None, memory_tracker: MemoryTracker = None) -> Tuple[float, Dict[str, float]]:
    
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    metrics = {'iou': 0, 'dice': 0}
    num_batches = 0
    optimizer.zero_grad()
    
    # Memory tracking at epoch start
    if memory_tracker:
        memory_tracker.log_memory_usage(logger, "at epoch start")
    
    progress_bar = tqdm(dataloader, desc='Training')
    batch_times = []
    
    for batch_idx, (images, masks) in enumerate(progress_bar):
        batch_start_time = time.time()
        
        try:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks) / accumulation_steps
            
            # Calculate metrics
            batch_iou = MetricsCalculator.calculate_iou(outputs, masks)
            batch_dice = MetricsCalculator.calculate_dice(outputs, masks)
            
            metrics['iou'] += batch_iou
            metrics['dice'] += batch_dice
            num_batches += 1
            
            # Backward pass
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * accumulation_steps:.4f}',
                'IoU': f'{batch_iou:.4f}',
                'Dice': f'{batch_dice:.4f}',
                'Time': f'{batch_time:.2f}s'
            })
            
            # Cleanup
            del images, masks, outputs, loss
            
            # Memory leak detection and cleanup
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if memory_tracker and batch_idx % 50 == 0:
                    memory_tracker.detect_memory_leak(logger, threshold_mb=200.0)
                    
        except RuntimeError as e:
            if logger:
                logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Handle remaining gradients
    if len(dataloader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    for key in metrics:
        metrics[key] = metrics[key] / num_batches if num_batches > 0 else 0.0
    
    # Log epoch statistics
    if logger:
        logger.info(f"Training epoch completed in {epoch_time:.2f}s")
        logger.info(f"Average batch time: {avg_batch_time:.3f}s")
        logger.info(f"Training loss: {avg_loss:.4f}")
        logger.info(f"Training IoU: {metrics['iou']:.4f}")
        logger.info(f"Training Dice: {metrics['dice']:.4f}")
    
    if memory_tracker:
        memory_tracker.log_memory_usage(logger, "at epoch end")
    
    return avg_loss, metrics

def validate_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                  device: torch.device, use_amp: bool = True, 
                  logger: logging.Logger = None, memory_tracker: MemoryTracker = None) -> Tuple[float, Dict[str, float]]:
    
    epoch_start_time = time.time()
    model.eval()
    total_loss = 0
    metrics = {'iou': 0, 'dice': 0}
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation')
        batch_times = []
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            batch_start_time = time.time()
            
            try:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp, device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                # Calculate metrics
                batch_iou = MetricsCalculator.calculate_iou(outputs, masks)
                batch_dice = MetricsCalculator.calculate_dice(outputs, masks)
                
                metrics['iou'] += batch_iou
                metrics['dice'] += batch_dice
                num_batches += 1
                
                total_loss += loss.item()
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'IoU': f'{batch_iou:.4f}',
                    'Dice': f'{batch_dice:.4f}',
                    'Time': f'{batch_time:.2f}s'
                })
                
                del images, masks, outputs, loss
                
                if batch_idx % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except RuntimeError as e:
                if logger:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    for key in metrics:
        metrics[key] = metrics[key] / num_batches if num_batches > 0 else 0.0
    
    # Log epoch statistics
    if logger:
        logger.info(f"Validation epoch completed in {epoch_time:.2f}s")
        logger.info(f"Average batch time: {avg_batch_time:.3f}s")
        logger.info(f"Validation loss: {avg_loss:.4f}")
        logger.info(f"Validation IoU: {metrics['iou']:.4f}")
        logger.info(f"Validation Dice: {metrics['dice']:.4f}")
    
    return avg_loss, metrics

def train_model(config: TrainingConfig, logger:logging.Logger) -> Tuple[nn.Module, Dict]:
    """Enhanced main training function with comprehensive logging and checkpointing"""
    
    # Setup
    set_seed()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logging
    # logger = setup_logging(save_dir)
    logger.info("=" * 80)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 80)
    
    # Initialize memory tracking
    memory_tracker = MemoryTracker(enabled=config.memory_tracking)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(save_dir, keep_last_n=3)
    
    # Log training configuration
    logger.info("Training Configuration:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")
    
    # Device setup
    device = get_device_info(logger)
    
    # Save configuration
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")
    
    # Memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info("CUDA optimizations enabled")
    
    # Create dataset
    logger.info("Loading dataset...")
    full_dataset = SegmentationDataset(
        root_dir=config.root_dir,
        target_size=config.target_size
    )
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid samples found in {config.root_dir}")
    
    logger.info(f"Total samples found: {len(full_dataset)}")
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=config.val_split, 
        random_state=42
    )
    
    # Create datasets with transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms(config.target_size)
    val_dataset.dataset.transform = get_val_transforms(config.target_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=2 if config.num_workers > 0 else None
    )
    
    logger.info(f'Training samples: {len(train_dataset)}')
    logger.info(f'Validation samples: {len(val_dataset)}')
    logger.info(f'Effective batch size: {config.batch_size * config.accumulation_steps}')
    
    # Initialize model
    logger.info("Initializing model...")
    model = UNet(
        n_channels=3, 
        n_classes=config.num_classes, 
        bilinear=True,
        architecture=config.model_architecture
    ).to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    logger.info(f"Model architecture: {config.model_architecture}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    # Loss function and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7, verbose=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_metrics': {'iou': [], 'dice': []},
        'val_metrics': {'iou': [], 'dice': []},
        'lr_history': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    best_model_path = save_dir / 'best_model.pth'
    
    logger.info("Starting training...")
    memory_tracker.log_memory_usage(logger, "before training")
    
    training_start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        logger.info(f'\nEpoch {epoch+1}/{config.num_epochs}')
        logger.info('-' * 70)
        
        # Cleanup before each epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            config.accumulation_steps, config.use_amp, logger, memory_tracker
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, config.use_amp, logger, memory_tracker
        )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['lr_history'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        for metric_name in ['iou', 'dice']:
            history['train_metrics'][metric_name].append(train_metrics[metric_name])
            history['val_metrics'][metric_name].append(val_metrics[metric_name])
        
        # Log epoch summary
        logger.info("=" * 50)
        logger.info(f"EPOCH {epoch+1} SUMMARY")
        logger.info("=" * 50)
        logger.info(f'Epoch time: {epoch_time:.2f} seconds ({epoch_time/60:.1f} minutes)')
        logger.info(f'Train Loss: {train_loss:.4f}, Train IoU: {train_metrics["iou"]:.4f}, Train Dice: {train_metrics["dice"]:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}, Val IoU: {val_metrics["iou"]:.4f}, Val Dice: {val_metrics["dice"]:.4f}')
        logger.info(f'Learning Rate: {current_lr:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'config': config.__dict__,
                'history': history
            }, best_model_path)
            logger.info(f'✓ New best model saved with validation loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config.checkpoint_frequency == 0:
            checkpoint_manager.save_checkpoint(
                epoch, model, optimizer, scheduler, val_loss, val_metrics,
                train_loss, train_metrics, config, history, epoch_time, logger
            )
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
        
        # Memory leak detection
        if memory_tracker.detect_memory_leak(logger, threshold_mb=500.0):
            logger.warning("Significant memory increase detected. Consider reducing batch size or enabling gradient checkpointing.")
    
    total_training_time = time.time() - training_start_time
    
    # Plot training history
    plot_path = save_dir / 'training_history.png'
    plot_training_history(
        history['train_losses'], 
        history['val_losses'],
        history['train_metrics'],
        history['val_metrics'],
        history['epoch_times'],
        str(plot_path)
    )
    
    # Save final history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Training completion summary
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.1f} minutes)")
    logger.info(f"Average epoch time: {np.mean(history['epoch_times'][1:]):.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation loss: {val_loss:.4f}")
    logger.info(f"Best model saved at: {best_model_path}")
    logger.info(f"Training history saved at: {history_path}")
    logger.info(f"Training plots saved at: {plot_path}")
    
    memory_tracker.log_memory_usage(logger, "after training")
    memory_tracker.cleanup()
    
    return model, history


import os
import sys
import random
from pathlib import Path
import argparse

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training import TrainingConfig
from src.training.trainer import train_model
from src.synthesis.image_operations import ImageOperations
from src.core.logger_config import setup_application_logger


def argument_parser():
    parser = argparse.ArgumentParser(description='Train U-Net for image segmentation')
    parser.add_argument('--root_dir', type=str, default='tests/test_images/output/synthetic_dataset_by_superimposing', 
                       help='Root directory containing training data')
    parser.add_argument('--save_dir', type=str, default='models/new_test', 
                       help='Directory to save models and results')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--target_size', type=int, nargs=2, default=[1792, 1792], 
                       help='Target image size (height width)')
    parser.add_argument('--architecture', type=str, choices=['standard', 'lightweight', "modified"], 
                       default='modified', help='Model architecture')
    parser.add_argument('--no_amp', action='store_true', help='Disable automatic mixed precision')
    parser.add_argument('--checkpoint_freq', type=int, default=5, help='Checkpoint frequency (epochs)')
    parser.add_argument('--no_memory_tracking', action='store_true', help='Disable memory leak detection')
    parser.add_argument('--arch_config', type=str, default="config/models/mini_unet.yaml")
    
    args = parser.parse_args()
    
    return args


def main(args):

    

    # Create configuration
    config = TrainingConfig(
        root_dir=args.root_dir,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        target_size=tuple(args.target_size),
        model_architecture=args.architecture,
        use_amp=not args.no_amp,
        checkpoint_frequency=args.checkpoint_freq,
        memory_tracking=not args.no_memory_tracking
    )
    image_operations = ImageOperations(app_logger)
    
    try:
        model, history = train_model(config, image_operations, app_logger, args.arch_config)
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    os.environ["DEBUG"] = "false"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

    app_logger = setup_application_logger(
        app_name="trainer", 
        log_file_name="logs/trainer.log"
    )

    args = argument_parser()

    main(args=args)
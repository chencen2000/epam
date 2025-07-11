from typing import Tuple
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    root_dir: str = "tests/temp"
    save_dir: str = "tests/model_saves"
    num_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    num_classes: int = 3  # Changed from 2 to 3 (background, dirt, scratches)
    class_names: Tuple[str, ...] = ("background", "scratches", "dirt")  # Added class names
    class_weights: Tuple[float, ...] = (0.3, 1.0, 1.2)  # Added class weights for imbalanced data
    target_size: Tuple[int, int] = (1024, 1024)
    accumulation_steps: int = 1
    val_split: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    early_stopping_patience: int = 10
    model_architecture: str = "lightweight"  # "standard" or "lightweight"
    checkpoint_frequency: int = 1  # Save checkpoint every N epochs
    memory_tracking: bool = True  # Enable memory leak detection
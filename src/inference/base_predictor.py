import os
from logging import Logger
from typing import Optional, Dict
from abc import ABC, abstractmethod

from numpy import ndarray
from torch import load

from src.models.unet import UNet
from src.core.device_utils import get_device_info
from src.core.logger_config import setup_application_logger


class BasePredictor(ABC):

    def __init__(self,
                 model_path:str,
                 device:str = "auto",
                 confidence_threshold:float = 0.5,
                 app_logger:Optional[Logger]=None) -> None:
        if app_logger is None:
            app_logger = setup_application_logger()
        self.logger = app_logger.getChild('BasePredictor')

        self.device = get_device_info(self.logger)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.config = None

        # Updated color scheme for multi-class visualization
        self.colors = {
            'background': (0, 0, 0),        # Black
            'dirt': (0, 255, 0),           # Green
            'scratches': (255, 0, 0),      # Red
            'ground_truth': (0, 255, 255), # Cyan
            'true_positive': (255, 255, 0), # Yellow
            'false_positive': (255, 0, 255), # Magenta
            'false_negative': (0, 255, 255), # Cyan
            'original': (128, 128, 128),
            'screen_boundary': (0, 255, 0),
            'patch_boundary': (255, 255, 0),
        }
        
        # Class names for multi-class
        self.class_names = ['background', 'dirt', 'scratches']
        
        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load the trained model from checkpoint."""
        self.logger.debug(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            checkpoint = load(model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            self.logger.warning(f"Failed to load with weights_only=True: {e}")
            self.logger.warning("Falling back to weights_only=False. Ensure the checkpoint is from a trusted source.")
            checkpoint = load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint.get('config', {})
        self.architecture = self.config.get('model_architecture', 'standard')
        self.logger.debug(f"architecture = {self.architecture}")
        
        # FIXED: Use correct channel count and classes for multi-class
        num_classes = self.config.get('num_classes', 3)  # Default to 3 for multi-class
        self.logger.debug(f"num_class = {num_classes}")
        
        self.model = UNet(
            n_channels=1,  # FIXED: Grayscale input, not 3
            n_classes=num_classes,  # FIXED: Use actual number of classes
            bilinear=True,
            architecture=self.architecture,
            app_logger=self.logger,
            config_path="config/models/light_weight.yaml"
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.debug(f"Model loaded successfully!")
        self.logger.debug(f"Architecture: {self.architecture}")
        self.logger.debug(f"Device: {self.device}")
        self.logger.debug(f"Number of classes: {num_classes}")
        self.logger.debug(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    @abstractmethod
    def predict(self, image: ndarray, return_raw: bool = False) -> Dict:
        pass

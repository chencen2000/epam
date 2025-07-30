import os
import time
from logging import Logger
from typing import Optional, Dict
from abc import ABC, abstractmethod

import cv2
from numpy import ndarray
import numpy as np
from torch import load
from ultralytics import YOLO

from src.models.unet import UNet
from src.target_labels import TargetLabels
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
        self.suction_cup_model = None

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
        # self.class_names = ['background',  'scratches', 'dirt',]
        self.class_names = TargetLabels.values()
        
        self.load_model(model_path)
        self._load_suction_cup_model()

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
            config_path=None
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.debug(f"Model loaded successfully!")
        self.logger.debug(f"Architecture: {self.architecture}")
        self.logger.debug(f"Device: {self.device}")
        self.logger.debug(f"Number of classes: {num_classes}")
        self.logger.debug(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _load_suction_cup_model(self, model_path:str="models/suction_cup_model/best.pt") -> None:
        self.suction_cup_model = YOLO(model_path)

    def classify(self, img):
        # Calculate horizontal projection (mean of pixel values along rows)
        horizontal_projection = np.mean(img, axis=1)
        mean = np.mean(horizontal_projection) # mean projection value (aka ~background color)
        if np.any(horizontal_projection > mean + 50): # some line projection is much higher than mean projection   
            return True
        return False

    def get_device_bbox(self, image, scale, log_file_name):
        resized = cv2.resize(image, None, fx=scale, fy=scale)
        filtered = cv2.medianBlur(resized, 13) # to remove bright stripes from PowerON samples, glares, etc.
        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Calculate the histogram
        hist = cv2.calcHist([normalized], [0], None, [256], [0, 256])
        # Calculate the cumulative histogram
        cumulative_hist = np.cumsum(hist)
        max_val = cumulative_hist.max()
        # Normalize the cumulative histogram
        cumulative_hist_normalized = cumulative_hist / max_val

        # Auto threshold adjustment
        threshold = 64 # suitable for most of the cases, except some rear surfaces
        if cumulative_hist_normalized[threshold] > 0.62: # area detected with this threshold is less than 38% of the image
            while cumulative_hist_normalized[threshold] > 0.62: # loop until area above threshold becomes at least 38%
                threshold -= 1
            if threshold > 40: # if it is still above 40
                threshold = max(40, threshold-4) # safe to make it even lower unless it is abobe 40 (there has to be plateau)
        print(threshold)

        # Threshold normalized image to binary
        _, binary_image = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            
        # Default fall-off (if binarization didn't produce anything, which is weird)
        h, w = image.shape
        xmin = int(w * 0.12)
        ymin = int(h * 0.15)
        xmax = int(w * (1.0 - 0.12))
        ymax = int(h * (1.0 - 0.15))
        return xmin, ymin, xmax-xmin, ymax-ymin
    
    def check_power_on_device_algo(self, folder_path:str, image, image_name:str):
        # Image for drawing
        img_h, img_w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        start_time = time.time()

         # Detect device region
        x, y, w, h = self.get_device_bbox(image, 0.04, "")
        detect_time = time.time()
        cv2.rectangle(rgb_image, (x,y), (x+w,y+h), (0,255,0), thickness=15) # draw

        # Define overall region for PowerON detection
        rx = 0
        rw = img_w
        ry = y + int(h * 0.01)
        rh = int(h * 0.052)
        cv2.rectangle(rgb_image, (rx,ry), (rx+rw,ry+rh), (255,0,0), thickness=15) # draw

        # Define left sub-regions
        rw = x
        left_outside_region = image[ry:ry+rh, rx:rx+rw]
        cv2.rectangle(rgb_image, (rx,ry), (rx+rw,ry+rh), (0,0,255), thickness=15) # draw
        rx = x + int(w * 0.14)
        rw = int(w * 0.12)
        left_inside_region = image[ry:ry+rh, rx:rx+rw]
        cv2.rectangle(rgb_image, (rx,ry), (rx+rw,ry+rh), (0,0,255), thickness=15) # draw

        # Define right sub-regions
        rx = x + w
        rw = img_w - rx
        right_outside_region = image[ry:ry+rh, rx:rx+rw]
        cv2.rectangle(rgb_image, (rx,ry), (rx+rw,ry+rh), (0,0,255), thickness=15) # draw
        rx = x + int(w * 0.74)
        rw = int(w * 0.12)
        right_inside_region = image[ry:ry+rh, rx:rx+rw]
        cv2.rectangle(rgb_image, (rx,ry), (rx+rw,ry+rh), (0,0,255), thickness=15) # draw

        classify_time = time.time()
        # Classify & print result
        classify_result = self.classify(left_outside_region) or self.classify(left_inside_region) or self.classify(right_outside_region) or self.classify(right_inside_region)
        end_time = time.time()

        if classify_result:
            self.logger.info('power ON')
        else:
            self.logger.info('power OFF')

        bbox_detection_time = round((detect_time - start_time) * 1000, 2) # in milliseconds
        power_on_classify_time = round((classify_time - detect_time) * 1000, 2) # in milliseconds
        self.logger.info(f'Bbox detection time = {bbox_detection_time}ms, PowerON classification time = {power_on_classify_time}ms')
        
        resized_rgb = cv2.resize(rgb_image, None, fx=0.2, fy=0.2) # downscale log image to save space           
        cv2.imwrite(f'{folder_path}/{image_name}_power_on_classification{"_POWER_ON_DETECTED" if classify_result else ""}.png', resized_rgb) # write log image

    @abstractmethod
    def predict(self, image: ndarray, return_raw: bool = False) -> Dict:
        pass

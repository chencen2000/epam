
import os
import psutil
import logging

import torch


def get_device_info(logger: logging.Logger) -> torch.device:
    """Get device information and set optimal settings"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name()})"
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        cpu_count = os.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024**3
        device_name = f"CPU ({cpu_count} cores)"
        logger.info(f"CPU Cores: {cpu_count}")
        logger.info(f"System Memory: {memory_gb:.1f}GB")
    
    logger.info(f"Using device: {device_name}")
    return device
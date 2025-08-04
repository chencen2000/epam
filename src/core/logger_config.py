import os
import logging
from logging.handlers import RotatingFileHandler


def setup_application_logger(app_name:str="my_app", log_file_name:str='application.log'):
    """Set up the main application logger"""
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate logs if logger already configured
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_level = logging.DEBUG if os.getenv('DEBUG', 'False').lower() == 'true' else logging.INFO
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file_name, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    # file_handler.setLevel(console_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# # Initialize the logger when module is imported
# app_logger = setup_application_logger()

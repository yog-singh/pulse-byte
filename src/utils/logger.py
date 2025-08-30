"""
Logging utilities for PulseByte.
"""

import sys
from pathlib import Path
from loguru import logger
from config.settings import LOGGING_CONFIG


def setup_logging():
    """Set up logging configuration."""
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        level=LOGGING_CONFIG['level'],
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True
    )
    
    # File handler
    log_file = LOGGING_CONFIG['log_file']
    log_file.parent.mkdir(exist_ok=True)
    
    logger.add(
        log_file,
        level=LOGGING_CONFIG['level'],
        format=LOGGING_CONFIG['format'],
        rotation=LOGGING_CONFIG['rotation'],
        retention=LOGGING_CONFIG['retention'],
        compression="zip"
    )
    
    logger.info("Logging initialized successfully")


def get_logger(name: str):
    """Get a logger instance with the specified name."""
    return logger.bind(name=name)

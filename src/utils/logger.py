"""
Production Logging System
Structured logging with file rotation and levels
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

class ProductionLogger:
    """Production-grade logging system"""
    
    def __init__(self, name: str, config: dict):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config['level']))
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Create formatters
        formatter = logging.Formatter(config['format'])
        
        # File handler with rotation
        log_file = Path(config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config['max_bytes'],
            backupCount=config['backup_count']
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if config.get('console', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, config['level']))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message"""
        if error:
            self.logger.error(f"{message}: {str(error)}", exc_info=True, extra=kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message"""
        if error:
            self.logger.critical(f"{message}: {str(error)}", exc_info=True, extra=kwargs)
        else:
            self.logger.critical(message, extra=kwargs)

# Global logger instance
_loggers = {}

def get_logger(name: str = "financial_agent") -> ProductionLogger:
    """Get logger instance (singleton per name)"""
    global _loggers
    
    if name not in _loggers:
        from src.config.config import get_config
        config = get_config()
        _loggers[name] = ProductionLogger(name, config['logging'])
    
    return _loggers[name]

"""
Production Logging System
Structured logging with file rotation and levels
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Global flag to ensure we only configure console encoding once
_CONSOLE_CONFIGURED = False

class ProductionLogger:
    """Production-grade logging system"""
    
    def __init__(self, name: str, config: dict):
        global _CONSOLE_CONFIGURED
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config['level']))
        self.logger.handlers.clear()  # Clear existing handlers
        
        # Create formatters
        formatter = logging.Formatter(config['format'])
        
        # File handler with rotation (UTF-8 encoding)
        log_file = Path(config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config['max_bytes'],
            backupCount=config['backup_count'],
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if config.get('console', True):
            # Configure Windows console for UTF-8 (only once)
            if sys.platform == 'win32' and not _CONSOLE_CONFIGURED:
                try:
                    # Try to set console code page to UTF-8
                    os.system('chcp 65001 > nul 2>&1')
                    _CONSOLE_CONFIGURED = True
                except Exception:
                    pass
            
            # Create console handler with error handling for encoding
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, config['level']))
            
            # Create a custom formatter that handles encoding errors
            class SafeFormatter(logging.Formatter):
                def format(self, record):
                    msg = super().format(record)
                    # Replace problematic Unicode characters with ASCII equivalents
                    replacements = {
                        '\u2192': '->',  # Right arrow
                        '\u2705': '[OK]',  # Check mark
                        '\u274c': '[X]',  # Cross mark
                        '\u26a0': '[!]',  # Warning sign
                        '\u20b9': 'Rs.',  # Indian Rupee
                        '\u00a3': 'GBP',  # Pound
                        '\u20ac': 'EUR',  # Euro
                    }
                    for char, replacement in replacements.items():
                        msg = msg.replace(char, replacement)
                    return msg
            
            safe_formatter = SafeFormatter(config['format'])
            console_handler.setFormatter(safe_formatter)
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

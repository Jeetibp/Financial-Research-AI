"""
Input Validation Utilities
Validates and sanitizes user inputs to prevent errors
"""

import re
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InputValidator:
    """Validates various types of inputs"""
    
    @staticmethod
    def validate_query(query: str, min_length: int = 1, max_length: int = 5000) -> tuple[bool, Optional[str]]:
        """
        Validate search/research query
        
        Args:
            query: Query string to validate
            min_length: Minimum query length
            max_length: Maximum query length
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check type
            if not isinstance(query, str):
                return False, "Query must be a string"
            
            # Strip whitespace
            query = query.strip()
            
            # Check length
            if len(query) < min_length:
                return False, f"Query too short (minimum {min_length} characters)"
            
            if len(query) > max_length:
                return False, f"Query too long (maximum {max_length} characters)"
            
            # Check for suspicious patterns (basic SQL injection prevention)
            suspicious_patterns = [
                r"(?i)(union.*select|drop\s+table|insert\s+into|delete\s+from)",
                r"(?i)(exec\s*\(|execute\s*\()",
                r"(?i)(<script|javascript:|onerror=)",
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, query):
                    logger.warning(f"Suspicious pattern detected in query: {pattern}")
                    return False, "Query contains suspicious patterns"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_extensions: Optional[List[str]] = None) -> tuple[bool, Optional[str]]:
        """
        Validate file path
        
        Args:
            file_path: Path to validate
            allowed_extensions: List of allowed file extensions (e.g., ['.pdf', '.docx'])
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check type
            if not isinstance(file_path, str):
                return False, "File path must be a string"
            
            # Create Path object
            path = Path(file_path)
            
            # Check if exists
            if not path.exists():
                return False, f"File not found: {file_path}"
            
            # Check if it's a file (not directory)
            if not path.is_file():
                return False, f"Path is not a file: {file_path}"
            
            # Check file extension
            if allowed_extensions:
                if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                    return False, f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            
            # Check file size (max 50MB by default)
            max_size = 50 * 1024 * 1024  # 50MB
            file_size = path.stat().st_size
            if file_size > max_size:
                return False, f"File too large ({file_size / 1024 / 1024:.2f}MB). Maximum: 50MB"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating file path: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        try:
            # Remove path components
            filename = Path(filename).name
            
            # Remove dangerous characters
            filename = re.sub(r'[^\w\s\-\.]', '', filename)
            
            # Limit length
            if len(filename) > 255:
                name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
                filename = name[:250] + ('.' + ext if ext else '')
            
            return filename
            
        except Exception as e:
            logger.error(f"Error sanitizing filename: {e}")
            return "file.txt"
    
    @staticmethod
    def validate_ticker(ticker: str) -> tuple[bool, Optional[str]]:
        """
        Validate stock ticker symbol
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check type
            if not isinstance(ticker, str):
                return False, "Ticker must be a string"
            
            ticker = ticker.strip().upper()
            
            # Check length (typically 1-5 characters)
            if len(ticker) < 1 or len(ticker) > 10:
                return False, "Ticker length must be between 1-10 characters"
            
            # Check format (letters, numbers, dots, hyphens)
            if not re.match(r'^[A-Z0-9.\-]+$', ticker):
                return False, "Ticker contains invalid characters"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating ticker: {e}")
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_session_id(session_id: str) -> tuple[bool, Optional[str]]:
        """
        Validate session ID
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check type
            if not isinstance(session_id, str):
                return False, "Session ID must be a string"
            
            session_id = session_id.strip()
            
            # Check length
            if len(session_id) < 8 or len(session_id) > 128:
                return False, "Session ID length must be between 8-128 characters"
            
            # Check format (alphanumeric and hyphens only)
            if not re.match(r'^[a-zA-Z0-9\-]+$', session_id):
                return False, "Session ID contains invalid characters"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating session ID: {e}")
            return False, f"Validation error: {str(e)}"


def validate_and_sanitize_query(query: str) -> str:
    """
    Validate and sanitize query string
    
    Args:
        query: Raw query string
        
    Returns:
        Sanitized query string
        
    Raises:
        ValueError: If query is invalid
    """
    validator = InputValidator()
    is_valid, error = validator.validate_query(query)
    
    if not is_valid:
        raise ValueError(error)
    
    # Sanitize: strip, normalize whitespace
    query = ' '.join(query.split())
    
    return query

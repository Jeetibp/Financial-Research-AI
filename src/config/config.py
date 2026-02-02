"""
Configuration Management System
Handles loading and validation of configuration
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Central configuration manager"""
    
    def __init__(self, config_path: str = "src/config/settings.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._load_env_vars()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def _load_env_vars(self):
        """Load environment variables and inject into config"""
        load_dotenv()
        
        # Get API keys from environment
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        # Inject API key into config
        if 'openai' in self.config:
            self.config['openai']['api_key'] = openai_key
    
    def _validate_config(self):
        """Validate required configuration sections"""
        required_sections = ['app', 'logging', 'rag', 'research', 'openai']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Create logs directory
        logs_dir = Path(self.config['logging']['file']).parent
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create vector store directory
        if 'rag' in self.config:
            vector_store_path = Path(self.config['rag']['vector_store_path'])
            os.makedirs(vector_store_path, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation
        Example: config.get('openai.model')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.config[key]


# Global config instance
_config = None


def get_config() -> Config:
    """Get global config instance (singleton)"""
    global _config
    if _config is None:
        _config = Config()
    return _config

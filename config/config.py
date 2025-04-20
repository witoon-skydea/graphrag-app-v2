"""
Configuration Module for GraphRAG

This module manages configuration settings for the GraphRAG system,
loading them from environment variables or configuration files.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration manager for GraphRAG system.
    
    Manages configuration settings, loading them from environment variables
    or configuration files.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to a configuration file (optional)
        """
        # Default configuration
        self.config = {
            # General settings
            "persist_path": os.getenv("GRAPHRAG_PERSIST_PATH", "./graphrag_data"),
            
            # Embedding settings
            "embedding_source": os.getenv("GRAPHRAG_EMBEDDING_SOURCE", "ollama"),
            "model_name": os.getenv("GRAPHRAG_MODEL_NAME", "llama2"),
            "api_key": os.getenv("GRAPHRAG_API_KEY", ""),
            
            # Knowledge Graph settings
            "extraction_method": os.getenv("GRAPHRAG_EXTRACTION_METHOD", "ollama"),
            "identification_method": os.getenv("GRAPHRAG_IDENTIFICATION_METHOD", "ollama"),
            "similarity_threshold": float(os.getenv("GRAPHRAG_SIMILARITY_THRESHOLD", "0.85")),
            "confidence_threshold": float(os.getenv("GRAPHRAG_CONFIDENCE_THRESHOLD", "0.5")),
            
            # Vector Database settings
            "weaviate_url": os.getenv("GRAPHRAG_WEAVIATE_URL", "http://localhost:8080"),
            "weaviate_api_key": os.getenv("GRAPHRAG_WEAVIATE_API_KEY", ""),
            
            # Document processing settings
            "chunk_size": int(os.getenv("GRAPHRAG_CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "200")),
            
            # OCR settings
            "ocr_language": os.getenv("GRAPHRAG_OCR_LANGUAGE", "eng+tha"),
            
            # Logging settings
            "log_level": os.getenv("GRAPHRAG_LOG_LEVEL", "INFO"),
        }
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # Set up logging level
        self._setup_logging()
        
        logger.info("GraphRAG configuration loaded")
    
    def _load_config_file(self, config_file: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update configuration with values from file
            self.config.update(file_config)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
    
    def _setup_logging(self) -> None:
        """Set up logging level based on configuration."""
        log_level = self.config["log_level"].upper()
        if log_level == "DEBUG":
            logging.getLogger().setLevel(logging.DEBUG)
        elif log_level == "INFO":
            logging.getLogger().setLevel(logging.INFO)
        elif log_level == "WARNING":
            logging.getLogger().setLevel(logging.WARNING)
        elif log_level == "ERROR":
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.INFO)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def save(self, config_file: str) -> None:
        """
        Save configuration to a file.
        
        Args:
            config_file: Path to the configuration file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration to {config_file}: {e}")
    
    def as_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()

# Create a default configuration instance
config = Config()

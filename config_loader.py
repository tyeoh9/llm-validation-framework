"""Configuration loader for API keys and other settings."""

import configparser
import os
from typing import Optional


def load_api_key(config_path: Optional[str] = None, provider: str = "ANTHROPIC") -> str:
    """
    Load API_KEY from config file using configparser.
    
    Args:
        config_path: Path to config file. If None, looks for config.ini in root directory.
        provider: Provider name (e.g., "ANTHROPIC", "GEMINI"). Defaults to "ANTHROPIC".
    """
    if config_path is None:
        # Get root directory (parent of this file's directory)
        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(root_dir, "config.ini")
    elif not os.path.isabs(config_path):
        # If relative path, resolve from root directory
        root_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(root_dir, config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    if provider not in config:
        raise ValueError(f"Provider '{provider}' not found in config file: {config_path}")
    
    api_key = config[provider].get("API_KEY")
    if not api_key:
        raise ValueError(f"API_KEY not found for provider '{provider}' in config file: {config_path}")
    
    return api_key


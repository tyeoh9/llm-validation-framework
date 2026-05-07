"""Configuration loader for API keys and other settings."""

import configparser
import os
from pathlib import Path
from typing import Optional


def load_api_key(config_path: Optional[str] = None, provider: str = "ANTHROPIC") -> str:
    """
    Load an API key for the given provider.

    Resolution order:
    1. Environment variable  <PROVIDER>_API_KEY  (e.g. ANTHROPIC_API_KEY)
    2. config.ini at the explicit ``config_path`` if supplied
    3. config.ini in the current working directory

    Args:
        config_path: Path to a config.ini file. Ignored when the env-var is set.
        provider: Section name in config.ini (e.g. "ANTHROPIC", "GEMINI").
    """
    env_key = f"{provider.upper()}_API_KEY"
    from_env = os.environ.get(env_key)
    if from_env:
        return from_env

    resolved = Path(config_path) if config_path else Path.cwd() / "config.ini"
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved

    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found: {resolved}\n"
            f"Set the {env_key} environment variable or provide a config.ini file."
        )

    config = configparser.ConfigParser()
    config.read(resolved)

    if provider not in config:
        raise ValueError(f"Provider '{provider}' not found in config file: {resolved}")

    api_key = config[provider].get("API_KEY")
    if not api_key:
        raise ValueError(f"API_KEY not found for provider '{provider}' in config file: {resolved}")

    return api_key

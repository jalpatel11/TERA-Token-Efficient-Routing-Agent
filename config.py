"""Configuration module for the routing agent.

Loads and validates all environment variables required at runtime.
"""

from __future__ import annotations

import os


class Config:
    """Manages system configuration and environment variables."""

    def __init__(self) -> None:
        """Initializes and validates configuration settings."""
        self.api_key: str = self._get_required_env("FIREWORKS_API_KEY")
        self.base_url: str = self._get_required_env("FIREWORKS_BASE_URL")
        
        allowed_models_str = self._get_required_env("ALLOWED_MODELS")
        self.allowed_models: list[str] = [
            model.strip()
            for model in allowed_models_str.split(",")
            if model.strip()
        ]
        
        if not self.allowed_models:
            raise ValueError(
                "ALLOWED_MODELS environment variable must contain at least "
                "one non-empty model identifier."
            )

    def _get_required_env(self, key: str) -> str:
        """Retrieves an environment variable, raising ValueError if not set."""
        value = os.environ.get(key)
        if not value:
            raise ValueError(
                f"Required environment variable '{key}' is not set."
            )
        return value.strip()

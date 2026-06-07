"""
config.py
---------
Application configuration module.

Loads all environment variables using Pydantic's BaseSettings, providing:
  - Automatic type coercion and validation at startup
  - Fail-fast behaviour on missing required values
  - A single importable `settings` singleton used across the entire app

Environment variables are read from a `.env` file (development) or injected
directly into the environment (production / Docker / CI).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the Auto Categorization Webhook service.

    All fields are read from environment variables (case-insensitive).
    Required fields will raise a ``ValidationError`` at import time if absent,
    preventing the application from starting with an invalid configuration.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Silently ignore undeclared env vars
    )

    # ------------------------------------------------------------------ #
    # OpenAI Settings                                                      #
    # ------------------------------------------------------------------ #

    openai_api_key: str = Field(
        ...,
        description="OpenAI API key. Required. Never log or expose this value.",
    )

    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model identifier used for classification.",
    )

    openai_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=120.0,
        description="Request timeout for OpenAI API calls, in seconds.",
    )

    openai_max_retries: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of retry attempts on transient OpenAI errors.",
    )

    # ------------------------------------------------------------------ #
    # Classification Settings                                              #
    # ------------------------------------------------------------------ #

    confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence score (0.0–1.0) below which a classification "
            "is flagged as low-confidence in the response."
        ),
    )

    mock_mode: bool = Field(
        default=True,
        description="Run classifier in mock mode without calling OpenAI.",
    )

    samples_path: str = Field(
        default="samples.json",
        description="Path to the JSON file containing few-shot examples.",
    )

    # ------------------------------------------------------------------ #
    # API / Security Settings                                              #
    # ------------------------------------------------------------------ #

    api_key: str = Field(
        ...,
        description=(
            "Secret key required in the `X-API-Key` header for all classify "
            "requests. Use a cryptographically random string (≥ 32 chars)."
        ),
    )

    # ------------------------------------------------------------------ #
    # Logging Settings                                                     #
    # ------------------------------------------------------------------ #

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Application log verbosity level.",
    )

    predictions_log_path: str = Field(
        default="logs/predictions.log",
        description="File path for the JSONL prediction audit log.",
    )

    log_max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum size in bytes before the log file is rotated.",
    )

    log_backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of rotated log file backups to retain.",
    )

    # ------------------------------------------------------------------ #
    # Server Settings                                                      #
    # ------------------------------------------------------------------ #

    app_host: str = Field(
        default="0.0.0.0",
        description="Host address for the Uvicorn server.",
    )

    app_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port number for the Uvicorn server.",
    )

    app_workers: int = Field(
        default=1,
        ge=1,
        description="Number of Uvicorn worker processes.",
    )

    debug: bool = Field(
        default=False,
        description=(
            "Enable FastAPI debug mode. "
            "MUST be False in production. Enables detailed error responses."
        ),
    )

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @field_validator("api_key")
    @classmethod
    def api_key_min_length(cls, value: str) -> str:
        """Enforce a minimum API key length to prevent weak secrets."""
        if len(value) < 16:
            raise ValueError(
                "API_KEY must be at least 16 characters long. "
                "Use a cryptographically random string."
            )
        return value

    @field_validator("openai_api_key")
    @classmethod
    def openai_key_must_start_with_sk(cls, value: str) -> str:
        """Perform a basic sanity check on the OpenAI key format."""
        if not value.startswith("sk-"):
            raise ValueError(
                "OPENAI_API_KEY does not look valid (expected prefix 'sk-'). "
                "Check your .env file."
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    Using ``lru_cache`` ensures the `.env` file is only parsed once per
    process lifetime, regardless of how many modules call ``get_settings()``.

    Returns
    -------
    Settings
        The validated, fully-populated settings object.

    Raises
    ------
    pydantic.ValidationError
        If any required environment variable is missing or invalid.
        This is intentionally raised at startup to fail fast.
    """
    return Settings()  # type: ignore[call-arg]


# Do NOT instantiate settings at module level.
# Doing so would cause an immediate crash on import in any environment
# (tests, Docker build, CI) where a .env file is absent.
#
# Instead, call get_settings() wherever settings are needed.
# The lru_cache decorator ensures the Settings object is constructed only once
# per process, giving identical singleton behaviour without the import-time risk.
#
# Usage:
#   from config import get_settings
#   settings = get_settings()
#
# Or inject via FastAPI dependency:
#   from fastapi import Depends
#   def my_route(settings: Settings = Depends(get_settings)): ...

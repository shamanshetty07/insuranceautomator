from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables or backend/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Vehicle Damage AI"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # Required: no default, so a missing value fails fast at startup.
    database_url: str = Field(min_length=1)

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    # Relative paths resolve against the process working directory (backend/).
    storage_root: Path = Path("../storage/inspections")

    # Image upload limits; enforced on the server regardless of frontend checks.
    max_upload_bytes: int = Field(default=15 * 1024 * 1024, gt=0)
    max_images_per_inspection: int = Field(default=20, gt=0)
    min_image_side_px: int = Field(default=32, gt=0)
    max_image_pixels: int = Field(default=40_000_000, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment

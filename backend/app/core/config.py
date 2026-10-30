from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.claims import MAX_CLAIMS_PER_REPORT
from app.schemas.runs import check_image_size_multiple


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

    # Damage detection (local inference only). "none" disables it. Ultralytics needs
    # the optional `detection` extra and a local weights file; relative paths resolve
    # against the process working directory (backend/).
    detector_backend: Literal["none", "ultralytics"] = "none"
    detector_weights_path: Path | None = None
    detector_model_name: str | None = Field(default=None, max_length=100)
    detector_model_version: str | None = Field(default=None, max_length=100)
    detector_confidence_threshold: float = Field(default=0.25, ge=0, le=1)
    detector_iou_threshold: float = Field(default=0.45, ge=0, le=1)
    # Must be a positive multiple of the model stride (32 for standard YOLO detection
    # models); never rounded. The loaded model's own stride is checked again at load.
    detector_image_size: int = Field(default=640, gt=0)
    detector_max_detections: int = Field(default=100, gt=0)
    detector_device: str | None = "cpu"
    # Opt-in restricted checkpoint loading (torch weights_only=True + Ultralytics'
    # allow-list). When true, loading FAILS rather than falling back if Ultralytics
    # cannot actually apply it (imported earlier without it, torch < 2.6, unknown
    # Ultralytics internals). Off by default: no checkpoint has been confirmed under it.
    detector_safe_load: bool = False
    # Expected lowercase hex SHA-256 of the approved weights file. When set, a file with
    # any other digest is refused before it is loaded. Unset: no pin (a warning is logged).
    detector_weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # A `running` detection run whose worker has not sent a heartbeat for this long is
    # treated as abandoned by the recovery command (manual; see detection/recovery.py).
    detector_stale_run_minutes: int = Field(default=30, gt=0)
    # How often a worker refreshes the heartbeat of the run it is processing.
    detector_heartbeat_seconds: int = Field(default=30, gt=0)

    # Local vision-language model (Qwen2-VL / Qwen2.5-VL) for report generation and claim
    # verification; needs the optional `vlm` extra and a LOCAL model directory (never
    # downloaded at runtime). "none" disables it. Decoding is greedy (temperature 0).
    vlm_backend: Literal["none", "transformers"] = "none"
    vlm_model_path: Path | None = None
    vlm_model_name: str | None = Field(default=None, max_length=100)
    vlm_model_version: str | None = Field(default=None, max_length=100)
    # Expected digest of the model directory (vlm.transformers_runner.model_dir_digest);
    # any other content is refused before loading. Unset: no pin.
    vlm_weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    vlm_device: str = "cpu"
    vlm_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    vlm_timeout_seconds: int = Field(default=180, gt=0)
    vlm_max_new_tokens: int = Field(default=1024, gt=0)
    vlm_max_images: int = Field(default=6, gt=0)
    vlm_max_image_pixels: int = Field(default=802_816, gt=0)  # ~896 x 896
    vlm_max_detections: int = Field(default=30, gt=0)
    vlm_max_claims: int = Field(default=20, gt=0, le=MAX_CLAIMS_PER_REPORT)

    @field_validator("detector_image_size")
    @classmethod
    def check_image_size(cls, v: int) -> int:
        return check_image_size_multiple(v)

    @field_validator("detector_weights_sha256", "vlm_weights_sha256", mode="before")
    @classmethod
    def lowercase_sha256(cls, v: object) -> object:
        return v.strip().lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_stale_threshold(self) -> "Settings":
        if self.detector_stale_run_minutes * 60 < 4 * self.detector_heartbeat_seconds:
            raise ValueError(
                "DETECTOR_STALE_RUN_MINUTES must be at least 4 heartbeat intervals "
                "(DETECTOR_HEARTBEAT_SECONDS), or a live run could look abandoned"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment

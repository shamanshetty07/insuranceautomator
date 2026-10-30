"""Decode, validate and normalize an uploaded image."""

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import AppError

logger = logging.getLogger(__name__)

EXIF_ORIENTATION_TAG = 0x0112

SUPPORTED_FORMATS: dict[str, tuple[str, str]] = {
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
}


@dataclass(frozen=True)
class ImageLimits:
    min_side_px: int
    max_pixels: int


@dataclass(frozen=True)
class ProcessedImage:
    data: bytes
    media_type: str
    extension: str
    width_px: int
    height_px: int
    exif_orientation: int | None


def _original_orientation(image: Image.Image) -> int | None:
    """Return the EXIF orientation if present and valid (1-8), else None."""
    try:
        value = image.getexif().get(EXIF_ORIENTATION_TAG)
    except Exception:
        logger.warning("Could not read EXIF orientation; ignoring")
        return None
    return value if isinstance(value, int) and 1 <= value <= 8 else None


def process_image(raw: bytes, limits: ImageLimits) -> ProcessedImage:
    if not raw:
        raise AppError(422, "empty_file", "The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError:
        raise AppError(422, "invalid_dimensions", "Image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError):
        raise AppError(
            415, "unsupported_media_type", "File is not a supported image (JPEG, PNG or WebP)."
        ) from None

    with image:
        image_format = "JPEG" if image.format == "MPO" else image.format
        if image_format not in SUPPORTED_FORMATS:
            raise AppError(
                415, "unsupported_media_type", "File is not a supported image (JPEG, PNG or WebP)."
            )
        media_type, extension = SUPPORTED_FORMATS[image_format]

        width, height = image.size
        if min(width, height) < limits.min_side_px:
            raise AppError(
                422,
                "invalid_dimensions",
                f"Image is too small; each side must be at least {limits.min_side_px}px.",
            )
        if width * height > limits.max_pixels:
            raise AppError(
                422,
                "invalid_dimensions",
                f"Image is too large; at most {limits.max_pixels} pixels are allowed.",
            )

        orientation = _original_orientation(image)
        try:
            image.load()
            normalized = ImageOps.exif_transpose(image)
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            raise AppError(422, "invalid_image", "The image is corrupt or cannot be decoded.") from None

        icc_profile = image.info.get("icc_profile")
        out = io.BytesIO()
        save_kwargs: dict[str, object] = {}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        if image_format == "JPEG":
            if normalized.mode not in ("RGB", "L"):
                normalized = normalized.convert("RGB")
            save_kwargs.update(quality=95)
        elif image_format == "WEBP":
            save_kwargs.update(quality=95)
        normalized.info.pop("comment", None)
        normalized.save(out, format=image_format, **save_kwargs)

        return ProcessedImage(
            data=out.getvalue(),
            media_type=media_type,
            extension=extension,
            width_px=normalized.width,
            height_px=normalized.height,
            exif_orientation=orientation,
        )

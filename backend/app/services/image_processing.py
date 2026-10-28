"""Decode, validate and normalize an uploaded image.

Trust model: only the decoded content matters. The client's filename and
Content-Type are never used to decide what the file is.
"""

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.errors import AppError

logger = logging.getLogger(__name__)

EXIF_ORIENTATION_TAG = 0x0112

# Pillow format -> (media type, file extension). Anything else is rejected.
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
    data: bytes  # normalized, re-encoded, metadata-stripped bytes to store
    media_type: str
    extension: str
    width_px: int  # of `data`, i.e. after orientation normalization
    height_px: int
    exif_orientation: int | None  # original value (1-8), informational only


def _original_orientation(image: Image.Image) -> int | None:
    """Return the EXIF orientation if present and valid (1-8), else None."""
    try:
        value = image.getexif().get(EXIF_ORIENTATION_TAG)
    except Exception:  # malformed EXIF must never fail the upload
        logger.warning("Could not read EXIF orientation; ignoring")
        return None
    return value if isinstance(value, int) and 1 <= value <= 8 else None


def process_image(raw: bytes, limits: ImageLimits) -> ProcessedImage:
    if not raw:
        raise AppError(422, "empty_file", "The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw))  # reads the header only
    except Image.DecompressionBombError:
        raise AppError(422, "invalid_dimensions", "Image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError):
        raise AppError(
            415, "unsupported_media_type", "File is not a supported image (JPEG, PNG or WebP)."
        ) from None

    with image:
        if image.format not in SUPPORTED_FORMATS:
            raise AppError(
                415, "unsupported_media_type", "File is not a supported image (JPEG, PNG or WebP)."
            )
        media_type, extension = SUPPORTED_FORMATS[image.format]

        # Check size from the header before any expensive pixel decoding.
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
            image.load()  # full decode: detects truncated/corrupt data
            normalized = ImageOps.exif_transpose(image)  # applies the rotation to pixels
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            raise AppError(422, "invalid_image", "The image is corrupt or cannot be decoded.") from None

        icc_profile = image.info.get("icc_profile")
        out = io.BytesIO()
        save_kwargs: dict[str, object] = {}
        if icc_profile:  # colour fidelity, not personal data
            save_kwargs["icc_profile"] = icc_profile
        if image.format == "JPEG":
            if normalized.mode not in ("RGB", "L"):
                normalized = normalized.convert("RGB")
            save_kwargs.update(quality=95)
        elif image.format == "WEBP":
            save_kwargs.update(quality=95)
        # No exif= argument and ImageOps clears the orientation tag: GPS, camera
        # model, timestamps etc. are not carried into the stored file.
        normalized.save(out, format=image.format, **save_kwargs)

        return ProcessedImage(
            data=out.getvalue(),
            media_type=media_type,
            extension=extension,
            width_px=normalized.width,
            height_px=normalized.height,
            exif_orientation=orientation,
        )

import io

import pytest
from PIL import Image

from app.core.errors import AppError
from app.services.image_processing import ImageLimits, process_image
from tests.image_utils import make_image_bytes, png_header_only

LIMITS = ImageLimits(min_side_px=16, max_pixels=1_000_000)


def _reject(raw: bytes) -> AppError:
    with pytest.raises(AppError) as info:
        process_image(raw, LIMITS)
    return info.value


@pytest.mark.parametrize(
    "fmt, media, ext", [("JPEG", "image/jpeg", "jpg"), ("PNG", "image/png", "png"), ("WEBP", "image/webp", "webp")]
)
def test_valid_formats(fmt, media, ext):
    result = process_image(make_image_bytes(fmt, (64, 32)), LIMITS)
    assert (result.media_type, result.extension) == (media, ext)
    assert (result.width_px, result.height_px) == (64, 32)
    assert result.exif_orientation is None
    decoded = Image.open(io.BytesIO(result.data))
    assert decoded.format == fmt and decoded.size == (64, 32)


def test_exif_rotation_is_applied_to_pixels_and_dimensions():
    raw = make_image_bytes("JPEG", (64, 32), orientation=6)
    assert Image.open(io.BytesIO(raw)).size == (64, 32)
    result = process_image(raw, LIMITS)
    assert (result.width_px, result.height_px) == (32, 64)
    assert result.exif_orientation == 6
    stored = Image.open(io.BytesIO(result.data))
    assert stored.size == (32, 64)
    assert stored.getexif().get(0x0112) in (None, 1)
    assert stored.convert("RGB").getpixel((stored.width - 2, 2))[2] > 150


def test_metadata_is_stripped():
    result = process_image(make_image_bytes("JPEG", orientation=3, gps=True), LIMITS)
    assert b"SecretCameraMaker" not in result.data
    assert not dict(Image.open(io.BytesIO(result.data)).getexif()).get(0x010F)


def test_invalid_orientation_value_is_ignored_safely():
    result = process_image(make_image_bytes("JPEG", orientation=99), LIMITS)
    assert result.exif_orientation is None
    assert (result.width_px, result.height_px) == (64, 32)


def test_empty_upload():
    err = _reject(b"")
    assert (err.status_code, err.code) == (422, "empty_file")


def test_not_an_image_even_with_image_extension():
    err = _reject(b"<html>definitely not a jpeg</html>")
    assert (err.status_code, err.code) == (415, "unsupported_media_type")


@pytest.mark.parametrize("fmt", ["GIF", "BMP"])
def test_unsupported_format(fmt):
    err = _reject(make_image_bytes(fmt))
    assert (err.status_code, err.code) == (415, "unsupported_media_type")


def test_corrupt_image_truncated():
    raw = make_image_bytes("JPEG", (256, 256))
    err = _reject(raw[: len(raw) // 2])
    assert (err.status_code, err.code) == (422, "invalid_image")


def test_corrupt_png_header_only_decode_failure_is_clean():
    err = _reject(png_header_only(64, 64))
    assert err.code == "invalid_image"


def test_too_small():
    err = _reject(make_image_bytes("PNG", (8, 8)))
    assert (err.status_code, err.code) == (422, "invalid_dimensions")


def test_too_many_pixels_rejected_from_header_without_decoding():
    err = _reject(png_header_only(5000, 5000))
    assert (err.status_code, err.code) == (422, "invalid_dimensions")


def test_decompression_bomb_header_rejected():
    err = _reject(png_header_only(100_000, 100_000))
    assert (err.status_code, err.code) == (422, "invalid_dimensions")

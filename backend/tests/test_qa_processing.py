"""QA gap tests for image normalization (no database needed)."""

import io
import struct

import pytest
from PIL import Image, ImageFile, PngImagePlugin

from app.core.errors import AppError
from app.services.image_processing import ImageLimits, process_image
from tests.image_utils import make_image_bytes

LIMITS = ImageLimits(min_side_px=16, max_pixels=1_000_000)


def _save(image: Image.Image, fmt: str, **kwargs) -> bytes:
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


@pytest.mark.parametrize(
    ("mode", "fmt"),
    [("RGBA", "PNG"), ("LA", "PNG"), ("P", "PNG"), ("I;16", "PNG"), ("1", "PNG"),
     ("RGBA", "WEBP"), ("CMYK", "JPEG"), ("L", "JPEG")],
)
def test_non_rgb_modes_are_accepted_and_stored_decodable(mode, fmt):
    result = process_image(_save(Image.new(mode, (64, 48)), fmt), LIMITS)
    stored = Image.open(io.BytesIO(result.data))
    stored.load()
    assert stored.format == fmt and stored.size == (64, 48)
    assert (result.width_px, result.height_px) == (64, 48)
    if fmt == "JPEG":
        assert stored.mode in ("RGB", "L")


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_multi_frame_input_is_stored_as_a_single_frame(fmt):
    frames = [Image.new("RGB", (64, 48), (i * 60, 0, 0)) for i in range(4)]
    raw = _save(frames[0], fmt, save_all=True, append_images=frames[1:])
    stored = Image.open(io.BytesIO(process_image(raw, LIMITS).data))
    assert getattr(stored, "n_frames", 1) == 1


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_exif_orientation_is_applied_for_png_and_webp(fmt):
    raw = make_image_bytes(fmt, (64, 32), orientation=6)
    result = process_image(raw, LIMITS)
    assert (result.width_px, result.height_px, result.exif_orientation) == (32, 64, 6)
    stored = Image.open(io.BytesIO(result.data))
    assert stored.size == (32, 64)
    assert stored.getexif().get(0x0112) in (None, 1)


def test_pixel_limit_is_inclusive():
    exact = ImageLimits(min_side_px=16, max_pixels=64 * 32)
    assert process_image(make_image_bytes("PNG", (64, 32)), exact).width_px == 64
    with pytest.raises(AppError) as exc:
        process_image(make_image_bytes("PNG", (65, 32)), exact)
    assert exc.value.code == "invalid_dimensions"


def _jpeg_with_declared_size(width: int, height: int) -> bytes:
    raw = bytearray(make_image_bytes("JPEG", (64, 32)))
    sof = raw.index(b"\xff\xc0")
    raw[sof + 5 : sof + 9] = struct.pack(">HH", height, width)
    return bytes(raw)


def test_oversized_jpeg_header_is_rejected_without_decoding(monkeypatch):
    def no_decode(self):
        raise AssertionError("pixel data must not be decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", no_decode)
    with pytest.raises(AppError) as exc:
        process_image(_jpeg_with_declared_size(2000, 2000), LIMITS)
    assert exc.value.code == "invalid_dimensions"


def test_jpeg_header_bomb_beyond_pillow_threshold_is_rejected():
    with pytest.raises(AppError) as exc:
        process_image(_jpeg_with_declared_size(60000, 60000), LIMITS)
    assert exc.value.code == "invalid_dimensions"


def test_png_text_chunks_and_xmp_are_not_stored():
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "SECRET-owner-name")
    info.add_itxt("XML:com.adobe.xmp", "SECRET-gps")
    raw = _save(Image.new("RGB", (64, 48)), "PNG", pnginfo=info)
    assert b"SECRET" not in process_image(raw, LIMITS).data


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
def test_xmp_packets_are_not_stored(fmt):
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/">SECRET-gps</x:xmpmeta>'
    raw = _save(Image.new("RGB", (64, 48)), fmt, xmp=xmp)
    assert b"SECRET" in raw
    assert b"SECRET" not in process_image(raw, LIMITS).data


def test_jpeg_comment_segment_is_not_stored():
    raw = _save(Image.new("RGB", (64, 48)), "JPEG", comment=b"SECRET-owner-name")
    assert b"SECRET" not in process_image(raw, LIMITS).data


def test_multi_picture_jpeg_from_cameras_is_accepted():
    frames = [Image.new("RGB", (64, 48), (200, 0, 0)), Image.new("RGB", (64, 48), (0, 200, 0))]
    raw = _save(frames[0], "MPO", save_all=True, append_images=frames[1:])
    result = process_image(raw, LIMITS)
    assert result.media_type == "image/jpeg"

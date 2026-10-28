"""Builders for in-memory test images."""

import io
import struct
import zlib

from PIL import Image


def make_image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (64, 32),
    orientation: int | None = None,
    color: tuple[int, int, int] = (200, 30, 30),
    gps: bool = False,
) -> bytes:
    """A solid image with a distinct top-left marker so rotations are detectable."""
    image = Image.new("RGB", size, color)
    image.paste((0, 0, 255), (0, 0, size[0] // 4, size[1] // 4))
    kwargs: dict[str, object] = {}
    if orientation is not None or gps:
        exif = Image.Exif()
        if orientation is not None:
            exif[0x0112] = orientation
        if gps:
            exif[0x010F] = "SecretCameraMaker"  # Make
        kwargs["exif"] = exif
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


def png_header_only(width: int, height: int) -> bytes:
    """A PNG whose header claims a size but has no pixel data (for size-limit checks)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")

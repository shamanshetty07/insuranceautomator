"""Reject oversized request bodies before Starlette parses (and spools) them.

The image route checks the size of its `file` part, but multipart parsing writes
every part to temporary storage first, so without this cap a client could send
arbitrarily large extra parts. The limit is the upload limit plus a small allowance
for multipart framing. Bodies without a Content-Length (chunked) are refused, so the
declared length is authoritative (the server never delivers more than it).
"""

from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import get_settings
from app.core.errors import _response

MULTIPART_OVERHEAD_BYTES = 64 * 1024


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def _limit(self, scope: Scope) -> int:
        # Resolve through the app's dependency overrides so tests' settings apply.
        fastapi_app = scope.get("app")
        overrides = getattr(fastapi_app, "dependency_overrides", {})
        settings = overrides.get(get_settings, get_settings)()
        return settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        length = headers.get(b"content-length")
        if length is None:
            if b"transfer-encoding" in headers:
                await _response(411, "length_required", "Content-Length is required.")(scope, receive, send)
                return
        else:
            limit = self._limit(scope)
            if not length.isdigit() or int(length) > limit:
                message = f"Request body exceeds the maximum size of {limit} bytes."
                # Same code as the route's own check: to clients both mean "upload too big".
                await _response(413, "file_too_large", message)(scope, receive, send)
                return
        await self.app(scope, receive, send)

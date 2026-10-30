"""Refuse TEST_DATABASE_URL values that could point at a database someone cares about; tests wipe it."""

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1"})


def disposable_database_problem(url: str) -> str | None:
    """None if `url` names a local PostgreSQL database whose name contains "test"; otherwise why not."""
    try:
        parsed = make_url(url)
    except ArgumentError:
        return "TEST_DATABASE_URL is not a valid database URL."
    if parsed.get_backend_name() != "postgresql":
        return "TEST_DATABASE_URL must be a PostgreSQL URL."
    if (parsed.host or "").lower() not in LOCAL_HOSTS:
        return "TEST_DATABASE_URL must point at localhost or a local socket; remote databases are never wiped."
    if "test" not in (parsed.database or "").lower():
        return "TEST_DATABASE_URL's database name must contain 'test' to mark it as disposable."
    if (parsed.query or {}).get("host"):
        return "TEST_DATABASE_URL must not override the host through query parameters."
    return None

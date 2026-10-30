import enum

from sqlalchemy import Enum


class RunStatus(str, enum.Enum):
    """Lifecycle of a detection or verification run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def run_status_type(name: str) -> Enum:
    """VARCHAR + CHECK (not a native PG enum), matching the other enumerations."""
    return Enum(
        RunStatus,
        native_enum=False,
        length=16,
        create_constraint=True,
        name=name,
        values_callable=lambda e: [m.value for m in e],
    )


COMPLETED_AT_CHECK = "(status IN ('completed', 'failed')) = (completed_at IS NOT NULL)"
SHA256_HEX_CHECK = "{col} ~ '^[0-9a-f]{{64}}$'"

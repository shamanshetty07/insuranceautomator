"""detection run heartbeat: last sign of life of the worker holding a running run

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-29

Adds nullable detection_runs.heartbeat_at. Existing rows keep NULL: stale-run recovery
falls back to created_at for them, which is exactly the old behaviour. No backfill.
Downgrade drops the column (heartbeat history is lost; nothing else depends on it).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("detection_runs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("detection_runs", "heartbeat_at")

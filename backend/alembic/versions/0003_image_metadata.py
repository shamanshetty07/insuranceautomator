"""image metadata: normalized dimensions, EXIF orientation, per-inspection checksum uniqueness"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inspection_images", sa.Column("width_px", sa.Integer(), nullable=True))
    op.add_column("inspection_images", sa.Column("height_px", sa.Integer(), nullable=True))
    op.add_column("inspection_images", sa.Column("exif_orientation", sa.SmallInteger(), nullable=True))
    op.create_check_constraint(
        "ck_inspection_images_dimensions", "inspection_images", "width_px > 0 AND height_px > 0"
    )
    op.create_check_constraint(
        "ck_inspection_images_exif_orientation", "inspection_images", "exif_orientation BETWEEN 1 AND 8"
    )
    op.create_index(
        "uq_inspection_images_sha256",
        "inspection_images",
        ["inspection_id", "sha256"],
        unique=True,
        postgresql_where=sa.text("sha256 IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_inspection_images_sha256", table_name="inspection_images")
    op.drop_column("inspection_images", "exif_orientation")
    op.drop_column("inspection_images", "height_px")
    op.drop_column("inspection_images", "width_px")

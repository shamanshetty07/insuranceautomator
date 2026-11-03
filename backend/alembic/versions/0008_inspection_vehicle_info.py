"""optional vehicle details on inspections"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("inspections", sa.Column("vehicle_make", sa.String(length=100), nullable=True))
    op.add_column("inspections", sa.Column("vehicle_model", sa.String(length=100), nullable=True))
    op.add_column("inspections", sa.Column("vehicle_year", sa.Integer(), nullable=True))
    op.add_column("inspections", sa.Column("vehicle_registration", sa.String(length=32), nullable=True))
    op.add_column("inspections", sa.Column("vehicle_color", sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column("inspections", "vehicle_color")
    op.drop_column("inspections", "vehicle_registration")
    op.drop_column("inspections", "vehicle_year")
    op.drop_column("inspections", "vehicle_model")
    op.drop_column("inspections", "vehicle_make")

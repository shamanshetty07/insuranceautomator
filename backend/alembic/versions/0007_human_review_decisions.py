"""persist immutable human decisions for review-required verification results"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "human_review_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("verification_result_id", sa.Uuid(), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("outcome IN ('approved', 'rejected')", name="human_review_outcome"),
        sa.ForeignKeyConstraint(["verification_result_id"], ["verification_results.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("verification_result_id"),
    )
    op.create_index("ix_human_review_decisions_verification_result_id", "human_review_decisions", ["verification_result_id"])


def downgrade() -> None:
    op.drop_index("ix_human_review_decisions_verification_result_id", table_name="human_review_decisions")
    op.drop_table("human_review_decisions")

"""structured report claims, claim image links, verification_results.report_claim_id"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Frozen on purpose: must not change when the Python enum gains members.
CLAIM_TYPES = (
    "visible_damage_presence",
    "damage_type",
    "damage_location",
    "damage_severity",
    "hidden_structural_damage",
    "repair_recommendation",
    "repair_cost",
    "damage_cause",
    "other",
)
CLAIM_ID_CHECK = "claim_id ~ '^[A-Za-z0-9_-]{1,64}$'"
JSON_ARRAYS_CHECK = (
    "jsonb_typeof(image_ids) = 'array' AND jsonb_typeof(prediction_ids) = 'array' "
    "AND jsonb_typeof(evidence) = 'array'"
)


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def _json_array(name: str) -> sa.Column:
    return sa.Column(name, postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"),
                     nullable=False)


def upgrade() -> None:
    op.create_table(
        "report_claims",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("generated_report_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.String(64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("claim_type", sa.String(32), nullable=False),
        sa.Column("damage_type", sa.String(100), nullable=True),
        _json_array("image_ids"),
        _json_array("prediction_ids"),
        _json_array("evidence"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["generated_report_id"], ["generated_reports.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("generated_report_id", "claim_id", name="uq_report_claims_report_claim"),
        sa.UniqueConstraint("generated_report_id", "position", name="uq_report_claims_report_position"),
        sa.CheckConstraint(_in_list("claim_type", CLAIM_TYPES), name="report_claim_type"),
        sa.CheckConstraint(CLAIM_ID_CHECK, name="ck_report_claims_claim_id"),
        sa.CheckConstraint("char_length(text) > 0", name="ck_report_claims_text"),
        sa.CheckConstraint("position >= 0", name="ck_report_claims_position"),
        sa.CheckConstraint(JSON_ARRAYS_CHECK, name="ck_report_claims_json_arrays"),
    )
    op.create_table(
        "report_claim_images",
        sa.Column("report_claim_id", sa.Uuid(), nullable=False),
        sa.Column("inspection_image_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["report_claim_id"], ["report_claims.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["inspection_image_id"], ["inspection_images.id"], deferrable=True, initially="DEFERRED"
        ),
        sa.PrimaryKeyConstraint("report_claim_id", "inspection_image_id"),
    )
    op.create_index(
        "ix_report_claim_images_inspection_image_id", "report_claim_images", ["inspection_image_id"]
    )

    op.add_column("verification_results", sa.Column("report_claim_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "verification_results_report_claim_id_fkey",
        "verification_results",
        "report_claims",
        ["report_claim_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_verification_results_report_claim_id", "verification_results", ["report_claim_id"])
    op.create_index(
        "uq_verification_results_run_report_claim",
        "verification_results",
        ["verification_run_id", "report_claim_id"],
        unique=True,
        postgresql_where=sa.text("report_claim_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_verification_results_run_report_claim", table_name="verification_results")
    op.drop_index("ix_verification_results_report_claim_id", table_name="verification_results")
    op.drop_column("verification_results", "report_claim_id")
    op.drop_index("ix_report_claim_images_inspection_image_id", table_name="report_claim_images")
    op.drop_table("report_claim_images")
    op.drop_table("report_claims")

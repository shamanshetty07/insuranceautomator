"""claim-level verification: verdict, claim type/id, review flag, policy reason codes"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "verification_results"


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
VERDICTS = ("supported", "contradicted", "insufficient_evidence")


def _in_list(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.alter_column(TABLE, "verification_label", existing_type=sa.String(32), nullable=True)
    op.add_column(TABLE, sa.Column("claim_id", sa.String(64), nullable=True))
    op.add_column(TABLE, sa.Column("claim_type", sa.String(32), nullable=True))
    op.add_column(TABLE, sa.Column("verdict", sa.String(32), nullable=True))
    op.add_column(
        TABLE,
        sa.Column("needs_human_review", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.add_column(
        TABLE,
        sa.Column(
            "policy_reason_codes",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.create_check_constraint("claim_type", TABLE, _in_list("claim_type", CLAIM_TYPES))
    op.create_check_constraint("verdict", TABLE, _in_list("verdict", VERDICTS))

    op.execute(
        """
        UPDATE verification_results SET verdict = CASE verification_label
            WHEN 'supported' THEN 'supported'
            WHEN 'contradicted' THEN 'contradicted'
            WHEN 'wrong_location' THEN 'contradicted'
            WHEN 'wrong_damage_type' THEN 'contradicted'
            WHEN 'unverifiable' THEN 'insufficient_evidence'
            ELSE NULL
        END
        """
    )

    op.create_check_constraint(
        "ck_verification_results_omitted_has_no_verdict",
        TABLE,
        "verification_label IS DISTINCT FROM 'omitted_damage' OR verdict IS NULL",
    )
    op.create_index(
        "uq_verification_results_run_claim",
        TABLE,
        ["verification_run_id", "claim_id"],
        unique=True,
        postgresql_where=sa.text("claim_id IS NOT NULL"),
    )
    op.create_index(
        "ix_verification_results_review",
        TABLE,
        ["verification_run_id"],
        postgresql_where=sa.text("needs_human_review"),
    )


def downgrade() -> None:
    op.drop_index("ix_verification_results_review", table_name=TABLE)
    op.drop_index("uq_verification_results_run_claim", table_name=TABLE)
    op.drop_constraint("ck_verification_results_omitted_has_no_verdict", TABLE, type_="check")
    op.execute("UPDATE verification_results SET verification_label = 'unverifiable' WHERE verification_label IS NULL")
    op.alter_column(TABLE, "verification_label", existing_type=sa.String(32), nullable=False)
    op.drop_column(TABLE, "policy_reason_codes")
    op.drop_column(TABLE, "needs_human_review")
    op.drop_column(TABLE, "verdict")
    op.drop_column(TABLE, "claim_type")
    op.drop_column(TABLE, "claim_id")

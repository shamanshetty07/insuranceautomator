"""reproducibility: detection/verification runs, report provenance, image checksum

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-28

Phase 1 had no code path that creates predictions, reports or verification
results, so the tables restructured here are expected to be empty. Rather than
invent run metadata for old rows, upgrade() refuses to run if any exist.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

RUN_STATUSES = ("pending", "running", "completed", "failed")
STRATEGIES = ("full_image", "cropped_region", "image_with_segmentation_mask")
COMPLETED_AT = "(status IN ('completed', 'failed')) = (completed_at IS NOT NULL)"
SHA256 = "{col} ~ '^[0-9a-f]{{64}}$'"


def _status(name: str) -> sa.Enum:
    return sa.Enum(
        *RUN_STATUSES, name=name, native_enum=False, create_constraint=True, length=16
    )


def _empty_json() -> sa.TextClause:
    return sa.text("'{}'::jsonb")


def upgrade() -> None:
    bind = op.get_bind()
    for table in ("damage_predictions", "generated_reports", "verification_results"):
        if bind.execute(sa.text(f"SELECT EXISTS (SELECT 1 FROM {table})")).scalar():
            raise RuntimeError(
                f"Table {table} contains rows. Their run/strategy metadata cannot be "
                "inferred, so this migration will not guess; backfill manually first."
            )

    # Image integrity (populated by Phase 2 upload code).
    op.add_column("inspection_images", sa.Column("sha256", sa.String(length=64), nullable=True))
    op.create_check_constraint(
        "ck_inspection_images_sha256", "inspection_images", SHA256.format(col="sha256")
    )

    # Detection runs; predictions now hang off a run instead of directly off an image.
    op.create_table(
        "detection_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inspection_image_id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=_empty_json(), nullable=False),
        sa.Column("weights_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "status", _status("detection_run_status"), server_default="pending", nullable=False
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(COMPLETED_AT, name="ck_detection_runs_completed_at"),
        sa.CheckConstraint(SHA256.format(col="weights_sha256"), name="ck_detection_runs_sha256"),
        sa.ForeignKeyConstraint(["inspection_image_id"], ["inspection_images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_detection_runs_inspection_image_id", "detection_runs", ["inspection_image_id"])
    op.create_index("ix_detection_runs_status", "detection_runs", ["status"])

    op.drop_index("ix_damage_predictions_inspection_image_id", table_name="damage_predictions")
    op.drop_column("damage_predictions", "inspection_image_id")  # also drops its FK
    op.drop_column("damage_predictions", "model_name")
    op.drop_column("damage_predictions", "model_version")
    op.add_column("damage_predictions", sa.Column("detection_run_id", sa.Uuid(), nullable=False))
    op.create_foreign_key(
        "damage_predictions_detection_run_id_fkey",
        "damage_predictions", "detection_runs", ["detection_run_id"], ["id"], ondelete="CASCADE",
    )
    op.create_index("ix_damage_predictions_detection_run_id", "damage_predictions", ["detection_run_id"])

    # Report reproducibility metadata and exact-prediction provenance.
    op.add_column(
        "generated_reports",
        sa.Column(
            "evidence_strategy",
            sa.Enum(*STRATEGIES, name="evidence_strategy", native_enum=False, create_constraint=True, length=32),
            nullable=False,
        ),
    )
    op.add_column(
        "generated_reports",
        sa.Column("generation_config", postgresql.JSONB(), server_default=_empty_json(), nullable=False),
    )
    op.create_index("ix_generated_reports_evidence_strategy", "generated_reports", ["evidence_strategy"])

    op.create_table(
        "report_predictions",
        sa.Column("generated_report_id", sa.Uuid(), nullable=False),
        sa.Column("damage_prediction_id", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(["generated_report_id"], ["generated_reports.id"], ondelete="CASCADE"),
        # NO ACTION, checked at commit: a prediction used by a report cannot be deleted
        # on its own, yet deleting a whole inspection works whatever the statement order.
        sa.ForeignKeyConstraint(
            ["damage_prediction_id"], ["damage_predictions.id"], deferrable=True, initially="DEFERRED"
        ),
        sa.PrimaryKeyConstraint("generated_report_id", "damage_prediction_id"),
    )
    op.create_index("ix_report_predictions_damage_prediction_id", "report_predictions", ["damage_prediction_id"])

    # Verification runs; results now hang off a run instead of directly off a report.
    op.create_table(
        "verification_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("generated_report_id", sa.Uuid(), nullable=False),
        sa.Column("verifier_model", sa.String(length=100), nullable=False),
        sa.Column("verifier_version", sa.String(length=100), nullable=False),
        sa.Column("config", postgresql.JSONB(), server_default=_empty_json(), nullable=False),
        sa.Column(
            "status", _status("verification_run_status"), server_default="pending", nullable=False
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(COMPLETED_AT, name="ck_verification_runs_completed_at"),
        sa.ForeignKeyConstraint(["generated_report_id"], ["generated_reports.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_verification_runs_generated_report_id", "verification_runs", ["generated_report_id"])
    op.create_index("ix_verification_runs_status", "verification_runs", ["status"])

    op.drop_index("ix_verification_results_generated_report_id", table_name="verification_results")
    op.drop_column("verification_results", "generated_report_id")  # also drops its FK
    op.drop_column("verification_results", "verifier_model")
    op.add_column("verification_results", sa.Column("verification_run_id", sa.Uuid(), nullable=False))
    op.create_foreign_key(
        "verification_results_verification_run_id_fkey",
        "verification_results", "verification_runs", ["verification_run_id"], ["id"], ondelete="CASCADE",
    )
    op.create_index("ix_verification_results_verification_run_id", "verification_results", ["verification_run_id"])


def downgrade() -> None:
    # Restore the 0001 columns from the runs so downgrading keeps rows. Lost by
    # design: verifier_version, run config/status, weights checksum, report
    # provenance (report_predictions), evidence strategy and generation config.
    op.add_column("verification_results", sa.Column("generated_report_id", sa.Uuid(), nullable=True))
    op.add_column("verification_results", sa.Column("verifier_model", sa.String(length=100), nullable=True))
    op.execute(
        "UPDATE verification_results r SET generated_report_id = v.generated_report_id, "
        "verifier_model = v.verifier_model FROM verification_runs v WHERE r.verification_run_id = v.id"
    )
    op.alter_column("verification_results", "generated_report_id", nullable=False)
    op.alter_column("verification_results", "verifier_model", nullable=False)
    op.drop_index("ix_verification_results_verification_run_id", table_name="verification_results")
    op.drop_column("verification_results", "verification_run_id")
    op.create_foreign_key(
        "verification_results_generated_report_id_fkey",
        "verification_results", "generated_reports", ["generated_report_id"], ["id"], ondelete="CASCADE",
    )
    op.create_index("ix_verification_results_generated_report_id", "verification_results", ["generated_report_id"])
    op.drop_table("verification_runs")

    op.drop_table("report_predictions")
    op.drop_index("ix_generated_reports_evidence_strategy", table_name="generated_reports")
    op.drop_column("generated_reports", "generation_config")
    op.drop_column("generated_reports", "evidence_strategy")

    op.add_column("damage_predictions", sa.Column("inspection_image_id", sa.Uuid(), nullable=True))
    op.add_column("damage_predictions", sa.Column("model_name", sa.String(length=100), nullable=True))
    op.add_column("damage_predictions", sa.Column("model_version", sa.String(length=100), nullable=True))
    op.execute(
        "UPDATE damage_predictions p SET inspection_image_id = d.inspection_image_id, "
        "model_name = d.model_name, model_version = d.model_version "
        "FROM detection_runs d WHERE p.detection_run_id = d.id"
    )
    for col in ("inspection_image_id", "model_name", "model_version"):
        op.alter_column("damage_predictions", col, nullable=False)
    op.drop_index("ix_damage_predictions_detection_run_id", table_name="damage_predictions")
    op.drop_column("damage_predictions", "detection_run_id")
    op.create_foreign_key(
        "damage_predictions_inspection_image_id_fkey",
        "damage_predictions", "inspection_images", ["inspection_image_id"], ["id"], ondelete="CASCADE",
    )
    op.create_index("ix_damage_predictions_inspection_image_id", "damage_predictions", ["inspection_image_id"])
    op.drop_table("detection_runs")

    op.drop_column("inspection_images", "sha256")

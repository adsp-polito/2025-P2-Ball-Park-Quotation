"""Add RLHF tables for continuous learning

Revision ID: 20250106_001
Revises: 20241218_001_initial_schema
Create Date: 2025-01-06

Tables added:
- preference_pairs: DPO training pairs from user corrections
- ab_experiments: A/B testing experiments
- ab_predictions: Predictions with shadow mode support
- rlhf_training_jobs: Training job history

Design decisions:
- NO FK constraint on session_id (soft reference) to allow session purging
- shadow_prediction column for instant drift analysis
- Partial indexes for performance
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "20250106_001"
down_revision = "001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Preference pairs for DPO training
    op.create_table(
        "preference_pairs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id", postgresql.UUID(as_uuid=True), nullable=True
        ),  # Soft reference - no FK
        sa.Column("context_hash", sa.String(64), nullable=True),  # For deduplication
        sa.Column("chosen_reasoning", sa.Text(), nullable=False),
        sa.Column("rejected_reasoning", sa.Text(), nullable=False),
        sa.Column("chosen_breakdown", postgresql.JSONB(), nullable=False),
        sa.Column("rejected_breakdown", postgresql.JSONB(), nullable=False),
        sa.Column(
            "signal_source", sa.String(50), nullable=False
        ),  # user_edit, actual_outcome, synthetic_negative, explicit_approval
        sa.Column("reward_delta", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("validated", sa.Boolean(), server_default="false"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column("used_in_training", sa.DateTime(timezone=True), nullable=True),
    )

    # Indexes for preference_pairs
    op.create_index(
        "idx_preference_pairs_unused",
        "preference_pairs",
        ["used_in_training"],
        postgresql_where=sa.text("used_in_training IS NULL"),
    )
    op.create_index(
        "idx_preference_pairs_source", "preference_pairs", ["signal_source"]
    )
    op.create_index("idx_preference_pairs_session", "preference_pairs", ["session_id"])

    # A/B test experiments
    op.create_table(
        "ab_experiments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("candidate_model_version", sa.String(50), nullable=False),
        sa.Column("production_model_version", sa.String(50), nullable=False),
        sa.Column(
            "status", sa.String(20), server_default="shadow"
        ),  # shadow, canary, gradual, complete, rolled_back
        sa.Column("candidate_weight", sa.Float(), server_default="0.0"),
        sa.Column("shadow_mode", sa.Boolean(), server_default="true"),
        sa.Column("kill_switch_triggered", sa.Boolean(), server_default="false"),
        sa.Column(
            "metrics_snapshot", postgresql.JSONB(), nullable=True
        ),  # Current metrics
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Partial index for active experiments
    op.create_index(
        "idx_ab_experiments_active",
        "ab_experiments",
        ["status"],
        postgresql_where=sa.text("status NOT IN ('complete', 'rolled_back')"),
    )

    # A/B predictions with shadow mode support
    op.create_table(
        "ab_predictions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "experiment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("ab_experiments.id"),
            nullable=False,
        ),
        sa.Column(
            "session_id", postgresql.UUID(as_uuid=True), nullable=True
        ),  # Soft reference - no FK
        sa.Column("model_used", sa.String(20), nullable=False),  # production, candidate
        sa.Column("prediction", postgresql.JSONB(), nullable=False),
        sa.Column(
            "shadow_prediction", postgresql.JSONB(), nullable=True
        ),  # Candidate result when shadow_mode=true
        sa.Column(
            "actual_outcome", sa.Float(), nullable=True
        ),  # Filled later if available
        sa.Column("user_edited", sa.Boolean(), server_default="false"),
        sa.Column(
            "sizing_category", sa.String(20), nullable=True
        ),  # For spread analysis
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
    )

    op.create_index(
        "idx_ab_predictions_experiment",
        "ab_predictions",
        ["experiment_id", "created_at"],
    )
    op.create_index("idx_ab_predictions_session", "ab_predictions", ["session_id"])

    # RLHF training jobs
    op.create_table(
        "rlhf_training_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("job_type", sa.String(20), nullable=False),  # ml_retrain, llm_dpo
        sa.Column(
            "status", sa.String(20), server_default="pending"
        ),  # pending, running, completed, failed
        sa.Column("samples_used", sa.Integer(), nullable=True),
        sa.Column("metrics_before", postgresql.JSONB(), nullable=True),
        sa.Column("metrics_after", postgresql.JSONB(), nullable=True),
        sa.Column("model_version_created", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")
        ),
    )

    # Partial index for active jobs
    op.create_index(
        "idx_training_jobs_active",
        "rlhf_training_jobs",
        ["status"],
        postgresql_where=sa.text("status IN ('pending', 'running')"),
    )


def downgrade() -> None:
    op.drop_table("rlhf_training_jobs")
    op.drop_table("ab_predictions")
    op.drop_table("ab_experiments")
    op.drop_table("preference_pairs")

"""Add R&D Cost Table version and correction tables

Revision ID: 20250106_002
Revises: 20250106_001
Create Date: 2025-01-06

Tables added:
- rd_cost_table_versions: Version history for R&D cost tables
- rd_cost_corrections: User corrections with DPO reasoning capture

Design decisions:
- Full JSONB data storage for versions (enables complete rollback)
- Separate corrections table for DPO training data extraction
- change_tags as TEXT[] for efficient filtering
- Indexes optimized for training data queries
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "20250106_002"
down_revision = "20250106_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # R&D Cost Table versions
    op.create_table(
        "rd_cost_table_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column(
            "data", postgresql.JSONB(), nullable=False
        ),  # Full table data snapshot
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "is_finalized", sa.Boolean(), server_default="false"
        ),  # Locked after finalization
        sa.Column(
            "change_count", sa.Integer(), server_default="0"
        ),  # Number of changes from previous
        sa.Column(
            "changes_from_previous", postgresql.JSONB(), nullable=True
        ),  # Diff details
        # Unique constraint on session + version number
        sa.UniqueConstraint(
            "session_id", "version_number", name="uq_rd_version_session_number"
        ),
    )

    # Indexes for rd_cost_table_versions
    op.create_index(
        "idx_rd_versions_session",
        "rd_cost_table_versions",
        ["session_id", "version_number"],
    )
    op.create_index(
        "idx_rd_versions_latest",
        "rd_cost_table_versions",
        ["session_id", "created_at"],
    )

    # R&D Cost corrections (DPO training data)
    op.create_table(
        "rd_cost_corrections",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "version_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("rd_cost_table_versions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("row_id", sa.String(100), nullable=False),
        sa.Column("column_name", sa.String(50), nullable=False),
        sa.Column("original_value", sa.Numeric(12, 2), nullable=False),
        sa.Column("new_value", sa.Numeric(12, 2), nullable=False),
        sa.Column(
            "change_percent", sa.Float(), nullable=True
        ),  # Pre-computed for filtering
        sa.Column("reasoning_text", sa.Text(), nullable=True),
        sa.Column(
            "change_tags",
            postgresql.ARRAY(sa.String(50)),
            nullable=True,
        ),  # Array of tag values
        sa.Column(
            "confidence_at_prediction", sa.Float(), nullable=True
        ),  # AI confidence when prediction was made
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # For DPO training export tracking
        sa.Column("exported_for_training", sa.DateTime(timezone=True), nullable=True),
    )

    # Indexes for rd_cost_corrections
    op.create_index(
        "idx_rd_corrections_session",
        "rd_cost_corrections",
        ["session_id", "created_at"],
    )
    op.create_index(
        "idx_rd_corrections_row",
        "rd_cost_corrections",
        ["session_id", "row_id"],
    )
    # Partial index for training data with reasoning
    op.create_index(
        "idx_rd_corrections_for_training",
        "rd_cost_corrections",
        ["session_id", "created_at"],
        postgresql_where=sa.text("reasoning_text IS NOT NULL"),
    )
    # Index for large corrections (>15% change)
    op.create_index(
        "idx_rd_corrections_large",
        "rd_cost_corrections",
        ["change_percent"],
        postgresql_where=sa.text("ABS(change_percent) > 15"),
    )
    # Index for unexported training data
    op.create_index(
        "idx_rd_corrections_unexported",
        "rd_cost_corrections",
        ["exported_for_training"],
        postgresql_where=sa.text(
            "exported_for_training IS NULL AND reasoning_text IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_table("rd_cost_corrections")
    op.drop_table("rd_cost_table_versions")

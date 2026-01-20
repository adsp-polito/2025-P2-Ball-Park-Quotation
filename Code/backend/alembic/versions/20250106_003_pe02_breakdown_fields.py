"""Add PE02 fields to quotation_breakdown table

Revision ID: 20250106_003
Revises: 20250106_002
Create Date: 2025-01-06

Adds PE02 standard fields for R&D cost estimation:
- function_id: PE02 activity code (A1, B1, C, etc.)
- 5 effort columns: manpower, bench_dev, bench_special, bench_dur, vehicle
- investment_keur: Cost in k€ (PE02 standard)
- hourly_rate_eur: Cluster-specific rate used for calculation
- source: Tracks origin (llm, ml, rule, manual)
- reasoning: LLM reasoning text
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = "20250106_003"
down_revision = "20250106_002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add PE02 function identifier
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "function_id",
            sa.String(10),
            nullable=True,
            comment="PE02 code: A1, A2, B1, B2, C, D1, D2, D3, E, F, G",
        ),
    )

    # Add PE02 effort columns (5 columns)
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "effort_manpower",
            sa.Numeric(10, 1),
            nullable=True,
            comment="Manpower hours",
        ),
    )
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "effort_bench_dev",
            sa.Numeric(10, 1),
            nullable=True,
            comment="Bench Development hours",
        ),
    )
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "effort_bench_special",
            sa.Numeric(10, 1),
            nullable=True,
            comment="Bench Special hours (NVH, climatic)",
        ),
    )
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "effort_bench_dur",
            sa.Numeric(10, 1),
            nullable=True,
            comment="Bench Durability hours",
        ),
    )
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "effort_vehicle",
            sa.Numeric(10, 1),
            nullable=True,
            comment="Vehicle testing hours",
        ),
    )

    # Add cost fields
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "investment_keur",
            sa.Numeric(10, 2),
            nullable=True,
            comment="Investment in k€ (PE02 standard)",
        ),
    )
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "hourly_rate_eur",
            sa.Numeric(6, 2),
            nullable=True,
            comment="Cluster-specific hourly rate used",
        ),
    )

    # Add source tracking
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "source",
            sa.String(50),
            nullable=True,
            comment="llm, ml, rule, manual, similar_pr",
        ),
    )
    op.add_column(
        "quotation_breakdown",
        sa.Column(
            "reasoning",
            sa.Text(),
            nullable=True,
            comment="LLM reasoning for this estimate",
        ),
    )

    # Add index on function_id
    op.create_index(
        "idx_breakdown_function_id",
        "quotation_breakdown",
        ["function_id"],
    )

    # Migrate existing data: Copy legacy columns to new PE02 columns
    # hours_manpower -> effort_manpower (as float)
    # cost_eur -> investment_keur (convert € to k€)
    op.execute("""
        UPDATE quotation_breakdown
        SET
            effort_manpower = hours_manpower::numeric,
            investment_keur = cost_eur / 1000.0,
            source = CASE WHEN ai_generated THEN 'llm' ELSE 'manual' END
        WHERE hours_manpower IS NOT NULL OR cost_eur IS NOT NULL
    """)


def downgrade() -> None:
    # Drop index
    op.drop_index("idx_breakdown_function_id", table_name="quotation_breakdown")

    # Drop new columns (reverse order)
    op.drop_column("quotation_breakdown", "reasoning")
    op.drop_column("quotation_breakdown", "source")
    op.drop_column("quotation_breakdown", "hourly_rate_eur")
    op.drop_column("quotation_breakdown", "investment_keur")
    op.drop_column("quotation_breakdown", "effort_vehicle")
    op.drop_column("quotation_breakdown", "effort_bench_dur")
    op.drop_column("quotation_breakdown", "effort_bench_special")
    op.drop_column("quotation_breakdown", "effort_bench_dev")
    op.drop_column("quotation_breakdown", "effort_manpower")
    op.drop_column("quotation_breakdown", "function_id")

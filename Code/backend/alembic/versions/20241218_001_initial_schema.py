"""Initial schema with all 15 tables

Revision ID: 001_initial
Revises:
Create Date: 2024-12-18

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="engineer"),
        sa.Column("language", sa.String(10), server_default="en"),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("last_login", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    # Product Requests table
    op.create_table(
        "product_requests",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pr_number", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("platform", sa.String(50), nullable=True),
        sa.Column("vehicle_models", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("engine_type", sa.String(50), nullable=True),
        sa.Column("tier", sa.String(50), nullable=True),
        sa.Column("plant", sa.String(100), nullable=True),
        sa.Column("markets", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("program_size", sa.String(20), nullable=True),
        sa.Column("program_size_confidence", sa.Numeric(3, 2), nullable=True),
        sa.Column("status", sa.String(50), server_default="draft"),
        sa.Column("original_file_path", sa.String(500), nullable=True),
        sa.Column("original_file_name", sa.String(255), nullable=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pr_number"),
    )
    op.create_index("idx_pr_status", "product_requests", ["status"])
    op.create_index("idx_pr_platform", "product_requests", ["platform"])
    op.create_index("idx_pr_created", "product_requests", ["created_at"])
    op.create_index("idx_pr_created_by", "product_requests", ["created_by"])

    # Quotations table
    op.create_table(
        "quotations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "pr_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), server_default="1"),
        sa.Column("total_hours_manpower", sa.Integer(), nullable=True),
        sa.Column("total_hours_bench", sa.Integer(), nullable=True),
        sa.Column("total_cost_eur", sa.Numeric(12, 2), nullable=True),
        sa.Column("confidence_score", sa.Numeric(3, 2), nullable=True),
        sa.Column("ml_prediction_eur", sa.Numeric(12, 2), nullable=True),
        sa.Column("similar_pr_weighted_avg", sa.Numeric(12, 2), nullable=True),
        sa.Column("status", sa.String(50), server_default="draft"),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column(
            "approved_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_quotation_pr", "quotations", ["pr_id"])
    op.create_index("idx_quotation_status", "quotations", ["status"])

    # Quotation Breakdown table
    op.create_table(
        "quotation_breakdown",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "quotation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("quotations.id"),
            nullable=False,
        ),
        sa.Column("pe_function", sa.String(100), nullable=False),
        sa.Column("sub_function", sa.String(100), nullable=True),
        sa.Column("activity_description", sa.Text(), nullable=True),
        sa.Column("hours_manpower", sa.Integer(), nullable=True),
        sa.Column("hours_bench", sa.Integer(), nullable=True),
        sa.Column("cost_eur", sa.Numeric(10, 2), nullable=True),
        sa.Column("ai_generated", sa.Boolean(), server_default="true"),
        sa.Column("manually_adjusted", sa.Boolean(), server_default="false"),
        sa.Column("original_hours_manpower", sa.Integer(), nullable=True),
        sa.Column("original_cost_eur", sa.Numeric(10, 2), nullable=True),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=True),
        sa.Column("basis", sa.Text(), nullable=True),
        sa.Column("display_order", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_breakdown_quotation", "quotation_breakdown", ["quotation_id"])
    op.create_index("idx_breakdown_function", "quotation_breakdown", ["pe_function"])

    # Estimation Questions table
    op.create_table(
        "estimation_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "pr_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id"),
            nullable=False,
        ),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("question_type", sa.String(50), nullable=True),
        sa.Column("reasoning", sa.Text(), nullable=True),
        sa.Column("impact_field", sa.String(100), nullable=True),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("answer_options", postgresql.JSONB(), nullable=True),
        sa.Column("answered_at", sa.DateTime(), nullable=True),
        sa.Column(
            "answered_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("display_order", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    # ML Features table
    op.create_table(
        "ml_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "pr_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id"),
            nullable=False,
        ),
        sa.Column("feature_name", sa.String(100), nullable=False),
        sa.Column("feature_value", postgresql.JSONB(), nullable=False),
        sa.Column("feature_type", sa.String(50), nullable=True),
        sa.Column("source", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_features_pr", "ml_features", ["pr_id"])
    op.create_index("idx_features_name", "ml_features", ["feature_name"])

    # Learned Rules table
    op.create_table(
        "learned_rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rule_name", sa.String(200), nullable=False),
        sa.Column("rule_description", sa.Text(), nullable=True),
        sa.Column("condition_type", sa.String(50), nullable=True),
        sa.Column("condition_config", postgresql.JSONB(), nullable=True),
        sa.Column("effect_type", sa.String(50), nullable=True),
        sa.Column("effect_field", sa.String(100), nullable=True),
        sa.Column("effect_value", sa.Numeric(10, 4), nullable=True),
        sa.Column("confidence", sa.Numeric(3, 2), nullable=True),
        sa.Column("feedback_count", sa.Integer(), server_default="0"),
        sa.Column("last_validated", sa.DateTime(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("auto_generated", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_rules_active", "learned_rules", ["is_active"])
    op.create_index("idx_rules_confidence", "learned_rules", ["confidence"])

    # Feedback Corrections table
    op.create_table(
        "feedback_corrections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "quotation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("quotations.id"),
            nullable=True,
        ),
        sa.Column(
            "breakdown_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("quotation_breakdown.id"),
            nullable=True,
        ),
        sa.Column(
            "pr_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id"),
            nullable=True,
        ),
        sa.Column("field_path", sa.String(200), nullable=True),
        sa.Column("original_value", sa.Numeric(12, 2), nullable=True),
        sa.Column("corrected_value", sa.Numeric(12, 2), nullable=True),
        sa.Column("change_percentage", sa.Numeric(5, 2), nullable=True),
        sa.Column("reason_category", sa.String(100), nullable=True),
        sa.Column("reason_text", sa.Text(), nullable=True),
        sa.Column("rule_extracted", sa.Boolean(), server_default="false"),
        sa.Column(
            "rule_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learned_rules.id"),
            nullable=True,
        ),
        sa.Column("included_in_retrain", sa.Boolean(), server_default="false"),
        sa.Column(
            "corrected_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_feedback_pr", "feedback_corrections", ["pr_id"])
    op.create_index(
        "idx_feedback_category", "feedback_corrections", ["reason_category"]
    )
    op.create_index(
        "idx_feedback_included", "feedback_corrections", ["included_in_retrain"]
    )

    # Model Versions table
    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=True),
        sa.Column("mae", sa.Numeric(12, 2), nullable=True),
        sa.Column("r2_score", sa.Numeric(5, 4), nullable=True),
        sa.Column("mape", sa.Numeric(5, 2), nullable=True),
        sa.Column("training_samples", sa.Integer(), nullable=True),
        sa.Column("corrections_included", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="false"),
        sa.Column("promoted_at", sa.DateTime(), nullable=True),
        sa.Column("model_path", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    # Retrain Jobs table
    op.create_table(
        "retrain_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("trigger_reason", sa.String(100), nullable=True),
        sa.Column("corrections_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(50), nullable=True),
        sa.Column(
            "old_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column(
            "new_model_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column("improvement_percentage", sa.Numeric(5, 2), nullable=True),
        sa.Column("auto_promoted", sa.Boolean(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Knowledge Documents table
    op.create_table(
        "knowledge_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("doc_type", sa.String(100), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("is_indexed", sa.Boolean(), server_default="false"),
        sa.Column("chunk_count", sa.Integer(), nullable=True),
        sa.Column("source_file_path", sa.String(500), nullable=True),
        sa.Column(
            "uploaded_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Acronyms table
    op.create_table(
        "acronyms",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("acronym", sa.String(50), nullable=False),
        sa.Column("full_form", sa.String(500), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("usage_examples", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("acronym"),
    )

    # Audit Log table
    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("old_value", postgresql.JSONB(), nullable=True),
        sa.Column("new_value", postgresql.JSONB(), nullable=True),
        sa.Column("ip_address", sa.String(50), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_audit_user", "audit_log", ["user_id"])
    op.create_index("idx_audit_action", "audit_log", ["action"])
    op.create_index("idx_audit_entity", "audit_log", ["entity_type", "entity_id"])
    op.create_index("idx_audit_created", "audit_log", ["created_at"])

    # Chat Sessions table
    op.create_table(
        "chat_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "pr_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("product_requests.id"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.Column("current_step", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_chat_session_pr", "chat_sessions", ["pr_id"])

    # Chat Messages table
    op.create_table(
        "chat_messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("sources", postgresql.JSONB(), nullable=True),
        sa.Column("tools_used", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("step", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_chat_messages_session", "chat_messages", ["session_id"])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
    op.drop_table("audit_log")
    op.drop_table("acronyms")
    op.drop_table("knowledge_documents")
    op.drop_table("retrain_jobs")
    op.drop_table("model_versions")
    op.drop_table("feedback_corrections")
    op.drop_table("learned_rules")
    op.drop_table("ml_features")
    op.drop_table("estimation_questions")
    op.drop_table("quotation_breakdown")
    op.drop_table("quotations")
    op.drop_table("product_requests")
    op.drop_table("users")

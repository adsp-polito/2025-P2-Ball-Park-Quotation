"""
FPT Cost Brain 2.0 - SQLAlchemy ORM Models
Complete database schema with 15 tables and 19 indexes
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# ===== ENUMS =====


class UserRole(str, Enum):
    ENGINEER = "engineer"
    MANAGER = "manager"
    HEAD = "head"
    EXECUTIVE = "executive"


class PRStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class QuotationStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    EXPORTED = "exported"


class ProgramSize(str, Enum):
    X_SMALL = "X_SMALL"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    FULL = "FULL"


class RetrainStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ChatStep(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    ESTIMATION = "estimation"
    REVIEW = "review"


# ===== AUTHENTICATION =====


class User(Base):
    """User accounts for authentication."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="engineer", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    product_requests: Mapped[list["ProductRequest"]] = relationship(
        back_populates="created_by_user"
    )
    quotations_approved: Mapped[list["Quotation"]] = relationship(
        back_populates="approved_by_user"
    )


# ===== CORE TABLES =====


class ProductRequest(Base):
    """Product Request documents."""

    __tablename__ = "product_requests"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pr_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Classification
    platform: Mapped[str | None] = mapped_column(String(50), nullable=True)
    vehicle_models: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True
    )
    engine_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tier: Mapped[str | None] = mapped_column(String(50), nullable=True)
    plant: Mapped[str | None] = mapped_column(String(100), nullable=True)
    markets: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)

    # Program sizing
    program_size: Mapped[str | None] = mapped_column(String(20), nullable=True)
    program_size_confidence: Mapped[float | None] = mapped_column(
        Numeric(3, 2), nullable=True
    )

    # Status
    status: Mapped[str] = mapped_column(String(50), default="draft")

    # File metadata
    original_file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    original_file_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Foreign keys
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    created_by_user: Mapped["User"] = relationship(back_populates="product_requests")
    quotations: Mapped[list["Quotation"]] = relationship(
        back_populates="product_request"
    )
    questions: Mapped[list["EstimationQuestion"]] = relationship(
        back_populates="product_request"
    )
    ml_features: Mapped[list["MLFeature"]] = relationship(
        back_populates="product_request"
    )
    feedback_corrections: Mapped[list["FeedbackCorrection"]] = relationship(
        back_populates="product_request"
    )
    chat_sessions: Mapped[list["ChatSession"]] = relationship(
        back_populates="product_request"
    )

    __table_args__ = (
        Index("idx_pr_status", "status"),
        Index("idx_pr_platform", "platform"),
        Index("idx_pr_created", "created_at"),
        Index("idx_pr_created_by", "created_by"),
    )


class Quotation(Base):
    """Cost quotations for Product Requests."""

    __tablename__ = "quotations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pr_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_requests.id"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    # Totals
    total_hours_manpower: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_hours_bench: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost_eur: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)

    # AI metrics
    confidence_score: Mapped[float | None] = mapped_column(Numeric(3, 2), nullable=True)
    ml_prediction_eur: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    similar_pr_weighted_avg: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )

    # Status
    status: Mapped[str] = mapped_column(String(50), default="draft")

    # Approval
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    approved_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    product_request: Mapped["ProductRequest"] = relationship(
        back_populates="quotations"
    )
    approved_by_user: Mapped["User"] = relationship(
        back_populates="quotations_approved"
    )
    breakdown_items: Mapped[list["QuotationBreakdown"]] = relationship(
        back_populates="quotation"
    )
    feedback_corrections: Mapped[list["FeedbackCorrection"]] = relationship(
        back_populates="quotation"
    )

    __table_args__ = (
        Index("idx_quotation_pr", "pr_id"),
        Index("idx_quotation_status", "status"),
    )


class QuotationBreakdown(Base):
    """
    Line items in a quotation breakdown.

    Updated for PE02 format with:
    - 5 effort columns (manpower, bench_dev, bench_special, bench_dur, vehicle)
    - investment_keur (k€) as primary cost field
    - function_id for PE02 codes (A1, B1, etc.)
    - cluster-based hourly_rate_eur
    - source tracking (llm, ml, rule, manual)
    """

    __tablename__ = "quotation_breakdown"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    quotation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("quotations.id"), nullable=False
    )

    # PE02 Function identifiers
    function_id: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
        comment="PE02 code: A1, A2, B1, B2, C, D1, D2, D3, E, F, G",
    )
    pe_function: Mapped[str] = mapped_column(String(100), nullable=False)
    sub_function: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Activity details
    activity_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # PE02 Effort Columns (5 columns)
    effort_manpower: Mapped[float | None] = mapped_column(
        Numeric(10, 1), nullable=True, comment="Manpower hours"
    )
    effort_bench_dev: Mapped[float | None] = mapped_column(
        Numeric(10, 1), nullable=True, comment="Bench Development hours"
    )
    effort_bench_special: Mapped[float | None] = mapped_column(
        Numeric(10, 1), nullable=True, comment="Bench Special hours (NVH, climatic)"
    )
    effort_bench_dur: Mapped[float | None] = mapped_column(
        Numeric(10, 1), nullable=True, comment="Bench Durability hours"
    )
    effort_vehicle: Mapped[float | None] = mapped_column(
        Numeric(10, 1), nullable=True, comment="Vehicle testing hours"
    )

    # Cost fields
    investment_keur: Mapped[float | None] = mapped_column(
        Numeric(10, 2), nullable=True, comment="Investment in k€ (PE02 standard)"
    )
    hourly_rate_eur: Mapped[float | None] = mapped_column(
        Numeric(6, 2), nullable=True, comment="Cluster-specific hourly rate used"
    )

    # Legacy hours & cost (for backward compatibility)
    hours_manpower: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hours_bench: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_eur: Mapped[float | None] = mapped_column(
        Numeric(10, 2), nullable=True, comment="DEPRECATED: Use investment_keur"
    )

    # AI tracking & source
    source: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="llm, ml, rule, manual, similar_pr"
    )
    reasoning: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="LLM reasoning for this estimate"
    )
    ai_generated: Mapped[bool] = mapped_column(Boolean, default=True)
    manually_adjusted: Mapped[bool] = mapped_column(Boolean, default=False)
    original_hours_manpower: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_cost_eur: Mapped[float | None] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    confidence: Mapped[float | None] = mapped_column(Numeric(3, 2), nullable=True)
    basis: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Ordering
    display_order: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    quotation: Mapped["Quotation"] = relationship(back_populates="breakdown_items")
    feedback_corrections: Mapped[list["FeedbackCorrection"]] = relationship(
        back_populates="breakdown_item"
    )

    __table_args__ = (
        Index("idx_breakdown_quotation", "quotation_id"),
        Index("idx_breakdown_function", "pe_function"),
        Index("idx_breakdown_function_id", "function_id"),
    )


# ===== Q&A TABLES =====


class EstimationQuestion(Base):
    """Smart questions for estimation clarification."""

    __tablename__ = "estimation_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pr_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_requests.id"), nullable=False
    )

    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    impact_field: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Answer
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_options: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    answered_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    display_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    product_request: Mapped["ProductRequest"] = relationship(back_populates="questions")


# ===== ML & LEARNING TABLES =====


class MLFeature(Base):
    """Extracted ML features for Product Requests."""

    __tablename__ = "ml_features"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pr_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_requests.id"), nullable=False
    )

    feature_name: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    feature_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    product_request: Mapped["ProductRequest"] = relationship(
        back_populates="ml_features"
    )

    __table_args__ = (
        Index("idx_features_pr", "pr_id"),
        Index("idx_features_name", "feature_name"),
    )


class LearnedRule(Base):
    """Rules extracted from user corrections."""

    __tablename__ = "learned_rules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    rule_name: Mapped[str] = mapped_column(String(200), nullable=False)
    rule_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Condition
    condition_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    condition_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Effect
    effect_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    effect_field: Mapped[str | None] = mapped_column(String(100), nullable=True)
    effect_value: Mapped[float | None] = mapped_column(Numeric(10, 4), nullable=True)

    # Confidence & validation
    confidence: Mapped[float | None] = mapped_column(Numeric(3, 2), nullable=True)
    feedback_count: Mapped[int] = mapped_column(Integer, default=0)
    last_validated: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_generated: Mapped[bool] = mapped_column(Boolean, default=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    feedback_corrections: Mapped[list["FeedbackCorrection"]] = relationship(
        back_populates="learned_rule"
    )

    __table_args__ = (
        Index("idx_rules_active", "is_active"),
        Index("idx_rules_confidence", "confidence"),
    )


class FeedbackCorrection(Base):
    """User corrections for learning."""

    __tablename__ = "feedback_corrections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    quotation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("quotations.id"), nullable=True
    )
    breakdown_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("quotation_breakdown.id"), nullable=True
    )
    pr_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_requests.id"), nullable=True
    )

    # What was changed
    field_path: Mapped[str | None] = mapped_column(String(200), nullable=True)
    original_value: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    corrected_value: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    change_percentage: Mapped[float | None] = mapped_column(
        Numeric(5, 2), nullable=True
    )

    # Why (for learning)
    reason_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    reason_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Learning status
    rule_extracted: Mapped[bool] = mapped_column(Boolean, default=False)
    rule_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("learned_rules.id"), nullable=True
    )
    included_in_retrain: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metadata
    corrected_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    quotation: Mapped["Quotation"] = relationship(back_populates="feedback_corrections")
    breakdown_item: Mapped["QuotationBreakdown"] = relationship(
        back_populates="feedback_corrections"
    )
    product_request: Mapped["ProductRequest"] = relationship(
        back_populates="feedback_corrections"
    )
    learned_rule: Mapped["LearnedRule"] = relationship(
        back_populates="feedback_corrections"
    )

    __table_args__ = (
        Index("idx_feedback_pr", "pr_id"),
        Index("idx_feedback_category", "reason_category"),
        Index("idx_feedback_included", "included_in_retrain"),
    )


class ModelVersion(Base):
    """ML model versions for tracking and rollback."""

    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    version_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Performance metrics
    mae: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    r2_score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    mape: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)

    # Training info
    training_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    corrections_included: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Storage
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())


class RetrainJob(Base):
    """Batch retraining job tracking."""

    __tablename__ = "retrain_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    trigger_reason: Mapped[str | None] = mapped_column(String(100), nullable=True)
    corrections_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Results
    status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    old_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=True
    )
    new_model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=True
    )

    improvement_percentage: Mapped[float | None] = mapped_column(
        Numeric(5, 2), nullable=True
    )
    auto_promoted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# ===== KNOWLEDGE BASE TABLES =====


class KnowledgeDocument(Base):
    """Enterprise knowledge documents for RAG."""

    __tablename__ = "knowledge_documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    doc_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)

    # For RAG
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Metadata
    source_file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )


class Acronym(Base):
    """FPT domain acronyms."""

    __tablename__ = "acronyms"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    acronym: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    full_form: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)

    usage_examples: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )


# ===== AUDIT TRAIL =====


class AuditLog(Base):
    """Complete audit trail for all changes."""

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )

    # Details
    old_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_action", "action"),
        Index("idx_audit_entity", "entity_type", "entity_id"),
        Index("idx_audit_created", "created_at"),
    )


# ===== CHAT HISTORY =====


class ChatSession(Base):
    """RAG chat sessions."""

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pr_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("product_requests.id"), nullable=True
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    current_step: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=func.now(), onupdate=func.now()
    )

    # Relationships
    product_request: Mapped["ProductRequest"] = relationship(
        back_populates="chat_sessions"
    )
    messages: Mapped[list["ChatMessage"]] = relationship(back_populates="session")

    __table_args__ = (Index("idx_chat_session_pr", "pr_id"),)


class ChatMessage(Base):
    """Individual chat messages."""

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False
    )

    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # RAG metadata
    sources: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    tools_used: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)

    step: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    session: Mapped["ChatSession"] = relationship(back_populates="messages")

    __table_args__ = (Index("idx_chat_messages_session", "session_id"),)


# ===== RLHF TABLES =====


class PreferencePair(Base):
    """DPO training pairs from user corrections."""

    __tablename__ = "preference_pairs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )  # Soft reference - no FK to allow session purging

    # Deduplication
    context_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # DPO training data
    chosen_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    rejected_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    chosen_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False)
    rejected_breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Signal metadata
    signal_source: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # user_edit, actual_outcome, synthetic_negative, explicit_approval
    reward_delta: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False)

    # Validation
    validated: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    used_in_training: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index(
            "idx_preference_pairs_unused",
            "used_in_training",
            postgresql_where="used_in_training IS NULL",
        ),
        Index("idx_preference_pairs_source", "signal_source"),
        Index("idx_preference_pairs_session", "session_id"),
    )


class ABExperiment(Base):
    """A/B testing experiments for model deployment."""

    __tablename__ = "ab_experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    # Model versions
    candidate_model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    production_model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Experiment state
    status: Mapped[str] = mapped_column(
        String(20), default="shadow"
    )  # shadow, canary, gradual, complete, rolled_back
    candidate_weight: Mapped[float] = mapped_column(Numeric(3, 2), default=0.0)
    shadow_mode: Mapped[bool] = mapped_column(Boolean, default=True)
    kill_switch_triggered: Mapped[bool] = mapped_column(Boolean, default=False)

    # Metrics
    metrics_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    predictions: Mapped[list["ABPrediction"]] = relationship(
        back_populates="experiment"
    )

    __table_args__ = (
        Index(
            "idx_ab_experiments_active",
            "status",
            postgresql_where="status NOT IN ('complete', 'rolled_back')",
        ),
    )


class ABPrediction(Base):
    """Predictions with shadow mode support for A/B testing."""

    __tablename__ = "ab_predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ab_experiments.id"), nullable=False
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )  # Soft reference - no FK

    # Prediction data
    model_used: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # production, candidate
    prediction: Mapped[dict] = mapped_column(JSONB, nullable=False)
    shadow_prediction: Mapped[dict | None] = mapped_column(
        JSONB, nullable=True
    )  # Candidate result when shadow_mode=true

    # Outcome tracking
    actual_outcome: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )  # Filled later if available
    user_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    sizing_category: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # For spread analysis

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Relationships
    experiment: Mapped["ABExperiment"] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("idx_ab_predictions_experiment", "experiment_id", "created_at"),
        Index("idx_ab_predictions_session", "session_id"),
    )


class RLHFTrainingJob(Base):
    """Training job history for ML retraining and LLM DPO."""

    __tablename__ = "rlhf_training_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # Job info
    job_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # ml_retrain, llm_dpo
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, running, completed, failed

    # Training data
    samples_used: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Metrics
    metrics_before: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metrics_after: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Output
    model_version_created: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    __table_args__ = (
        Index(
            "idx_training_jobs_active",
            "status",
            postgresql_where="status IN ('pending', 'running')",
        ),
    )


# ===== R&D COST TABLE TABLES =====


class RDCostTableVersion(Base):
    """Version history for R&D cost tables."""

    __tablename__ = "rd_cost_table_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # PR reference (serves as session identifier)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("product_requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Full table data snapshot
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Metadata
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # Status
    is_finalized: Mapped[bool] = mapped_column(Boolean, default=False)
    change_count: Mapped[int] = mapped_column(Integer, default=0)
    changes_from_previous: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    corrections: Mapped[list["RDCostCorrection"]] = relationship(
        back_populates="version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_rd_versions_session", "session_id", "version_number"),
        Index("idx_rd_versions_latest", "session_id", "created_at"),
    )


class RDCostCorrection(Base):
    """User corrections with DPO reasoning capture."""

    __tablename__ = "rd_cost_corrections"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    # References (session_id = PR id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("product_requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("rd_cost_table_versions.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Cell identification
    row_id: Mapped[str] = mapped_column(String(100), nullable=False)
    column_name: Mapped[str] = mapped_column(String(50), nullable=False)

    # Values
    original_value: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    new_value: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    change_percent: Mapped[float | None] = mapped_column(Numeric(6, 2), nullable=True)

    # DPO training data
    reasoning_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    change_tags: Mapped[list[str] | None] = mapped_column(
        ARRAY(String(50)), nullable=True
    )
    confidence_at_prediction: Mapped[float | None] = mapped_column(
        Numeric(4, 3), nullable=True
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    exported_for_training: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    # Relationships
    version: Mapped["RDCostTableVersion"] = relationship(back_populates="corrections")

    __table_args__ = (
        Index("idx_rd_corrections_session", "session_id", "created_at"),
        Index("idx_rd_corrections_row", "session_id", "row_id"),
        Index(
            "idx_rd_corrections_for_training",
            "session_id",
            "created_at",
            postgresql_where="reasoning_text IS NOT NULL",
        ),
    )

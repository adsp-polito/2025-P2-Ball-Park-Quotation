"""
FPT Cost Brain 2.0 - Estimation State Definition
TypedDict for LangGraph state management
"""

from enum import Enum
from typing import Any, TypedDict


class StepStatus(str, Enum):
    """Status for each estimation step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class ParsedPR(TypedDict, total=False):
    """
    Parsed Product Request data.

    Includes comprehensive feature extraction ported from v1:
    - Basic fields (pr_code, title, description, etc.)
    - Platform/Engine/Tier fields
    - Product family detection (E0C0, NEF, CURSOR, F1, E5F0)
    - Emissions detection (Stage V, Tier 4B, China NRIV, Euro VI)
    - Sector detection (AG, CE, PT)
    - ATS technology detection
    - Component-specific flags for sizing classification
    - Power/torque detection
    - Boolean flags for change types
    - Raw text for LLM processing
    """

    # Basic fields
    pr_code: str
    revision: str
    title: str
    description: str
    program_family: str
    customer: str
    project_phase: str

    # FPT-specific fields
    platform: str
    engine: str
    tier: str

    # Detected features (from v1 parser)
    product_family: str  # E0C0, NEF, CURSOR, F1, E5F0
    emissions: str  # Stage V, Tier 4B, China NRIV, Euro VI
    sector: str  # AG, CE, PT
    sector_source: (
        str  # How sector was determined (explicit_field, product_family, etc.)
    )
    ats_tech: str  # DOC_SCRoF, DOC_SCR-T, SCR_only, DOC_only

    # Boolean flags for change types
    hardware_change: bool
    calibration_change: bool
    ats_change: bool
    software_vcu_change: bool

    # Component-specific flags (for sizing classification)
    turbo_related: bool
    injectors_related: bool
    fuel_rail_related: bool
    EGR_related: bool
    cooling_system_related: bool

    # Power and torque detection
    power_kw: float | None
    torque_nm: float | None
    power_increase_kw: float | None
    torque_increase_nm: float | None

    # Consolidated extracted features dict (for sizing and estimation)
    extracted_features: dict[str, Any]

    # Raw data
    raw_text: str  # All text for LLM processing
    raw_activities: list[dict[str, Any]]
    raw_data: dict[str, Any]
    validation_errors: list[str]


class Question(TypedDict):
    """Q&A question structure."""

    id: str
    question: str
    reason: str
    category: str  # scope, complexity, dependencies, technical, timeline
    priority: str  # high, medium, low
    suggested_answers: list[str]
    answer: str | None
    is_answered: bool


class PRSummary(TypedDict, total=False):
    """PR summary after analysis."""

    summary_text: str
    program_size: str  # small, medium, large, xl
    complexity_score: float
    activity_count: int  # Number of technical activities/functions identified
    key_features: list[str]
    dependencies: list[str]
    risk_factors: list[str]
    special_requirements: list[str]


class MLFeature(TypedDict):
    """ML feature for prediction."""

    name: str
    value: float | str | bool
    source: str  # extracted, inferred, default


class SimilarPRRequired(TypedDict):
    """Required fields for Similar PR from vector search."""

    id: str
    pr_code: str
    title: str
    program_family: str
    similarity_score: float
    total_hours: float
    total_cost_eur: float


class SimilarPR(SimilarPRRequired, total=False):
    """
    Similar PR from vector search with optional enriched fields.

    Required fields: id, pr_code, title, program_family, similarity_score,
                     total_hours, total_cost_eur

    Optional fields (from enriched pr_embeddings payload):
    - customer_platform: e.g., "Wheel Loader", "Cash Crop Medium Tractor"
    - sector: AG (Agriculture), CE (Construction Equipment), PT (Powertrain)
    - sizing: Program size classification (X-small, Small, Mid, Full, Large)
    - emissions: Emission standard (Stage V, Tier 4B, China NRIV, etc.)
    """

    # Enriched fields from pr_embeddings v3 payload
    customer_platform: str  # Machine/vehicle type
    sector: str  # Business sector (AG, CE, PT)
    sizing: str  # Program sizing classification
    emissions: str  # Emission regulation standard

    # R&D breakdown for CBR context (from db_RandD_output.csv)
    rd_breakdown: dict[str, Any] | None  # PE02 function-level hours/cost breakdown


class SizingResult(TypedDict):
    """Result of sizing classification for a domain."""

    sizing: str  # Full, Large, Medium, Small, X-small
    confidence: float  # 0.0-1.0
    reasoning: str
    rule_id: str  # Reference to applied rule (e.g., PE_BASE_L_001)
    method: str  # "llm", "keyword", or "default"


class ProgramSizing(TypedDict):
    """Complete program sizing across all 9 domains + aggregated.

    Domains from ref_sizing.json:
    - Product Engineering (3): Base Engine, System (engine+ATS), Installation/Application
    - Manufacturing (2): Plant - base engine, Plant - ATS
    - Purchasing (2): Sourcing, Supplier Quality
    - Customer Manager (1): Build stages
    - Program Manager (1): Overall
    """

    # Product Engineering (3)
    pe_base_powertrain: SizingResult
    pe_system_assembly: SizingResult
    pe_installation_application: SizingResult

    # Manufacturing (2)
    manufacturing_base_engine: SizingResult
    manufacturing_ats: SizingResult

    # Purchasing (2)
    purchasing_sourcing: SizingResult
    purchasing_supplier_quality: SizingResult

    # Customer Manager (1)
    customer_build_stages: SizingResult

    # Program Manager (1)
    program_manager_overall: SizingResult

    # Aggregated (max of all domains)
    program_overall: SizingResult


class BreakdownItem(TypedDict, total=False):
    """
    Quotation breakdown item in PE02 format.

    PE02 Functions: A1, A2, A3, A4, B1, B1-C, B2, B3, C, D1+D2, D3, E, F, G
    Effort columns: Manpower (hrs), Bench Dev/Special/Dur (hrs), Vehicle (hrs), Cost (k€)
    """

    id: str

    # PE02 function identifiers
    code: str  # A1, A2, B1, etc.
    function: str  # Project Management, Design & Release, etc.
    description: str

    # PE02 effort breakdown columns
    effort_manpower: float  # Hours
    effort_bench_dev: float  # Hours - Bench Development
    effort_bench_special: float  # Hours - Bench Special (NVH, climatic)
    effort_bench_dur: float  # Hours - Bench Durability
    effort_vehicle: float  # Hours - Vehicle testing
    investment_keur: float  # Cost in k€

    # Legacy fields (for backward compatibility)
    activity_code: str
    activity_name: str
    hours: float  # Total hours (sum of all effort columns)
    hourly_rate_eur: float
    cost_eur: float  # Cost in €

    # Metadata
    confidence_score: float
    reasoning: str
    source: str  # pe02_baseline, pe02_llm_adjusted, model, rule, similar_pr, manual
    user_edited: bool
    edit_reason: str | None


class AppliedRule(TypedDict):
    """Rule that was applied to estimation."""

    rule_id: str
    rule_name: str
    description: str
    adjustment_type: str
    adjustment_value: float
    target_activity: str | None


class UserEdit(TypedDict):
    """User edit in review step."""

    breakdown_id: str
    original_hours: float
    new_hours: float
    reason: str
    timestamp: str


class ExportResult(TypedDict, total=False):
    """Export generation result."""

    pptx_path: str | None
    xlsx_path: str | None
    pptx_bytes: bytes | None
    xlsx_bytes: bytes | None
    generated_at: str


class LearningResult(TypedDict, total=False):
    """Learning extraction result."""

    extracted_rules: list[dict[str, Any]]
    feedback_ids: list[str]
    queued_for_retrain: bool


class EstimationState(TypedDict, total=False):
    """
    Complete state for the estimation workflow.

    This TypedDict defines all data that flows through the LangGraph
    estimation pipeline. Each step reads and writes specific keys.

    Step Flow:
    1. Intake: Parse PR file, validate, extract initial data
    2. Q&A: Generate questions, collect answers
    3. Summary: Analyze PR, extract features, find similar PRs
    4. Estimation: Run ML prediction, apply rules, generate breakdown
    5. Review: Allow user edits, capture feedback
    6. Export: Generate PE02 documents
    7. Learning: Extract rules from corrections, queue for retraining
    """

    # ===== Session Metadata =====
    session_id: str
    user_id: str
    pr_id: str
    created_at: str
    updated_at: str

    # ===== Step Tracking =====
    current_step: str  # intake, qa, summary, estimation, review, export, complete
    step_status: dict[str, StepStatus]  # Status for each step
    error_message: str | None
    error_step: str | None

    # ===== Step 1: Intake =====
    pr_file_bytes: bytes | None
    pr_filename: str | None
    parsed_pr: ParsedPR | None
    validation_result: dict[str, Any] | None
    is_valid: bool

    # ===== Step 2: Q&A =====
    questions: list[Question]
    answers: dict[str, str]  # question_id -> answer
    qa_complete: bool
    qa_skipped: bool

    # ===== Step 3: Summary =====
    pr_summary: PRSummary | None
    ml_features: list[MLFeature]
    similar_prs: list[SimilarPR]
    embedding: list[float] | None
    embedding_id: str | None

    # ===== Step 4: Estimation =====
    quotation_id: str | None
    breakdown: list[BreakdownItem]
    total_hours: float
    total_cost_eur: float
    overall_confidence: float
    applied_rules: list[AppliedRule]
    estimation_method: str  # model_only, rules_only, hybrid, pe02_hybrid
    ml_prediction: dict[str, Any] | None
    program_sizing: dict[str, Any] | None  # ProgramSizing dict from sizing classifier
    # LLM-based sizing predictions (from ref_sizing.json rules)
    ml_sizing: str | None  # Overall sizing: Full, Large, Medium, Small, X-small
    sizing_predictions: (
        dict[str, Any] | None
    )  # Per-domain: {domain: {size, confidence, reason}}
    sizing_confidence: float  # Overall sizing confidence 0-1
    ml_interval: tuple[float, float] | None  # HCQE prediction interval (low, high)
    ml_recommendations: list[str]  # HCQE recommendations

    # ===== Step 5: Review =====
    user_edits: list[UserEdit]
    feedback_reasons: dict[str, str]  # breakdown_id -> reason
    is_finalized: bool
    finalized_by: str | None
    finalized_at: str | None

    # ===== Step 6: Export =====
    export_result: ExportResult | None
    export_format: str  # pptx, xlsx, bundle
    export_language: str  # en, it

    # ===== Step 7: Learning =====
    learning_result: LearningResult | None

    # ===== Chat Context =====
    chat_messages: list[dict[str, Any]]
    chat_context: dict[str, Any]


def create_initial_state(
    session_id: str,
    user_id: str,
) -> EstimationState:
    """Create initial state for a new estimation session."""
    from datetime import datetime, timezone

    return EstimationState(
        # Session metadata
        session_id=session_id,
        user_id=user_id,
        pr_id="",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        # Step tracking
        current_step="intake",
        step_status={
            "intake": StepStatus.PENDING,
            "qa": StepStatus.PENDING,
            "summary": StepStatus.PENDING,
            "estimation": StepStatus.PENDING,
            "review": StepStatus.PENDING,
            "export": StepStatus.PENDING,
        },
        error_message=None,
        error_step=None,
        # Intake
        pr_file_bytes=None,
        pr_filename=None,
        parsed_pr=None,
        validation_result=None,
        is_valid=False,
        # Q&A
        questions=[],
        answers={},
        qa_complete=False,
        qa_skipped=False,
        # Summary
        pr_summary=None,
        ml_features=[],
        similar_prs=[],
        embedding=None,
        embedding_id=None,
        # Estimation
        quotation_id=None,
        breakdown=[],
        total_hours=0.0,
        total_cost_eur=0.0,
        overall_confidence=0.0,
        applied_rules=[],
        estimation_method="pe02_hybrid",
        ml_prediction=None,
        program_sizing=None,
        ml_sizing=None,
        sizing_predictions=None,
        sizing_confidence=0.5,
        ml_interval=None,
        ml_recommendations=[],
        # Review
        user_edits=[],
        feedback_reasons={},
        is_finalized=False,
        finalized_by=None,
        finalized_at=None,
        # Export
        export_result=None,
        export_format="pptx",
        export_language="en",
        # Learning
        learning_result=None,
        # Chat
        chat_messages=[],
        chat_context={},
    )


def get_state_for_step(state: EstimationState, step: str) -> dict[str, Any]:
    """Get relevant state keys for a specific step."""
    step_keys = {
        "intake": [
            "pr_file_bytes",
            "pr_filename",
            "parsed_pr",
            "validation_result",
            "is_valid",
        ],
        "qa": [
            "parsed_pr",
            "similar_prs",
            "questions",
            "answers",
            "qa_complete",
        ],
        "summary": [
            "parsed_pr",
            "answers",
            "pr_summary",
            "ml_features",
            "similar_prs",
            "embedding",
        ],
        "estimation": [
            "parsed_pr",
            "pr_summary",
            "ml_features",
            "similar_prs",
            "breakdown",
            "total_hours",
            "total_cost_eur",
            "overall_confidence",
            "applied_rules",
            "ml_prediction",
            "program_sizing",
            "ml_sizing",
            "sizing_predictions",
            "sizing_confidence",
            "ml_interval",
            "ml_recommendations",
            "answers",
            "questions",
        ],
        "review": [
            "breakdown",
            "total_hours",
            "total_cost_eur",
            "user_edits",
            "feedback_reasons",
            "is_finalized",
        ],
        "export": [
            "pr_summary",
            "breakdown",
            "total_hours",
            "total_cost_eur",
            "export_result",
            "export_format",
        ],
    }

    keys = step_keys.get(step, [])
    return {k: state.get(k) for k in keys if k in state}

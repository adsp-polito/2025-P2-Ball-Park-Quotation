"""
FPT Cost Brain 2.0 - Estimation API
Endpoints for the 5-step estimation workflow with LangGraph integration
"""

from typing import Annotated, Any, Literal
from uuid import UUID

from app.dependencies import get_current_user, get_db
from db.models import User
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from services.estimation_service import EstimationService
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["Estimation"])


# ===== Schemas =====


class QuestionSchema(BaseModel):
    """Schema for Q&A question."""

    id: str
    question: str
    reason: str | None = None
    category: str
    priority: str
    suggested_answers: list[str] = []
    answer: str | None = None
    is_answered: bool = False


class BreakdownItemSchema(BaseModel):
    """Schema for breakdown item with PE02 fields."""

    id: str
    # PE02 Standard Fields
    code: str | None = None  # PE02 function code: A1, A2, B1, etc.
    function: str | None = None  # PE Function name: Project Management, etc.
    description: str | None = None  # Activity description
    # PE02 Effort Columns (hours)
    effort_manpower: float = 0
    effort_bench_dev: float = 0
    effort_bench_special: float = 0
    effort_bench_dur: float = 0
    effort_vehicle: float = 0
    # PE02 Cost
    investment_keur: float = 0  # Cost in k€
    # Legacy fields (backward compatibility)
    activity_code: str
    activity_name: str
    hours: float
    hourly_rate_eur: float = 150.0
    cost_eur: float
    confidence_score: float
    reasoning: str | None = None
    source: str
    user_edited: bool = False
    edit_reason: str | None = None


class SimilarPRSchema(BaseModel):
    """Schema for similar PR."""

    id: str
    pr_code: str
    title: str
    program_family: str | None = None
    similarity_score: float
    total_hours: float
    total_cost_eur: float


class EstimationStateResponse(BaseModel):
    """Full estimation state response - matches frontend EstimationState."""

    session_id: str
    pr_id: str | None = None
    current_step: str
    status: str = "active"
    created_at: str | None = None
    updated_at: str | None = None

    # Step completion status (for navigation)
    step_status: dict[
        str, str
    ] = {}  # e.g., {"intake": "COMPLETED", "qa": "WAITING_INPUT"}

    # Parsed PR data
    parsed_pr: dict[str, Any] | None = None

    # Q&A - with LAZY LOADING support
    questions: list[QuestionSchema] = []
    questions_ready: bool = False  # True when questions have been generated
    questions_generating: bool = False  # True while generation in progress
    answers: dict[str, str] = {}

    # Summary
    pr_summary: dict[str, Any] | None = None
    similar_prs: list[SimilarPRSchema] = []

    # Estimation
    breakdown: list[BreakdownItemSchema] = []
    total_hours: float = 0
    total_cost_eur: float = 0
    overall_confidence: float = 0
    applied_rules: list[dict[str, Any]] = []
    # ML prediction metadata
    ml_sizing: str | None = None
    sizing_predictions: dict[str, Any] | None = None
    sizing_confidence: float | None = None
    ml_prediction: dict[str, Any] | None = None
    ml_interval: list[float | None] | None = None  # [lower_bound, upper_bound]
    ml_recommendations: list[str] = []
    estimation_method: str | None = None

    # Review
    user_edits: list[dict[str, Any]] = []
    is_finalized: bool = False

    # Export
    export_result: dict[str, Any] | None = None

    # Error
    error_message: str | None = None


class EstimationSessionResponse(BaseModel):
    """Simple session response for start endpoint."""

    session_id: str
    pr_id: str | None = None
    current_step: str
    status: str
    created_at: str | None = None


class AnswersSubmitRequest(BaseModel):
    """Request to submit Q&A answers."""

    answers: dict[str, str] = Field(..., description="Map of question_id to answer")


class BreakdownUpdateRequest(BaseModel):
    """Request to update a breakdown item."""

    hours: float | None = None
    hourly_rate_eur: float | None = None
    reason: str | None = None


class RegenerateQuestionsRequest(BaseModel):
    """Request to regenerate questions based on chat context."""

    message: str = Field(..., description="User message that triggers question update")


class StepAdvanceRequest(BaseModel):
    """Request to advance to next step."""

    data: dict[str, Any] | None = None


class GoToStepRequest(BaseModel):
    """Request to navigate to a specific completed step."""

    step: Literal["upload", "qa", "summary", "estimation", "review"]


# ===== Helper Functions =====


def get_estimation_service(db: AsyncSession) -> EstimationService:
    """Create estimation service instance."""
    return EstimationService(db)


# ===== Endpoints =====


@router.post(
    "/start",
    response_model=EstimationSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_estimation(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(None),
):
    """
    Start a new estimation session.

    Upload a PR Excel file to begin the estimation process.
    The system will:
    1. Parse the Excel file and extract PR data
    2. Generate smart clarifying questions
    3. Transition to the Q&A step

    Returns session_id to use for subsequent API calls.
    """
    # Validate file
    file_bytes = None
    filename = None

    if file and file.filename:
        if not file.filename.endswith((".xls", ".xlsx")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Please upload an Excel file (.xls or .xlsx)",
            )
        file_bytes = await file.read()
        filename = file.filename

    # Use service to start session
    service = get_estimation_service(db)

    try:
        result = await service.start_session(
            user_id=str(current_user.id),
            file_bytes=file_bytes,
            filename=filename,
        )

        return EstimationSessionResponse(
            session_id=result["session_id"],
            pr_id=result.get("pr_id"),
            current_step=result["current_step"],
            status=result["status"],
            created_at=result.get("created_at"),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start estimation: {str(e)}",
        )


@router.get("/{session_id}", response_model=EstimationStateResponse)
async def get_estimation_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Get the full state of an estimation session.

    Returns all data for the current step including:
    - Parsed PR data
    - Questions and answers
    - PR summary and similar PRs
    - Estimation breakdown
    - User edits and export results
    """
    service = get_estimation_service(db)

    result = await service.get_session(session_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Estimation session not found",
        )

    return EstimationStateResponse(**result)


@router.post("/{session_id}/answers", response_model=EstimationStateResponse)
async def submit_answers(
    session_id: str,
    request: AnswersSubmitRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Submit answers to Q&A questions.

    After submitting answers, the workflow advances to:
    1. Summary step - analyzes PR and finds similar projects
    2. Estimation step - generates cost breakdown

    The session will be in 'summary' or 'estimation' step after this.
    """
    service = get_estimation_service(db)

    try:
        result = await service.submit_answers(session_id, request.answers)
        return EstimationStateResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit answers: {str(e)}",
        )


@router.post("/{session_id}/skip-qa", response_model=EstimationStateResponse)
async def skip_qa(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Skip the Q&A step and proceed directly to summary.

    Use this when the PR contains enough information to proceed
    without additional clarification.
    """
    service = get_estimation_service(db)

    try:
        result = await service.skip_qa(session_id)
        return EstimationStateResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{session_id}/generate-questions", response_model=EstimationStateResponse)
async def generate_questions(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Generate Q&A questions for the session (LAZY LOADING).

    This endpoint triggers question generation when the user enters
    the Q&A step. Questions are NOT generated during file upload to
    provide a faster initial upload experience.

    The frontend should:
    1. Call this endpoint when entering Q&A step
    2. Show a loading animation while questions_generating is true
    3. Display questions once questions_ready is true

    Returns the updated session state with generated questions.
    """
    service = get_estimation_service(db)

    try:
        result = await service.generate_questions(session_id)
        return EstimationStateResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate questions: {str(e)}",
        )


@router.post(
    "/{session_id}/regenerate-questions", response_model=EstimationStateResponse
)
async def regenerate_questions(
    session_id: str,
    request: RegenerateQuestionsRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Regenerate questions dynamically based on chat interaction.

    When the user provides information through chat, this endpoint
    updates the question list:
    - Removes questions already answered by the message
    - Adds new relevant questions based on revealed information
    - Reprioritizes remaining questions

    This enables interactive Q&A where the frontend can update
    the question list in real-time as the user chats.
    """
    service = get_estimation_service(db)

    try:
        result = await service.regenerate_questions(session_id, request.message)
        return EstimationStateResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to regenerate questions: {str(e)}",
        )


@router.post("/{session_id}/next", response_model=EstimationStateResponse)
async def advance_to_next_step(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    request: StepAdvanceRequest | None = None,
):
    """
    Advance to the next step in the estimation workflow.

    Steps:
    1. intake -> qa (automatic after start)
    2. qa -> summary (after submitting answers or skipping)
    3. summary -> estimation (automatic after analysis)
    4. estimation -> review (user views and edits)
    5. review -> export (after finalization)
    6. export -> complete (learning triggers)
    """
    import logging
    import time

    logger = logging.getLogger(__name__)
    api_start = time.time()

    logger.info("=" * 70)
    logger.info(f"🌐 API /next called for session {session_id[:8]}...")
    logger.info("=" * 70)
    service = get_estimation_service(db)

    try:
        logger.info(f"⏳ Calling advance_step...")
        result = await service.advance_step(
            session_id,
            request.data if request else None,
        )
        elapsed = time.time() - api_start
        logger.info(f"✅ advance_step completed in {elapsed:.2f}s")
        logger.info(f"📦 Result keys: {list(result.keys()) if result else 'None'}")
        # Wrap in try-except to catch Pydantic validation errors
        try:
            response = EstimationStateResponse(**result)
            logger.info(f"[API] Response built successfully")
            return response
        except Exception as pydantic_error:
            # Log detailed error info for debugging
            logger.error(f"[API] Pydantic validation failed: {pydantic_error}")
            logger.error(
                f"[API] Result keys: {list(result.keys()) if result else 'None'}"
            )
            # Log problematic fields
            for key in [
                "step_status",
                "similar_prs",
                "breakdown",
                "pr_summary",
                "ml_interval",
            ]:
                value = result.get(key)
                logger.error(
                    f"[API] {key}: type={type(value).__name__}, value={str(value)[:200] if value else 'None'}"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Response validation failed: {str(pydantic_error)}",
            )
    except ValueError as e:
        logger.error(f"[API] ValueError in /next: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except HTTPException:
        raise  # Re-raise HTTPExceptions as-is
    except Exception as e:
        logger.exception(f"[API] Exception in /next for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to advance step: {str(e)}",
        )


@router.post("/{session_id}/go-to-step", response_model=EstimationStateResponse)
async def go_to_step(
    session_id: str,
    request: GoToStepRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Navigate to a specific completed step (view only, no re-processing).

    Allows users to go back and review previous steps without restarting.
    Only completed steps can be navigated to - this is validated server-side.
    """
    service = get_estimation_service(db)

    try:
        result = await service.go_to_step(session_id, request.step)
        return EstimationStateResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to navigate to step: {str(e)}",
        )


@router.patch("/{session_id}/breakdown/{item_id}")
async def update_breakdown_item(
    session_id: str,
    item_id: str,
    update: BreakdownUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Update a specific breakdown item during review.

    User edits are recorded for the learning system to improve
    future predictions. Provide a reason for better rule extraction.
    """
    service = get_estimation_service(db)

    try:
        result = await service.update_breakdown_item(
            session_id,
            item_id,
            hours=update.hours,
            reason=update.reason,
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{session_id}/finalize", response_model=EstimationStateResponse)
async def finalize_estimation(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Finalize the estimation and trigger export/learning.

    This will:
    1. Mark the estimation as finalized
    2. Generate PE02 export documents
    3. Extract learning rules from user corrections
    4. Queue for model retraining if needed
    """
    service = get_estimation_service(db)

    try:
        result = await service.finalize_session(session_id, str(current_user.id))
        return EstimationStateResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to finalize: {str(e)}",
        )


# ===== HCQE Prediction Endpoint =====


class HCQEPredictionRequest(BaseModel):
    """Schema for HCQE prediction request."""

    features: dict[str, Any]


class HCQEPredictionResponse(BaseModel):
    """Schema for HCQE prediction response."""

    predicted_cost_keur: float
    predicted_hours: float
    confidence: float
    method: str
    prediction_interval: dict[str, float] | None = None
    quantiles: dict[str, float] | None = None
    sizing: dict[str, Any] | None = None
    cluster_estimates: dict[str, float] = {}
    reasoning: str = ""
    recommendations: list[str] = []


@router.post("/predict", response_model=HCQEPredictionResponse)
async def predict_cost(
    request: HCQEPredictionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Get HCQE cost prediction for given features.

    Uses Hierarchical Conformal Quantile Ensemble (HCQE) for:
    - 78.8% within-30% accuracy
    - Calibrated prediction intervals with 81.8% coverage
    - Sizing-aware predictions with confidence scores
    """
    try:
        from ml.hcqe_predictor import get_hcqe_predictor

        hcqe = get_hcqe_predictor()
        if hcqe:
            result = hcqe.predict(request.features)
            return HCQEPredictionResponse(
                predicted_cost_keur=result.get("predicted_cost_keur", 0),
                predicted_hours=result.get("predicted_hours", 0),
                confidence=result.get("confidence", 0.5),
                method="hcqe",
                prediction_interval=result.get("prediction_interval"),
                quantiles=result.get("quantiles"),
                sizing=result.get("sizing"),
                cluster_estimates=result.get("cluster_estimates", {}),
                reasoning=result.get("reasoning", ""),
                recommendations=result.get("recommendations", []),
            )
        else:
            # Fallback prediction
            return HCQEPredictionResponse(
                predicted_cost_keur=500.0,
                predicted_hours=3000.0,
                confidence=0.3,
                method="fallback",
                reasoning="HCQE model not loaded, using default values",
                recommendations=["Upload historical data to improve predictions"],
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get("/predict/health")
async def prediction_health():
    """Check if HCQE model is loaded and ready."""
    try:
        from ml.hcqe_predictor import get_hcqe_predictor

        hcqe = get_hcqe_predictor()
        if hcqe:
            return {
                "status": "healthy",
                "model": "hcqe",
                "version": "1.0",
                "accuracy": "78.8% within 30%",
                "interval_coverage": "81.8%",
            }
    except Exception:
        pass

    return {
        "status": "degraded",
        "model": "fallback",
        "message": "HCQE model not loaded, using fallback predictions",
    }

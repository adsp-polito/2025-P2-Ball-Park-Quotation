"""
FPT Cost Brain 2.0 - RLHF API Endpoints
Preference pairs, A/B experiments, and training jobs
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from db.models import User
from services.rlhf_service import (
    ABExperimentService,
    PreferencePairService,
    RLHFTrainingService,
)
from ml.ab_router import (
    ABRouterManager,
    ExperimentMetrics,
    ExperimentStatus,
    check_kill_switch,
    get_ab_router_manager,
)
from services.dpo_dataset_generator import (
    DPODatasetGenerator,
    FineTuningProvider,
    FineTuningUploadPipeline,
)
from services.rlhf_monitoring import (
    RLHFMonitoringService,
    export_prometheus_metrics,
    AlertSeverity,
    AlertType,
)
from llm.client import LLMClient
from fastapi.responses import PlainTextResponse

router = APIRouter(prefix="/rlhf", tags=["rlhf"])


# ===== Pydantic Schemas =====


class PreferencePairCreate(BaseModel):
    """Schema for creating a preference pair manually."""

    session_id: uuid.UUID | None = None
    chosen_reasoning: str
    rejected_reasoning: str
    chosen_breakdown: dict[str, Any]
    rejected_breakdown: dict[str, Any]
    signal_source: str = Field(
        ..., pattern="^(user_edit|actual_outcome|synthetic_negative|explicit_approval)$"
    )
    reward_delta: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class PreferencePairResponse(BaseModel):
    """Schema for preference pair response."""

    id: uuid.UUID
    session_id: uuid.UUID | None
    chosen_reasoning: str
    rejected_reasoning: str
    chosen_breakdown: dict[str, Any]
    rejected_breakdown: dict[str, Any]
    signal_source: str
    reward_delta: float
    confidence: float
    validated: bool
    created_at: str
    used_in_training: str | None

    class Config:
        from_attributes = True


class PreferencePairValidate(BaseModel):
    """Schema for validating a preference pair."""

    validated: bool
    confidence_override: float | None = Field(None, ge=0.0, le=1.0)


class ABExperimentCreate(BaseModel):
    """Schema for creating an A/B experiment."""

    name: str = Field(..., min_length=3, max_length=100)
    candidate_model_version: str
    production_model_version: str


class ABExperimentResponse(BaseModel):
    """Schema for A/B experiment response."""

    id: uuid.UUID
    name: str
    candidate_model_version: str
    production_model_version: str
    status: str
    candidate_weight: float
    shadow_mode: bool
    kill_switch_triggered: bool
    metrics_snapshot: dict[str, Any] | None
    created_at: str
    completed_at: str | None

    class Config:
        from_attributes = True


class ABExperimentPromote(BaseModel):
    """Schema for promoting an experiment."""

    new_status: str = Field(..., pattern="^(canary|gradual|complete|rolled_back)$")
    new_weight: float | None = Field(None, ge=0.0, le=1.0)


class ABExperimentStats(BaseModel):
    """Schema for experiment statistics."""

    prediction_counts: dict[str, int]
    sizing_spread: dict[str, int]
    edit_counts: dict[str, int]
    total_predictions: int


class TrainingJobCreate(BaseModel):
    """Schema for creating a training job."""

    job_type: str = Field(..., pattern="^(ml_retrain|llm_dpo)$")


class TrainingJobResponse(BaseModel):
    """Schema for training job response."""

    id: uuid.UUID
    job_type: str
    status: str
    samples_used: int | None
    metrics_before: dict[str, Any] | None
    metrics_after: dict[str, Any] | None
    model_version_created: str | None
    error_message: str | None
    started_at: str | None
    completed_at: str | None
    created_at: str

    class Config:
        from_attributes = True


class RLHFMetricsOverview(BaseModel):
    """Schema for RLHF system overview."""

    total_pairs: int
    pairs_by_source: dict[str, int]
    unvalidated_pairs: int
    pending_training_pairs: int
    active_experiment: ABExperimentResponse | None
    active_jobs: list[TrainingJobResponse]


class ABRouterStatus(BaseModel):
    """Schema for A/B router runtime status."""

    status: str  # "active" or "no_active_experiment"
    experiment_id: str | None = None
    production_version: str | None = None
    candidate_version: str | None = None
    candidate_weight: float = 0.0
    shadow_mode: bool = False
    shadow_stats: dict[str, Any] | None = None


class KillSwitchEvaluation(BaseModel):
    """Schema for kill switch evaluation request."""

    production_mape: float = Field(..., ge=0.0)
    candidate_mape: float = Field(..., ge=0.0)
    production_confidence: float = Field(..., ge=0.0, le=1.0)
    candidate_confidence: float = Field(..., ge=0.0, le=1.0)
    production_edit_rate: float = Field(..., ge=0.0, le=1.0)
    candidate_edit_rate: float = Field(..., ge=0.0, le=1.0)
    production_samples: int = Field(..., ge=0)
    candidate_samples: int = Field(..., ge=0)


class KillSwitchResponse(BaseModel):
    """Schema for kill switch evaluation response."""

    should_kill: bool
    reason: str
    threshold_breached: str | None = None
    auto_rolled_back: bool = False


class DPOExportRequest(BaseModel):
    """Schema for DPO dataset export request."""

    provider: str = Field(..., pattern="^(openai|fireworks|together|anyscale)$")
    validated_only: bool = True
    min_confidence: float = Field(0.5, ge=0.0, le=1.0)
    generate_synthetic: bool = True
    limit: int | None = Field(None, ge=1, le=10000)
    mark_as_used: bool = False


class DPOExportResponse(BaseModel):
    """Schema for DPO dataset export response."""

    file_path: str
    total_examples: int
    examples_with_synthetic: int
    provider_format: str
    export_timestamp: str
    validation_passed: bool
    validation_errors: list[str]


class DPOExportStats(BaseModel):
    """Schema for DPO export statistics."""

    total_pairs: int
    validated_pairs: int
    unused_pairs: int
    pairs_by_source: dict[str, int]
    ready_for_export: int


class UploadConfigRequest(BaseModel):
    """Schema for upload config generation."""

    provider: str = Field(..., pattern="^(openai|fireworks|together)$")
    jsonl_path: str
    model_suffix: str = "fpt-costbrain"
    n_epochs: int = Field(3, ge=1, le=10)


class DatasetValidationResponse(BaseModel):
    """Schema for dataset validation response."""

    valid: bool
    examples_count: int
    issues: list[str]
    file_path: str


# ===== Monitoring Schemas =====


class TimeSeriesPointResponse(BaseModel):
    """Schema for time series data point."""

    timestamp: str
    value: float
    labels: dict[str, str]


class AlertResponse(BaseModel):
    """Schema for monitoring alert."""

    alert_type: str
    severity: str
    message: str
    details: dict[str, Any]
    timestamp: str
    acknowledged: bool


class DashboardMetricsResponse(BaseModel):
    """Schema for dashboard metrics."""

    # Summary stats
    total_predictions: int
    total_corrections: int
    total_preference_pairs: int
    active_experiment: str | None

    # Performance metrics
    current_mape: float | None
    avg_confidence: float | None
    user_edit_rate: float

    # Time series (last 7 days)
    predictions_by_day: list[TimeSeriesPointResponse]
    corrections_by_day: list[TimeSeriesPointResponse]
    mape_trend: list[TimeSeriesPointResponse]

    # Health indicators
    model_health: str
    data_pipeline_health: str
    alerts: list[AlertResponse]


class ExperimentComparisonResponse(BaseModel):
    """Schema for experiment comparison metrics."""

    status: str
    experiment_id: str | None = None
    experiment_name: str | None = None
    production: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    candidate_weight: float | None = None
    shadow_mode: bool | None = None


# ===== Preference Pairs Endpoints =====


@router.get("/preference-pairs", response_model=list[PreferencePairResponse])
async def list_preference_pairs(
    limit: int = Query(50, ge=1, le=200),
    unused_only: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """List preference pairs."""
    service = PreferencePairService(db)
    if unused_only:
        pairs = await service.get_pending_pairs(limit)
    else:
        from sqlalchemy import select
        from db.models import PreferencePair

        result = await db.execute(
            select(PreferencePair)
            .order_by(PreferencePair.created_at.desc())
            .limit(limit)
        )
        pairs = list(result.scalars().all())

    return [
        PreferencePairResponse(
            id=p.id,
            session_id=p.session_id,
            chosen_reasoning=p.chosen_reasoning,
            rejected_reasoning=p.rejected_reasoning,
            chosen_breakdown=p.chosen_breakdown,
            rejected_breakdown=p.rejected_breakdown,
            signal_source=p.signal_source,
            reward_delta=float(p.reward_delta),
            confidence=float(p.confidence),
            validated=p.validated,
            created_at=p.created_at.isoformat() if p.created_at else "",
            used_in_training=p.used_in_training.isoformat()
            if p.used_in_training
            else None,
        )
        for p in pairs
    ]


@router.get("/preference-pairs/pending", response_model=list[PreferencePairResponse])
async def get_pending_pairs(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Get unvalidated pairs needing human review."""
    service = PreferencePairService(db)
    pairs = await service.get_unvalidated_pairs(limit)

    return [
        PreferencePairResponse(
            id=p.id,
            session_id=p.session_id,
            chosen_reasoning=p.chosen_reasoning,
            rejected_reasoning=p.rejected_reasoning,
            chosen_breakdown=p.chosen_breakdown,
            rejected_breakdown=p.rejected_breakdown,
            signal_source=p.signal_source,
            reward_delta=float(p.reward_delta),
            confidence=float(p.confidence),
            validated=p.validated,
            created_at=p.created_at.isoformat() if p.created_at else "",
            used_in_training=p.used_in_training.isoformat()
            if p.used_in_training
            else None,
        )
        for p in pairs
    ]


@router.patch("/preference-pairs/{pair_id}")
async def validate_preference_pair(
    pair_id: uuid.UUID,
    data: PreferencePairValidate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Validate or reject a preference pair."""
    service = PreferencePairService(db)
    success = await service.validate_pair(
        pair_id, data.validated, data.confidence_override
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preference pair not found",
        )
    await db.commit()
    return {"status": "ok", "validated": data.validated}


@router.delete("/preference-pairs/{pair_id}")
async def delete_preference_pair(
    pair_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Delete an invalid preference pair."""
    from sqlalchemy import delete
    from db.models import PreferencePair

    result = await db.execute(
        delete(PreferencePair).where(PreferencePair.id == pair_id)
    )
    if result.rowcount == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preference pair not found",
        )
    await db.commit()
    return {"status": "deleted"}


# ===== A/B Experiments Endpoints =====


@router.post("/experiments", response_model=ABExperimentResponse)
async def create_experiment(
    data: ABExperimentCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Create a new A/B experiment (starts in shadow mode)."""
    service = ABExperimentService(db)

    # Check for existing active experiment
    active = await service.get_active_experiment()
    if active:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Active experiment already exists: {active.name}",
        )

    experiment = await service.create_experiment(
        data.name, data.candidate_model_version, data.production_model_version
    )
    await db.commit()

    return ABExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        candidate_model_version=experiment.candidate_model_version,
        production_model_version=experiment.production_model_version,
        status=experiment.status,
        candidate_weight=float(experiment.candidate_weight),
        shadow_mode=experiment.shadow_mode,
        kill_switch_triggered=experiment.kill_switch_triggered,
        metrics_snapshot=experiment.metrics_snapshot,
        created_at=experiment.created_at.isoformat() if experiment.created_at else "",
        completed_at=experiment.completed_at.isoformat()
        if experiment.completed_at
        else None,
    )


@router.get("/experiments", response_model=list[ABExperimentResponse])
async def list_experiments(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """List A/B experiments."""
    from sqlalchemy import select
    from db.models import ABExperiment

    result = await db.execute(
        select(ABExperiment).order_by(ABExperiment.created_at.desc()).limit(limit)
    )
    experiments = list(result.scalars().all())

    return [
        ABExperimentResponse(
            id=e.id,
            name=e.name,
            candidate_model_version=e.candidate_model_version,
            production_model_version=e.production_model_version,
            status=e.status,
            candidate_weight=float(e.candidate_weight),
            shadow_mode=e.shadow_mode,
            kill_switch_triggered=e.kill_switch_triggered,
            metrics_snapshot=e.metrics_snapshot,
            created_at=e.created_at.isoformat() if e.created_at else "",
            completed_at=e.completed_at.isoformat() if e.completed_at else None,
        )
        for e in experiments
    ]


@router.get("/experiments/{experiment_id}", response_model=ABExperimentResponse)
async def get_experiment(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Get experiment details."""
    from sqlalchemy import select
    from db.models import ABExperiment

    result = await db.execute(
        select(ABExperiment).where(ABExperiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    return ABExperimentResponse(
        id=experiment.id,
        name=experiment.name,
        candidate_model_version=experiment.candidate_model_version,
        production_model_version=experiment.production_model_version,
        status=experiment.status,
        candidate_weight=float(experiment.candidate_weight),
        shadow_mode=experiment.shadow_mode,
        kill_switch_triggered=experiment.kill_switch_triggered,
        metrics_snapshot=experiment.metrics_snapshot,
        created_at=experiment.created_at.isoformat() if experiment.created_at else "",
        completed_at=experiment.completed_at.isoformat()
        if experiment.completed_at
        else None,
    )


@router.get("/experiments/{experiment_id}/stats", response_model=ABExperimentStats)
async def get_experiment_stats(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Get experiment statistics."""
    service = ABExperimentService(db)
    stats = await service.get_experiment_stats(experiment_id)
    return ABExperimentStats(**stats)


@router.patch("/experiments/{experiment_id}/promote")
async def promote_experiment(
    experiment_id: uuid.UUID,
    data: ABExperimentPromote,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Promote experiment to next stage."""
    service = ABExperimentService(db)
    success = await service.promote_experiment(
        experiment_id, data.new_status, data.new_weight
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    await db.commit()
    return {"status": "ok", "new_status": data.new_status}


@router.post("/experiments/{experiment_id}/rollback")
async def rollback_experiment(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Emergency rollback of an experiment."""
    service = ABExperimentService(db)
    success = await service.promote_experiment(experiment_id, "rolled_back", 0.0)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )
    await db.commit()
    return {"status": "rolled_back"}


# ===== Training Jobs Endpoints =====


@router.post("/training/trigger", response_model=TrainingJobResponse)
async def trigger_training(
    data: TrainingJobCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Manually trigger a training job."""
    service = RLHFTrainingService(db)

    # Check for existing active jobs
    active_jobs = await service.get_active_jobs()
    if active_jobs:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training job already in progress",
        )

    job = await service.create_job(data.job_type)
    await db.commit()

    return TrainingJobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        samples_used=job.samples_used,
        metrics_before=job.metrics_before,
        metrics_after=job.metrics_after,
        model_version_created=job.model_version_created,
        error_message=job.error_message,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        created_at=job.created_at.isoformat() if job.created_at else "",
    )


@router.get("/training/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    limit: int = Query(20, ge=1, le=100),
    job_type: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """List training jobs."""
    from sqlalchemy import select
    from db.models import RLHFTrainingJob

    query = select(RLHFTrainingJob).order_by(RLHFTrainingJob.created_at.desc())
    if job_type:
        query = query.where(RLHFTrainingJob.job_type == job_type)
    result = await db.execute(query.limit(limit))
    jobs = list(result.scalars().all())

    return [
        TrainingJobResponse(
            id=j.id,
            job_type=j.job_type,
            status=j.status,
            samples_used=j.samples_used,
            metrics_before=j.metrics_before,
            metrics_after=j.metrics_after,
            model_version_created=j.model_version_created,
            error_message=j.error_message,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            created_at=j.created_at.isoformat() if j.created_at else "",
        )
        for j in jobs
    ]


@router.get("/training/status", response_model=TrainingJobResponse | None)
async def get_training_status(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Get current/latest training job status."""
    service = RLHFTrainingService(db)
    job = await service.get_latest_job()
    if not job:
        return None

    return TrainingJobResponse(
        id=job.id,
        job_type=job.job_type,
        status=job.status,
        samples_used=job.samples_used,
        metrics_before=job.metrics_before,
        metrics_after=job.metrics_after,
        model_version_created=job.model_version_created,
        error_message=job.error_message,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        created_at=job.created_at.isoformat() if job.created_at else "",
    )


# ===== Overview Endpoint =====


@router.get("/metrics/overview", response_model=RLHFMetricsOverview)
async def get_rlhf_overview(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Get RLHF system health overview."""
    from sqlalchemy import func, select
    from db.models import PreferencePair, ABExperiment, RLHFTrainingJob

    pair_service = PreferencePairService(db)
    experiment_service = ABExperimentService(db)
    training_service = RLHFTrainingService(db)

    # Total pairs
    total_result = await db.execute(select(func.count(PreferencePair.id)))
    total_pairs = total_result.scalar() or 0

    # Pairs by source
    pairs_by_source = await pair_service.get_pair_count_by_source()

    # Unvalidated pairs
    unvalidated_result = await db.execute(
        select(func.count(PreferencePair.id)).where(PreferencePair.validated.is_(False))
    )
    unvalidated_pairs = unvalidated_result.scalar() or 0

    # Pending training pairs
    pending_result = await db.execute(
        select(func.count(PreferencePair.id)).where(
            PreferencePair.used_in_training.is_(None)
        )
    )
    pending_training_pairs = pending_result.scalar() or 0

    # Active experiment
    active_experiment = await experiment_service.get_active_experiment()
    active_experiment_response = None
    if active_experiment:
        active_experiment_response = ABExperimentResponse(
            id=active_experiment.id,
            name=active_experiment.name,
            candidate_model_version=active_experiment.candidate_model_version,
            production_model_version=active_experiment.production_model_version,
            status=active_experiment.status,
            candidate_weight=float(active_experiment.candidate_weight),
            shadow_mode=active_experiment.shadow_mode,
            kill_switch_triggered=active_experiment.kill_switch_triggered,
            metrics_snapshot=active_experiment.metrics_snapshot,
            created_at=active_experiment.created_at.isoformat()
            if active_experiment.created_at
            else "",
            completed_at=active_experiment.completed_at.isoformat()
            if active_experiment.completed_at
            else None,
        )

    # Active jobs
    active_jobs = await training_service.get_active_jobs()
    active_jobs_response = [
        TrainingJobResponse(
            id=j.id,
            job_type=j.job_type,
            status=j.status,
            samples_used=j.samples_used,
            metrics_before=j.metrics_before,
            metrics_after=j.metrics_after,
            model_version_created=j.model_version_created,
            error_message=j.error_message,
            started_at=j.started_at.isoformat() if j.started_at else None,
            completed_at=j.completed_at.isoformat() if j.completed_at else None,
            created_at=j.created_at.isoformat() if j.created_at else "",
        )
        for j in active_jobs
    ]

    return RLHFMetricsOverview(
        total_pairs=total_pairs,
        pairs_by_source=pairs_by_source,
        unvalidated_pairs=unvalidated_pairs,
        pending_training_pairs=pending_training_pairs,
        active_experiment=active_experiment_response,
        active_jobs=active_jobs_response,
    )


# ===== A/B Router Control Endpoints =====


@router.get("/router/status", response_model=ABRouterStatus)
async def get_router_status(
    _: User = Depends(get_current_user),
):
    """Get current A/B router runtime status."""
    manager = get_ab_router_manager()
    status = manager.get_experiment_status()

    return ABRouterStatus(
        status=status.get("status", "no_active_experiment"),
        experiment_id=status.get("experiment_id"),
        production_version=status.get("production_version"),
        candidate_version=status.get("candidate_version"),
        candidate_weight=status.get("candidate_weight", 0.0),
        shadow_mode=status.get("shadow_mode", False),
        shadow_stats=status.get("shadow_stats"),
    )


@router.post("/router/evaluate-kill-switch", response_model=KillSwitchResponse)
async def evaluate_kill_switch(
    data: KillSwitchEvaluation,
    auto_rollback: bool = Query(
        True, description="Automatically rollback if kill triggered"
    ),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Evaluate kill switch conditions against provided metrics.

    If auto_rollback=True and kill switch triggers, the experiment will be
    automatically rolled back in both the database and the runtime router.
    """
    # Convert Pydantic model to ExperimentMetrics
    metrics = ExperimentMetrics(
        production_mape=data.production_mape,
        candidate_mape=data.candidate_mape,
        production_confidence=data.production_confidence,
        candidate_confidence=data.candidate_confidence,
        production_edit_rate=data.production_edit_rate,
        candidate_edit_rate=data.candidate_edit_rate,
        production_samples=data.production_samples,
        candidate_samples=data.candidate_samples,
    )

    result = check_kill_switch(metrics)
    auto_rolled_back = False

    if result.should_kill and auto_rollback:
        # Rollback in runtime router
        manager = get_ab_router_manager()
        if manager.get_router():
            manager.evaluate_kill_switch(metrics)  # This auto-rolls back
            auto_rolled_back = True

        # Also update database
        experiment_service = ABExperimentService(db)
        active_experiment = await experiment_service.get_active_experiment()
        if active_experiment:
            await experiment_service.promote_experiment(
                active_experiment.id, "rolled_back", 0.0
            )
            # Update kill_switch_triggered flag
            active_experiment.kill_switch_triggered = True
            await db.commit()

    return KillSwitchResponse(
        should_kill=result.should_kill,
        reason=result.reason,
        threshold_breached=result.threshold_breached,
        auto_rolled_back=auto_rolled_back,
    )


@router.post("/router/sync-from-db")
async def sync_router_from_db(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Synchronize runtime A/B router state from database.

    Use this after a server restart or when the router state
    needs to be reconciled with the database.
    """
    experiment_service = ABExperimentService(db)
    active_experiment = await experiment_service.get_active_experiment()

    manager = get_ab_router_manager()

    if not active_experiment:
        # No active experiment - router should have no candidate
        router = manager.get_router()
        if router and router.candidate:
            router.rollback()
        return {"status": "synced", "message": "No active experiment - router cleared"}

    # For now, just return status - actual model loading would require
    # loading models from disk which depends on the model storage implementation
    return {
        "status": "info",
        "message": "Active experiment found. Model loading requires manual initialization.",
        "experiment": {
            "id": str(active_experiment.id),
            "name": active_experiment.name,
            "status": active_experiment.status,
            "candidate_weight": float(active_experiment.candidate_weight),
            "shadow_mode": active_experiment.shadow_mode,
        },
    }


@router.get("/metrics/drift")
async def get_drift_metrics(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Get model drift indicators from recent predictions.

    Compares production vs candidate performance when shadow mode is active.
    """
    manager = get_ab_router_manager()
    router = manager.get_router()

    if not router:
        return {"status": "no_router", "message": "A/B router not initialized"}

    if not router.shadow_mode:
        return {"status": "not_shadow", "message": "Shadow mode not active"}

    shadow_stats = router.get_shadow_stats()

    # Calculate drift indicators
    drift_indicators = {
        "shadow_predictions_count": shadow_stats.get("count", 0),
        "avg_delta_pct": shadow_stats.get("avg_delta_pct", 0),
        "max_delta_pct": shadow_stats.get("max_delta_pct", 0),
        "candidate_higher_count": shadow_stats.get("candidate_higher_count", 0),
        "candidate_lower_count": shadow_stats.get("candidate_lower_count", 0),
        "bias_direction": (
            "candidate_higher"
            if shadow_stats.get("candidate_higher_count", 0)
            > shadow_stats.get("candidate_lower_count", 0)
            else "candidate_lower"
            if shadow_stats.get("candidate_lower_count", 0)
            > shadow_stats.get("candidate_higher_count", 0)
            else "balanced"
        ),
    }

    # Recommendations based on drift
    recommendations = []
    avg_delta = abs(shadow_stats.get("avg_delta_pct", 0))

    if avg_delta < 0.05:
        recommendations.append("Models are well-aligned. Safe to proceed to canary.")
    elif avg_delta < 0.15:
        recommendations.append("Moderate drift detected. Continue shadow testing.")
    else:
        recommendations.append("High drift detected. Investigate before proceeding.")

    if shadow_stats.get("count", 0) < 10:
        recommendations.append("Insufficient samples. Continue shadow testing.")

    return {
        "status": "ok",
        "drift_indicators": drift_indicators,
        "recommendations": recommendations,
    }


# ===== DPO Dataset Export Endpoints =====


@router.get("/dpo/stats", response_model=DPOExportStats)
async def get_dpo_export_stats(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Get statistics about exportable preference pairs."""
    generator = DPODatasetGenerator(db=db)
    stats = await generator.get_export_stats()

    return DPOExportStats(**stats)


@router.post("/dpo/export", response_model=DPOExportResponse)
async def export_dpo_dataset(
    data: DPOExportRequest,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Export preference pairs to JSONL format for fine-tuning.

    Supports multiple providers:
    - openai: Messages format with chosen/rejected
    - fireworks: Simple prompt/chosen/rejected
    - together: Instruction/Response format
    - anyscale: OpenAI-compatible format

    If generate_synthetic=True and LLM is available, will generate
    synthetic rejected reasoning for pairs that only have chosen.
    """
    # Initialize LLM client for synthetic generation if requested
    llm_client = None
    if data.generate_synthetic:
        try:
            llm_client = LLMClient()
        except Exception as e:
            # Continue without synthetic generation
            pass

    generator = DPODatasetGenerator(db=db, llm_client=llm_client)

    result = await generator.export_dataset(
        provider=FineTuningProvider(data.provider),
        validated_only=data.validated_only,
        min_confidence=data.min_confidence,
        generate_synthetic=data.generate_synthetic,
        limit=data.limit,
        mark_as_used=data.mark_as_used,
    )

    return DPOExportResponse(
        file_path=result.file_path,
        total_examples=result.total_examples,
        examples_with_synthetic=result.examples_with_synthetic,
        provider_format=result.provider_format,
        export_timestamp=result.export_timestamp,
        validation_passed=result.validation_passed,
        validation_errors=result.validation_errors,
    )


@router.post("/dpo/generate-upload-config")
async def generate_upload_config(
    data: UploadConfigRequest,
    _: User = Depends(get_current_user),
):
    """
    Generate fine-tuning upload configuration for a provider.

    Returns the configuration payload that can be used with the
    provider's fine-tuning API.
    """
    if data.provider == "openai":
        config = FineTuningUploadPipeline.prepare_openai_upload(
            jsonl_path=data.jsonl_path,
            model_suffix=data.model_suffix,
            n_epochs=data.n_epochs,
        )
    elif data.provider == "fireworks":
        config = FineTuningUploadPipeline.prepare_fireworks_upload(
            jsonl_path=data.jsonl_path,
            job_name=data.model_suffix,
        )
    elif data.provider == "together":
        config = FineTuningUploadPipeline.prepare_together_upload(
            jsonl_path=data.jsonl_path,
            job_name=data.model_suffix,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported provider: {data.provider}",
        )

    return {
        "provider": data.provider,
        "config": config,
        "notes": [
            "Use this config with the provider's fine-tuning API",
            "Ensure the JSONL file is accessible to the provider",
            "For OpenAI: Upload file first, then create fine-tuning job",
            "For Fireworks/Together: Config includes all needed fields",
        ],
    }


@router.post("/dpo/validate-dataset", response_model=DatasetValidationResponse)
async def validate_dpo_dataset(
    file_path: str = Query(..., description="Path to JSONL file to validate"),
    _: User = Depends(get_current_user),
):
    """
    Validate a JSONL dataset file before upload.

    Checks for:
    - Valid JSON on each line
    - Required fields (prompt, chosen, rejected)
    - Structural consistency
    """
    result = FineTuningUploadPipeline.validate_dataset_for_upload(file_path)

    return DatasetValidationResponse(
        valid=result["valid"],
        examples_count=result["examples_count"],
        issues=result["issues"],
        file_path=result.get("file_path", file_path),
    )


# ===== Monitoring Endpoints =====


@router.get("/monitoring/dashboard", response_model=DashboardMetricsResponse)
async def get_dashboard_metrics(
    days: int = Query(7, ge=1, le=30, description="Number of days for time series"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Get aggregated dashboard metrics for RLHF monitoring.

    Returns:
    - Summary statistics (predictions, corrections, pairs)
    - Performance metrics (MAPE, confidence, edit rate)
    - Time series data (last N days)
    - Health indicators (model health, pipeline health)
    - Active alerts
    """
    service = RLHFMonitoringService(db)
    metrics = await service.get_dashboard_metrics(days=days)

    return DashboardMetricsResponse(
        total_predictions=metrics.total_predictions,
        total_corrections=metrics.total_corrections,
        total_preference_pairs=metrics.total_preference_pairs,
        active_experiment=metrics.active_experiment,
        current_mape=metrics.current_mape,
        avg_confidence=metrics.avg_confidence,
        user_edit_rate=metrics.user_edit_rate,
        predictions_by_day=[
            TimeSeriesPointResponse(
                timestamp=p.timestamp.isoformat(),
                value=p.value,
                labels=p.labels,
            )
            for p in metrics.predictions_by_day
        ],
        corrections_by_day=[
            TimeSeriesPointResponse(
                timestamp=p.timestamp.isoformat(),
                value=p.value,
                labels=p.labels,
            )
            for p in metrics.corrections_by_day
        ],
        mape_trend=[
            TimeSeriesPointResponse(
                timestamp=p.timestamp.isoformat(),
                value=p.value,
                labels=p.labels,
            )
            for p in metrics.mape_trend
        ],
        model_health=metrics.model_health,
        data_pipeline_health=metrics.data_pipeline_health,
        alerts=[
            AlertResponse(
                alert_type=a.alert_type.value,
                severity=a.severity.value,
                message=a.message,
                details=a.details,
                timestamp=a.timestamp.isoformat(),
                acknowledged=a.acknowledged,
            )
            for a in metrics.alerts
        ],
    )


@router.get("/monitoring/alerts", response_model=list[AlertResponse])
async def get_active_alerts(
    severity: str | None = Query(
        None, pattern="^(info|warning|critical)$", description="Filter by severity"
    ),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Get current active alerts for the RLHF system.

    Optionally filter by severity level.
    """
    service = RLHFMonitoringService(db)
    metrics = await service.get_dashboard_metrics(days=7)

    alerts = metrics.alerts
    if severity:
        alerts = [a for a in alerts if a.severity.value == severity]

    return [
        AlertResponse(
            alert_type=a.alert_type.value,
            severity=a.severity.value,
            message=a.message,
            details=a.details,
            timestamp=a.timestamp.isoformat(),
            acknowledged=a.acknowledged,
        )
        for a in alerts
    ]


@router.get(
    "/monitoring/experiment-comparison", response_model=ExperimentComparisonResponse
)
async def get_experiment_comparison(
    experiment_id: str | None = Query(None, description="Specific experiment ID"),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Get detailed comparison between production and candidate models.

    If no experiment_id is provided, uses the active experiment.
    """
    service = RLHFMonitoringService(db)
    comparison = await service.get_experiment_comparison(experiment_id)

    return ExperimentComparisonResponse(**comparison)


@router.get("/monitoring/health")
async def get_system_health(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """
    Quick health check for the RLHF system.

    Returns simplified health status suitable for monitoring dashboards.
    """
    service = RLHFMonitoringService(db)
    metrics = await service.get_dashboard_metrics(days=3)

    # Count alerts by severity
    critical_count = len(
        [a for a in metrics.alerts if a.severity == AlertSeverity.CRITICAL]
    )
    warning_count = len(
        [a for a in metrics.alerts if a.severity == AlertSeverity.WARNING]
    )

    return {
        "status": metrics.model_health,
        "model_health": metrics.model_health,
        "pipeline_health": metrics.data_pipeline_health,
        "alerts": {
            "critical": critical_count,
            "warning": warning_count,
            "info": len(metrics.alerts) - critical_count - warning_count,
        },
        "metrics": {
            "current_mape": metrics.current_mape,
            "avg_confidence": metrics.avg_confidence,
            "user_edit_rate": metrics.user_edit_rate,
        },
        "active_experiment": metrics.active_experiment,
    }


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics(
    db: AsyncSession = Depends(get_db),
):
    """
    Export metrics in Prometheus format for scraping.

    This endpoint is typically called by Prometheus server at regular intervals.
    Returns plain text in Prometheus exposition format.

    Note: No authentication required for Prometheus scraping.
    Configure network security to restrict access.
    """
    service = RLHFMonitoringService(db)
    metrics = await service.get_dashboard_metrics(days=7)

    prometheus_output = export_prometheus_metrics(metrics)

    return PlainTextResponse(
        content=prometheus_output,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

"""
FPT Cost Brain 2.0 - RLHF Monitoring Service
Production monitoring, alerting, and dashboard metrics aggregation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    ABExperiment,
    ABPrediction,
    FeedbackCorrection,
    PreferencePair,
    RLHFTrainingJob,
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of RLHF alerts."""

    MODEL_DRIFT = "model_drift"
    HIGH_EDIT_RATE = "high_edit_rate"
    LOW_CONFIDENCE = "low_confidence"
    TRAINING_FAILURE = "training_failure"
    EXPERIMENT_KILLED = "experiment_killed"
    DATA_QUALITY = "data_quality"
    RETRAIN_NEEDED = "retrain_needed"


@dataclass
class Alert:
    """A monitoring alert."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False


@dataclass
class TimeSeriesPoint:
    """A single time-series data point."""

    timestamp: datetime
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class DashboardMetrics:
    """Aggregated metrics for dashboard display."""

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
    predictions_by_day: list[TimeSeriesPoint]
    corrections_by_day: list[TimeSeriesPoint]
    mape_trend: list[TimeSeriesPoint]

    # Health indicators
    model_health: str  # "healthy", "degraded", "critical"
    data_pipeline_health: str
    alerts: list[Alert]


# Alert thresholds
ALERT_THRESHOLDS = {
    "mape_warning": 0.25,  # 25% MAPE triggers warning
    "mape_critical": 0.35,  # 35% MAPE triggers critical
    "edit_rate_warning": 0.15,  # 15% edit rate triggers warning
    "edit_rate_critical": 0.25,  # 25% edit rate triggers critical
    "confidence_warning": 0.65,  # Below 65% confidence triggers warning
    "confidence_critical": 0.55,  # Below 55% triggers critical
    "days_since_retrain_warning": 14,
    "days_since_retrain_critical": 30,
    "min_daily_predictions": 1,  # Alert if no predictions in a day
}


class RLHFMonitoringService:
    """
    Production monitoring for the RLHF system.

    Provides:
    - Metrics aggregation for dashboards
    - Alert generation based on thresholds
    - Time-series data for trend analysis
    - Health status indicators
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_dashboard_metrics(
        self,
        days: int = 7,
    ) -> DashboardMetrics:
        """
        Get aggregated metrics for dashboard display.

        Args:
            days: Number of days for time-series data
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Get summary stats
        total_predictions = await self._count_predictions(cutoff)
        total_corrections = await self._count_corrections(cutoff)
        total_pairs = await self._count_preference_pairs()
        active_exp = await self._get_active_experiment_name()

        # Get performance metrics
        current_mape = await self._calculate_current_mape(days=3)
        avg_confidence = await self._calculate_avg_confidence(days=3)
        edit_rate = await self._calculate_edit_rate(days=7)

        # Get time series
        predictions_ts = await self._get_predictions_time_series(days)
        corrections_ts = await self._get_corrections_time_series(days)
        mape_ts = await self._get_mape_time_series(days)

        # Generate alerts
        alerts = await self._generate_alerts(current_mape, avg_confidence, edit_rate)

        # Determine health status
        model_health = self._determine_model_health(
            current_mape, avg_confidence, edit_rate
        )
        pipeline_health = await self._determine_pipeline_health()

        return DashboardMetrics(
            total_predictions=total_predictions,
            total_corrections=total_corrections,
            total_preference_pairs=total_pairs,
            active_experiment=active_exp,
            current_mape=current_mape,
            avg_confidence=avg_confidence,
            user_edit_rate=edit_rate,
            predictions_by_day=predictions_ts,
            corrections_by_day=corrections_ts,
            mape_trend=mape_ts,
            model_health=model_health,
            data_pipeline_health=pipeline_health,
            alerts=alerts,
        )

    async def _count_predictions(self, since: datetime) -> int:
        """Count predictions since a given date."""
        result = await self.db.execute(
            select(func.count(ABPrediction.id)).where(ABPrediction.created_at >= since)
        )
        return result.scalar() or 0

    async def _count_corrections(self, since: datetime) -> int:
        """Count user corrections since a given date."""
        result = await self.db.execute(
            select(func.count(FeedbackCorrection.id)).where(
                FeedbackCorrection.created_at >= since
            )
        )
        return result.scalar() or 0

    async def _count_preference_pairs(self) -> int:
        """Count total preference pairs."""
        result = await self.db.execute(select(func.count(PreferencePair.id)))
        return result.scalar() or 0

    async def _get_active_experiment_name(self) -> str | None:
        """Get name of active experiment."""
        result = await self.db.execute(
            select(ABExperiment.name).where(
                ABExperiment.status.in_(["shadow", "canary", "gradual"])
            )
        )
        row = result.first()
        return row[0] if row else None

    async def _calculate_current_mape(self, days: int = 3) -> float | None:
        """Calculate MAPE from recent predictions with actual outcomes."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        result = await self.db.execute(
            select(
                ABPrediction.prediction,
                ABPrediction.actual_outcome,
            ).where(
                and_(
                    ABPrediction.created_at >= cutoff,
                    ABPrediction.actual_outcome.isnot(None),
                )
            )
        )
        rows = result.all()

        if not rows:
            return None

        errors = []
        for pred_json, actual in rows:
            if actual and actual > 0:
                pred_value = pred_json.get("point_estimate", 0) if pred_json else 0
                if pred_value > 0:
                    errors.append(abs(pred_value - actual) / actual)

        return sum(errors) / len(errors) if errors else None

    async def _calculate_avg_confidence(self, days: int = 3) -> float | None:
        """Calculate average prediction confidence."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        result = await self.db.execute(
            select(ABPrediction.prediction).where(ABPrediction.created_at >= cutoff)
        )
        rows = result.all()

        if not rows:
            return None

        confidences = []
        for (pred_json,) in rows:
            if pred_json and "calibrated_confidence" in pred_json:
                confidences.append(pred_json["calibrated_confidence"])

        return sum(confidences) / len(confidences) if confidences else None

    async def _calculate_edit_rate(self, days: int = 7) -> float:
        """Calculate user edit rate (corrections / predictions)."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        predictions = await self._count_predictions(cutoff)
        corrections = await self._count_corrections(cutoff)

        if predictions == 0:
            return 0.0

        return corrections / predictions

    async def _get_predictions_time_series(self, days: int) -> list[TimeSeriesPoint]:
        """Get daily prediction counts."""
        points = []
        for i in range(days, -1, -1):
            day_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)

            result = await self.db.execute(
                select(func.count(ABPrediction.id)).where(
                    and_(
                        ABPrediction.created_at >= day_start,
                        ABPrediction.created_at < day_end,
                    )
                )
            )
            count = result.scalar() or 0

            points.append(
                TimeSeriesPoint(
                    timestamp=day_start,
                    value=float(count),
                    labels={"metric": "predictions"},
                )
            )

        return points

    async def _get_corrections_time_series(self, days: int) -> list[TimeSeriesPoint]:
        """Get daily correction counts."""
        points = []
        for i in range(days, -1, -1):
            day_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)

            result = await self.db.execute(
                select(func.count(FeedbackCorrection.id)).where(
                    and_(
                        FeedbackCorrection.created_at >= day_start,
                        FeedbackCorrection.created_at < day_end,
                    )
                )
            )
            count = result.scalar() or 0

            points.append(
                TimeSeriesPoint(
                    timestamp=day_start,
                    value=float(count),
                    labels={"metric": "corrections"},
                )
            )

        return points

    async def _get_mape_time_series(self, days: int) -> list[TimeSeriesPoint]:
        """Get daily MAPE trend (requires actual outcomes)."""
        # Simplified: return empty for now as MAPE calculation per day
        # requires actual outcomes which may not be available daily
        return []

    async def _generate_alerts(
        self,
        current_mape: float | None,
        avg_confidence: float | None,
        edit_rate: float,
    ) -> list[Alert]:
        """Generate alerts based on current metrics."""
        alerts = []

        # MAPE alerts
        if current_mape is not None:
            if current_mape > ALERT_THRESHOLDS["mape_critical"]:
                alerts.append(
                    Alert(
                        alert_type=AlertType.MODEL_DRIFT,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Model MAPE at {current_mape:.1%} exceeds critical threshold",
                        details={"current_mape": current_mape, "threshold": 0.35},
                    )
                )
            elif current_mape > ALERT_THRESHOLDS["mape_warning"]:
                alerts.append(
                    Alert(
                        alert_type=AlertType.MODEL_DRIFT,
                        severity=AlertSeverity.WARNING,
                        message=f"Model MAPE at {current_mape:.1%} exceeds warning threshold",
                        details={"current_mape": current_mape, "threshold": 0.25},
                    )
                )

        # Confidence alerts
        if avg_confidence is not None:
            if avg_confidence < ALERT_THRESHOLDS["confidence_critical"]:
                alerts.append(
                    Alert(
                        alert_type=AlertType.LOW_CONFIDENCE,
                        severity=AlertSeverity.CRITICAL,
                        message=f"Model confidence at {avg_confidence:.1%} below critical threshold",
                        details={"avg_confidence": avg_confidence, "threshold": 0.55},
                    )
                )
            elif avg_confidence < ALERT_THRESHOLDS["confidence_warning"]:
                alerts.append(
                    Alert(
                        alert_type=AlertType.LOW_CONFIDENCE,
                        severity=AlertSeverity.WARNING,
                        message=f"Model confidence at {avg_confidence:.1%} below warning threshold",
                        details={"avg_confidence": avg_confidence, "threshold": 0.65},
                    )
                )

        # Edit rate alerts
        if edit_rate > ALERT_THRESHOLDS["edit_rate_critical"]:
            alerts.append(
                Alert(
                    alert_type=AlertType.HIGH_EDIT_RATE,
                    severity=AlertSeverity.CRITICAL,
                    message=f"User edit rate at {edit_rate:.1%} indicates model issues",
                    details={"edit_rate": edit_rate, "threshold": 0.25},
                )
            )
        elif edit_rate > ALERT_THRESHOLDS["edit_rate_warning"]:
            alerts.append(
                Alert(
                    alert_type=AlertType.HIGH_EDIT_RATE,
                    severity=AlertSeverity.WARNING,
                    message=f"User edit rate at {edit_rate:.1%} elevated",
                    details={"edit_rate": edit_rate, "threshold": 0.15},
                )
            )

        # Check for recent training failures
        training_alerts = await self._check_training_health()
        alerts.extend(training_alerts)

        # Check for killed experiments
        experiment_alerts = await self._check_experiment_health()
        alerts.extend(experiment_alerts)

        return alerts

    async def _check_training_health(self) -> list[Alert]:
        """Check for training job failures."""
        alerts = []

        # Check for recent failures
        result = await self.db.execute(
            select(RLHFTrainingJob).where(
                and_(
                    RLHFTrainingJob.status == "failed",
                    RLHFTrainingJob.created_at
                    >= datetime.now(timezone.utc) - timedelta(days=7),
                )
            )
        )
        failed_jobs = result.scalars().all()

        for job in failed_jobs:
            alerts.append(
                Alert(
                    alert_type=AlertType.TRAINING_FAILURE,
                    severity=AlertSeverity.WARNING,
                    message=f"Training job {job.id} failed: {job.error_message or 'Unknown error'}",
                    details={"job_id": str(job.id), "job_type": job.job_type},
                    timestamp=job.created_at,
                )
            )

        # Check time since last successful retrain
        result = await self.db.execute(
            select(RLHFTrainingJob)
            .where(RLHFTrainingJob.status == "completed")
            .order_by(RLHFTrainingJob.completed_at.desc())
            .limit(1)
        )
        last_job = result.scalar_one_or_none()

        if last_job and last_job.completed_at:
            days_since = (datetime.now(timezone.utc) - last_job.completed_at).days
            if days_since > ALERT_THRESHOLDS["days_since_retrain_critical"]:
                alerts.append(
                    Alert(
                        alert_type=AlertType.RETRAIN_NEEDED,
                        severity=AlertSeverity.CRITICAL,
                        message=f"No model retrain in {days_since} days",
                        details={"days_since_retrain": days_since},
                    )
                )
            elif days_since > ALERT_THRESHOLDS["days_since_retrain_warning"]:
                alerts.append(
                    Alert(
                        alert_type=AlertType.RETRAIN_NEEDED,
                        severity=AlertSeverity.WARNING,
                        message=f"Model retrain recommended ({days_since} days since last)",
                        details={"days_since_retrain": days_since},
                    )
                )

        return alerts

    async def _check_experiment_health(self) -> list[Alert]:
        """Check for killed experiments."""
        alerts = []

        # Check for recently killed experiments
        result = await self.db.execute(
            select(ABExperiment).where(
                and_(
                    ABExperiment.kill_switch_triggered.is_(True),
                    ABExperiment.completed_at
                    >= datetime.now(timezone.utc) - timedelta(days=7),
                )
            )
        )
        killed_experiments = result.scalars().all()

        for exp in killed_experiments:
            alerts.append(
                Alert(
                    alert_type=AlertType.EXPERIMENT_KILLED,
                    severity=AlertSeverity.WARNING,
                    message=f"Experiment '{exp.name}' was killed by kill switch",
                    details={
                        "experiment_id": str(exp.id),
                        "candidate_version": exp.candidate_model_version,
                    },
                    timestamp=exp.completed_at,
                )
            )

        return alerts

    def _determine_model_health(
        self,
        mape: float | None,
        confidence: float | None,
        edit_rate: float,
    ) -> str:
        """Determine overall model health status."""
        critical_conditions = [
            mape is not None and mape > ALERT_THRESHOLDS["mape_critical"],
            confidence is not None
            and confidence < ALERT_THRESHOLDS["confidence_critical"],
            edit_rate > ALERT_THRESHOLDS["edit_rate_critical"],
        ]

        warning_conditions = [
            mape is not None and mape > ALERT_THRESHOLDS["mape_warning"],
            confidence is not None
            and confidence < ALERT_THRESHOLDS["confidence_warning"],
            edit_rate > ALERT_THRESHOLDS["edit_rate_warning"],
        ]

        if any(critical_conditions):
            return "critical"
        elif any(warning_conditions):
            return "degraded"
        else:
            return "healthy"

    async def _determine_pipeline_health(self) -> str:
        """Determine data pipeline health."""
        # Check for recent preference pair creation
        result = await self.db.execute(
            select(func.count(PreferencePair.id)).where(
                PreferencePair.created_at
                >= datetime.now(timezone.utc) - timedelta(days=7)
            )
        )
        recent_pairs = result.scalar() or 0

        if recent_pairs == 0:
            return "degraded"

        # Check for pending jobs stuck
        result = await self.db.execute(
            select(func.count(RLHFTrainingJob.id)).where(
                and_(
                    RLHFTrainingJob.status == "running",
                    RLHFTrainingJob.started_at
                    <= datetime.now(timezone.utc) - timedelta(hours=2),
                )
            )
        )
        stuck_jobs = result.scalar() or 0

        if stuck_jobs > 0:
            return "degraded"

        return "healthy"

    async def get_experiment_comparison(
        self,
        experiment_id: str | None = None,
    ) -> dict:
        """Get detailed experiment comparison metrics."""
        query = select(ABExperiment)
        if experiment_id:
            query = query.where(ABExperiment.id == experiment_id)
        else:
            query = query.where(
                ABExperiment.status.in_(["shadow", "canary", "gradual"])
            )

        result = await self.db.execute(query.limit(1))
        experiment = result.scalar_one_or_none()

        if not experiment:
            return {"status": "no_experiment"}

        # Get predictions by model
        prod_result = await self.db.execute(
            select(func.count(ABPrediction.id)).where(
                and_(
                    ABPrediction.experiment_id == experiment.id,
                    ABPrediction.model_used == "production",
                )
            )
        )
        prod_count = prod_result.scalar() or 0

        cand_result = await self.db.execute(
            select(func.count(ABPrediction.id)).where(
                and_(
                    ABPrediction.experiment_id == experiment.id,
                    ABPrediction.model_used == "candidate",
                )
            )
        )
        cand_count = cand_result.scalar() or 0

        # Get edit rates
        prod_edits = await self.db.execute(
            select(func.count(ABPrediction.id)).where(
                and_(
                    ABPrediction.experiment_id == experiment.id,
                    ABPrediction.model_used == "production",
                    ABPrediction.user_edited.is_(True),
                )
            )
        )
        prod_edit_count = prod_edits.scalar() or 0

        cand_edits = await self.db.execute(
            select(func.count(ABPrediction.id)).where(
                and_(
                    ABPrediction.experiment_id == experiment.id,
                    ABPrediction.model_used == "candidate",
                    ABPrediction.user_edited.is_(True),
                )
            )
        )
        cand_edit_count = cand_edits.scalar() or 0

        return {
            "experiment_id": str(experiment.id),
            "experiment_name": experiment.name,
            "status": experiment.status,
            "production": {
                "model_version": experiment.production_model_version,
                "predictions": prod_count,
                "edits": prod_edit_count,
                "edit_rate": prod_edit_count / prod_count if prod_count > 0 else 0,
            },
            "candidate": {
                "model_version": experiment.candidate_model_version,
                "predictions": cand_count,
                "edits": cand_edit_count,
                "edit_rate": cand_edit_count / cand_count if cand_count > 0 else 0,
            },
            "candidate_weight": float(experiment.candidate_weight),
            "shadow_mode": experiment.shadow_mode,
        }


# Prometheus-style metrics export
def export_prometheus_metrics(metrics: DashboardMetrics) -> str:
    """Export metrics in Prometheus format for scraping."""
    lines = [
        "# HELP rlhf_predictions_total Total number of predictions",
        "# TYPE rlhf_predictions_total counter",
        f"rlhf_predictions_total {metrics.total_predictions}",
        "",
        "# HELP rlhf_corrections_total Total number of user corrections",
        "# TYPE rlhf_corrections_total counter",
        f"rlhf_corrections_total {metrics.total_corrections}",
        "",
        "# HELP rlhf_preference_pairs_total Total preference pairs",
        "# TYPE rlhf_preference_pairs_total gauge",
        f"rlhf_preference_pairs_total {metrics.total_preference_pairs}",
        "",
        "# HELP rlhf_user_edit_rate Current user edit rate",
        "# TYPE rlhf_user_edit_rate gauge",
        f"rlhf_user_edit_rate {metrics.user_edit_rate:.4f}",
        "",
    ]

    if metrics.current_mape is not None:
        lines.extend(
            [
                "# HELP rlhf_model_mape Current model MAPE",
                "# TYPE rlhf_model_mape gauge",
                f"rlhf_model_mape {metrics.current_mape:.4f}",
                "",
            ]
        )

    if metrics.avg_confidence is not None:
        lines.extend(
            [
                "# HELP rlhf_model_confidence Average model confidence",
                "# TYPE rlhf_model_confidence gauge",
                f"rlhf_model_confidence {metrics.avg_confidence:.4f}",
                "",
            ]
        )

    # Health status as numeric (0=healthy, 1=degraded, 2=critical)
    health_map = {"healthy": 0, "degraded": 1, "critical": 2}
    lines.extend(
        [
            "# HELP rlhf_model_health Model health status (0=healthy, 1=degraded, 2=critical)",
            "# TYPE rlhf_model_health gauge",
            f"rlhf_model_health {health_map.get(metrics.model_health, 2)}",
            "",
            "# HELP rlhf_pipeline_health Pipeline health status",
            "# TYPE rlhf_pipeline_health gauge",
            f"rlhf_pipeline_health {health_map.get(metrics.data_pipeline_health, 2)}",
            "",
            "# HELP rlhf_alerts_count Number of active alerts",
            "# TYPE rlhf_alerts_count gauge",
            f"rlhf_alerts_count {len(metrics.alerts)}",
        ]
    )

    return "\n".join(lines)

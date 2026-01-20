"""
FPT Cost Brain 2.0 - RLHF Service
Preference pair collection, reward calculation, and training coordination
"""

import hashlib
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    ABExperiment,
    ABPrediction,
    FeedbackCorrection,
    PreferencePair,
    Quotation,
    QuotationBreakdown,
    RLHFTrainingJob,
)


@dataclass
class RewardResult:
    """Result of reward calculation."""

    reward: float
    error_pct: float
    is_severe_error: bool


@dataclass
class PairGenerationResult:
    """Result of preference pair generation."""

    pair_id: uuid.UUID
    signal_source: str
    reward_delta: float
    confidence: float


def calculate_reward(prediction: float, target: float) -> RewardResult:
    """
    Calculate reward score [-1.0, 1.0]. Negative for severe errors.

    Scale:
      0% error  → +1.0 (perfect)
      25% error → +0.5 (good)
      50% error →  0.0 (neutral)
      75% error → -0.5 (bad)
      100% error → -1.0 (hallucination)
    """
    if target is None or target == 0:
        return RewardResult(reward=0.0, error_pct=0.0, is_severe_error=False)

    error_pct = abs(prediction - target) / target
    reward = 1.0 - (error_pct * 2)  # Linear: 50% error = 0.0
    reward = max(min(reward, 1.0), -1.0)

    return RewardResult(
        reward=reward,
        error_pct=error_pct,
        is_severe_error=error_pct > 0.5,
    )


def generate_synthetic_negative(
    approved_breakdown: dict[str, float],
) -> dict[str, float]:
    """
    Create a plausible-but-wrong breakdown for preference pair.
    User approved the original, so perturbed version is 'rejected'.
    """
    synthetic = approved_breakdown.copy()
    # Exclude bools since bool is a subclass of int in Python
    clusters = [
        k
        for k in synthetic.keys()
        if isinstance(synthetic[k], (int, float)) and not isinstance(synthetic[k], bool)
    ]

    if not clusters:
        return synthetic

    strategy = random.choice(["swap", "inflate", "deflate"])

    if strategy == "swap" and len(clusters) >= 2:
        # Swap two cluster hours
        c1, c2 = random.sample(clusters, 2)
        synthetic[c1], synthetic[c2] = synthetic[c2], synthetic[c1]

    elif strategy == "inflate":
        # Inflate total by 30-50%
        factor = random.uniform(1.3, 1.5)
        for k in clusters:
            synthetic[k] = int(float(synthetic[k]) * factor)

    else:  # deflate
        # Reduce by 25-40%
        factor = random.uniform(0.6, 0.75)
        for k in clusters:
            synthetic[k] = int(float(synthetic[k]) * factor)

    return synthetic


def compute_context_hash(context: dict[str, Any]) -> str:
    """Compute hash of context for deduplication."""
    import json

    serialized = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:64]


class PreferencePairService:
    """Service for collecting and managing DPO preference pairs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_from_user_edit(
        self,
        session_id: uuid.UUID,
        correction: FeedbackCorrection,
        original_breakdown: dict[str, Any],
        edited_breakdown: dict[str, Any],
        original_reasoning: str,
        edited_reasoning: str,
        context: dict[str, Any],
    ) -> PairGenerationResult:
        """
        Create preference pair from user correction.

        chosen = edited (what user wanted)
        rejected = original (what model produced)
        """
        # Calculate reward delta
        original_total = sum(
            v for v in original_breakdown.values() if isinstance(v, (int, float))
        )
        edited_total = sum(
            v for v in edited_breakdown.values() if isinstance(v, (int, float))
        )

        if original_total > 0:
            change_pct = abs(edited_total - original_total) / original_total
            # User edit implies original was wrong - negative reward for original
            reward_delta = min(change_pct * 2, 1.0)  # Cap at 1.0
        else:
            reward_delta = 0.5  # Default moderate delta

        pair = PreferencePair(
            session_id=session_id,
            context_hash=compute_context_hash(context),
            chosen_reasoning=edited_reasoning,
            rejected_reasoning=original_reasoning,
            chosen_breakdown=edited_breakdown,
            rejected_breakdown=original_breakdown,
            signal_source="user_edit",
            reward_delta=reward_delta,
            confidence=0.7,  # User edits have 0.7 confidence
            validated=False,
        )

        self.db.add(pair)
        await self.db.flush()

        return PairGenerationResult(
            pair_id=pair.id,
            signal_source="user_edit",
            reward_delta=reward_delta,
            confidence=0.7,
        )

    async def create_from_actual_outcome(
        self,
        session_id: uuid.UUID,
        predicted_breakdown: dict[str, Any],
        actual_cost: float,
        reasoning: str,
        context: dict[str, Any],
    ) -> PairGenerationResult:
        """
        Create preference pair from actual outcome data.

        Uses actual cost to determine if prediction was good or bad.
        """
        predicted_total = sum(
            v for v in predicted_breakdown.values() if isinstance(v, (int, float))
        )

        reward_result = calculate_reward(predicted_total, actual_cost)

        if reward_result.reward >= 0:
            # Prediction was good - create chosen=prediction, rejected=synthetic worse
            synthetic = generate_synthetic_negative(predicted_breakdown)
            chosen_breakdown = predicted_breakdown
            rejected_breakdown = synthetic
            chosen_reasoning = reasoning
            rejected_reasoning = "Standard estimation without domain context."
        else:
            # Prediction was bad - create chosen=adjusted, rejected=original
            # Adjust breakdown to match actual
            adjustment_factor = actual_cost / max(predicted_total, 1)
            adjusted = {
                k: int(float(v) * adjustment_factor)
                if isinstance(v, (int, float))
                else v
                for k, v in predicted_breakdown.items()
            }
            chosen_breakdown = adjusted
            rejected_breakdown = predicted_breakdown
            chosen_reasoning = f"Adjusted based on actual outcome: {actual_cost} K€"
            rejected_reasoning = reasoning

        pair = PreferencePair(
            session_id=session_id,
            context_hash=compute_context_hash(context),
            chosen_reasoning=chosen_reasoning,
            rejected_reasoning=rejected_reasoning,
            chosen_breakdown=chosen_breakdown,
            rejected_breakdown=rejected_breakdown,
            signal_source="actual_outcome",
            reward_delta=abs(reward_result.reward),
            confidence=1.0,  # Actual outcomes have 1.0 confidence
            validated=True,  # Auto-validated since it's ground truth
        )

        self.db.add(pair)
        await self.db.flush()

        return PairGenerationResult(
            pair_id=pair.id,
            signal_source="actual_outcome",
            reward_delta=abs(reward_result.reward),
            confidence=1.0,
        )

    async def create_from_explicit_approval(
        self,
        session_id: uuid.UUID,
        approved_breakdown: dict[str, Any],
        reasoning: str,
        context: dict[str, Any],
    ) -> PairGenerationResult:
        """
        Create preference pair from explicit approval (no edit).

        User approved without changes = implicit positive signal.
        Generate synthetic negative for training.
        """
        synthetic = generate_synthetic_negative(approved_breakdown)

        pair = PreferencePair(
            session_id=session_id,
            context_hash=compute_context_hash(context),
            chosen_reasoning=reasoning,
            rejected_reasoning="Baseline estimation without project-specific adjustments.",
            chosen_breakdown=approved_breakdown,
            rejected_breakdown=synthetic,
            signal_source="explicit_approval",
            reward_delta=0.5,  # Moderate delta for approvals
            confidence=0.6,  # Lower confidence than edits
            validated=False,
        )

        self.db.add(pair)
        await self.db.flush()

        return PairGenerationResult(
            pair_id=pair.id,
            signal_source="explicit_approval",
            reward_delta=0.5,
            confidence=0.6,
        )

    async def get_pending_pairs(self, limit: int = 50) -> list[PreferencePair]:
        """Get pairs not yet used in training."""
        result = await self.db.execute(
            select(PreferencePair)
            .where(PreferencePair.used_in_training.is_(None))
            .order_by(PreferencePair.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_unvalidated_pairs(self, limit: int = 20) -> list[PreferencePair]:
        """Get pairs needing human validation."""
        result = await self.db.execute(
            select(PreferencePair)
            .where(PreferencePair.validated.is_(False))
            .order_by(PreferencePair.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def validate_pair(
        self,
        pair_id: uuid.UUID,
        validated: bool,
        confidence_override: float | None = None,
    ) -> bool:
        """Mark pair as validated or rejected."""
        values = {"validated": validated}
        if confidence_override is not None:
            values["confidence"] = confidence_override

        result = await self.db.execute(
            update(PreferencePair).where(PreferencePair.id == pair_id).values(**values)
        )
        return result.rowcount > 0

    async def mark_used_in_training(self, pair_ids: list[uuid.UUID]) -> int:
        """Mark pairs as used in training."""
        now = datetime.now(timezone.utc)
        result = await self.db.execute(
            update(PreferencePair)
            .where(PreferencePair.id.in_(pair_ids))
            .values(used_in_training=now)
        )
        return result.rowcount

    async def get_pair_count_by_source(self) -> dict[str, int]:
        """Get count of pairs by signal source."""
        from sqlalchemy import func

        result = await self.db.execute(
            select(
                PreferencePair.signal_source, func.count(PreferencePair.id)
            ).group_by(PreferencePair.signal_source)
        )
        return {row[0]: row[1] for row in result.all()}


class ABExperimentService:
    """Service for managing A/B testing experiments."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_experiment(
        self,
        name: str,
        candidate_model_version: str,
        production_model_version: str,
    ) -> ABExperiment:
        """Create a new A/B experiment in shadow mode."""
        experiment = ABExperiment(
            name=name,
            candidate_model_version=candidate_model_version,
            production_model_version=production_model_version,
            status="shadow",
            candidate_weight=0.0,
            shadow_mode=True,
        )
        self.db.add(experiment)
        await self.db.flush()
        return experiment

    async def get_active_experiment(self) -> ABExperiment | None:
        """Get the currently active experiment."""
        result = await self.db.execute(
            select(ABExperiment)
            .where(ABExperiment.status.not_in(["complete", "rolled_back"]))
            .order_by(ABExperiment.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def promote_experiment(
        self, experiment_id: uuid.UUID, new_status: str, new_weight: float | None = None
    ) -> bool:
        """Promote experiment to next stage."""
        values = {"status": new_status}
        if new_weight is not None:
            values["candidate_weight"] = new_weight
            if new_weight > 0:
                values["shadow_mode"] = False

        if new_status in ["complete", "rolled_back"]:
            values["completed_at"] = datetime.now(timezone.utc)

        result = await self.db.execute(
            update(ABExperiment)
            .where(ABExperiment.id == experiment_id)
            .values(**values)
        )
        return result.rowcount > 0

    async def record_prediction(
        self,
        experiment_id: uuid.UUID,
        session_id: uuid.UUID | None,
        model_used: str,
        prediction: dict[str, Any],
        shadow_prediction: dict[str, Any] | None = None,
        sizing_category: str | None = None,
    ) -> ABPrediction:
        """Record a prediction for A/B comparison."""
        pred = ABPrediction(
            experiment_id=experiment_id,
            session_id=session_id,
            model_used=model_used,
            prediction=prediction,
            shadow_prediction=shadow_prediction,
            sizing_category=sizing_category,
        )
        self.db.add(pred)
        await self.db.flush()
        return pred

    async def get_experiment_stats(self, experiment_id: uuid.UUID) -> dict[str, Any]:
        """Get statistics for an experiment."""
        from sqlalchemy import func

        # Count predictions by model
        count_result = await self.db.execute(
            select(ABPrediction.model_used, func.count(ABPrediction.id))
            .where(ABPrediction.experiment_id == experiment_id)
            .group_by(ABPrediction.model_used)
        )
        prediction_counts = {row[0]: row[1] for row in count_result.all()}

        # Count by sizing category
        sizing_result = await self.db.execute(
            select(ABPrediction.sizing_category, func.count(ABPrediction.id))
            .where(ABPrediction.experiment_id == experiment_id)
            .where(ABPrediction.sizing_category.isnot(None))
            .group_by(ABPrediction.sizing_category)
        )
        sizing_spread = {row[0]: row[1] for row in sizing_result.all()}

        # Count user edits
        edit_result = await self.db.execute(
            select(ABPrediction.model_used, func.count(ABPrediction.id))
            .where(ABPrediction.experiment_id == experiment_id)
            .where(ABPrediction.user_edited.is_(True))
            .group_by(ABPrediction.model_used)
        )
        edit_counts = {row[0]: row[1] for row in edit_result.all()}

        return {
            "prediction_counts": prediction_counts,
            "sizing_spread": sizing_spread,
            "edit_counts": edit_counts,
            "total_predictions": sum(prediction_counts.values()),
        }


class RLHFTrainingService:
    """Service for managing RLHF training jobs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(self, job_type: str) -> RLHFTrainingJob:
        """Create a new training job."""
        job = RLHFTrainingJob(
            job_type=job_type,
            status="pending",
        )
        self.db.add(job)
        await self.db.flush()
        return job

    async def start_job(
        self, job_id: uuid.UUID, metrics_before: dict[str, Any] | None = None
    ) -> bool:
        """Mark job as started."""
        result = await self.db.execute(
            update(RLHFTrainingJob)
            .where(RLHFTrainingJob.id == job_id)
            .values(
                status="running",
                started_at=datetime.now(timezone.utc),
                metrics_before=metrics_before,
            )
        )
        return result.rowcount > 0

    async def complete_job(
        self,
        job_id: uuid.UUID,
        success: bool,
        samples_used: int | None = None,
        metrics_after: dict[str, Any] | None = None,
        model_version_created: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Mark job as completed or failed."""
        result = await self.db.execute(
            update(RLHFTrainingJob)
            .where(RLHFTrainingJob.id == job_id)
            .values(
                status="completed" if success else "failed",
                completed_at=datetime.now(timezone.utc),
                samples_used=samples_used,
                metrics_after=metrics_after,
                model_version_created=model_version_created,
                error_message=error_message,
            )
        )
        return result.rowcount > 0

    async def get_latest_job(
        self, job_type: str | None = None
    ) -> RLHFTrainingJob | None:
        """Get the most recent training job."""
        query = select(RLHFTrainingJob).order_by(RLHFTrainingJob.created_at.desc())
        if job_type:
            query = query.where(RLHFTrainingJob.job_type == job_type)
        result = await self.db.execute(query.limit(1))
        return result.scalar_one_or_none()

    async def get_active_jobs(self) -> list[RLHFTrainingJob]:
        """Get currently running jobs."""
        result = await self.db.execute(
            select(RLHFTrainingJob)
            .where(RLHFTrainingJob.status.in_(["pending", "running"]))
            .order_by(RLHFTrainingJob.created_at.desc())
        )
        return list(result.scalars().all())

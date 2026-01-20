"""
FPT Cost Brain 2.0 - A/B Router for Model Deployment
Routes predictions between production and candidate models with kill switch monitoring.
"""

import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    """A/B experiment deployment stages."""

    SHADOW = "shadow"  # 0% traffic, candidate runs in parallel
    CANARY = "canary"  # 10% traffic to candidate
    GRADUAL = "gradual"  # 10% → 50% → 100% incremental
    COMPLETE = "complete"  # Candidate is now production
    ROLLED_BACK = "rolled_back"  # Experiment failed


@dataclass
class ABPredictionResult:
    """Result from A/B router prediction."""

    result: dict[str, Any]  # Prediction result (point_estimate, interval, breakdown)
    model_used: str  # "production" or "candidate"
    shadow_result: dict[str, Any] | None = None  # Candidate result in shadow mode
    experiment_id: str | None = None
    prediction_id: str | None = None


@dataclass
class ExperimentMetrics:
    """Metrics for kill switch evaluation."""

    production_mape: float
    candidate_mape: float
    production_confidence: float
    candidate_confidence: float
    production_edit_rate: float
    candidate_edit_rate: float
    production_samples: int
    candidate_samples: int


@dataclass
class KillSwitchResult:
    """Result of kill switch evaluation."""

    should_kill: bool
    reason: str
    metrics: ExperimentMetrics
    threshold_breached: str | None = None


# Kill switch thresholds from design
KILL_SWITCH_THRESHOLDS = {
    "mape_absolute": 0.35,  # >35% MAPE = immediate rollback
    "mape_relative_increase": 0.10,  # >10% worse than production
    "confidence_min": 0.55,  # Confidence below 55%
    "user_edit_rate_increase": 0.20,  # 20% more user edits
    "min_samples_for_evaluation": 5,  # Minimum samples before evaluating
}


def check_kill_switch(metrics: ExperimentMetrics) -> KillSwitchResult:
    """
    Evaluate whether experiment should be killed based on metrics.

    Returns KillSwitchResult with should_kill flag and reason.
    """
    # Not enough samples for reliable evaluation
    if metrics.candidate_samples < KILL_SWITCH_THRESHOLDS["min_samples_for_evaluation"]:
        return KillSwitchResult(
            should_kill=False,
            reason=f"Insufficient samples ({metrics.candidate_samples}/{KILL_SWITCH_THRESHOLDS['min_samples_for_evaluation']})",
            metrics=metrics,
        )

    # Check 1: Absolute MAPE threshold
    if metrics.candidate_mape > KILL_SWITCH_THRESHOLDS["mape_absolute"]:
        return KillSwitchResult(
            should_kill=True,
            reason=f"Candidate MAPE {metrics.candidate_mape:.1%} exceeds absolute threshold {KILL_SWITCH_THRESHOLDS['mape_absolute']:.0%}",
            metrics=metrics,
            threshold_breached="mape_absolute",
        )

    # Check 2: Relative MAPE increase vs production
    if metrics.production_mape > 0:
        mape_increase = (
            metrics.candidate_mape - metrics.production_mape
        ) / metrics.production_mape
        if mape_increase > KILL_SWITCH_THRESHOLDS["mape_relative_increase"]:
            return KillSwitchResult(
                should_kill=True,
                reason=f"Candidate MAPE {mape_increase:.1%} worse than production (threshold: {KILL_SWITCH_THRESHOLDS['mape_relative_increase']:.0%})",
                metrics=metrics,
                threshold_breached="mape_relative_increase",
            )

    # Check 3: Minimum confidence
    if metrics.candidate_confidence < KILL_SWITCH_THRESHOLDS["confidence_min"]:
        return KillSwitchResult(
            should_kill=True,
            reason=f"Candidate confidence {metrics.candidate_confidence:.1%} below minimum {KILL_SWITCH_THRESHOLDS['confidence_min']:.0%}",
            metrics=metrics,
            threshold_breached="confidence_min",
        )

    # Check 4: User edit rate increase
    if metrics.production_edit_rate > 0:
        edit_increase = (
            metrics.candidate_edit_rate - metrics.production_edit_rate
        ) / metrics.production_edit_rate
        if edit_increase > KILL_SWITCH_THRESHOLDS["user_edit_rate_increase"]:
            return KillSwitchResult(
                should_kill=True,
                reason=f"User edit rate {edit_increase:.1%} higher than production (threshold: {KILL_SWITCH_THRESHOLDS['user_edit_rate_increase']:.0%})",
                metrics=metrics,
                threshold_breached="user_edit_rate_increase",
            )

    # All checks passed
    return KillSwitchResult(
        should_kill=False,
        reason="All metrics within acceptable thresholds",
        metrics=metrics,
    )


@dataclass
class ModelWrapper:
    """Wrapper for HCQE model with version tracking."""

    model: Any  # HCQEPredictor instance
    version: str
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run prediction and return result dict."""
        prediction = self.model.predict(features)
        return {
            "point_estimate": prediction.point_estimate,
            "prediction_interval": prediction.prediction_interval,
            "cluster_breakdown": prediction.cluster_breakdown,
            "calibrated_confidence": prediction.calibrated_confidence,
            "sizing_category": prediction.sizing_category,
            "model_version": self.version,
        }


class ABRouter:
    """
    Route predictions between production and candidate models.

    Supports:
    - Shadow mode: Run candidate in parallel, log results, serve production
    - Weighted A/B split: Route X% of traffic to candidate
    - Kill switch: Automatic rollback on metric degradation
    """

    def __init__(
        self,
        production_model: Any,
        production_version: str,
        candidate_model: Any | None = None,
        candidate_version: str | None = None,
        candidate_weight: float = 0.0,
        shadow_mode: bool = False,
        experiment_id: str | None = None,
    ):
        """
        Initialize A/B router.

        Args:
            production_model: Currently deployed HCQE model
            production_version: Version identifier for production
            candidate_model: New model being tested (optional)
            candidate_version: Version identifier for candidate
            candidate_weight: Fraction of traffic to candidate [0.0, 1.0]
            shadow_mode: If True, always serve production but log candidate
            experiment_id: ID of active A/B experiment
        """
        self.production = ModelWrapper(production_model, production_version)
        self.candidate_weight = candidate_weight
        self.shadow_mode = shadow_mode
        self.experiment_id = experiment_id

        # CRITICAL: Load candidate if weight > 0 OR shadow mode enabled
        should_load_candidate = candidate_weight > 0 or shadow_mode

        if should_load_candidate and candidate_model is not None:
            self.candidate = ModelWrapper(
                candidate_model, candidate_version or "candidate"
            )
            logger.info(
                f"ABRouter initialized: candidate_weight={candidate_weight}, shadow_mode={shadow_mode}"
            )
        else:
            self.candidate = None
            if should_load_candidate and candidate_model is None:
                logger.warning(
                    "ABRouter: shadow_mode or candidate_weight set but no candidate model provided"
                )

        # Track predictions for metrics
        self._prediction_log: list[dict] = []

    def predict(self, features: dict[str, Any]) -> ABPredictionResult:
        """
        Route prediction to appropriate model.

        1. Always runs production model
        2. In shadow mode: also runs candidate, logs comparison, serves production
        3. With A/B split: routes X% to candidate based on weight

        Returns ABPredictionResult with prediction and metadata.
        """
        prediction_id = str(uuid.uuid4())

        # Always run production
        prod_result = self.production.predict(features)

        # Shadow mode: run candidate, log, but serve production
        if self.shadow_mode and self.candidate:
            try:
                candidate_result = self.candidate.predict(features)
                self._log_shadow_comparison(
                    prediction_id, features, prod_result, candidate_result
                )

                return ABPredictionResult(
                    result=prod_result,
                    model_used="production",
                    shadow_result=candidate_result,
                    experiment_id=self.experiment_id,
                    prediction_id=prediction_id,
                )
            except Exception as e:
                logger.error(f"Candidate model failed in shadow mode: {e}")
                # Fall through to return production result

        # A/B split based on weight
        if self.candidate and random.random() < self.candidate_weight:
            try:
                candidate_result = self.candidate.predict(features)
                self._log_prediction(
                    prediction_id, "candidate", features, candidate_result
                )

                return ABPredictionResult(
                    result=candidate_result,
                    model_used="candidate",
                    experiment_id=self.experiment_id,
                    prediction_id=prediction_id,
                )
            except Exception as e:
                logger.error(f"Candidate model failed, falling back to production: {e}")
                # Fall through to return production result

        # Default: return production
        self._log_prediction(prediction_id, "production", features, prod_result)

        return ABPredictionResult(
            result=prod_result,
            model_used="production",
            experiment_id=self.experiment_id,
            prediction_id=prediction_id,
        )

    def _log_shadow_comparison(
        self,
        prediction_id: str,
        features: dict,
        prod_result: dict,
        candidate_result: dict,
    ) -> None:
        """Log shadow mode comparison for drift analysis."""
        comparison = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_served": "production",
            "production_estimate": prod_result.get("point_estimate"),
            "candidate_estimate": candidate_result.get("point_estimate"),
            "delta_pct": self._calculate_delta(
                prod_result.get("point_estimate"),
                candidate_result.get("point_estimate"),
            ),
            "production_confidence": prod_result.get("calibrated_confidence"),
            "candidate_confidence": candidate_result.get("calibrated_confidence"),
        }

        self._prediction_log.append(comparison)
        logger.debug(
            f"Shadow comparison: prod={prod_result.get('point_estimate')}, "
            f"cand={candidate_result.get('point_estimate')}, "
            f"delta={comparison['delta_pct']:.1%}"
        )

    def _log_prediction(
        self, prediction_id: str, model_used: str, features: dict, result: dict
    ) -> None:
        """Log prediction for metrics tracking."""
        log_entry = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_used": model_used,
            "estimate": result.get("point_estimate"),
            "confidence": result.get("calibrated_confidence"),
        }
        self._prediction_log.append(log_entry)

    def _calculate_delta(
        self, prod_value: float | None, cand_value: float | None
    ) -> float:
        """Calculate percentage difference between production and candidate."""
        if prod_value is None or cand_value is None or prod_value == 0:
            return 0.0
        return (cand_value - prod_value) / prod_value

    def get_shadow_stats(self) -> dict:
        """Get statistics from shadow mode predictions."""
        shadow_predictions = [
            p for p in self._prediction_log if "candidate_estimate" in p
        ]

        if not shadow_predictions:
            return {"count": 0, "message": "No shadow predictions logged"}

        deltas = [p["delta_pct"] for p in shadow_predictions]
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(abs(d) for d in deltas)

        return {
            "count": len(shadow_predictions),
            "avg_delta_pct": avg_delta,
            "max_delta_pct": max_delta,
            "candidate_higher_count": sum(1 for d in deltas if d > 0),
            "candidate_lower_count": sum(1 for d in deltas if d < 0),
        }

    def update_weights(self, new_weight: float) -> None:
        """Update candidate traffic weight (for gradual rollout)."""
        if new_weight < 0 or new_weight > 1:
            raise ValueError(f"Weight must be between 0 and 1, got {new_weight}")

        old_weight = self.candidate_weight
        self.candidate_weight = new_weight
        logger.info(f"ABRouter weight updated: {old_weight} → {new_weight}")

    def disable_shadow_mode(self) -> None:
        """Disable shadow mode (after gathering enough data)."""
        self.shadow_mode = False
        logger.info("ABRouter shadow mode disabled")

    def promote_candidate(self) -> None:
        """Promote candidate to production (after successful A/B test)."""
        if self.candidate is None:
            raise ValueError("No candidate model to promote")

        self.production = self.candidate
        self.candidate = None
        self.candidate_weight = 0.0
        self.shadow_mode = False
        self.experiment_id = None

        logger.info(f"Candidate promoted to production: {self.production.version}")

    def rollback(self) -> None:
        """Emergency rollback: disable candidate completely."""
        self.candidate = None
        self.candidate_weight = 0.0
        self.shadow_mode = False

        logger.warning("ABRouter: Emergency rollback executed")


class ABRouterManager:
    """
    Singleton manager for A/B router with experiment lifecycle.

    Handles:
    - Creating new experiments
    - Loading models from disk
    - Advancing experiment stages
    - Kill switch evaluation and rollback
    """

    _instance: "ABRouterManager | None" = None
    _router: ABRouter | None = None

    def __new__(cls) -> "ABRouterManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._router = None
            self._models_path = Path("models")

    def get_router(self) -> ABRouter | None:
        """Get current A/B router instance."""
        return self._router

    def initialize_production_only(
        self, production_model: Any, production_version: str
    ) -> ABRouter:
        """Initialize router with production model only (no experiment)."""
        self._router = ABRouter(
            production_model=production_model,
            production_version=production_version,
        )
        logger.info(f"ABRouter initialized with production model: {production_version}")
        return self._router

    def start_experiment(
        self,
        production_model: Any,
        production_version: str,
        candidate_model: Any,
        candidate_version: str,
        experiment_id: str,
        initial_stage: ExperimentStatus = ExperimentStatus.SHADOW,
    ) -> ABRouter:
        """
        Start new A/B experiment.

        Args:
            production_model: Current production model
            production_version: Production version identifier
            candidate_model: New model to test
            candidate_version: Candidate version identifier
            experiment_id: Experiment ID from database
            initial_stage: Starting stage (default: shadow)
        """
        # Determine initial settings based on stage
        if initial_stage == ExperimentStatus.SHADOW:
            candidate_weight = 0.0
            shadow_mode = True
        elif initial_stage == ExperimentStatus.CANARY:
            candidate_weight = 0.10
            shadow_mode = False
        elif initial_stage == ExperimentStatus.GRADUAL:
            candidate_weight = 0.10  # Start at 10%, can be increased
            shadow_mode = False
        else:
            candidate_weight = 0.0
            shadow_mode = False

        self._router = ABRouter(
            production_model=production_model,
            production_version=production_version,
            candidate_model=candidate_model,
            candidate_version=candidate_version,
            candidate_weight=candidate_weight,
            shadow_mode=shadow_mode,
            experiment_id=experiment_id,
        )

        logger.info(
            f"ABRouter experiment started: {experiment_id}, stage={initial_stage.value}"
        )
        return self._router

    def advance_stage(self, new_stage: ExperimentStatus) -> None:
        """
        Advance experiment to next stage.

        Shadow → Canary (10%) → Gradual (10-50-100%) → Complete
        """
        if self._router is None:
            raise ValueError("No active router to advance")

        if new_stage == ExperimentStatus.CANARY:
            self._router.disable_shadow_mode()
            self._router.update_weights(0.10)
            logger.info("Experiment advanced to CANARY (10%)")

        elif new_stage == ExperimentStatus.GRADUAL:
            self._router.update_weights(0.50)
            logger.info("Experiment advanced to GRADUAL (50%)")

        elif new_stage == ExperimentStatus.COMPLETE:
            self._router.promote_candidate()
            logger.info("Experiment COMPLETE - candidate promoted to production")

        elif new_stage == ExperimentStatus.ROLLED_BACK:
            self._router.rollback()
            logger.warning("Experiment ROLLED BACK")

    def evaluate_kill_switch(self, metrics: ExperimentMetrics) -> KillSwitchResult:
        """
        Evaluate kill switch conditions.

        If kill switch triggered, automatically rolls back.
        """
        result = check_kill_switch(metrics)

        if result.should_kill:
            logger.error(f"Kill switch triggered: {result.reason}")
            if self._router:
                self._router.rollback()

        return result

    def get_experiment_status(self) -> dict:
        """Get current experiment status."""
        if self._router is None:
            return {"status": "no_active_experiment"}

        return {
            "status": "active",
            "experiment_id": self._router.experiment_id,
            "production_version": self._router.production.version,
            "candidate_version": (
                self._router.candidate.version if self._router.candidate else None
            ),
            "candidate_weight": self._router.candidate_weight,
            "shadow_mode": self._router.shadow_mode,
            "shadow_stats": (
                self._router.get_shadow_stats() if self._router.shadow_mode else None
            ),
        }


# Global instance
_ab_router_manager: ABRouterManager | None = None


def get_ab_router_manager() -> ABRouterManager:
    """Get or create global ABRouterManager instance."""
    global _ab_router_manager
    if _ab_router_manager is None:
        _ab_router_manager = ABRouterManager()
    return _ab_router_manager

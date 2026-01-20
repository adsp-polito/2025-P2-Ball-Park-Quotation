"""
FPT Cost Brain 2.0 - A/B Router Tests
Unit tests for A/B routing, kill switch, and experiment management.
"""

import pytest
from unittest.mock import MagicMock, patch

from ml.ab_router import (
    ABRouter,
    ABRouterManager,
    ABPredictionResult,
    ExperimentMetrics,
    ExperimentStatus,
    KillSwitchResult,
    check_kill_switch,
    KILL_SWITCH_THRESHOLDS,
    get_ab_router_manager,
)


class MockModel:
    """Mock HCQE model for testing."""

    def __init__(self, base_estimate: float = 1000, confidence: float = 0.85):
        self.base_estimate = base_estimate
        self.confidence = confidence
        self.call_count = 0

    def predict(self, features: dict) -> MagicMock:
        self.call_count += 1
        result = MagicMock()
        result.point_estimate = self.base_estimate
        result.prediction_interval = (
            self.base_estimate * 0.8,
            self.base_estimate * 1.2,
        )
        result.cluster_breakdown = {"hardware": 500, "calibration": 300, "testing": 200}
        result.calibrated_confidence = self.confidence
        result.sizing_category = "medium"
        return result


class TestKillSwitch:
    """Tests for kill switch evaluation."""

    def test_insufficient_samples_no_kill(self):
        """Test kill switch not triggered with insufficient samples."""
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.50,  # Very bad, but...
            production_confidence=0.80,
            candidate_confidence=0.40,  # Very bad, but...
            production_edit_rate=0.10,
            candidate_edit_rate=0.50,  # Very bad, but...
            production_samples=100,
            candidate_samples=3,  # Not enough samples!
        )

        result = check_kill_switch(metrics)

        assert not result.should_kill
        assert "Insufficient samples" in result.reason

    def test_absolute_mape_kill(self):
        """Test kill switch on absolute MAPE threshold breach."""
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.40,  # > 35% threshold
            production_confidence=0.80,
            candidate_confidence=0.75,
            production_edit_rate=0.10,
            candidate_edit_rate=0.12,
            production_samples=100,
            candidate_samples=20,
        )

        result = check_kill_switch(metrics)

        assert result.should_kill
        assert result.threshold_breached == "mape_absolute"
        assert "exceeds absolute threshold" in result.reason

    def test_relative_mape_increase_kill(self):
        """Test kill switch on relative MAPE increase."""
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.18,  # 20% increase (> 10% threshold)
            production_confidence=0.80,
            candidate_confidence=0.78,
            production_edit_rate=0.10,
            candidate_edit_rate=0.11,
            production_samples=100,
            candidate_samples=20,
        )

        result = check_kill_switch(metrics)

        assert result.should_kill
        assert result.threshold_breached == "mape_relative_increase"
        assert "worse than production" in result.reason

    def test_low_confidence_kill(self):
        """Test kill switch on low confidence."""
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.14,  # Actually better
            production_confidence=0.80,
            candidate_confidence=0.50,  # < 55% threshold
            production_edit_rate=0.10,
            candidate_edit_rate=0.09,
            production_samples=100,
            candidate_samples=20,
        )

        result = check_kill_switch(metrics)

        assert result.should_kill
        assert result.threshold_breached == "confidence_min"
        assert "below minimum" in result.reason

    def test_high_edit_rate_kill(self):
        """Test kill switch on high user edit rate."""
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.14,
            production_confidence=0.80,
            candidate_confidence=0.78,
            production_edit_rate=0.10,
            candidate_edit_rate=0.15,  # 50% increase (> 20% threshold)
            production_samples=100,
            candidate_samples=20,
        )

        result = check_kill_switch(metrics)

        assert result.should_kill
        assert result.threshold_breached == "user_edit_rate_increase"
        assert "edit rate" in result.reason

    def test_all_metrics_good_no_kill(self):
        """Test no kill when all metrics are good."""
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.13,  # 13% better
            production_confidence=0.80,
            candidate_confidence=0.82,  # Better
            production_edit_rate=0.10,
            candidate_edit_rate=0.08,  # Better
            production_samples=100,
            candidate_samples=50,
        )

        result = check_kill_switch(metrics)

        assert not result.should_kill
        assert "within acceptable thresholds" in result.reason
        assert result.threshold_breached is None


class TestABRouter:
    """Tests for A/B router prediction routing."""

    def test_production_only_routing(self):
        """Test routing when no candidate model."""
        prod_model = MockModel(base_estimate=1000)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
        )

        result = router.predict({"feature1": 1})

        assert result.model_used == "production"
        assert result.result["point_estimate"] == 1000
        assert result.shadow_result is None
        assert prod_model.call_count == 1

    def test_shadow_mode_serves_production(self):
        """Test shadow mode runs both but serves production."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1200)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            shadow_mode=True,
        )

        result = router.predict({"feature1": 1})

        assert result.model_used == "production"
        assert result.result["point_estimate"] == 1000  # Production served
        assert result.shadow_result is not None
        assert result.shadow_result["point_estimate"] == 1200  # Candidate logged
        assert prod_model.call_count == 1
        assert cand_model.call_count == 1

    def test_shadow_mode_logs_comparison(self):
        """Test shadow mode logs comparison statistics."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1100)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            shadow_mode=True,
        )

        # Run multiple predictions
        for _ in range(5):
            router.predict({"feature1": 1})

        stats = router.get_shadow_stats()

        assert stats["count"] == 5
        assert stats["avg_delta_pct"] == pytest.approx(0.10, rel=0.01)  # 10% higher
        assert stats["candidate_higher_count"] == 5

    def test_ab_split_respects_weight(self):
        """Test A/B split routes based on weight."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1200)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            candidate_weight=0.50,  # 50% split
        )

        # Run many predictions
        results = [router.predict({"feature1": 1}) for _ in range(100)]

        prod_count = sum(1 for r in results if r.model_used == "production")
        cand_count = sum(1 for r in results if r.model_used == "candidate")

        # Should be roughly 50/50 (allow variance)
        assert 30 <= prod_count <= 70
        assert 30 <= cand_count <= 70

    def test_candidate_weight_zero_always_production(self):
        """Test weight=0 always routes to production."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1200)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            candidate_weight=0.0,
            shadow_mode=False,
        )

        results = [router.predict({"feature1": 1}) for _ in range(20)]

        assert all(r.model_used == "production" for r in results)
        assert cand_model.call_count == 0

    def test_candidate_failure_fallback(self):
        """Test fallback to production when candidate fails."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1200)

        # Make candidate throw exception
        cand_model.predict = MagicMock(side_effect=Exception("Model error"))

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            candidate_weight=1.0,  # Always try candidate
        )

        result = router.predict({"feature1": 1})

        # Should fallback to production
        assert result.model_used == "production"
        assert result.result["point_estimate"] == 1000

    def test_update_weights(self):
        """Test weight update for gradual rollout."""
        prod_model = MockModel()
        cand_model = MockModel()

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            candidate_weight=0.10,
        )

        assert router.candidate_weight == 0.10

        router.update_weights(0.50)
        assert router.candidate_weight == 0.50

        router.update_weights(1.0)
        assert router.candidate_weight == 1.0

    def test_update_weights_validation(self):
        """Test weight validation on update."""
        router = ABRouter(
            production_model=MockModel(),
            production_version="v1.0",
        )

        with pytest.raises(ValueError):
            router.update_weights(-0.1)

        with pytest.raises(ValueError):
            router.update_weights(1.5)

    def test_promote_candidate(self):
        """Test candidate promotion to production."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1200)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            candidate_weight=0.50,
        )

        router.promote_candidate()

        assert router.production.version == "v2.0"
        assert router.candidate is None
        assert router.candidate_weight == 0.0
        assert router.shadow_mode is False

        # New production should serve v2.0 estimates
        result = router.predict({"feature1": 1})
        assert result.result["point_estimate"] == 1200

    def test_promote_without_candidate_fails(self):
        """Test promotion fails when no candidate."""
        router = ABRouter(
            production_model=MockModel(),
            production_version="v1.0",
        )

        with pytest.raises(ValueError, match="No candidate model"):
            router.promote_candidate()

    def test_rollback(self):
        """Test emergency rollback."""
        prod_model = MockModel(base_estimate=1000)
        cand_model = MockModel(base_estimate=1200)

        router = ABRouter(
            production_model=prod_model,
            production_version="v1.0",
            candidate_model=cand_model,
            candidate_version="v2.0",
            candidate_weight=0.50,
            shadow_mode=True,
        )

        router.rollback()

        assert router.candidate is None
        assert router.candidate_weight == 0.0
        assert router.shadow_mode is False

        # Should only serve production
        result = router.predict({"feature1": 1})
        assert result.model_used == "production"


class TestABRouterManager:
    """Tests for A/B router lifecycle management."""

    def test_singleton_pattern(self):
        """Test manager is singleton."""
        manager1 = ABRouterManager()
        manager2 = ABRouterManager()
        assert manager1 is manager2

    def test_initialize_production_only(self):
        """Test initializing with production model only."""
        manager = ABRouterManager()
        prod_model = MockModel()

        router = manager.initialize_production_only(prod_model, "v1.0")

        assert router is not None
        assert manager.get_router() is router
        assert router.candidate is None

    def test_start_experiment_shadow(self):
        """Test starting experiment in shadow mode."""
        manager = ABRouterManager()

        router = manager.start_experiment(
            production_model=MockModel(base_estimate=1000),
            production_version="v1.0",
            candidate_model=MockModel(base_estimate=1200),
            candidate_version="v2.0",
            experiment_id="exp-001",
            initial_stage=ExperimentStatus.SHADOW,
        )

        assert router.shadow_mode is True
        assert router.candidate_weight == 0.0
        assert router.experiment_id == "exp-001"

    def test_start_experiment_canary(self):
        """Test starting experiment in canary mode."""
        manager = ABRouterManager()

        router = manager.start_experiment(
            production_model=MockModel(),
            production_version="v1.0",
            candidate_model=MockModel(),
            candidate_version="v2.0",
            experiment_id="exp-002",
            initial_stage=ExperimentStatus.CANARY,
        )

        assert router.shadow_mode is False
        assert router.candidate_weight == 0.10

    def test_advance_stage_shadow_to_canary(self):
        """Test advancing from shadow to canary."""
        manager = ABRouterManager()

        manager.start_experiment(
            production_model=MockModel(),
            production_version="v1.0",
            candidate_model=MockModel(),
            candidate_version="v2.0",
            experiment_id="exp-003",
            initial_stage=ExperimentStatus.SHADOW,
        )

        manager.advance_stage(ExperimentStatus.CANARY)

        router = manager.get_router()
        assert router.shadow_mode is False
        assert router.candidate_weight == 0.10

    def test_advance_stage_to_gradual(self):
        """Test advancing to gradual rollout."""
        manager = ABRouterManager()

        manager.start_experiment(
            production_model=MockModel(),
            production_version="v1.0",
            candidate_model=MockModel(),
            candidate_version="v2.0",
            experiment_id="exp-004",
        )

        manager.advance_stage(ExperimentStatus.GRADUAL)

        router = manager.get_router()
        assert router.candidate_weight == 0.50

    def test_advance_stage_to_complete(self):
        """Test completing experiment (promotion)."""
        manager = ABRouterManager()

        manager.start_experiment(
            production_model=MockModel(),
            production_version="v1.0",
            candidate_model=MockModel(),
            candidate_version="v2.0",
            experiment_id="exp-005",
        )

        manager.advance_stage(ExperimentStatus.COMPLETE)

        router = manager.get_router()
        assert router.production.version == "v2.0"
        assert router.candidate is None

    def test_evaluate_kill_switch_auto_rollback(self):
        """Test kill switch evaluation auto-rolls back."""
        manager = ABRouterManager()

        manager.start_experiment(
            production_model=MockModel(),
            production_version="v1.0",
            candidate_model=MockModel(),
            candidate_version="v2.0",
            experiment_id="exp-006",
            initial_stage=ExperimentStatus.CANARY,
        )

        # Bad metrics that trigger kill
        metrics = ExperimentMetrics(
            production_mape=0.15,
            candidate_mape=0.50,  # Very bad
            production_confidence=0.80,
            candidate_confidence=0.75,
            production_edit_rate=0.10,
            candidate_edit_rate=0.12,
            production_samples=100,
            candidate_samples=20,
        )

        result = manager.evaluate_kill_switch(metrics)

        assert result.should_kill
        router = manager.get_router()
        assert router.candidate is None  # Auto-rolled back

    def test_get_experiment_status(self):
        """Test getting experiment status."""
        manager = ABRouterManager()

        manager.start_experiment(
            production_model=MockModel(),
            production_version="v1.0",
            candidate_model=MockModel(),
            candidate_version="v2.0",
            experiment_id="exp-007",
            initial_stage=ExperimentStatus.SHADOW,
        )

        status = manager.get_experiment_status()

        assert status["status"] == "active"
        assert status["experiment_id"] == "exp-007"
        assert status["production_version"] == "v1.0"
        assert status["candidate_version"] == "v2.0"
        assert status["shadow_mode"] is True


class TestKillSwitchThresholds:
    """Tests for kill switch threshold values."""

    def test_threshold_values_exist(self):
        """Test all threshold values are defined."""
        assert "mape_absolute" in KILL_SWITCH_THRESHOLDS
        assert "mape_relative_increase" in KILL_SWITCH_THRESHOLDS
        assert "confidence_min" in KILL_SWITCH_THRESHOLDS
        assert "user_edit_rate_increase" in KILL_SWITCH_THRESHOLDS
        assert "min_samples_for_evaluation" in KILL_SWITCH_THRESHOLDS

    def test_threshold_values_reasonable(self):
        """Test threshold values are reasonable."""
        assert 0.3 <= KILL_SWITCH_THRESHOLDS["mape_absolute"] <= 0.5
        assert 0.05 <= KILL_SWITCH_THRESHOLDS["mape_relative_increase"] <= 0.20
        assert 0.5 <= KILL_SWITCH_THRESHOLDS["confidence_min"] <= 0.7
        assert 0.1 <= KILL_SWITCH_THRESHOLDS["user_edit_rate_increase"] <= 0.3
        assert KILL_SWITCH_THRESHOLDS["min_samples_for_evaluation"] >= 3

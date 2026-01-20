"""
FPT Cost Brain 2.0 - RLHF Monitoring Service Tests
Unit tests for dashboard metrics, alerts, and Prometheus export.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.rlhf_monitoring import (
    Alert,
    AlertSeverity,
    AlertType,
    ALERT_THRESHOLDS,
    DashboardMetrics,
    RLHFMonitoringService,
    TimeSeriesPoint,
    export_prometheus_metrics,
)


class TestAlertGeneration:
    """Tests for alert generation based on thresholds."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            alert_type=AlertType.MODEL_DRIFT,
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            details={"mape": 0.30},
        )

        assert alert.alert_type == AlertType.MODEL_DRIFT
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert message"
        assert alert.details["mape"] == 0.30
        assert alert.acknowledged is False

    def test_alert_timestamp_default(self):
        """Test alert has default timestamp."""
        alert = Alert(
            alert_type=AlertType.HIGH_EDIT_RATE,
            severity=AlertSeverity.CRITICAL,
            message="High edit rate",
        )

        assert alert.timestamp is not None
        assert isinstance(alert.timestamp, datetime)


class TestAlertThresholds:
    """Tests for alert threshold values."""

    def test_threshold_values_defined(self):
        """Test all required thresholds are defined."""
        required_thresholds = [
            "mape_warning",
            "mape_critical",
            "edit_rate_warning",
            "edit_rate_critical",
            "confidence_warning",
            "confidence_critical",
            "days_since_retrain_warning",
            "days_since_retrain_critical",
        ]

        for threshold in required_thresholds:
            assert threshold in ALERT_THRESHOLDS
            assert isinstance(ALERT_THRESHOLDS[threshold], (int, float))

    def test_critical_thresholds_higher_than_warning(self):
        """Test critical thresholds are more severe than warning."""
        # For MAPE and edit rate, critical > warning (higher is worse)
        assert ALERT_THRESHOLDS["mape_critical"] > ALERT_THRESHOLDS["mape_warning"]
        assert (
            ALERT_THRESHOLDS["edit_rate_critical"]
            > ALERT_THRESHOLDS["edit_rate_warning"]
        )

        # For confidence, critical < warning (lower is worse)
        assert (
            ALERT_THRESHOLDS["confidence_critical"]
            < ALERT_THRESHOLDS["confidence_warning"]
        )

        # For days since retrain, critical > warning (more days is worse)
        assert (
            ALERT_THRESHOLDS["days_since_retrain_critical"]
            > ALERT_THRESHOLDS["days_since_retrain_warning"]
        )


class TestTimeSeriesPoint:
    """Tests for time series data point."""

    def test_create_time_series_point(self):
        """Test creating a time series point."""
        point = TimeSeriesPoint(
            timestamp=datetime.now(timezone.utc),
            value=42.5,
            labels={"metric": "predictions"},
        )

        assert point.value == 42.5
        assert point.labels["metric"] == "predictions"

    def test_default_labels(self):
        """Test default empty labels."""
        point = TimeSeriesPoint(
            timestamp=datetime.now(timezone.utc),
            value=10.0,
        )

        assert point.labels == {}


class TestDashboardMetrics:
    """Tests for dashboard metrics dataclass."""

    def test_create_dashboard_metrics(self):
        """Test creating dashboard metrics."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment="test-experiment",
            current_mape=0.15,
            avg_confidence=0.85,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=[],
        )

        assert metrics.total_predictions == 100
        assert metrics.total_corrections == 10
        assert metrics.model_health == "healthy"
        assert metrics.user_edit_rate == 0.10


class TestRLHFMonitoringService:
    """Tests for the monitoring service."""

    @pytest.mark.asyncio
    async def test_determine_model_health_healthy(self):
        """Test model health determination when metrics are good."""
        db_mock = MagicMock()
        service = RLHFMonitoringService(db_mock)

        health = service._determine_model_health(
            mape=0.10,  # Below warning threshold
            confidence=0.80,  # Above warning threshold
            edit_rate=0.05,  # Below warning threshold
        )

        assert health == "healthy"

    @pytest.mark.asyncio
    async def test_determine_model_health_degraded(self):
        """Test model health determination when metrics show warning."""
        db_mock = MagicMock()
        service = RLHFMonitoringService(db_mock)

        health = service._determine_model_health(
            mape=0.30,  # Above warning, below critical
            confidence=0.70,  # Above critical, below warning
            edit_rate=0.10,  # Below warning
        )

        assert health == "degraded"

    @pytest.mark.asyncio
    async def test_determine_model_health_critical(self):
        """Test model health determination when metrics are critical."""
        db_mock = MagicMock()
        service = RLHFMonitoringService(db_mock)

        health = service._determine_model_health(
            mape=0.40,  # Above critical threshold
            confidence=0.80,
            edit_rate=0.10,
        )

        assert health == "critical"

    @pytest.mark.asyncio
    async def test_determine_model_health_none_mape(self):
        """Test model health determination when MAPE is None."""
        db_mock = MagicMock()
        service = RLHFMonitoringService(db_mock)

        health = service._determine_model_health(
            mape=None,
            confidence=0.80,
            edit_rate=0.10,
        )

        assert health == "healthy"

    @pytest.mark.asyncio
    async def test_generate_alerts_mape_critical(self):
        """Test MAPE critical alert generation."""
        db_mock = AsyncMock()
        # Mock training job query
        db_mock.execute = AsyncMock(
            return_value=MagicMock(scalars=lambda: MagicMock(all=lambda: []))
        )

        service = RLHFMonitoringService(db_mock)

        # Patch the internal methods that do DB queries
        with (
            patch.object(service, "_check_training_health", return_value=[]),
            patch.object(service, "_check_experiment_health", return_value=[]),
        ):
            alerts = await service._generate_alerts(
                current_mape=0.40,  # Critical
                avg_confidence=0.80,
                edit_rate=0.10,
            )

        mape_alerts = [a for a in alerts if a.alert_type == AlertType.MODEL_DRIFT]
        assert len(mape_alerts) == 1
        assert mape_alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_generate_alerts_mape_warning(self):
        """Test MAPE warning alert generation."""
        db_mock = AsyncMock()
        service = RLHFMonitoringService(db_mock)

        with (
            patch.object(service, "_check_training_health", return_value=[]),
            patch.object(service, "_check_experiment_health", return_value=[]),
        ):
            alerts = await service._generate_alerts(
                current_mape=0.28,  # Warning level
                avg_confidence=0.80,
                edit_rate=0.10,
            )

        mape_alerts = [a for a in alerts if a.alert_type == AlertType.MODEL_DRIFT]
        assert len(mape_alerts) == 1
        assert mape_alerts[0].severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_generate_alerts_low_confidence(self):
        """Test low confidence alert generation."""
        db_mock = AsyncMock()
        service = RLHFMonitoringService(db_mock)

        with (
            patch.object(service, "_check_training_health", return_value=[]),
            patch.object(service, "_check_experiment_health", return_value=[]),
        ):
            alerts = await service._generate_alerts(
                current_mape=0.10,
                avg_confidence=0.50,  # Below critical
                edit_rate=0.10,
            )

        confidence_alerts = [
            a for a in alerts if a.alert_type == AlertType.LOW_CONFIDENCE
        ]
        assert len(confidence_alerts) == 1
        assert confidence_alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_generate_alerts_high_edit_rate(self):
        """Test high edit rate alert generation."""
        db_mock = AsyncMock()
        service = RLHFMonitoringService(db_mock)

        with (
            patch.object(service, "_check_training_health", return_value=[]),
            patch.object(service, "_check_experiment_health", return_value=[]),
        ):
            alerts = await service._generate_alerts(
                current_mape=0.10,
                avg_confidence=0.80,
                edit_rate=0.30,  # Above critical
            )

        edit_alerts = [a for a in alerts if a.alert_type == AlertType.HIGH_EDIT_RATE]
        assert len(edit_alerts) == 1
        assert edit_alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_generate_alerts_no_alerts_when_healthy(self):
        """Test no alerts generated when all metrics are healthy."""
        db_mock = AsyncMock()
        service = RLHFMonitoringService(db_mock)

        with (
            patch.object(service, "_check_training_health", return_value=[]),
            patch.object(service, "_check_experiment_health", return_value=[]),
        ):
            alerts = await service._generate_alerts(
                current_mape=0.10,
                avg_confidence=0.85,
                edit_rate=0.05,
            )

        # Only training/experiment health alerts might be present
        core_alerts = [
            a
            for a in alerts
            if a.alert_type
            in [
                AlertType.MODEL_DRIFT,
                AlertType.LOW_CONFIDENCE,
                AlertType.HIGH_EDIT_RATE,
            ]
        ]
        assert len(core_alerts) == 0


class TestPrometheusExport:
    """Tests for Prometheus metrics export."""

    def test_export_basic_metrics(self):
        """Test basic metrics export."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment="test",
            current_mape=0.15,
            avg_confidence=0.85,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_predictions_total 100" in output
        assert "rlhf_corrections_total 10" in output
        assert "rlhf_preference_pairs_total 50" in output
        assert "rlhf_user_edit_rate 0.1000" in output

    def test_export_includes_mape_when_present(self):
        """Test MAPE is included when available."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=0.25,
            avg_confidence=0.80,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_model_mape 0.2500" in output
        assert "rlhf_model_confidence 0.8000" in output

    def test_export_excludes_mape_when_none(self):
        """Test MAPE is excluded when None."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=None,
            avg_confidence=None,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_model_mape" not in output
        assert "rlhf_model_confidence" not in output

    def test_export_health_status(self):
        """Test health status is exported as numeric."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=None,
            avg_confidence=None,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_model_health 0" in output  # healthy = 0
        assert "rlhf_pipeline_health 0" in output

    def test_export_health_status_degraded(self):
        """Test degraded health status."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=None,
            avg_confidence=None,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="degraded",
            data_pipeline_health="healthy",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_model_health 1" in output  # degraded = 1

    def test_export_health_status_critical(self):
        """Test critical health status."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=None,
            avg_confidence=None,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="critical",
            data_pipeline_health="critical",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_model_health 2" in output  # critical = 2
        assert "rlhf_pipeline_health 2" in output

    def test_export_alerts_count(self):
        """Test alerts count is exported."""
        alerts = [
            Alert(
                alert_type=AlertType.MODEL_DRIFT,
                severity=AlertSeverity.WARNING,
                message="Test",
            ),
            Alert(
                alert_type=AlertType.HIGH_EDIT_RATE,
                severity=AlertSeverity.CRITICAL,
                message="Test",
            ),
        ]

        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=None,
            avg_confidence=None,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=alerts,
        )

        output = export_prometheus_metrics(metrics)

        assert "rlhf_alerts_count 2" in output

    def test_export_prometheus_format(self):
        """Test output follows Prometheus format."""
        metrics = DashboardMetrics(
            total_predictions=100,
            total_corrections=10,
            total_preference_pairs=50,
            active_experiment=None,
            current_mape=None,
            avg_confidence=None,
            user_edit_rate=0.10,
            predictions_by_day=[],
            corrections_by_day=[],
            mape_trend=[],
            model_health="healthy",
            data_pipeline_health="healthy",
            alerts=[],
        )

        output = export_prometheus_metrics(metrics)

        # Check for Prometheus format: # HELP, # TYPE, metric_name value
        assert "# HELP rlhf_predictions_total" in output
        assert "# TYPE rlhf_predictions_total counter" in output


class TestAlertTypes:
    """Tests for alert type enum."""

    def test_all_alert_types_defined(self):
        """Test all expected alert types exist."""
        assert AlertType.MODEL_DRIFT.value == "model_drift"
        assert AlertType.HIGH_EDIT_RATE.value == "high_edit_rate"
        assert AlertType.LOW_CONFIDENCE.value == "low_confidence"
        assert AlertType.TRAINING_FAILURE.value == "training_failure"
        assert AlertType.EXPERIMENT_KILLED.value == "experiment_killed"
        assert AlertType.DATA_QUALITY.value == "data_quality"
        assert AlertType.RETRAIN_NEEDED.value == "retrain_needed"


class TestAlertSeverities:
    """Tests for alert severity enum."""

    def test_all_severities_defined(self):
        """Test all expected severities exist."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"

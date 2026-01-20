"""
FPT Cost Brain 2.0 - RLHF Retrainer Tests
Unit tests for sample weighting, rare tag detection, and trigger checking
"""

import pytest
from datetime import datetime, timezone, timedelta

from ml.rlhf_retrainer import (
    TrainingSample,
    get_rare_tags,
    calculate_sample_weight,
    RetrainTriggerChecker,
    RetrainDatasetBuilder,
    RETRAIN_TRIGGERS,
)


def make_sample(
    cost: float = 1000,
    source: str = "user_correction",
    tags: list[str] | None = None,
    days_old: int = 0,
) -> TrainingSample:
    """Helper to create test samples."""
    return TrainingSample(
        features={"feature1": 1, "feature2": 0},
        cost=cost,
        source=source,
        tags=tags or ["MK08"],
        created_at=datetime.now(timezone.utc) - timedelta(days=days_old),
        session_id="test-session",
    )


class TestRareTagDetection:
    """Tests for rare tag identification."""

    def test_identifies_rare_tags(self):
        """Test rare tags are correctly identified."""
        # 10 samples with MK08, 1 with NEW_FAMILY
        samples = [make_sample(tags=["MK08"]) for _ in range(10)]
        samples.append(make_sample(tags=["NEW_FAMILY"]))

        rare = get_rare_tags(samples, threshold=0.15)  # <15%

        assert "NEW_FAMILY" in rare
        assert "MK08" not in rare

    def test_empty_samples(self):
        """Test empty samples returns empty set."""
        assert get_rare_tags([]) == set()

    def test_all_common_tags(self):
        """Test no rare tags when all common."""
        samples = [make_sample(tags=["MK08", "CURSOR"]) for _ in range(10)]
        rare = get_rare_tags(samples, threshold=0.05)
        assert len(rare) == 0


class TestSampleWeighting:
    """Tests for sample weight calculation."""

    def test_actual_outcome_highest_confidence(self):
        """Test actual_outcome gets highest confidence weight."""
        rare_tags = set()
        actual = make_sample(source="actual_outcome")
        correction = make_sample(source="user_correction")

        w_actual = calculate_sample_weight(actual, rare_tags, 1000, 200)
        w_correction = calculate_sample_weight(correction, rare_tags, 1000, 200)

        assert w_actual > w_correction

    def test_recency_decay(self):
        """Test older samples get lower weight."""
        rare_tags = set()
        new = make_sample(days_old=0)
        old = make_sample(days_old=90)  # 3 months

        w_new = calculate_sample_weight(new, rare_tags, 1000, 200)
        w_old = calculate_sample_weight(old, rare_tags, 1000, 200)

        assert w_new > w_old

    def test_innovation_outlier_not_penalized(self):
        """Test outliers with rare tags are not penalized."""
        rare_tags = {"NEW_TURBO_SYSTEM"}

        # Outlier WITH rare tag (innovation)
        innovation = make_sample(cost=5000, tags=["NEW_TURBO_SYSTEM"])
        # Outlier WITHOUT rare tag (noise)
        noise = make_sample(cost=5000, tags=["MK08"])

        w_innovation = calculate_sample_weight(innovation, rare_tags, 1000, 500)
        w_noise = calculate_sample_weight(noise, rare_tags, 1000, 500)

        # Innovation should have higher weight than noise outlier
        assert w_innovation > w_noise

    def test_minimum_weight(self):
        """Test weight never goes below minimum."""
        rare_tags = set()
        # Very old, low confidence, outlier
        sample = make_sample(
            cost=10000,  # Extreme outlier
            source="rule_inference",  # Low confidence
            days_old=365,  # 1 year old
        )

        weight = calculate_sample_weight(sample, rare_tags, 1000, 500)
        assert weight >= 0.1


class TestRetrainTriggers:
    """Tests for retrain trigger checking."""

    def test_force_retrain_on_many_samples(self):
        """Test force retrain when sample count exceeds threshold."""
        checker = RetrainTriggerChecker(last_retrain_date=datetime.now(timezone.utc))

        status = checker.check_triggers(
            new_sample_count=RETRAIN_TRIGGERS["force_retrain_samples"],
            avg_correction_pct=0.05,
        )

        assert status.should_retrain
        assert "Force retrain" in status.reason

    def test_time_trigger(self):
        """Test retrain triggered by time since last retrain."""
        old_retrain = datetime.now(timezone.utc) - timedelta(days=15)
        checker = RetrainTriggerChecker(last_retrain_date=old_retrain)

        status = checker.check_triggers(
            new_sample_count=RETRAIN_TRIGGERS["min_new_samples"],
            avg_correction_pct=0.05,
        )

        assert status.should_retrain
        assert "Time trigger" in status.reason

    def test_drift_trigger(self):
        """Test retrain triggered by MAPE drift."""
        checker = RetrainTriggerChecker(
            last_retrain_date=datetime.now(timezone.utc),
            production_mape=0.20,  # 20% baseline
        )

        status = checker.check_triggers(
            new_sample_count=RETRAIN_TRIGGERS["min_new_samples"],
            avg_correction_pct=0.05,
            current_mape=0.30,  # 50% increase
        )

        assert status.should_retrain
        assert status.drift_detected
        assert "Drift" in status.reason

    def test_high_correction_rate_trigger(self):
        """Test retrain triggered by high correction rate."""
        checker = RetrainTriggerChecker(last_retrain_date=datetime.now(timezone.utc))

        status = checker.check_triggers(
            new_sample_count=RETRAIN_TRIGGERS["min_new_samples"],
            avg_correction_pct=0.20,  # 20% average correction
        )

        assert status.should_retrain
        assert "correction rate" in status.reason

    def test_no_trigger_insufficient_samples(self):
        """Test no trigger when samples below minimum."""
        checker = RetrainTriggerChecker(
            last_retrain_date=datetime.now(timezone.utc) - timedelta(days=20)
        )

        status = checker.check_triggers(
            new_sample_count=RETRAIN_TRIGGERS["min_new_samples"] - 1,
            avg_correction_pct=0.05,
        )

        # Time trigger exists but samples insufficient
        assert not status.should_retrain


class TestDatasetBuilder:
    """Tests for training dataset construction."""

    def test_build_with_new_samples_only(self):
        """Test building dataset with only new samples."""
        builder = RetrainDatasetBuilder()
        samples = [make_sample() for _ in range(5)]

        df = builder.build_training_set(samples)

        assert len(df) == 5
        assert "weight" in df.columns
        assert all(df["is_new"])

    def test_empty_samples_returns_historical(self):
        """Test empty samples returns historical baseline."""
        builder = RetrainDatasetBuilder()
        df = builder.build_training_set([])

        # Should return empty historical (no baseline loaded)
        assert len(df) == 0 or "features" in df.columns

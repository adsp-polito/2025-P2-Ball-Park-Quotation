"""
FPT Cost Brain 2.0 - RLHF Service Tests
Unit tests for preference pair generation and reward calculation
"""

import pytest
from services.rlhf_service import (
    calculate_reward,
    generate_synthetic_negative,
    compute_context_hash,
    RewardResult,
)


class TestRewardCalculation:
    """Tests for reward calculation with [-1, 1] scale."""

    def test_perfect_prediction(self):
        """Test perfect prediction gets +1.0 reward."""
        result = calculate_reward(1000, 1000)
        assert result.reward == 1.0
        assert result.error_pct == 0.0
        assert not result.is_severe_error

    def test_good_prediction_25_percent_error(self):
        """Test 25% error gets +0.5 reward."""
        result = calculate_reward(1250, 1000)
        assert result.reward == 0.5
        assert result.error_pct == 0.25
        assert not result.is_severe_error

    def test_neutral_prediction_50_percent_error(self):
        """Test 50% error gets 0.0 reward (neutral)."""
        result = calculate_reward(1500, 1000)
        assert result.reward == 0.0
        assert result.error_pct == 0.5
        assert not result.is_severe_error

    def test_bad_prediction_75_percent_error(self):
        """Test 75% error gets -0.5 reward."""
        result = calculate_reward(1750, 1000)
        assert result.reward == -0.5
        assert result.error_pct == 0.75
        assert result.is_severe_error

    def test_hallucination_100_percent_error(self):
        """Test 100% error gets -1.0 reward (hallucination)."""
        result = calculate_reward(2000, 1000)
        assert result.reward == -1.0
        assert result.error_pct == 1.0
        assert result.is_severe_error

    def test_extreme_overestimate_capped(self):
        """Test reward is capped at -1.0 for extreme errors."""
        result = calculate_reward(5000, 1000)  # 400% error
        assert result.reward == -1.0
        assert result.error_pct == 4.0
        assert result.is_severe_error

    def test_zero_target_returns_neutral(self):
        """Test zero target returns neutral reward."""
        result = calculate_reward(1000, 0)
        assert result.reward == 0.0
        assert result.error_pct == 0.0
        assert not result.is_severe_error

    def test_none_target_returns_neutral(self):
        """Test None target returns neutral reward."""
        result = calculate_reward(1000, None)
        assert result.reward == 0.0
        assert result.error_pct == 0.0
        assert not result.is_severe_error

    def test_underestimate_same_as_overestimate(self):
        """Test underestimate penalized same as overestimate."""
        over = calculate_reward(1250, 1000)  # +25%
        under = calculate_reward(750, 1000)  # -25%
        assert over.reward == under.reward == 0.5


class TestSyntheticNegativeGeneration:
    """Tests for synthetic negative pair generation."""

    def test_swap_strategy(self):
        """Test swap strategy swaps two cluster values."""
        breakdown = {"hardware": 500, "calibration": 300, "testing": 200}

        # Run multiple times to test randomness
        swapped_count = 0
        for _ in range(20):
            synthetic = generate_synthetic_negative(breakdown)
            # Check that values are rearranged or inflated/deflated
            if set(synthetic.values()) != set(breakdown.values()):
                # Inflation or deflation happened
                pass
            else:
                # Values might be swapped
                if synthetic != breakdown:
                    swapped_count += 1

        # At least some should be different
        assert swapped_count > 0 or any(
            generate_synthetic_negative(breakdown) != breakdown for _ in range(10)
        )

    def test_inflation_strategy(self):
        """Test inflation increases values by 30-50%."""
        breakdown = {"hardware": 100, "calibration": 100}

        inflated = False
        for _ in range(50):
            synthetic = generate_synthetic_negative(breakdown)
            total_original = sum(breakdown.values())
            total_synthetic = sum(synthetic.values())
            if total_synthetic > total_original * 1.2:
                inflated = True
                break

        assert inflated, "Inflation strategy should increase total by 30-50%"

    def test_deflation_strategy(self):
        """Test deflation reduces values by 25-40%."""
        breakdown = {"hardware": 100, "calibration": 100}

        deflated = False
        for _ in range(50):
            synthetic = generate_synthetic_negative(breakdown)
            total_original = sum(breakdown.values())
            total_synthetic = sum(synthetic.values())
            if total_synthetic < total_original * 0.8:
                deflated = True
                break

        assert deflated, "Deflation strategy should decrease total by 25-40%"

    def test_empty_breakdown(self):
        """Test empty breakdown returns empty."""
        breakdown = {}
        synthetic = generate_synthetic_negative(breakdown)
        assert synthetic == {}

    def test_single_cluster(self):
        """Test single cluster only uses inflate/deflate."""
        breakdown = {"hardware": 100}

        different_count = 0
        for _ in range(20):
            synthetic = generate_synthetic_negative(breakdown)
            if synthetic != breakdown:
                different_count += 1

        # Should mostly be different (inflated or deflated)
        assert different_count > 10

    def test_non_numeric_values_preserved(self):
        """Test non-numeric values are preserved."""
        breakdown = {"hardware": 100, "notes": "some text", "flag": True}
        synthetic = generate_synthetic_negative(breakdown)

        # Non-numeric values should be unchanged
        assert synthetic.get("notes") == "some text"
        assert synthetic.get("flag") == True


class TestContextHash:
    """Tests for context hash computation."""

    def test_same_context_same_hash(self):
        """Test identical contexts produce same hash."""
        ctx1 = {"pr_id": "123", "family": "MK08"}
        ctx2 = {"pr_id": "123", "family": "MK08"}
        assert compute_context_hash(ctx1) == compute_context_hash(ctx2)

    def test_different_context_different_hash(self):
        """Test different contexts produce different hashes."""
        ctx1 = {"pr_id": "123", "family": "MK08"}
        ctx2 = {"pr_id": "456", "family": "MK08"}
        assert compute_context_hash(ctx1) != compute_context_hash(ctx2)

    def test_hash_is_64_chars(self):
        """Test hash is truncated to 64 characters."""
        ctx = {"pr_id": "123"}
        hash_value = compute_context_hash(ctx)
        assert len(hash_value) == 64

    def test_order_independent(self):
        """Test hash is order-independent (sorted keys)."""
        ctx1 = {"a": 1, "b": 2, "c": 3}
        ctx2 = {"c": 3, "a": 1, "b": 2}
        assert compute_context_hash(ctx1) == compute_context_hash(ctx2)


class TestRewardResultDataclass:
    """Tests for RewardResult dataclass."""

    def test_dataclass_creation(self):
        """Test RewardResult can be created."""
        result = RewardResult(reward=0.5, error_pct=0.25, is_severe_error=False)
        assert result.reward == 0.5
        assert result.error_pct == 0.25
        assert result.is_severe_error is False

    def test_severe_error_threshold(self):
        """Test severe error is correctly identified at >50% error."""
        # 49% error - not severe
        result1 = calculate_reward(1490, 1000)
        assert not result1.is_severe_error

        # 51% error - severe
        result2 = calculate_reward(1510, 1000)
        assert result2.is_severe_error

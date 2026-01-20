"""
Unit tests for PR Embedding Text module.
Verifies unified format consistency and feature similarity calculation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.pr_embedding_text import (
    build_pr_embedding_text,
    calculate_feature_similarity,
    calculate_ensemble_score,
    get_sizing_filter_values,
    SIZING_PROXIMITY,
)


class TestBuildPREmbeddingText:
    """Test build_pr_embedding_text function for deterministic output."""

    def test_identical_output_for_same_input(self):
        """Same PR dict should produce identical text every time."""
        pr = {
            "sector": "CE",
            "customer_platform": "Wheel Loader",
            "product_family": "E0C0",
            "emissions": "Stage V",
            "sizing": "Small",
            "pr_type": "BOM",
            "hardware_change": 1,
            "calibration_change": 0,
            "ats_change": 1,
            "software_vcu_change": 0,
            "pr_name": "21031_C",
        }

        text1 = build_pr_embedding_text(pr)
        text2 = build_pr_embedding_text(pr)

        assert text1 == text2, "Same input should produce identical output"

    def test_contains_all_required_fields(self):
        """Output should contain all required fields in correct format."""
        pr = {
            "sector": "AG",
            "customer_platform": "Tractor",
            "product_family": "E0N0",
            "emissions": "Tier 4B",
            "sizing": "Mid",
            "pr_type": "New engine",
        }

        text = build_pr_embedding_text(pr)

        assert "Sector: AG" in text
        assert "Platform: Tractor" in text
        assert "Engine family: E0N0" in text
        assert "Emissions: Tier 4B" in text
        assert "Sizing: Mid" in text
        assert "Type: New engine" in text

    def test_boolean_flags_yes_no_format(self):
        """Boolean flags should be converted to yes/no."""
        pr_with_changes = {
            "hardware_change": 1,
            "calibration_change": True,
            "ats_change": "yes",
            "software_vcu_change": 0,
        }

        text = build_pr_embedding_text(pr_with_changes)

        assert "Hardware change: yes" in text
        assert "Calibration change: yes" in text
        assert "ATS change: yes" in text
        assert "Software VCU change: no" in text

    def test_missing_fields_use_default(self):
        """Missing fields should use 'Unknown' default."""
        pr_minimal = {"pr_name": "TEST_001"}

        text = build_pr_embedding_text(pr_minimal)

        assert "Sector: Unknown" in text
        assert "Platform: Unknown" in text

    def test_handles_nan_values(self):
        """NaN values should be treated as Unknown."""
        pr_with_nan = {
            "sector": "nan",
            "customer_platform": None,
            "emissions": "NaN",
        }

        text = build_pr_embedding_text(pr_with_nan)

        assert "Sector: Unknown" in text
        assert "Platform: Unknown" in text
        assert "Emissions: Unknown" in text

    def test_field_order_is_fixed(self):
        """Fields should appear in fixed order for consistency."""
        pr = {
            "sector": "CE",
            "customer_platform": "Loader",
            "product_family": "E5F0",
        }

        text = build_pr_embedding_text(pr)

        sector_pos = text.find("Sector:")
        platform_pos = text.find("Platform:")
        family_pos = text.find("Engine family:")

        assert sector_pos < platform_pos < family_pos, "Fields should be in fixed order"


class TestCalculateFeatureSimilarity:
    """Test feature-based similarity calculation."""

    def test_identical_prs_score_high(self):
        """Identical PRs should have high similarity score."""
        pr1 = {
            "sector": "CE",
            "sizing": "Small",
            "product_family": "E5F0",
            "customer_platform": "Wheel Loader",
            "emissions": "Stage V",
            "hardware_change": 1,
            "calibration_change": 1,
        }
        pr2 = pr1.copy()

        score = calculate_feature_similarity(pr1, pr2)

        assert score >= 0.8, f"Identical PRs should score >= 0.8, got {score}"

    def test_different_sector_lower_score(self):
        """Different sector should significantly reduce score."""
        pr1 = {"sector": "CE", "sizing": "Small"}
        pr2 = {"sector": "AG", "sizing": "Small"}

        score = calculate_feature_similarity(pr1, pr2)

        # Sector weight is 0.25, so score should be reduced
        assert score < 0.8, f"Different sector should reduce score, got {score}"

    def test_adjacent_sizing_partial_match(self):
        """Adjacent sizing should get partial score."""
        pr1 = {"sizing": "Small"}
        pr2 = {"sizing": "Mid"}  # Adjacent to Small

        score1 = calculate_feature_similarity(pr1, pr2)

        pr3 = {"sizing": "Large"}  # Not adjacent to Small
        score2 = calculate_feature_similarity(pr1, pr3)

        assert score1 > score2, "Adjacent sizing should score higher than distant"

    def test_unknown_values_no_match(self):
        """Unknown values should not contribute to sector match bonus."""
        pr1 = {"sector": "CE", "hardware_change": 1}
        pr2 = {"sector": "Unknown", "hardware_change": 0}

        score = calculate_feature_similarity(pr1, pr2)

        # Should not get sector match bonus (0.25), but may get some flag matches
        # Since hardware_change differs, score should be lower than full match
        assert score < 0.9, (
            f"Unknown sector and different flags should not score high: {score}"
        )


class TestCalculateEnsembleScore:
    """Test ensemble scoring (0.6 feature + 0.4 vector)."""

    def test_weights_applied_correctly(self):
        """Ensemble should use 0.6/0.4 weights."""
        vector_score = 1.0
        feature_score = 0.0

        ensemble = calculate_ensemble_score(vector_score, feature_score)

        # 0.6 * 0.0 + 0.4 * 1.0 = 0.4
        assert abs(ensemble - 0.4) < 0.001

    def test_perfect_scores(self):
        """Perfect vector and feature should give 1.0."""
        ensemble = calculate_ensemble_score(1.0, 1.0)
        assert abs(ensemble - 1.0) < 0.001

    def test_feature_weighted_higher(self):
        """Feature score should have more weight than vector."""
        # High feature, low vector
        score1 = calculate_ensemble_score(vector_score=0.5, feature_score=0.9)
        # Low feature, high vector
        score2 = calculate_ensemble_score(vector_score=0.9, feature_score=0.5)

        assert score1 > score2, "Feature should be weighted higher (0.6 vs 0.4)"


class TestSizingProximity:
    """Test sizing filter values."""

    def test_small_includes_adjacent(self):
        """Small sizing should include X-small, Small, Mid."""
        values = get_sizing_filter_values("Small")
        assert "X-small" in values
        assert "Small" in values
        assert "Mid" in values

    def test_unknown_includes_all(self):
        """Unknown sizing should include all values."""
        values = get_sizing_filter_values("Unknown")
        assert len(values) >= 5

    def test_proximity_groups_defined(self):
        """All standard sizing values should have proximity groups."""
        for sizing in ["X-small", "Small", "Mid", "Large", "Full"]:
            assert sizing in SIZING_PROXIMITY


class TestIndexQueryConsistency:
    """Integration test: verify index and query produce same text."""

    def test_csv_format_matches_parsed_pr_format(self):
        """
        PR data from CSV (indexing) should produce same text as
        parsed PR dict (querying).
        """
        # Simulated CSV row (as used in populate_pr_embeddings_v3.py)
        csv_row = {
            "pr_name": "21031_C",
            "sector": "CE",
            "customer_platform": "Compact Wheel Loader",
            "product_family": "E5F0",
            "emissions": "Stage V",
            "sizing": "Small",
            "pr_type": "BOM",
            "hardware_change": 1,
            "calibration_change": 1,
            "ats_change": 0,
            "software_vcu_change": 0,
        }

        # Simulated parsed PR (as used in summary_node.py)
        parsed_pr = {
            "pr_code": "21031_C",
            "sector": "CE",
            "customer_platform": "Compact Wheel Loader",
            "product_family": "E5F0",
            "emissions": "Stage V",
            "sizing": "Small",
            "pr_type": "BOM",
            "hardware_change": True,
            "calibration_change": True,
            "ats_change": False,
            "software_vcu_change": False,
        }

        text_from_csv = build_pr_embedding_text(csv_row)
        text_from_parsed = build_pr_embedding_text(parsed_pr)

        assert text_from_csv == text_from_parsed, (
            f"Index and query should produce identical text!\n"
            f"CSV: {text_from_csv}\n"
            f"Parsed: {text_from_parsed}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

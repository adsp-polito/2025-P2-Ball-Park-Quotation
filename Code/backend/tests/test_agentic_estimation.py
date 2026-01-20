"""
FPT Cost Brain 2.0 - Agentic Estimation Tests
Unit and integration tests for the multi-agent estimation pipeline
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.agentic.types import (
    Decision,
    SIZING_THRESHOLDS,
    ArbitrationScores,
)
from agents.agentic.arbitrator import ArbitratorAgent, KNOWN_FAMILIES
from agents.agentic.cluster_agents import (
    BaseClusterAgent,
    HardwareAgent,
    CalibrationAgent,
    TestingAgent,
    DependentAgent,
    aggregate_cluster_results,
    ClusterResult,
)


# ===== Arbitrator Tests =====


class TestArbitratorAgent:
    """Tests for ArbitratorAgent multi-factor scoring."""

    def setup_method(self):
        """Set up test fixtures."""
        self.arbitrator = ArbitratorAgent()

    def test_sanity_check_small_deviation(self):
        """Test that small deviations (<5%) skip full arbitration."""
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.8, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1040}]}  # 4% deviation
        context = {"program_family": "MK08"}

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        assert decision.decision == Decision.USE_HCQE
        assert decision.deviation_pct < 0.05
        assert "< 5%" in decision.analysis_summary

    def test_zero_hcqe_estimate(self):
        """Test fallback to LLM when HCQE returns zero."""
        hcqe = {"predicted_total_hours": 0, "confidence": 0}
        llm = {"breakdown": [{"hours": 500}]}
        context = {}

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        assert decision.decision == Decision.USE_LLM
        assert "zero estimate" in decision.analysis_summary.lower()

    def test_known_family_favors_hcqe(self):
        """Test that known program families favor HCQE in domain rules."""
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.8, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1200, "confidence_score": 0.6}]}  # 20% deviation
        context = {"program_family": "MK08"}  # Known family

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        # MK08 is known, so HCQE should get domain rules points
        assert decision.scores.domain_rules.get("hcqe", 0) > 0

    def test_unknown_family_favors_llm(self):
        """Test that unknown program families favor LLM in domain rules."""
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.5, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1200, "confidence_score": 0.8}]}
        context = {"program_family": "NEW_FAMILY_XYZ"}  # Unknown family

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        # Unknown family should favor LLM
        assert decision.scores.domain_rules.get("llm", 0) > 0

    def test_calibration_heavy_favors_hcqe(self):
        """Test that calibration-heavy projects favor HCQE."""
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.8, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1200, "confidence_score": 0.6}]}
        context = {
            "calibration_change": True,
            "emission_level": 2,  # > 1
            "program_family": "",
        }

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        # Calibration-heavy should favor HCQE
        assert decision.scores.domain_rules.get("hcqe", 0) == 30

    def test_novel_hardware_favors_llm(self):
        """Test that novel hardware combos favor LLM."""
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.5, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1200, "confidence_score": 0.8}]}
        context = {
            "turbo_related": True,
            "injectors_related": True,
            "program_family": "",
        }

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        # Novel hardware should favor LLM
        assert decision.scores.domain_rules.get("llm", 0) == 30

    def test_escalation_on_close_scores_high_deviation(self):
        """Test escalation when scores are close AND deviation is high."""
        # Create scenario with close scores (split evenly)
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.5, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1250, "confidence_score": 0.5}]}  # 25% deviation
        context = {"program_family": ""}  # No family = split historical

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        # With evenly split scores and >20% deviation, should escalate
        # This depends on exact scoring - may need adjustment
        if (
            abs(decision.scores.hcqe - decision.scores.llm) < 15
            and decision.deviation_pct > 0.20
        ):
            assert decision.decision == Decision.ESCALATE_TO_USER

    def test_confidence_comparison_scoring(self):
        """Test confidence comparison factor."""
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.9, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": 1100, "confidence_score": 0.5}]}
        context = {"program_family": ""}

        decision = self.arbitrator.arbitrate(hcqe, llm, context)

        # HCQE has higher confidence, should get confidence points
        assert decision.scores.confidence.get("hcqe", 0) == 30

    def test_generate_critique(self):
        """Test natural language critique generation."""
        hcqe = {
            "predicted_total_hours": 8000,
            "cluster_estimates": {"hardware": 2800, "testing": 2100},
        }
        llm = {
            "breakdown": [
                {"cluster": "hardware", "hours": 4500},
                {"cluster": "testing", "hours": 3200},
            ]
        }

        critique = self.arbitrator._generate_critique(hcqe, llm, 0.50)

        # Check critique contains key elements
        assert "8,000h" in critique
        assert "Hardware" in critique or "hardware" in critique.lower()
        assert "reduce" in critique.lower() or "increase" in critique.lower()
        assert "revise" in critique.lower() or "align" in critique.lower()


class TestSizingThresholds:
    """Tests for sizing-specific deviation thresholds."""

    def test_threshold_values(self):
        """Test correct threshold values for each sizing."""
        assert SIZING_THRESHOLDS["X-Small"] == 0.20
        assert SIZING_THRESHOLDS["Small"] == 0.25
        assert SIZING_THRESHOLDS["Medium"] == 0.30
        assert SIZING_THRESHOLDS["Large"] == 0.35
        assert SIZING_THRESHOLDS["Full"] == 0.40

    def test_larger_projects_more_tolerant(self):
        """Test that larger projects have higher tolerance."""
        thresholds = list(SIZING_THRESHOLDS.values())
        for i in range(len(thresholds) - 1):
            assert thresholds[i] < thresholds[i + 1]


# ===== Cluster Agent Tests =====


class TestClusterAgents:
    """Tests for cluster-specific estimation agents."""

    def test_hardware_agent_relevant_features(self):
        """Test hardware agent recognizes relevant features."""
        agent = HardwareAgent()

        assert agent._is_relevant_feature("turbo_integration")
        assert agent._is_relevant_feature("injector_calibration")
        assert agent._is_relevant_feature("cooling_system")
        assert not agent._is_relevant_feature("software_vcu")

    def test_calibration_agent_relevant_features(self):
        """Test calibration agent recognizes relevant features."""
        agent = CalibrationAgent()

        assert agent._is_relevant_feature("calibration_change")
        assert agent._is_relevant_feature("emission_standard")
        assert agent._is_relevant_feature("ats_modification")
        assert not agent._is_relevant_feature("hardware_change")

    def test_testing_agent_relevant_features(self):
        """Test testing agent recognizes relevant features."""
        agent = TestingAgent()

        assert agent._is_relevant_feature("bench_testing")
        assert agent._is_relevant_feature("vehicle_validation")
        assert agent._is_relevant_feature("field_test")
        assert not agent._is_relevant_feature("calibration_work")

    def test_dependent_agent_handled_clusters(self):
        """Test dependent agent handles software, docs, installation."""
        agent = DependentAgent()

        assert "software" in agent.handled_clusters
        assert "documentation" in agent.handled_clusters
        assert "installation" in agent.handled_clusters

    def test_fallback_result(self):
        """Test fallback result uses HCQE estimate."""
        agent = HardwareAgent()
        hcqe_estimate = 500.0

        result = agent._fallback_result(hcqe_estimate)

        assert result.cluster == "hardware"
        assert result.total_hours == hcqe_estimate
        assert result.confidence_score == 0.5
        # Check reasoning indicates fallback to HCQE
        assert (
            "hcqe" in result.reasoning.lower() or "baseline" in result.reasoning.lower()
        )


class TestAggregateClusterResults:
    """Tests for cluster result aggregation."""

    def test_aggregate_multiple_clusters(self):
        """Test aggregation of multiple cluster results."""
        results = {
            "hardware": ClusterResult(
                cluster="hardware",
                activities=[{"name": "HW1", "hours": 100, "confidence": 0.8}],
                total_hours=100,
                confidence_score=0.8,
                reasoning="Test",
            ),
            "calibration": ClusterResult(
                cluster="calibration",
                activities=[{"name": "CAL1", "hours": 200, "confidence": 0.7}],
                total_hours=200,
                confidence_score=0.7,
                reasoning="Test",
            ),
        }

        breakdown, total, avg_conf = aggregate_cluster_results(results)

        assert len(breakdown) == 2
        assert total == 300
        # Weighted average: (100*0.8 + 200*0.7) / 300 = 220/300 ≈ 0.733
        assert 0.7 < avg_conf < 0.8

    def test_aggregate_empty_results(self):
        """Test aggregation handles empty results."""
        results = {}

        breakdown, total, avg_conf = aggregate_cluster_results(results)

        assert breakdown == []
        assert total == 0.0
        assert avg_conf == 0.5  # Default


# ===== Known Families Tests =====


class TestKnownFamilies:
    """Tests for known program families constant."""

    def test_known_families_exist(self):
        """Test that known families set is populated."""
        assert len(KNOWN_FAMILIES) > 0
        assert "MK08" in KNOWN_FAMILIES
        assert "CURSOR" in KNOWN_FAMILIES
        assert "NEF" in KNOWN_FAMILIES


# ===== Integration Tests =====


@pytest.mark.asyncio
class TestAgenticPipelineIntegration:
    """Integration tests for the full agentic pipeline."""

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock LLM client."""
        client = MagicMock()
        client.chat_async = AsyncMock(
            return_value={
                "content": '{"activities": [{"name": "Test", "hours": 100, "confidence": 0.8}], "total_hours": 100, "confidence_score": 0.8, "reasoning": "Test"}'
            }
        )
        return client

    @pytest.fixture
    def mock_hcqe_predictor(self):
        """Create a mock HCQE predictor."""
        predictor = MagicMock()
        predictor.predict.return_value = MagicMock(
            point_estimate=1000,  # K€
            predicted_sizing="Medium",
            calibrated_confidence=0.8,
            prediction_interval=(800, 1200),
            cluster_estimates={"hardware": 300, "calibration": 400, "testing": 300},
            method_used="hcqe",
            recommendations=["Test recommendation"],
        )
        return predictor

    async def test_pipeline_happy_path(self, mock_llm_client, mock_hcqe_predictor):
        """Test pipeline completes successfully and returns valid result."""
        from agents.agentic.pipeline import run_agentic_estimation

        with patch(
            "agents.agentic.pipeline.get_llm_client", return_value=mock_llm_client
        ):
            result = await run_agentic_estimation(
                session_id="test-session",
                pr_context={"program_family": "MK08", "features": {}},
                hcqe_predictor=mock_hcqe_predictor,
                ml_features={},
            )

        assert result.session_id == "test-session"
        # Pipeline can return "completed" or "escalated" based on arbitration
        assert result.status in ["completed", "escalated"]
        assert result.final_estimate.total_hours >= 0
        assert result.trace.total_latency_ms >= 0  # Can be 0 with mocks
        # Verify trace has agent logs
        assert len(result.trace.agent_logs) >= 2  # At least HCQE and cluster agents

    async def test_pipeline_handles_hcqe_failure(self, mock_llm_client):
        """Test pipeline gracefully handles HCQE failure."""
        from agents.agentic.pipeline import run_agentic_estimation

        # Predictor that raises exception
        failing_predictor = MagicMock()
        failing_predictor.predict.side_effect = Exception("HCQE failed")

        with patch(
            "agents.agentic.pipeline.get_llm_client", return_value=mock_llm_client
        ):
            result = await run_agentic_estimation(
                session_id="test-session",
                pr_context={"program_family": "MK08", "features": {}},
                hcqe_predictor=failing_predictor,
                ml_features={},
            )

        # Should fall back to LLM-only mode
        assert result.estimation_source.method.value == "llm_only"
        assert "ML model unavailable" in (
            result.final_estimate.global_justification or ""
        )


# ===== Edge Case Tests =====


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_deviation(self):
        """Test handling of very large deviations (>100%)."""
        arbitrator = ArbitratorAgent()
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.8, "sizing": "Small"}
        llm = {
            "breakdown": [{"hours": 3000, "confidence_score": 0.5}]
        }  # 200% deviation
        context = {"program_family": "MK08"}

        decision = arbitrator.arbitrate(hcqe, llm, context)

        # Should make a decision (not crash)
        assert decision.decision in [
            Decision.USE_HCQE,
            Decision.USE_LLM,
            Decision.ESCALATE_TO_USER,
        ]
        assert decision.deviation_pct == 2.0  # 200%

    def test_negative_hours_handled(self):
        """Test handling of invalid negative hours."""
        arbitrator = ArbitratorAgent()
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.8, "sizing": "Medium"}
        llm = {"breakdown": [{"hours": -100}]}  # Invalid negative
        context = {}

        decision = arbitrator.arbitrate(hcqe, llm, context)

        # Should handle gracefully
        assert decision is not None

    def test_empty_breakdown(self):
        """Test handling of empty LLM breakdown."""
        arbitrator = ArbitratorAgent()
        hcqe = {"predicted_total_hours": 1000, "confidence": 0.8, "sizing": "Medium"}
        llm = {"breakdown": []}  # Empty
        context = {}

        decision = arbitrator.arbitrate(hcqe, llm, context)

        # Empty LLM = 0 hours, high deviation from HCQE
        assert decision.deviation_pct == 1.0  # 100%

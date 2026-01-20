"""
Unit tests for SizingService - Rule-based sizing classifier.

Tests cover:
1. Rule loading from ref_sizing.json
2. Keyword matching (deterministic fallback)
3. Aggregation logic
4. Edge cases and defaults
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from services.sizing_service import (
    SizingService,
    SizingRule,
    SizingResult,
    ProgramSizingResult,
    SizingLevel,
    SizingMethod,
    DOMAIN_MAPPING,
    DOMAIN_PREFIXES,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_rules_json():
    """Sample ref_sizing.json data for testing."""
    return {
        "document_type": "reference",
        "title": "Program Sizing Reference",
        "sizing_rules": [
            {
                "function": "Product Engineering",
                "sub_function": "Base Engine",
                "development_effort": "New concept required; High level of New Content (NC); High validation effort.",
                "sizing": "Full",
            },
            {
                "function": "Product Engineering",
                "sub_function": "Base Engine",
                "development_effort": "Heavy modification with impact on manufacturing; High/Medium NC.",
                "sizing": "Large",
            },
            {
                "function": "Product Engineering",
                "sub_function": "Base Engine",
                "development_effort": "Medium modification (no manufacturing impact); Medium NC.",
                "sizing": "Medium",
            },
            {
                "function": "Product Engineering",
                "sub_function": "Base Engine",
                "development_effort": "Light modification; Low NC; Low validation.",
                "sizing": "Small",
            },
            {
                "function": "Product Engineering",
                "sub_function": "Base Engine",
                "development_effort": "Minimum modification (only adaptation); No validation.",
                "sizing": "X-small",
            },
            {
                "function": "Customer Manager",
                "sub_function": "Build stages",
                "development_effort": "All build stages required (Alpha, Beta, Gamma, PP, Pilot)",
                "sizing": "Full",
            },
            {
                "function": "Customer Manager",
                "sub_function": "Build stages",
                "development_effort": "Beta, Gamma, PP, Pilot required",
                "sizing": "Large",
            },
            {
                "function": "Customer Manager",
                "sub_function": "Build stages",
                "development_effort": "Gamma, PP, Pilot required",
                "sizing": "Medium",
            },
        ],
    }


@pytest.fixture
def sizing_service(sample_rules_json, tmp_path):
    """Create SizingService with test rules."""
    rules_file = tmp_path / "ref_sizing.json"
    rules_file.write_text(json.dumps(sample_rules_json))
    return SizingService(rules_path=str(rules_file))


@pytest.fixture
def sample_pr_text_full():
    """PR text that should match Full sizing."""
    return """
    This is a new concept engine development project.
    Requires high level of new content (NC) and high validation effort.
    All build stages required including Alpha, Beta, Gamma, PP, and Pilot.
    First installation with new emission standards.
    """


@pytest.fixture
def sample_pr_text_medium():
    """PR text that should match Medium sizing."""
    return """
    Medium modification of existing engine design.
    No impact on manufacturing process.
    Medium level of calibration changes required.
    Gamma, PP, and Pilot stages planned.
    """


@pytest.fixture
def sample_pr_text_small():
    """PR text that should match Small sizing."""
    return """
    Light modification to existing product.
    Low new content and low validation effort.
    Only PP and Pilot stages required.
    """


# ============================================================================
# RULE LOADING TESTS
# ============================================================================


class TestRuleLoading:
    """Tests for rule loading from ref_sizing.json."""

    def test_loads_rules_from_file(self, sizing_service):
        """Should load rules from JSON file."""
        assert len(sizing_service.rules) == 8
        assert len(sizing_service.rules_by_id) == 8

    def test_generates_unique_rule_ids(self, sizing_service):
        """Each rule should have a unique ID."""
        rule_ids = [r.rule_id for r in sizing_service.rules]
        assert len(rule_ids) == len(set(rule_ids))

    def test_rule_id_format(self, sizing_service):
        """Rule IDs should follow expected format: PREFIX_SIZING_NUM."""
        for rule in sizing_service.rules:
            parts = rule.rule_id.split("_")
            assert len(parts) >= 3
            assert parts[-1].isdigit()

    def test_indexes_rules_by_domain(self, sizing_service):
        """Rules should be indexed by domain key."""
        assert "pe_base_powertrain" in sizing_service.rules_by_domain
        assert "customer_build_stages" in sizing_service.rules_by_domain
        assert len(sizing_service.rules_by_domain["pe_base_powertrain"]) == 5
        assert len(sizing_service.rules_by_domain["customer_build_stages"]) == 3

    def test_extracts_keywords(self, sizing_service):
        """Rules should have extracted keywords."""
        pe_rules = sizing_service.rules_by_domain["pe_base_powertrain"]
        full_rule = [r for r in pe_rules if r.sizing == SizingLevel.FULL][0]
        assert "new" in full_rule.keywords or "high" in full_rule.keywords

    def test_handles_missing_file(self, tmp_path):
        """Should handle missing rules file gracefully."""
        service = SizingService(rules_path=str(tmp_path / "nonexistent.json"))
        assert len(service.rules) == 0

    def test_handles_invalid_json(self, tmp_path):
        """Should handle invalid JSON gracefully."""
        rules_file = tmp_path / "invalid.json"
        rules_file.write_text("{ invalid json }")
        service = SizingService(rules_path=str(rules_file))
        assert len(service.rules) == 0


# ============================================================================
# KEYWORD MATCHING TESTS
# ============================================================================


class TestKeywordMatching:
    """Tests for keyword-based rule matching."""

    def test_matches_full_sizing_keywords(self, sizing_service, sample_pr_text_full):
        """Should match Full sizing for PR with new concept keywords."""
        result = sizing_service._classify_domain_with_keywords(
            "pe_base_powertrain",
            sizing_service.rules_by_domain["pe_base_powertrain"],
            sample_pr_text_full,
            {},
        )
        # Should match Full or Large (high-level keywords)
        assert result.sizing in [SizingLevel.FULL.value, SizingLevel.LARGE.value]
        assert result.method == SizingMethod.KEYWORD.value
        assert result.confidence >= 0.4

    def test_matches_medium_sizing_keywords(
        self, sizing_service, sample_pr_text_medium
    ):
        """Should match Medium sizing for PR with medium modification keywords."""
        result = sizing_service._classify_domain_with_keywords(
            "pe_base_powertrain",
            sizing_service.rules_by_domain["pe_base_powertrain"],
            sample_pr_text_medium,
            {},
        )
        assert result.sizing == SizingLevel.MEDIUM.value
        assert result.method == SizingMethod.KEYWORD.value

    def test_matches_small_sizing_keywords(self, sizing_service, sample_pr_text_small):
        """Should match Small sizing for PR with light modification keywords."""
        result = sizing_service._classify_domain_with_keywords(
            "pe_base_powertrain",
            sizing_service.rules_by_domain["pe_base_powertrain"],
            sample_pr_text_small,
            {},
        )
        # Should match Small or X-small
        assert result.sizing in [SizingLevel.SMALL.value, SizingLevel.X_SMALL.value]

    def test_returns_default_for_no_matches(self, sizing_service):
        """Should return default Medium when no keywords match."""
        result = sizing_service._classify_domain_with_keywords(
            "pe_base_powertrain",
            sizing_service.rules_by_domain["pe_base_powertrain"],
            "This PR contains no relevant keywords whatsoever.",
            {},
        )
        assert result.sizing == SizingLevel.MEDIUM.value
        assert result.method == SizingMethod.DEFAULT.value
        assert result.confidence == 0.5

    def test_case_insensitive_matching(self, sizing_service):
        """Keyword matching should be case insensitive."""
        result = sizing_service._classify_domain_with_keywords(
            "pe_base_powertrain",
            sizing_service.rules_by_domain["pe_base_powertrain"],
            "NEW CONCEPT with HIGH LEVEL modifications",
            {},
        )
        assert result.sizing in [SizingLevel.FULL.value, SizingLevel.LARGE.value]


# ============================================================================
# AGGREGATION TESTS
# ============================================================================


class TestAggregation:
    """Tests for sizing aggregation logic (MODE-based)."""

    def test_aggregates_to_mode_sizing(self, sizing_service):
        """Should aggregate to MODE (most frequent) sizing level."""
        domain_results = {
            "pe_base_powertrain": SizingResult(
                sizing="Small",
                reasoning="test",
                rule_id="TEST_S_001",
                confidence=0.7,
                method="keyword",
            ),
            "pe_system_assembly": SizingResult(
                sizing="Small",
                reasoning="test",
                rule_id="TEST_S_002",
                confidence=0.7,
                method="keyword",
            ),
            "customer_build_stages": SizingResult(
                sizing="Medium",
                reasoning="test",
                rule_id="TEST_M_001",
                confidence=0.8,
                method="keyword",
            ),
        }

        result = sizing_service._aggregate_sizing(domain_results)
        assert result.sizing == "Small"  # Mode: Small:2, Medium:1 → Small

    def test_aggregates_to_full_when_majority(self, sizing_service):
        """Full should be selected when it's the MODE (most frequent)."""
        domain_results = {
            "pe_base_powertrain": SizingResult(
                sizing="Full",
                reasoning="test",
                rule_id="TEST_F_001",
                confidence=0.9,
                method="llm",
            ),
            "pe_system_assembly": SizingResult(
                sizing="Full",
                reasoning="test",
                rule_id="TEST_F_002",
                confidence=0.9,
                method="llm",
            ),
            "customer_build_stages": SizingResult(
                sizing="Small",
                reasoning="test",
                rule_id="TEST_S_001",
                confidence=0.7,
                method="keyword",
            ),
        }

        result = sizing_service._aggregate_sizing(domain_results)
        assert result.sizing == "Full"  # Mode: Full:2, Small:1 → Full

    def test_tie_prefers_smaller_sizing(self, sizing_service):
        """When tied, should prefer smaller sizing (conservative)."""
        domain_results = {
            "pe_base_powertrain": SizingResult(
                sizing="Large",
                reasoning="test",
                rule_id="TEST_L_001",
                confidence=0.8,
                method="keyword",
            ),
            "customer_build_stages": SizingResult(
                sizing="Small",
                reasoning="test",
                rule_id="TEST_S_001",
                confidence=0.7,
                method="keyword",
            ),
        }

        result = sizing_service._aggregate_sizing(domain_results)
        # Tie: Large:1, Small:1 → prefer smaller = Small
        assert result.sizing == "Small"

    def test_calculates_average_confidence(self, sizing_service):
        """Should calculate average confidence from all domains."""
        domain_results = {
            "pe_base_powertrain": SizingResult(
                sizing="Medium",
                reasoning="test",
                rule_id="TEST_M_001",
                confidence=0.6,
                method="keyword",
            ),
            "customer_build_stages": SizingResult(
                sizing="Medium",
                reasoning="test",
                rule_id="TEST_M_002",
                confidence=0.8,
                method="keyword",
            ),
        }

        result = sizing_service._aggregate_sizing(domain_results)
        assert result.confidence == 0.7  # Average of 0.6 and 0.8


# ============================================================================
# LLM CLASSIFICATION TESTS (with mocks)
# ============================================================================


class TestLLMClassification:
    """Tests for LLM-based rule selection."""

    @pytest.mark.asyncio
    async def test_parses_valid_llm_response(self, sizing_service):
        """Should parse valid JSON response from LLM."""
        domain_rules = sizing_service.rules_by_domain["pe_base_powertrain"]
        rule_id = domain_rules[0].rule_id

        response_text = f"""
        {{
            "selected_rule_id": "{rule_id}",
            "confidence": 0.85,
            "reasoning": "Matched new concept keywords"
        }}
        """

        result = sizing_service._parse_llm_response(response_text, domain_rules)
        assert result is not None
        assert result.rule_id == rule_id
        assert result.confidence == 0.85
        assert result.method == SizingMethod.LLM.value

    @pytest.mark.asyncio
    async def test_handles_invalid_llm_response(self, sizing_service):
        """Should return None for invalid LLM response."""
        result = sizing_service._parse_llm_response(
            "This is not valid JSON",
            sizing_service.rules_by_domain["pe_base_powertrain"],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_unknown_rule_id(self, sizing_service):
        """Should return None for unknown rule_id."""
        response_text = """
        {
            "selected_rule_id": "UNKNOWN_RULE_999",
            "confidence": 0.9,
            "reasoning": "Test"
        }
        """

        result = sizing_service._parse_llm_response(
            response_text, sizing_service.rules_by_domain["pe_base_powertrain"]
        )
        assert result is None


# ============================================================================
# FULL CLASSIFICATION TESTS
# ============================================================================


class TestFullClassification:
    """Tests for complete classify_sizing workflow."""

    @pytest.mark.asyncio
    async def test_classifies_without_llm(self, sizing_service, sample_pr_text_medium):
        """Should classify using keywords when LLM is None."""
        result = await sizing_service.classify_sizing(
            pr_text=sample_pr_text_medium,
            parsed_pr={},
            llm=None,
        )

        assert isinstance(result, ProgramSizingResult)
        assert result.pe_base_powertrain.sizing in [s.value for s in SizingLevel]
        assert result.program_overall.sizing in [s.value for s in SizingLevel]

    @pytest.mark.asyncio
    async def test_returns_all_domains(self, sizing_service, sample_pr_text_full):
        """Should return sizing for all domains."""
        result = await sizing_service.classify_sizing(
            pr_text=sample_pr_text_full,
            parsed_pr={},
            llm=None,
        )

        result_dict = result.to_dict()
        expected_domains = [
            "pe_base_powertrain",
            "pe_system_assembly",
            "pe_installation_application",
            "manufacturing_base_engine",
            "manufacturing_ats",
            "purchasing_sourcing",
            "purchasing_supplier_quality",
            "customer_build_stages",
            "program_manager_overall",
            "program_overall",
        ]

        for domain in expected_domains:
            assert domain in result_dict
            assert "sizing" in result_dict[domain]
            assert "rule_id" in result_dict[domain]
            assert "confidence" in result_dict[domain]

    @pytest.mark.asyncio
    async def test_with_mock_llm(self, sizing_service, sample_pr_text_full):
        """Should use LLM when provided and parse response."""
        mock_llm = AsyncMock()
        # LLMClient.fast_response returns a string directly, not an object with .content
        mock_llm.fast_response.return_value = """
        {
            "selected_rule_id": "PE_BASE_F_001",
            "confidence": 0.9,
            "reasoning": "Full sizing matches new concept"
        }
        """

        result = await sizing_service.classify_sizing(
            pr_text=sample_pr_text_full,
            parsed_pr={},
            llm=mock_llm,
        )

        assert isinstance(result, ProgramSizingResult)
        # LLM should have been called for domains with rules
        assert mock_llm.fast_response.called


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_pr_text(self, sizing_service):
        """Should handle empty PR text."""
        result = sizing_service._classify_domain_with_keywords(
            "pe_base_powertrain",
            sizing_service.rules_by_domain["pe_base_powertrain"],
            "",
            {},
        )
        assert result.sizing == SizingLevel.MEDIUM.value
        assert result.method == SizingMethod.DEFAULT.value

    def test_get_rule_by_id(self, sizing_service):
        """Should retrieve rule by ID."""
        rule = sizing_service.rules[0]
        retrieved = sizing_service.get_rule_by_id(rule.rule_id)
        assert retrieved is not None
        assert retrieved.rule_id == rule.rule_id

    def test_get_nonexistent_rule(self, sizing_service):
        """Should return None for nonexistent rule ID."""
        result = sizing_service.get_rule_by_id("NONEXISTENT_RULE")
        assert result is None

    def test_get_rules_context(self, sizing_service):
        """Should generate rules context for LLM prompts."""
        context = sizing_service.get_all_rules_context()
        assert "SIZING RULES REFERENCE" in context
        assert (
            "pe_base_powertrain" in context.lower().replace("_", " ").replace(" ", "_")
            or "Base" in context
        )


# ============================================================================
# DATACLASS TESTS
# ============================================================================


class TestDataClasses:
    """Tests for data classes."""

    def test_sizing_result_to_dict(self):
        """SizingResult should serialize to dict."""
        result = SizingResult(
            sizing="Large",
            reasoning="Test reasoning",
            rule_id="PE_BASE_L_001",
            confidence=0.85,
            method="llm",
        )

        d = result.to_dict()
        assert d["sizing"] == "Large"
        assert d["rule_id"] == "PE_BASE_L_001"
        assert d["confidence"] == 0.85

    def test_sizing_rule_keyword_extraction(self):
        """SizingRule should extract keywords from development_effort."""
        rule = SizingRule(
            rule_id="TEST_F_001",
            function="Product Engineering",
            sub_function="Base Engine",
            development_effort="New concept required with high level of validation",
            sizing=SizingLevel.FULL,
            domain_key="pe_base_powertrain",
        )

        assert len(rule.keywords) > 0
        # Should contain some extracted keywords
        assert any(
            kw in rule.keywords for kw in ["new", "high", "validation", "new concept"]
        )


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
FPT Cost Brain 2.0 - DPO Dataset Generator Tests
Unit tests for JSONL export, synthetic rejected generation, and upload pipeline.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.dpo_dataset_generator import (
    DPODatasetGenerator,
    DPOExample,
    FineTuningProvider,
    FineTuningUploadPipeline,
    SyntheticRejectedGenerator,
)


class TestDPOExample:
    """Tests for DPO example dataclass."""

    def test_create_example(self):
        """Test creating a DPO example."""
        example = DPOExample(
            prompt="Estimate the cost for this project",
            chosen="This is a complex project requiring...",
            rejected="This is a simple project...",
            metadata={"pair_id": "123"},
        )

        assert example.prompt == "Estimate the cost for this project"
        assert example.chosen == "This is a complex project requiring..."
        assert example.rejected == "This is a simple project..."
        assert example.metadata["pair_id"] == "123"

    def test_default_metadata(self):
        """Test default empty metadata."""
        example = DPOExample(
            prompt="prompt",
            chosen="chosen",
            rejected="rejected",
        )
        assert example.metadata == {}


class TestSyntheticRejectedGenerator:
    """Tests for synthetic rejected reasoning generation."""

    @pytest.mark.asyncio
    async def test_generate_rejected(self):
        """Test generating synthetic rejected reasoning."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value="This appears to be a standard project with minimal complexity."
        )

        generator = SyntheticRejectedGenerator(mock_llm)

        result = await generator.generate_rejected(
            chosen_reasoning="Complex project with turbo integration and EU7 compliance",
            context="MK08 engine family, 3 vehicle variants",
        )

        assert result is not None
        assert len(result) > 0
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        """Test fallback when LLM fails."""
        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(side_effect=Exception("API error"))

        generator = SyntheticRejectedGenerator(mock_llm)

        result = await generator.generate_rejected(
            chosen_reasoning="Complex reasoning here",
            context="Some context",
        )

        # Should return fallback reasoning
        assert "standard project" in result.lower()
        assert "baseline" in result.lower()

    def test_fallback_rejected_content(self):
        """Test fallback rejected content is sensible."""
        generator = SyntheticRejectedGenerator(MagicMock())
        fallback = generator._fallback_rejected("Some chosen reasoning")

        assert "standard" in fallback.lower()
        assert "typical" in fallback.lower() or "baseline" in fallback.lower()


class TestDPODatasetGeneratorValidation:
    """Tests for DPO example validation."""

    def test_validate_valid_example(self):
        """Test validation passes for valid example."""
        generator = DPODatasetGenerator(db=MagicMock())

        example = DPOExample(
            prompt="A sufficiently long prompt for cost estimation of this project",
            chosen="A detailed and thorough reasoning that explains the cost drivers "
            "and complexity factors in this R&D project.",
            rejected="A simpler reasoning that misses key factors and underestimates "
            "the project complexity and requirements.",
        )

        errors = generator._validate_example(example)
        assert len(errors) == 0

    def test_validate_short_prompt(self):
        """Test validation fails for short prompt."""
        generator = DPODatasetGenerator(db=MagicMock())

        example = DPOExample(
            prompt="Too short",
            chosen="A sufficiently long chosen response for validation",
            rejected="A sufficiently long rejected response for validation",
        )

        errors = generator._validate_example(example)
        assert any("Prompt too short" in e for e in errors)

    def test_validate_identical_responses(self):
        """Test validation fails for identical chosen/rejected."""
        generator = DPODatasetGenerator(db=MagicMock())

        example = DPOExample(
            prompt="A sufficiently long prompt for cost estimation",
            chosen="This is the exact same response text",
            rejected="This is the exact same response text",
        )

        errors = generator._validate_example(example)
        assert any("identical" in e.lower() for e in errors)

    def test_validate_too_similar_responses(self):
        """Test validation fails for very similar responses (>90% word overlap)."""
        generator = DPODatasetGenerator(db=MagicMock())

        # Use text with >90% Jaccard word overlap
        # 20 words, only 1 different = 19/21 = 0.904 similarity
        base_text = "This project requires extensive hardware calibration testing phases for MK08 engine turbo integration emission compliance validation and final certification work"
        example = DPOExample(
            prompt="A sufficiently long prompt for cost estimation",
            chosen=base_text,
            rejected=base_text.replace(
                "extensive", "basic"
            ),  # Only 1 word different out of 20
        )

        errors = generator._validate_example(example)
        assert any("similar" in e.lower() for e in errors)

    def test_calculate_similarity(self):
        """Test similarity calculation."""
        generator = DPODatasetGenerator(db=MagicMock())

        # Identical texts
        assert generator._calculate_similarity("hello world", "hello world") == 1.0

        # Completely different
        assert generator._calculate_similarity("hello", "goodbye") == 0.0

        # Partial overlap
        similarity = generator._calculate_similarity(
            "the quick brown fox", "the lazy brown dog"
        )
        assert 0 < similarity < 1


class TestDPOFormatters:
    """Tests for provider-specific formatting."""

    def setup_method(self):
        """Setup test fixtures."""
        self.example = DPOExample(
            prompt="Estimate the cost for MK08 turbo project",
            chosen="This is a complex project with turbo integration...",
            rejected="This is a simple standard project...",
            metadata={"pair_id": "test-123"},
        )
        self.generator = DPODatasetGenerator(db=MagicMock())

    def test_format_for_openai(self):
        """Test OpenAI format."""
        formatted = self.generator._format_for_openai(self.example)

        assert "messages" in formatted
        assert "rejected" in formatted
        assert "metadata" in formatted

        # Check messages structure
        messages = formatted["messages"]
        assert len(messages) == 3  # system, user, assistant
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == self.example.chosen

    def test_format_for_fireworks(self):
        """Test Fireworks format."""
        formatted = self.generator._format_for_fireworks(self.example)

        assert formatted["prompt"] == self.example.prompt
        assert formatted["chosen"] == self.example.chosen
        assert formatted["rejected"] == self.example.rejected

    def test_format_for_together(self):
        """Test Together AI format."""
        formatted = self.generator._format_for_together(self.example)

        assert "text" in formatted
        assert "text_rejected" in formatted
        assert "### Instruction" in formatted["text"]
        assert "### Response (Good)" in formatted["text"]
        assert "### Response (Bad)" in formatted["text_rejected"]

    def test_format_example_routing(self):
        """Test format routing to correct provider."""
        for provider in FineTuningProvider:
            formatted = self.generator._format_example(self.example, provider)
            assert isinstance(formatted, dict)


class TestIsGenericRejected:
    """Tests for generic rejected detection."""

    def test_empty_rejected(self):
        """Test empty string is generic."""
        generator = DPODatasetGenerator(db=MagicMock())
        assert generator._is_generic_rejected("") is True
        assert generator._is_generic_rejected(None) is True

    def test_generic_phrases(self):
        """Test known generic phrases are detected."""
        generator = DPODatasetGenerator(db=MagicMock())

        generic_texts = [
            "This is a standard project with typical complexity.",
            "Apply baseline estimates without significant adjustments.",
            "No special considerations needed for this work.",
        ]

        for text in generic_texts:
            assert generator._is_generic_rejected(text) is True

    def test_specific_rejected(self):
        """Test specific reasoning is not generic."""
        generator = DPODatasetGenerator(db=MagicMock())

        specific_text = (
            "The MK08 turbo integration requires only 200 hours because "
            "we have done similar work on the MK07 platform last year."
        )

        assert generator._is_generic_rejected(specific_text) is False


class TestFineTuningUploadPipeline:
    """Tests for upload pipeline preparation."""

    def test_prepare_openai_upload(self):
        """Test OpenAI upload config preparation."""
        config = FineTuningUploadPipeline.prepare_openai_upload(
            jsonl_path="/path/to/dataset.jsonl",
            model_suffix="fpt-v1",
            n_epochs=5,
        )

        assert config["training_file"] == "/path/to/dataset.jsonl"
        assert config["suffix"] == "fpt-v1"
        assert config["hyperparameters"]["n_epochs"] == 5
        assert config["method"]["type"] == "dpo"

    def test_prepare_fireworks_upload(self):
        """Test Fireworks upload config preparation."""
        config = FineTuningUploadPipeline.prepare_fireworks_upload(
            jsonl_path="/path/to/dataset.jsonl",
            job_name="my-model",
        )

        assert config["dataset"] == "/path/to/dataset.jsonl"
        assert config["job_name"] == "my-model"
        assert config["method"] == "dpo"
        assert "beta" in config["hyperparameters"]

    def test_prepare_together_upload(self):
        """Test Together AI upload config preparation."""
        config = FineTuningUploadPipeline.prepare_together_upload(
            jsonl_path="/path/to/dataset.jsonl",
            job_name="my-model",
        )

        assert config["training_file"] == "/path/to/dataset.jsonl"
        assert config["suffix"] == "my-model"
        assert config["method"] == "dpo"

    def test_validate_dataset_valid(self):
        """Test validation of valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write valid DPO examples
            examples = [
                {"prompt": "test prompt", "chosen": "good", "rejected": "bad"},
                {"prompt": "another", "chosen": "correct", "rejected": "wrong"},
            ]
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            f.flush()

            result = FineTuningUploadPipeline.validate_dataset_for_upload(f.name)

            assert result["valid"] is True
            assert result["examples_count"] == 2
            assert len(result["issues"]) == 0

    def test_validate_dataset_invalid_json(self):
        """Test validation catches invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": "json"}\n')
            f.write("not valid json\n")
            f.write('{"another": "valid"}\n')
            f.flush()

            result = FineTuningUploadPipeline.validate_dataset_for_upload(f.name)

            assert result["valid"] is False
            assert any("Invalid JSON" in issue for issue in result["issues"])

    def test_validate_dataset_missing_fields(self):
        """Test validation catches missing fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"only_prompt": "no responses"}\n')
            f.flush()

            result = FineTuningUploadPipeline.validate_dataset_for_upload(f.name)

            assert result["valid"] is False
            assert any("Missing" in issue for issue in result["issues"])

    def test_validate_dataset_file_not_found(self):
        """Test validation handles missing file."""
        result = FineTuningUploadPipeline.validate_dataset_for_upload(
            "/nonexistent/path.jsonl"
        )

        assert result["valid"] is False
        assert any("not found" in issue for issue in result["issues"])


class TestFineTuningProvider:
    """Tests for provider enum."""

    def test_all_providers_defined(self):
        """Test all expected providers exist."""
        assert FineTuningProvider.OPENAI.value == "openai"
        assert FineTuningProvider.FIREWORKS.value == "fireworks"
        assert FineTuningProvider.TOGETHER.value == "together"
        assert FineTuningProvider.ANYSCALE.value == "anyscale"

    def test_provider_from_string(self):
        """Test creating provider from string."""
        assert FineTuningProvider("openai") == FineTuningProvider.OPENAI
        assert FineTuningProvider("fireworks") == FineTuningProvider.FIREWORKS

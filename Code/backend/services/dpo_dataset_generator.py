"""
FPT Cost Brain 2.0 - DPO Dataset Generator
Exports preference pairs to JSONL format for fine-tuning providers.
Includes synthetic rejected reasoning generation.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import PreferencePair
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class FineTuningProvider(str, Enum):
    """Supported fine-tuning providers."""

    OPENAI = "openai"
    FIREWORKS = "fireworks"
    TOGETHER = "together"
    ANYSCALE = "anyscale"


@dataclass
class DPOExample:
    """A single DPO training example."""

    prompt: str  # The input context (PR features + question)
    chosen: str  # The preferred response (good reasoning)
    rejected: str  # The dispreferred response (bad reasoning)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetExportResult:
    """Result of dataset export operation."""

    file_path: str
    total_examples: int
    examples_with_synthetic: int
    provider_format: str
    export_timestamp: str
    validation_passed: bool
    validation_errors: list[str] = field(default_factory=list)


# Prompt templates for synthetic rejected generation
SYNTHETIC_REJECTED_SYSTEM_PROMPT = """You are an AI assistant that generates plausible but INCORRECT reasoning for R&D cost estimation.

Your task: Given a CORRECT reasoning chain, generate an alternative reasoning that would lead to a WRONG estimate. The bad reasoning should:
1. Sound professional but contain subtle errors
2. Miss key complexity factors or misinterpret project requirements
3. Use overly simplistic assumptions
4. Ignore relevant historical precedents
5. Make logical but ultimately wrong conclusions

DO NOT generate obviously wrong or nonsensical reasoning. The goal is to create realistic "mistakes" that a less experienced estimator might make."""

SYNTHETIC_REJECTED_USER_TEMPLATE = """Here is the CORRECT reasoning for a cost estimate:

<correct_reasoning>
{chosen_reasoning}
</correct_reasoning>

Context about the project:
{context}

Generate an alternative reasoning that sounds professional but would lead to an INCORRECT (typically underestimated) cost. The bad reasoning should:
- Miss at least one key complexity factor
- Use simpler assumptions than warranted
- Sound confident but be subtly wrong

Output ONLY the bad reasoning, no explanations or meta-commentary."""


class SyntheticRejectedGenerator:
    """
    Generates synthetic 'rejected' reasoning using a cheaper LLM.

    Used when we only have the user's corrected (chosen) response
    but need a rejected counterpart for DPO training.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "google/gemini-2.0-flash-001",  # Cheap, fast model
    ):
        self.llm = llm_client
        self.model = model

    async def generate_rejected(
        self,
        chosen_reasoning: str,
        context: str,
        temperature: float = 0.9,  # Higher temp for diverse bad examples
    ) -> str:
        """
        Generate a plausible-but-wrong reasoning given the correct one.

        Args:
            chosen_reasoning: The correct reasoning chain (from user edit)
            context: Project context (PR features, requirements)
            temperature: LLM temperature for diversity

        Returns:
            Generated bad reasoning that sounds professional but is wrong
        """
        messages = [
            {"role": "system", "content": SYNTHETIC_REJECTED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SYNTHETIC_REJECTED_USER_TEMPLATE.format(
                    chosen_reasoning=chosen_reasoning,
                    context=context,
                ),
            },
        ]

        try:
            rejected = await self.llm.chat(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=1024,
            )
            return rejected.strip()
        except Exception as e:
            logger.error(f"Failed to generate synthetic rejected: {e}")
            # Fallback: return a generic bad reasoning
            return self._fallback_rejected(chosen_reasoning)

    def _fallback_rejected(self, chosen_reasoning: str) -> str:
        """Fallback when LLM fails - creates a simplified version."""
        return (
            "This appears to be a standard project with typical complexity. "
            "Based on baseline estimates, no special considerations are needed. "
            "Historical averages should apply without significant adjustments."
        )


class DPODatasetGenerator:
    """
    Generates DPO training datasets from preference pairs.

    Supports multiple fine-tuning providers with their specific formats:
    - OpenAI: messages format with chosen/rejected
    - Fireworks: prompt/chosen/rejected format
    - Together: similar to Fireworks
    """

    def __init__(
        self,
        db: AsyncSession,
        llm_client: LLMClient | None = None,
        output_dir: Path | None = None,
    ):
        self.db = db
        self.llm = llm_client
        self.output_dir = output_dir or Path("data/dpo_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if llm_client:
            self.synthetic_generator = SyntheticRejectedGenerator(llm_client)
        else:
            self.synthetic_generator = None

    async def get_preference_pairs(
        self,
        validated_only: bool = True,
        min_confidence: float = 0.5,
        limit: int | None = None,
        exclude_used: bool = False,
    ) -> list[PreferencePair]:
        """
        Fetch preference pairs from database.

        Args:
            validated_only: Only include validated pairs
            min_confidence: Minimum confidence threshold
            limit: Maximum number of pairs
            exclude_used: Exclude pairs already used in training
        """
        query = select(PreferencePair).where(
            PreferencePair.confidence >= min_confidence
        )

        if validated_only:
            query = query.where(PreferencePair.validated.is_(True))

        if exclude_used:
            query = query.where(PreferencePair.used_in_training.is_(None))

        query = query.order_by(PreferencePair.created_at.desc())

        if limit:
            query = query.limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def generate_synthetic_rejected_for_pair(
        self,
        pair: PreferencePair,
    ) -> str | None:
        """
        Generate synthetic rejected reasoning for a pair that only has chosen.

        This is needed when signal_source is 'user_edit' and we captured
        the user's correction (chosen) but don't have a bad alternative (rejected).
        """
        if not self.synthetic_generator:
            logger.warning("No LLM client available for synthetic generation")
            return None

        # Build context from the pair's metadata
        context = self._build_context_from_pair(pair)

        return await self.synthetic_generator.generate_rejected(
            chosen_reasoning=pair.chosen_reasoning,
            context=context,
        )

    def _build_context_from_pair(self, pair: PreferencePair) -> str:
        """Build context string from pair metadata."""
        context_parts = []

        # Add breakdown summary
        if pair.chosen_breakdown:
            total = sum(
                v for v in pair.chosen_breakdown.values() if isinstance(v, (int, float))
            )
            context_parts.append(f"Total estimated hours: {total}")
            context_parts.append(f"Breakdown: {json.dumps(pair.chosen_breakdown)}")

        context_parts.append(f"Signal source: {pair.signal_source}")
        context_parts.append(f"Confidence: {pair.confidence}")

        return "\n".join(context_parts)

    def _pair_to_dpo_example(
        self,
        pair: PreferencePair,
        synthetic_rejected: str | None = None,
    ) -> DPOExample:
        """Convert a PreferencePair to DPOExample format."""
        # Build the prompt from context
        prompt = self._build_prompt_from_pair(pair)

        # Use synthetic rejected if the original rejected is empty/generic
        rejected = pair.rejected_reasoning
        if synthetic_rejected and self._is_generic_rejected(rejected):
            rejected = synthetic_rejected

        return DPOExample(
            prompt=prompt,
            chosen=pair.chosen_reasoning,
            rejected=rejected,
            metadata={
                "pair_id": str(pair.id),
                "signal_source": pair.signal_source,
                "reward_delta": float(pair.reward_delta),
                "confidence": float(pair.confidence),
                "created_at": pair.created_at.isoformat() if pair.created_at else None,
            },
        )

    def _build_prompt_from_pair(self, pair: PreferencePair) -> str:
        """Build the input prompt for DPO training."""
        # The prompt should represent the estimation task context
        prompt_parts = [
            "You are an R&D cost estimation expert for FPT Industrial.",
            "Given the following project context, provide detailed reasoning for your cost estimate.",
            "",
            "Project Details:",
        ]

        if pair.chosen_breakdown:
            prompt_parts.append(f"Activity breakdown targets: {pair.chosen_breakdown}")

        prompt_parts.append("")
        prompt_parts.append(
            "Provide your reasoning for this estimate, considering complexity factors, "
            "historical precedents, and risk areas."
        )

        return "\n".join(prompt_parts)

    def _is_generic_rejected(self, rejected: str) -> bool:
        """Check if rejected reasoning is too generic and needs replacement."""
        if not rejected:
            return True

        generic_phrases = [
            "standard project",
            "baseline estimates",
            "no special considerations",
            "typical complexity",
            "apply without significant adjustments",
        ]

        rejected_lower = rejected.lower()
        return any(phrase in rejected_lower for phrase in generic_phrases)

    # ===== Export Formats =====

    def _format_for_openai(self, example: DPOExample) -> dict:
        """
        Format example for OpenAI fine-tuning.

        OpenAI uses a messages format with system/user/assistant roles.
        For DPO, we need to provide two conversations: chosen and rejected.
        """
        base_messages = [
            {
                "role": "system",
                "content": "You are an expert R&D cost estimator for FPT Industrial.",
            },
            {"role": "user", "content": example.prompt},
        ]

        return {
            "messages": base_messages
            + [{"role": "assistant", "content": example.chosen}],
            "rejected": base_messages
            + [{"role": "assistant", "content": example.rejected}],
            "metadata": example.metadata,
        }

    def _format_for_fireworks(self, example: DPOExample) -> dict:
        """
        Format example for Fireworks AI fine-tuning.

        Fireworks uses a simpler prompt/chosen/rejected format.
        """
        return {
            "prompt": example.prompt,
            "chosen": example.chosen,
            "rejected": example.rejected,
            "metadata": example.metadata,
        }

    def _format_for_together(self, example: DPOExample) -> dict:
        """
        Format example for Together AI fine-tuning.

        Together uses a similar format to Fireworks with slight variations.
        """
        return {
            "text": f"### Instruction\n{example.prompt}\n\n### Response (Good)\n{example.chosen}",
            "text_rejected": f"### Instruction\n{example.prompt}\n\n### Response (Bad)\n{example.rejected}",
            "metadata": example.metadata,
        }

    def _format_for_anyscale(self, example: DPOExample) -> dict:
        """
        Format example for Anyscale fine-tuning.

        Anyscale uses OpenAI-compatible format.
        """
        return self._format_for_openai(example)

    def _format_example(
        self,
        example: DPOExample,
        provider: FineTuningProvider,
    ) -> dict:
        """Format example for the specified provider."""
        formatters = {
            FineTuningProvider.OPENAI: self._format_for_openai,
            FineTuningProvider.FIREWORKS: self._format_for_fireworks,
            FineTuningProvider.TOGETHER: self._format_for_together,
            FineTuningProvider.ANYSCALE: self._format_for_anyscale,
        }
        return formatters[provider](example)

    # ===== Validation =====

    def _validate_example(self, example: DPOExample) -> list[str]:
        """Validate a DPO example for common issues."""
        errors = []

        if not example.prompt or len(example.prompt) < 20:
            errors.append("Prompt too short")

        if not example.chosen or len(example.chosen) < 50:
            errors.append("Chosen response too short")

        if not example.rejected or len(example.rejected) < 50:
            errors.append("Rejected response too short")

        if example.chosen == example.rejected:
            errors.append("Chosen and rejected are identical")

        # Check for extremely similar responses
        if example.chosen and example.rejected:
            similarity = self._calculate_similarity(example.chosen, example.rejected)
            if similarity > 0.9:
                errors.append(f"Responses too similar: {similarity:.2%}")

        return errors

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple token overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    # ===== Export =====

    async def export_dataset(
        self,
        provider: FineTuningProvider,
        validated_only: bool = True,
        min_confidence: float = 0.5,
        generate_synthetic: bool = True,
        limit: int | None = None,
        mark_as_used: bool = False,
    ) -> DatasetExportResult:
        """
        Export preference pairs to JSONL format for fine-tuning.

        Args:
            provider: Target fine-tuning provider
            validated_only: Only include validated pairs
            min_confidence: Minimum confidence threshold
            generate_synthetic: Generate synthetic rejected for pairs missing it
            limit: Maximum number of examples
            mark_as_used: Mark pairs as used_in_training

        Returns:
            DatasetExportResult with export details
        """
        # Fetch pairs
        pairs = await self.get_preference_pairs(
            validated_only=validated_only,
            min_confidence=min_confidence,
            limit=limit,
        )

        if not pairs:
            return DatasetExportResult(
                file_path="",
                total_examples=0,
                examples_with_synthetic=0,
                provider_format=provider.value,
                export_timestamp=datetime.now(timezone.utc).isoformat(),
                validation_passed=False,
                validation_errors=["No preference pairs found matching criteria"],
            )

        # Convert to DPO examples
        examples = []
        synthetic_count = 0
        all_errors = []

        for pair in pairs:
            synthetic_rejected = None

            # Generate synthetic rejected if needed
            if generate_synthetic and self._is_generic_rejected(
                pair.rejected_reasoning
            ):
                if self.synthetic_generator:
                    synthetic_rejected = (
                        await self.generate_synthetic_rejected_for_pair(pair)
                    )
                    if synthetic_rejected:
                        synthetic_count += 1

            example = self._pair_to_dpo_example(pair, synthetic_rejected)

            # Validate
            errors = self._validate_example(example)
            if errors:
                all_errors.extend([f"Pair {pair.id}: {e}" for e in errors])
                continue

            examples.append(example)

        if not examples:
            return DatasetExportResult(
                file_path="",
                total_examples=0,
                examples_with_synthetic=synthetic_count,
                provider_format=provider.value,
                export_timestamp=datetime.now(timezone.utc).isoformat(),
                validation_passed=False,
                validation_errors=all_errors or ["All examples failed validation"],
            )

        # Write JSONL file
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"dpo_dataset_{provider.value}_{timestamp}.jsonl"
        file_path = self.output_dir / filename

        with open(file_path, "w") as f:
            for example in examples:
                formatted = self._format_example(example, provider)
                f.write(json.dumps(formatted) + "\n")

        logger.info(
            f"Exported {len(examples)} DPO examples to {file_path} "
            f"({synthetic_count} with synthetic rejected)"
        )

        # Mark pairs as used if requested
        if mark_as_used:
            pair_ids = [pair.id for pair in pairs]
            await self._mark_pairs_as_used(pair_ids)

        return DatasetExportResult(
            file_path=str(file_path),
            total_examples=len(examples),
            examples_with_synthetic=synthetic_count,
            provider_format=provider.value,
            export_timestamp=datetime.now(timezone.utc).isoformat(),
            validation_passed=len(all_errors) == 0,
            validation_errors=all_errors[:10]
            if all_errors
            else [],  # Limit errors shown
        )

    async def _mark_pairs_as_used(self, pair_ids: list[uuid.UUID]) -> None:
        """Mark preference pairs as used in training."""
        from sqlalchemy import update

        await self.db.execute(
            update(PreferencePair)
            .where(PreferencePair.id.in_(pair_ids))
            .values(used_in_training=datetime.now(timezone.utc))
        )
        await self.db.commit()

    async def get_export_stats(self) -> dict:
        """Get statistics about exportable pairs."""
        # Total pairs
        total_result = await self.db.execute(select(func.count(PreferencePair.id)))
        total = total_result.scalar() or 0

        # Validated pairs
        validated_result = await self.db.execute(
            select(func.count(PreferencePair.id)).where(
                PreferencePair.validated.is_(True)
            )
        )
        validated = validated_result.scalar() or 0

        # Unused pairs
        unused_result = await self.db.execute(
            select(func.count(PreferencePair.id)).where(
                PreferencePair.used_in_training.is_(None)
            )
        )
        unused = unused_result.scalar() or 0

        # Pairs by source
        source_result = await self.db.execute(
            select(
                PreferencePair.signal_source,
                func.count(PreferencePair.id),
            ).group_by(PreferencePair.signal_source)
        )
        by_source = {row[0]: row[1] for row in source_result}

        # Pairs needing synthetic rejected
        # (those with generic rejected reasoning)
        # Note: This is an approximation - actual check happens during export

        return {
            "total_pairs": total,
            "validated_pairs": validated,
            "unused_pairs": unused,
            "pairs_by_source": by_source,
            "ready_for_export": validated,  # Only validated pairs are exported
        }


# ===== Upload Pipeline Helpers =====


class FineTuningUploadPipeline:
    """
    Prepares and uploads datasets to fine-tuning providers.

    Note: Actual upload requires provider-specific API keys and endpoints.
    This class prepares the upload payload and provides validation.
    """

    @staticmethod
    def prepare_openai_upload(
        jsonl_path: str,
        model_suffix: str = "fpt-costbrain",
        n_epochs: int = 3,
    ) -> dict:
        """
        Prepare OpenAI fine-tuning job configuration.

        Returns configuration dict that can be used with OpenAI API.
        """
        return {
            "training_file": jsonl_path,
            "model": "gpt-4o-mini-2024-07-18",  # Base model for fine-tuning
            "suffix": model_suffix,
            "hyperparameters": {
                "n_epochs": n_epochs,
                "batch_size": "auto",
                "learning_rate_multiplier": "auto",
            },
            "method": {
                "type": "dpo",
                "dpo": {
                    "hyperparameters": {
                        "beta": 0.1,
                    }
                },
            },
            "integrations": [],
            "seed": 42,
        }

    @staticmethod
    def prepare_fireworks_upload(
        jsonl_path: str,
        base_model: str = "accounts/fireworks/models/llama-v3p1-8b-instruct",
        job_name: str = "fpt-costbrain-dpo",
    ) -> dict:
        """
        Prepare Fireworks AI fine-tuning job configuration.
        """
        return {
            "dataset": jsonl_path,
            "base_model": base_model,
            "job_name": job_name,
            "method": "dpo",
            "hyperparameters": {
                "learning_rate": 1e-5,
                "epochs": 3,
                "beta": 0.1,  # DPO temperature parameter
                "batch_size": 4,
            },
        }

    @staticmethod
    def prepare_together_upload(
        jsonl_path: str,
        base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        job_name: str = "fpt-costbrain-dpo",
    ) -> dict:
        """
        Prepare Together AI fine-tuning job configuration.
        """
        return {
            "training_file": jsonl_path,
            "model": base_model,
            "n_epochs": 3,
            "learning_rate": 1e-5,
            "suffix": job_name,
            "training_type": "Full",  # or "LoRA"
            "method": "dpo",
        }

    @staticmethod
    def validate_dataset_for_upload(jsonl_path: str) -> dict:
        """
        Validate a JSONL dataset file before upload.

        Returns validation results with any issues found.
        """
        issues = []
        examples_count = 0

        try:
            with open(jsonl_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        examples_count += 1

                        # Basic structure validation
                        if "prompt" not in data and "messages" not in data:
                            issues.append(
                                f"Line {line_num}: Missing prompt or messages"
                            )

                        if "chosen" not in data and "rejected" not in data:
                            # Check for alternative formats
                            if "text" not in data or "text_rejected" not in data:
                                issues.append(
                                    f"Line {line_num}: Missing chosen/rejected responses"
                                )

                    except json.JSONDecodeError as e:
                        issues.append(f"Line {line_num}: Invalid JSON - {e}")

        except FileNotFoundError:
            return {
                "valid": False,
                "examples_count": 0,
                "issues": [f"File not found: {jsonl_path}"],
            }

        return {
            "valid": len(issues) == 0,
            "examples_count": examples_count,
            "issues": issues[:20],  # Limit to first 20 issues
            "file_path": jsonl_path,
        }

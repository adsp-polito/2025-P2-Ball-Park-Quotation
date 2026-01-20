"""
FPT Cost Brain 2.0 - Online Continual Learning
3-layer learning architecture for continuous improvement
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ml.features import FeatureExtractor, FeatureSet
from ml.model import CostPredictionModel, ModelRegistry
from ml.trainer import ModelTrainer, TrainingConfig, TrainingResult


@dataclass
class RetrainConfig:
    """Configuration for automatic retraining triggers."""

    # Minimum corrections before considering retrain
    min_corrections: int = 5

    # Force retrain after this many corrections
    force_retrain_corrections: int = 20

    # Retrain if average correction percentage exceeds this
    drift_threshold_pct: float = 15.0

    # Maximum days between retrains
    max_days_between_retrain: int = 7

    # Improvement threshold for auto-promotion
    improvement_threshold_pct: float = 5.0

    # Enable auto-promotion
    auto_promote: bool = True


@dataclass
class FeedbackCorrection:
    """A single user correction."""

    correction_id: str
    quotation_id: str
    breakdown_id: str
    activity_code: str
    original_hours: float
    corrected_hours: float
    correction_pct: float
    reason: str
    created_at: str
    user_id: str
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_edit(
        cls,
        edit: dict[str, Any],
        quotation_id: str,
        user_id: str,
    ) -> "FeedbackCorrection":
        """Create from user edit."""
        original = edit.get("original_hours", 0)
        corrected = edit.get("new_hours", 0)
        correction_pct = (
            ((corrected - original) / original * 100) if original != 0 else 0
        )

        return cls(
            correction_id=str(uuid.uuid4()),
            quotation_id=quotation_id,
            breakdown_id=edit.get("breakdown_id", ""),
            activity_code=edit.get("activity_code", ""),
            original_hours=original,
            corrected_hours=corrected,
            correction_pct=correction_pct,
            reason=edit.get("reason", ""),
            created_at=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            context=edit.get("context", {}),
        )


@dataclass
class ExtractedRule:
    """Rule extracted from corrections."""

    rule_id: str
    rule_name: str
    conditions: dict[str, Any]
    adjustment: dict[str, Any]
    confidence: float
    source_corrections: list[str]
    created_at: str
    is_active: bool = False

    def matches(self, context: dict[str, Any]) -> bool:
        """Check if rule matches given context."""
        field = self.conditions.get("field")
        operator = self.conditions.get("operator")
        value = self.conditions.get("value")

        if not field or not operator:
            return False

        context_value = context.get(field)
        if context_value is None:
            return False

        if operator == "equals":
            return context_value == value
        elif operator == "contains":
            return value in str(context_value)
        elif operator == "greater_than":
            return float(context_value) > float(value)
        elif operator == "less_than":
            return float(context_value) < float(value)
        elif operator == "in":
            return context_value in value

        return False

    def apply(self, hours: float) -> float:
        """Apply rule adjustment to hours."""
        adj_type = self.adjustment.get("type")
        adj_value = self.adjustment.get("value", 0)

        if adj_type == "multiply":
            return hours * adj_value
        elif adj_type == "add":
            return hours + adj_value
        elif adj_type == "subtract":
            return hours - adj_value
        elif adj_type == "set":
            return adj_value

        return hours


@dataclass
class LearningStats:
    """Statistics about the learning system."""

    total_corrections: int
    corrections_since_retrain: int
    avg_correction_pct: float
    last_retrain: str | None
    next_scheduled_retrain: str | None
    active_rules: int
    pending_rules: int
    model_version: str | None
    model_metrics: dict[str, float] = field(default_factory=dict)


class OnlineLearningManager:
    """
    3-layer online learning system.

    Layer 1 - Immediate Learning:
        - Store feedback corrections
        - Extract rules via LLM
        - Update session cache
        - Update feedback embeddings

    Layer 2 - Rule Consolidation:
        - Find similar corrections (vector search)
        - Increase/decrease rule confidence
        - Flag conflicting rules

    Layer 3 - Batch Retrain:
        - Triggered by thresholds
        - Train candidate model
        - Shadow evaluation
        - Auto-promote if improved
    """

    def __init__(
        self,
        models_dir: Path | str,
        config: RetrainConfig | None = None,
        llm_client: Any = None,
        vector_store: Any = None,
    ):
        self.models_dir = Path(models_dir)
        self.config = config or RetrainConfig()
        self.llm = llm_client
        self.vector_store = vector_store

        self.feature_extractor = FeatureExtractor()
        self.model_registry = ModelRegistry(models_dir)
        self.trainer = ModelTrainer(models_dir)

        # In-memory state (would be DB in production)
        self._corrections: list[FeedbackCorrection] = []
        self._rules: dict[str, ExtractedRule] = {}
        self._last_retrain: datetime | None = None
        self._corrections_since_retrain = 0

    # ===== Layer 1: Immediate Learning =====

    async def process_feedback(
        self,
        user_edits: list[dict[str, Any]],
        quotation_id: str,
        user_id: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Process user feedback corrections (Layer 1).

        Args:
            user_edits: List of user edits
            quotation_id: Quotation being edited
            user_id: User making edits
            state: Current estimation state

        Returns:
            Processing results
        """
        results = {
            "corrections_stored": 0,
            "rules_extracted": [],
            "embeddings_updated": False,
        }

        for edit in user_edits:
            # Create correction record
            correction = FeedbackCorrection.from_edit(
                edit=edit,
                quotation_id=quotation_id,
                user_id=user_id,
            )
            correction.context = self._extract_context(state, edit)

            # Store correction
            self._corrections.append(correction)
            self._corrections_since_retrain += 1
            results["corrections_stored"] += 1

            # Extract rule if significant correction
            if abs(correction.correction_pct) > 10:
                rule = await self._extract_rule(correction, state)
                if rule:
                    results["rules_extracted"].append(rule.rule_name)

        # Update embeddings for similar correction search
        if self.vector_store and results["corrections_stored"] > 0:
            await self._update_feedback_embeddings(user_edits, state)
            results["embeddings_updated"] = True

        # Check if retrain should be triggered
        results["should_retrain"] = self._should_trigger_retrain()

        return results

    async def _extract_rule(
        self,
        correction: FeedbackCorrection,
        state: dict[str, Any],
    ) -> ExtractedRule | None:
        """Extract generalizable rule from correction using LLM."""
        if not self.llm:
            return None

        try:
            prompt = f"""Analyze this cost estimation correction and extract a generalizable rule.

Correction Details:
- Activity: {correction.activity_code}
- Original Hours: {correction.original_hours}
- Corrected Hours: {correction.corrected_hours}
- Change: {correction.correction_pct:+.1f}%
- Reason: {correction.reason}

Context:
- Program Family: {correction.context.get("program_family", "Unknown")}
- Program Size: {correction.context.get("program_size", "Unknown")}

Extract a rule with:
1. rule_name: Short descriptive name
2. conditions: {{field, operator, value}}
3. adjustment: {{type: multiply|add|subtract, value}}

Return JSON only."""

            result = await self.llm.extract_json(
                prompt=prompt,
                system_prompt="You extract patterns from cost estimation corrections.",
            )

            if result and result.get("rule_name"):
                rule = ExtractedRule(
                    rule_id=str(uuid.uuid4()),
                    rule_name=result["rule_name"],
                    conditions=result.get("conditions", {}),
                    adjustment=result.get("adjustment", {}),
                    confidence=0.5,  # Initial confidence
                    source_corrections=[correction.correction_id],
                    created_at=datetime.now(timezone.utc).isoformat(),
                )

                self._rules[rule.rule_id] = rule
                return rule

        except Exception:
            pass

        return None

    def _extract_context(
        self,
        state: dict[str, Any],
        edit: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract context for rule matching."""
        parsed_pr = state.get("parsed_pr", {})
        pr_summary = state.get("pr_summary", {})

        return {
            "program_family": parsed_pr.get("program_family"),
            "program_size": pr_summary.get("program_size") if pr_summary else None,
            "activity_code": edit.get("activity_code"),
            "activity_count": len(parsed_pr.get("raw_activities", [])),
            "has_similar_prs": len(state.get("similar_prs", [])) > 0,
        }

    async def _update_feedback_embeddings(
        self,
        edits: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> None:
        """Update vector store with feedback embeddings."""
        if not self.vector_store or not self.llm:
            return

        try:
            # Generate embedding for feedback context
            text = self._format_feedback_text(edits, state)
            embedding = await self.llm.embed(text)

            await self.vector_store.upsert(
                collection="feedback_patterns",
                id=f"fb_{uuid.uuid4().hex[:12]}",
                vector=embedding,
                payload={
                    "edit_count": len(edits),
                    "pr_code": state.get("parsed_pr", {}).get("pr_code"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            pass

    def _format_feedback_text(
        self,
        edits: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> str:
        """Format feedback for embedding."""
        parts = []
        parsed_pr = state.get("parsed_pr", {})

        parts.append(f"PR: {parsed_pr.get('pr_code', 'Unknown')}")
        parts.append(f"Program: {parsed_pr.get('program_family', 'Unknown')}")

        for edit in edits:
            parts.append(
                f"Correction: {edit.get('activity_code')} "
                f"{edit.get('original_hours')}h -> {edit.get('new_hours')}h"
            )
            if edit.get("reason"):
                parts.append(f"Reason: {edit['reason']}")

        return "\n".join(parts)

    # ===== Layer 2: Rule Consolidation =====

    async def consolidate_rules(self) -> dict[str, Any]:
        """
        Consolidate rules based on similar corrections (Layer 2).

        - Find corrections with similar context
        - Increase confidence for consistent patterns
        - Decrease confidence for conflicting patterns
        - Merge similar rules
        """
        results = {
            "rules_updated": 0,
            "rules_merged": 0,
            "conflicts_flagged": 0,
        }

        # Group corrections by context similarity
        context_groups = self._group_corrections_by_context()

        for context_key, corrections in context_groups.items():
            if len(corrections) < 2:
                continue

            # Check if corrections are consistent
            directions = [1 if c.correction_pct > 0 else -1 for c in corrections]
            is_consistent = len(set(directions)) == 1

            # Find rules matching this context
            matching_rules = [
                r for r in self._rules.values() if r.matches(corrections[0].context)
            ]

            for rule in matching_rules:
                if is_consistent:
                    # Increase confidence
                    rule.confidence = min(1.0, rule.confidence + 0.1)
                    results["rules_updated"] += 1
                else:
                    # Decrease confidence due to conflict
                    rule.confidence = max(0.0, rule.confidence - 0.1)
                    results["conflicts_flagged"] += 1

        # Activate rules with high confidence
        for rule in self._rules.values():
            if rule.confidence >= 0.7 and not rule.is_active:
                rule.is_active = True
                results["rules_updated"] += 1

        return results

    def _group_corrections_by_context(
        self,
    ) -> dict[str, list[FeedbackCorrection]]:
        """Group corrections by similar context."""
        groups: dict[str, list[FeedbackCorrection]] = {}

        for correction in self._corrections:
            # Create context key
            ctx = correction.context
            key = (
                f"{ctx.get('program_family', 'Unknown')}_"
                f"{ctx.get('activity_code', 'Unknown')}"
            )

            if key not in groups:
                groups[key] = []
            groups[key].append(correction)

        return groups

    async def update_rule_confidence(
        self,
        rule_id: str,
        was_helpful: bool,
    ) -> None:
        """Update rule confidence based on feedback."""
        rule = self._rules.get(rule_id)
        if not rule:
            return

        if was_helpful:
            rule.confidence = min(1.0, rule.confidence + 0.05)
        else:
            rule.confidence = max(0.0, rule.confidence - 0.1)

            # Deactivate if confidence too low
            if rule.confidence < 0.3:
                rule.is_active = False

    def get_applicable_rules(
        self,
        context: dict[str, Any],
    ) -> list[ExtractedRule]:
        """Get rules that apply to given context."""
        return [
            rule
            for rule in self._rules.values()
            if rule.is_active and rule.matches(context)
        ]

    # ===== Layer 3: Batch Retrain =====

    def _should_trigger_retrain(self) -> bool:
        """Check if batch retrain should be triggered."""
        # Check correction count
        if self._corrections_since_retrain >= self.config.force_retrain_corrections:
            return True

        if self._corrections_since_retrain < self.config.min_corrections:
            return False

        # Check average correction magnitude
        recent_corrections = self._corrections[-self.config.min_corrections :]
        avg_correction = sum(abs(c.correction_pct) for c in recent_corrections) / len(
            recent_corrections
        )

        if avg_correction > self.config.drift_threshold_pct:
            return True

        # Check time since last retrain
        if self._last_retrain:
            days_since = (datetime.now(timezone.utc) - self._last_retrain).days
            if days_since >= self.config.max_days_between_retrain:
                return True

        return False

    async def trigger_retrain(
        self,
        training_data: list[dict[str, Any]],
    ) -> TrainingResult:
        """
        Trigger batch model retraining (Layer 3).

        Args:
            training_data: Historical training data

        Returns:
            Training result
        """
        # Convert corrections to feedback format
        feedback_data = [
            {
                "corrected_value": c.corrected_hours,
                "original_item": c.context,
            }
            for c in self._corrections
        ]

        # Train new model
        result = await self.trainer.train(
            training_data=training_data,
            feedback_data=feedback_data,
        )

        if result.success:
            self._last_retrain = datetime.now(timezone.utc)
            self._corrections_since_retrain = 0

        return result

    async def get_stats(self) -> LearningStats:
        """Get learning system statistics."""
        # Calculate average correction percentage
        avg_correction = 0.0
        if self._corrections:
            avg_correction = sum(
                abs(c.correction_pct) for c in self._corrections
            ) / len(self._corrections)

        # Get active model info
        active_model = self.model_registry.get_active()
        model_version = None
        model_metrics = {}

        if active_model and active_model.version:
            model_version = active_model.version.version_id
            model_metrics = active_model.version.metrics

        # Calculate next scheduled retrain
        next_retrain = None
        if self._last_retrain:
            next_date = self._last_retrain + timedelta(
                days=self.config.max_days_between_retrain
            )
            next_retrain = next_date.isoformat()

        return LearningStats(
            total_corrections=len(self._corrections),
            corrections_since_retrain=self._corrections_since_retrain,
            avg_correction_pct=avg_correction,
            last_retrain=self._last_retrain.isoformat() if self._last_retrain else None,
            next_scheduled_retrain=next_retrain,
            active_rules=sum(1 for r in self._rules.values() if r.is_active),
            pending_rules=sum(1 for r in self._rules.values() if not r.is_active),
            model_version=model_version,
            model_metrics=model_metrics,
        )

    # ===== Public API =====

    def get_model(self) -> CostPredictionModel | None:
        """Get the active prediction model."""
        return self.model_registry.get_active()

    def get_rules(self, active_only: bool = True) -> list[ExtractedRule]:
        """Get all rules."""
        if active_only:
            return [r for r in self._rules.values() if r.is_active]
        return list(self._rules.values())

    def get_corrections(
        self,
        limit: int = 100,
    ) -> list[FeedbackCorrection]:
        """Get recent corrections."""
        return self._corrections[-limit:]

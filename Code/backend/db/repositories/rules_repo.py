"""
FPT Cost Brain 2.0 - Rules Repository
CRUD operations for learned rules
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import LearnedRule


class RulesRepository:
    """Repository for Learned Rules operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Create =====

    async def create(
        self,
        rule_name: str,
        description: str,
        condition_json: dict,
        adjustment_json: dict,
        confidence: float = 0.5,
        source_feedback_id: UUID | None = None,
    ) -> LearnedRule:
        """Create a new learned rule."""
        rule = LearnedRule(
            rule_name=rule_name,
            description=description,
            condition_json=condition_json,
            adjustment_json=adjustment_json,
            confidence=confidence,
            source_feedback_id=source_feedback_id,
        )
        self.db.add(rule)
        await self.db.commit()
        await self.db.refresh(rule)
        return rule

    async def create_from_extraction(
        self,
        extraction_result: dict,
        source_feedback_id: UUID | None = None,
    ) -> LearnedRule:
        """Create rule from LLM extraction result."""
        return await self.create(
            rule_name=extraction_result["rule_name"],
            description=extraction_result["description"],
            condition_json=extraction_result["conditions"],
            adjustment_json=extraction_result["adjustment"],
            confidence=extraction_result.get("confidence", 0.5),
            source_feedback_id=source_feedback_id,
        )

    # ===== Read =====

    async def get_by_id(self, rule_id: UUID) -> LearnedRule | None:
        """Get rule by ID."""
        result = await self.db.execute(
            select(LearnedRule).where(LearnedRule.id == rule_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, rule_name: str) -> LearnedRule | None:
        """Get rule by name."""
        result = await self.db.execute(
            select(LearnedRule).where(LearnedRule.rule_name == rule_name)
        )
        return result.scalar_one_or_none()

    async def get_active_rules(
        self,
        min_confidence: float = 0.3,
    ) -> list[LearnedRule]:
        """Get all active rules above confidence threshold."""
        result = await self.db.execute(
            select(LearnedRule)
            .where(
                and_(
                    LearnedRule.is_active == True,
                    LearnedRule.confidence >= min_confidence,
                )
            )
            .order_by(LearnedRule.confidence.desc())
        )
        return list(result.scalars().all())

    async def get_rules_for_context(
        self,
        context: dict[str, Any],
        min_confidence: float = 0.3,
    ) -> list[LearnedRule]:
        """
        Get rules that might apply to a given context.
        This is a simple implementation - for production,
        consider using vector similarity search.
        """
        active_rules = await self.get_active_rules(min_confidence)
        matching_rules = []

        for rule in active_rules:
            if self._rule_matches_context(rule, context):
                matching_rules.append(rule)

        return matching_rules

    def _rule_matches_context(
        self,
        rule: LearnedRule,
        context: dict[str, Any],
    ) -> bool:
        """Check if a rule's conditions match the context."""
        conditions = rule.condition_json

        if not conditions:
            return False

        field = conditions.get("field")
        operator = conditions.get("operator")
        value = conditions.get("value")

        if field not in context:
            return False

        context_value = context[field]

        if operator == "equals":
            return context_value == value
        elif operator == "contains":
            return value in str(context_value)
        elif operator == "greater_than":
            return float(context_value) > float(value)
        elif operator == "less_than":
            return float(context_value) < float(value)

        return False

    async def list_all(
        self,
        skip: int = 0,
        limit: int = 100,
        include_inactive: bool = False,
    ) -> list[LearnedRule]:
        """List all rules."""
        query = select(LearnedRule)

        if not include_inactive:
            query = query.where(LearnedRule.is_active == True)

        query = query.order_by(LearnedRule.confidence.desc()).offset(skip).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_pending_review(self) -> list[LearnedRule]:
        """Get rules pending human review."""
        result = await self.db.execute(
            select(LearnedRule)
            .where(LearnedRule.requires_review == True)
            .order_by(LearnedRule.created_at.desc())
        )
        return list(result.scalars().all())

    async def search(self, query: str, limit: int = 20) -> list[LearnedRule]:
        """Search rules by name or description."""
        search_term = f"%{query}%"
        result = await self.db.execute(
            select(LearnedRule)
            .where(
                (LearnedRule.rule_name.ilike(search_term))
                | (LearnedRule.description.ilike(search_term))
            )
            .order_by(LearnedRule.confidence.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ===== Update =====

    async def update(
        self,
        rule_id: UUID,
        **kwargs: Any,
    ) -> LearnedRule | None:
        """Update rule fields."""
        rule = await self.get_by_id(rule_id)
        if rule is None:
            return None

        for key, value in kwargs.items():
            if hasattr(rule, key):
                setattr(rule, key, value)

        # Use timezone-naive datetime for TIMESTAMP WITHOUT TIME ZONE column
        rule.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        await self.db.refresh(rule)
        return rule

    async def increase_confidence(
        self,
        rule_id: UUID,
        amount: float = 0.1,
        max_confidence: float = 0.95,
    ) -> LearnedRule | None:
        """Increase rule confidence (rule was applied successfully)."""
        rule = await self.get_by_id(rule_id)
        if rule is None:
            return None

        rule.confidence = min(rule.confidence + amount, max_confidence)
        rule.times_applied += 1
        rule.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        await self.db.commit()
        await self.db.refresh(rule)
        return rule

    async def decrease_confidence(
        self,
        rule_id: UUID,
        amount: float = 0.15,
        min_confidence: float = 0.1,
    ) -> LearnedRule | None:
        """Decrease rule confidence (rule was overridden)."""
        rule = await self.get_by_id(rule_id)
        if rule is None:
            return None

        rule.confidence = max(rule.confidence - amount, min_confidence)
        rule.times_overridden += 1
        rule.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Flag for review if confidence drops too low
        if rule.confidence < 0.3:
            rule.requires_review = True

        await self.db.commit()
        await self.db.refresh(rule)
        return rule

    async def activate(self, rule_id: UUID) -> LearnedRule | None:
        """Activate a rule."""
        return await self.update(rule_id, is_active=True, requires_review=False)

    async def deactivate(self, rule_id: UUID) -> LearnedRule | None:
        """Deactivate a rule."""
        return await self.update(rule_id, is_active=False)

    async def approve_rule(
        self,
        rule_id: UUID,
        reviewed_by: UUID,
    ) -> LearnedRule | None:
        """Approve a rule after human review."""
        return await self.update(
            rule_id,
            requires_review=False,
            is_active=True,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    async def reject_rule(
        self,
        rule_id: UUID,
        reviewed_by: UUID,
        rejection_reason: str | None = None,
    ) -> LearnedRule | None:
        """Reject a rule after human review."""
        return await self.update(
            rule_id,
            requires_review=False,
            is_active=False,
            reviewed_by=reviewed_by,
            reviewed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    # ===== Delete =====

    async def delete(self, rule_id: UUID) -> bool:
        """Delete a rule."""
        rule = await self.get_by_id(rule_id)
        if rule is None:
            return False

        await self.db.delete(rule)
        await self.db.commit()
        return True

    # ===== Analytics =====

    async def count(self, include_inactive: bool = False) -> int:
        """Count rules."""
        query = select(func.count(LearnedRule.id))
        if not include_inactive:
            query = query.where(LearnedRule.is_active == True)
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def get_statistics(self) -> dict[str, Any]:
        """Get rule statistics."""
        total = await self.count(include_inactive=True)
        active = await self.count(include_inactive=False)
        pending = await self.db.execute(
            select(func.count(LearnedRule.id)).where(
                LearnedRule.requires_review == True
            )
        )

        avg_confidence = await self.db.execute(
            select(func.avg(LearnedRule.confidence)).where(
                LearnedRule.is_active == True
            )
        )

        most_applied = await self.db.execute(
            select(LearnedRule)
            .where(LearnedRule.is_active == True)
            .order_by(LearnedRule.times_applied.desc())
            .limit(5)
        )

        return {
            "total": total,
            "active": active,
            "pending_review": pending.scalar() or 0,
            "avg_confidence": avg_confidence.scalar() or 0.0,
            "most_applied": [
                {"name": r.rule_name, "times_applied": r.times_applied}
                for r in most_applied.scalars().all()
            ],
        }

    async def find_conflicting_rules(
        self,
        new_rule_conditions: dict,
    ) -> list[LearnedRule]:
        """Find rules that might conflict with a new rule."""
        # Get active rules and check for potential conflicts
        active_rules = await self.get_active_rules()
        conflicts = []

        new_field = new_rule_conditions.get("field")

        for rule in active_rules:
            existing_field = rule.condition_json.get("field")
            if existing_field == new_field:
                # Same field - potential conflict
                conflicts.append(rule)

        return conflicts

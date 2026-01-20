"""
FPT Cost Brain 2.0 - Feedback Repository
CRUD operations for feedback corrections and learning
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import FeedbackCorrection, QuotationBreakdown


class FeedbackRepository:
    """Repository for Feedback Correction operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Create =====

    async def create(
        self,
        quotation_id: UUID,
        breakdown_id: UUID,
        original_value: float,
        corrected_value: float,
        field_name: str,
        reason: str | None = None,
        created_by: UUID | None = None,
    ) -> FeedbackCorrection:
        """Create a new feedback correction."""
        correction_percent = (
            ((corrected_value - original_value) / original_value * 100)
            if original_value != 0
            else 0.0
        )

        feedback = FeedbackCorrection(
            quotation_id=quotation_id,
            breakdown_id=breakdown_id,
            original_value=original_value,
            corrected_value=corrected_value,
            field_name=field_name,
            correction_percent=correction_percent,
            reason=reason,
            created_by=created_by,
        )
        self.db.add(feedback)
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    async def create_from_breakdown_edit(
        self,
        breakdown_id: UUID,
        original_hours: float,
        corrected_hours: float,
        reason: str | None = None,
        created_by: UUID | None = None,
    ) -> FeedbackCorrection:
        """Create feedback from a breakdown edit."""
        breakdown = await self.db.execute(
            select(QuotationBreakdown).where(QuotationBreakdown.id == breakdown_id)
        )
        bd = breakdown.scalar_one_or_none()

        if bd is None:
            raise ValueError(f"Breakdown {breakdown_id} not found")

        return await self.create(
            quotation_id=bd.quotation_id,
            breakdown_id=breakdown_id,
            original_value=original_hours,
            corrected_value=corrected_hours,
            field_name="hours",
            reason=reason,
            created_by=created_by,
        )

    # ===== Read =====

    async def get_by_id(self, feedback_id: UUID) -> FeedbackCorrection | None:
        """Get feedback by ID."""
        result = await self.db.execute(
            select(FeedbackCorrection).where(FeedbackCorrection.id == feedback_id)
        )
        return result.scalar_one_or_none()

    async def get_for_quotation(
        self,
        quotation_id: UUID,
    ) -> list[FeedbackCorrection]:
        """Get all feedback for a quotation."""
        result = await self.db.execute(
            select(FeedbackCorrection)
            .where(FeedbackCorrection.quotation_id == quotation_id)
            .order_by(FeedbackCorrection.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_for_breakdown(
        self,
        breakdown_id: UUID,
    ) -> list[FeedbackCorrection]:
        """Get all feedback for a specific breakdown."""
        result = await self.db.execute(
            select(FeedbackCorrection)
            .where(FeedbackCorrection.breakdown_id == breakdown_id)
            .order_by(FeedbackCorrection.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_unprocessed(self, limit: int = 100) -> list[FeedbackCorrection]:
        """Get feedback that hasn't been processed for rule extraction."""
        result = await self.db.execute(
            select(FeedbackCorrection)
            .where(FeedbackCorrection.processed_for_learning == False)
            .order_by(FeedbackCorrection.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_activity_code(
        self,
        activity_code: str,
        limit: int = 50,
    ) -> list[FeedbackCorrection]:
        """Get feedback for a specific activity code."""
        result = await self.db.execute(
            select(FeedbackCorrection)
            .join(QuotationBreakdown)
            .where(QuotationBreakdown.activity_code == activity_code)
            .order_by(FeedbackCorrection.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent(
        self,
        days: int = 30,
        limit: int = 100,
    ) -> list[FeedbackCorrection]:
        """Get recent feedback corrections."""
        cutoff = datetime.now(timezone.utc) - timezone.timedelta(days=days)
        result = await self.db.execute(
            select(FeedbackCorrection)
            .where(FeedbackCorrection.created_at >= cutoff)
            .order_by(FeedbackCorrection.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ===== Update =====

    async def mark_as_processed(
        self,
        feedback_id: UUID,
        extracted_rule_id: UUID | None = None,
    ) -> FeedbackCorrection | None:
        """Mark feedback as processed for learning."""
        feedback = await self.get_by_id(feedback_id)
        if feedback is None:
            return None

        feedback.processed_for_learning = True
        feedback.extracted_rule_id = extracted_rule_id
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    async def mark_batch_as_processed(
        self,
        feedback_ids: list[UUID],
        extracted_rule_id: UUID | None = None,
    ) -> int:
        """Mark multiple feedback as processed."""
        count = 0
        for fid in feedback_ids:
            result = await self.mark_as_processed(fid, extracted_rule_id)
            if result:
                count += 1
        return count

    async def update_embedding_id(
        self,
        feedback_id: UUID,
        embedding_id: str,
    ) -> FeedbackCorrection | None:
        """Set the embedding ID for a feedback."""
        feedback = await self.get_by_id(feedback_id)
        if feedback is None:
            return None

        feedback.embedding_id = embedding_id
        await self.db.commit()
        await self.db.refresh(feedback)
        return feedback

    # ===== Analytics =====

    async def count_unprocessed(self) -> int:
        """Count unprocessed feedback."""
        result = await self.db.execute(
            select(func.count(FeedbackCorrection.id)).where(
                FeedbackCorrection.processed_for_learning == False
            )
        )
        return result.scalar() or 0

    async def get_statistics(self) -> dict[str, Any]:
        """Get feedback statistics."""
        total = await self.db.execute(select(func.count(FeedbackCorrection.id)))
        unprocessed = await self.count_unprocessed()

        avg_correction = await self.db.execute(
            select(func.avg(func.abs(FeedbackCorrection.correction_percent)))
        )

        # Corrections by field
        by_field = await self.db.execute(
            select(
                FeedbackCorrection.field_name,
                func.count(FeedbackCorrection.id),
                func.avg(FeedbackCorrection.correction_percent),
            ).group_by(FeedbackCorrection.field_name)
        )

        return {
            "total": total.scalar() or 0,
            "unprocessed": unprocessed,
            "avg_correction_percent": avg_correction.scalar() or 0.0,
            "by_field": {
                row[0]: {"count": row[1], "avg_correction": row[2]}
                for row in by_field.all()
            },
        }

    async def get_correction_trend(
        self,
        days: int = 30,
    ) -> list[dict]:
        """Get correction trend over time."""
        cutoff = datetime.now(timezone.utc) - timezone.timedelta(days=days)

        result = await self.db.execute(
            select(
                func.date(FeedbackCorrection.created_at),
                func.count(FeedbackCorrection.id),
                func.avg(func.abs(FeedbackCorrection.correction_percent)),
            )
            .where(FeedbackCorrection.created_at >= cutoff)
            .group_by(func.date(FeedbackCorrection.created_at))
            .order_by(func.date(FeedbackCorrection.created_at))
        )

        return [
            {
                "date": row[0].isoformat() if row[0] else None,
                "count": row[1],
                "avg_correction": row[2],
            }
            for row in result.all()
        ]

    async def should_trigger_retrain(
        self,
        min_corrections: int = 5,
        max_corrections: int = 20,
        drift_threshold: float = 15.0,
    ) -> tuple[bool, str]:
        """
        Check if model retraining should be triggered.
        Returns (should_retrain, reason).
        """
        unprocessed = await self.count_unprocessed()

        if unprocessed >= max_corrections:
            return True, f"Max corrections reached ({unprocessed} >= {max_corrections})"

        if unprocessed >= min_corrections:
            # Check average correction magnitude
            result = await self.db.execute(
                select(func.avg(func.abs(FeedbackCorrection.correction_percent))).where(
                    FeedbackCorrection.processed_for_learning == False
                )
            )
            avg_correction = result.scalar() or 0.0

            if avg_correction > drift_threshold:
                return (
                    True,
                    f"High drift detected ({avg_correction:.1f}% > {drift_threshold}%)",
                )

        return False, "No retrain needed"

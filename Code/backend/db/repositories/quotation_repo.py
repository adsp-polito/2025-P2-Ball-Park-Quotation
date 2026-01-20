"""
FPT Cost Brain 2.0 - Quotation Repository
CRUD operations for Quotations and Breakdowns
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from db.models import Quotation, QuotationBreakdown
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


class QuotationRepository:
    """Repository for Quotation and Breakdown operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Quotation CRUD =====

    async def create_quotation(
        self,
        pr_id: UUID,
        version: int = 1,
        total_hours: float = 0.0,
        total_cost_eur: float = 0.0,
        confidence_score: float | None = None,
        estimation_method: str = "hybrid",
        created_by: UUID | None = None,
    ) -> Quotation:
        """Create a new quotation."""
        quotation = Quotation(
            pr_id=pr_id,
            version=version,
            total_hours=total_hours,
            total_cost_eur=total_cost_eur,
            confidence_score=confidence_score,
            estimation_method=estimation_method,
            created_by=created_by,
        )
        self.db.add(quotation)
        await self.db.commit()
        await self.db.refresh(quotation)
        return quotation

    async def get_quotation(self, quotation_id: UUID) -> Quotation | None:
        """Get quotation by ID with breakdowns."""
        result = await self.db.execute(
            select(Quotation)
            .where(Quotation.id == quotation_id)
            .options(selectinload(Quotation.breakdown_items))
        )
        return result.scalar_one_or_none()

    async def get_latest_for_pr(self, pr_id: UUID) -> Quotation | None:
        """Get the latest quotation for a PR."""
        result = await self.db.execute(
            select(Quotation)
            .where(Quotation.pr_id == pr_id)
            .order_by(Quotation.version.desc())
            .limit(1)
            .options(selectinload(Quotation.breakdown_items))
        )
        return result.scalar_one_or_none()

    async def get_all_for_pr(self, pr_id: UUID) -> list[Quotation]:
        """Get all quotation versions for a PR."""
        result = await self.db.execute(
            select(Quotation)
            .where(Quotation.pr_id == pr_id)
            .order_by(Quotation.version.desc())
            .options(selectinload(Quotation.breakdown_items))
        )
        return list(result.scalars().all())

    async def update_quotation(
        self,
        quotation_id: UUID,
        **kwargs: Any,
    ) -> Quotation | None:
        """Update quotation fields."""
        quotation = await self.get_quotation(quotation_id)
        if quotation is None:
            return None

        for key, value in kwargs.items():
            if hasattr(quotation, key):
                setattr(quotation, key, value)

        # Use timezone-naive datetime for TIMESTAMP WITHOUT TIME ZONE column
        quotation.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        await self.db.refresh(quotation)
        return quotation

    async def finalize_quotation(
        self,
        quotation_id: UUID,
        finalized_by: UUID,
    ) -> Quotation | None:
        """Mark quotation as finalized."""
        return await self.update_quotation(
            quotation_id,
            is_finalized=True,
            finalized_at=datetime.now(timezone.utc).replace(tzinfo=None),
            finalized_by=finalized_by,
        )

    async def create_new_version(self, pr_id: UUID) -> Quotation:
        """Create a new version of quotation for a PR."""
        latest = await self.get_latest_for_pr(pr_id)
        new_version = (latest.version + 1) if latest else 1

        return await self.create_quotation(
            pr_id=pr_id,
            version=new_version,
        )

    # ===== Breakdown CRUD =====

    async def add_breakdown(
        self,
        quotation_id: UUID,
        activity_code: str,
        activity_name: str,
        hours: float = 0.0,
        hourly_rate_eur: float = 75.0,
        confidence_score: float | None = None,
        reasoning: str | None = None,
        source: str = "model",
    ) -> QuotationBreakdown:
        """Add a breakdown item to a quotation."""
        cost = hours * hourly_rate_eur

        breakdown = QuotationBreakdown(
            quotation_id=quotation_id,
            activity_code=activity_code,
            activity_name=activity_name,
            hours=hours,
            hourly_rate_eur=hourly_rate_eur,
            cost_eur=cost,
            confidence_score=confidence_score,
            reasoning=reasoning,
            source=source,
        )
        self.db.add(breakdown)

        # Update quotation totals
        await self._update_quotation_totals(quotation_id)

        await self.db.commit()
        await self.db.refresh(breakdown)
        return breakdown

    async def add_breakdowns_batch(
        self,
        quotation_id: UUID,
        breakdowns: list[dict],
    ) -> list[QuotationBreakdown]:
        """Add multiple breakdown items at once."""
        items = []
        for bd in breakdowns:
            hours = bd.get("hours", 0.0)
            rate = bd.get("hourly_rate_eur", 75.0)

            item = QuotationBreakdown(
                quotation_id=quotation_id,
                activity_code=bd["activity_code"],
                activity_name=bd["activity_name"],
                hours=hours,
                hourly_rate_eur=rate,
                cost_eur=hours * rate,
                confidence_score=bd.get("confidence_score"),
                reasoning=bd.get("reasoning"),
                source=bd.get("source", "model"),
            )
            items.append(item)
            self.db.add(item)

        await self._update_quotation_totals(quotation_id)
        await self.db.commit()

        for item in items:
            await self.db.refresh(item)

        return items

    async def get_breakdown(self, breakdown_id: UUID) -> QuotationBreakdown | None:
        """Get a specific breakdown item."""
        result = await self.db.execute(
            select(QuotationBreakdown).where(QuotationBreakdown.id == breakdown_id)
        )
        return result.scalar_one_or_none()

    async def get_breakdowns_for_quotation(
        self,
        quotation_id: UUID,
    ) -> list[QuotationBreakdown]:
        """Get all breakdowns for a quotation."""
        result = await self.db.execute(
            select(QuotationBreakdown)
            .where(QuotationBreakdown.quotation_id == quotation_id)
            .order_by(QuotationBreakdown.activity_code)
        )
        return list(result.scalars().all())

    async def update_breakdown(
        self,
        breakdown_id: UUID,
        hours: float | None = None,
        hourly_rate_eur: float | None = None,
        user_edited: bool = True,
        edit_reason: str | None = None,
        **kwargs: Any,
    ) -> QuotationBreakdown | None:
        """Update a breakdown item."""
        breakdown = await self.get_breakdown(breakdown_id)
        if breakdown is None:
            return None

        if hours is not None:
            breakdown.hours = hours
        if hourly_rate_eur is not None:
            breakdown.hourly_rate_eur = hourly_rate_eur

        breakdown.cost_eur = breakdown.hours * breakdown.hourly_rate_eur
        breakdown.user_edited = user_edited
        breakdown.edit_reason = edit_reason

        for key, value in kwargs.items():
            if hasattr(breakdown, key):
                setattr(breakdown, key, value)

        await self._update_quotation_totals(breakdown.quotation_id)
        await self.db.commit()
        await self.db.refresh(breakdown)
        return breakdown

    async def delete_breakdown(self, breakdown_id: UUID) -> bool:
        """Delete a breakdown item."""
        breakdown = await self.get_breakdown(breakdown_id)
        if breakdown is None:
            return False

        quotation_id = breakdown.quotation_id
        await self.db.delete(breakdown)
        await self._update_quotation_totals(quotation_id)
        await self.db.commit()
        return True

    # ===== Helper Methods =====

    async def _update_quotation_totals(self, quotation_id: UUID) -> None:
        """Recalculate quotation totals from breakdowns."""
        result = await self.db.execute(
            select(
                func.sum(QuotationBreakdown.hours),
                func.sum(QuotationBreakdown.cost_eur),
                func.avg(QuotationBreakdown.confidence_score),
            ).where(QuotationBreakdown.quotation_id == quotation_id)
        )
        row = result.one()

        total_hours = row[0] or 0.0
        total_cost = row[1] or 0.0
        avg_confidence = row[2]

        quotation = await self.get_quotation(quotation_id)
        if quotation:
            quotation.total_hours = total_hours
            quotation.total_cost_eur = total_cost
            if avg_confidence is not None:
                quotation.confidence_score = avg_confidence

    # ===== Analytics =====

    async def get_statistics(
        self,
        program_family: str | None = None,
    ) -> dict[str, Any]:
        """Get quotation statistics."""
        base_query = select(Quotation).where(Quotation.status == "approved")

        if program_family:
            from db.models import ProductRequest

            base_query = base_query.join(ProductRequest).where(
                ProductRequest.platform == program_family
            )

        # Total count
        count_result = await self.db.execute(
            select(func.count(Quotation.id)).select_from(base_query.subquery())
        )
        total_count = count_result.scalar() or 0

        # Averages
        avg_result = await self.db.execute(
            select(
                func.avg(Quotation.total_hours),
                func.avg(Quotation.total_cost_eur),
                func.avg(Quotation.confidence_score),
            ).select_from(base_query.subquery())
        )
        row = avg_result.one()

        return {
            "total_quotations": total_count,
            "avg_hours": row[0] or 0.0,
            "avg_cost_eur": row[1] or 0.0,
            "avg_confidence": row[2] or 0.0,
        }

    async def get_activity_statistics(self) -> list[dict]:
        """Get statistics by activity code."""
        result = await self.db.execute(
            select(
                QuotationBreakdown.activity_code,
                QuotationBreakdown.activity_name,
                func.count(QuotationBreakdown.id),
                func.avg(QuotationBreakdown.hours),
                func.avg(QuotationBreakdown.cost_eur),
            )
            .group_by(
                QuotationBreakdown.activity_code,
                QuotationBreakdown.activity_name,
            )
            .order_by(func.count(QuotationBreakdown.id).desc())
        )

        return [
            {
                "activity_code": row[0],
                "activity_name": row[1],
                "count": row[2],
                "avg_hours": row[3],
                "avg_cost": row[4],
            }
            for row in result.all()
        ]

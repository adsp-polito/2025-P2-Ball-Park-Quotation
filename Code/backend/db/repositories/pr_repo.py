"""
FPT Cost Brain 2.0 - Product Request Repository
CRUD operations for Product Requests
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from db.models import ProductRequest, Quotation
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


class ProductRequestRepository:
    """Repository for Product Request operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Create =====

    async def create(
        self,
        pr_code: str,
        title: str,
        program_family: str | None = None,
        customer: str | None = None,
        raw_data: dict | None = None,
        uploaded_by: UUID | None = None,
    ) -> ProductRequest:
        """Create a new product request."""
        pr = ProductRequest(
            pr_number=pr_code,  # Model uses pr_number, not pr_code
            title=title,
            platform=program_family,  # Map to existing field
            description=str(raw_data)
            if raw_data
            else None,  # Store raw_data in description
            created_by=uploaded_by,
        )
        self.db.add(pr)
        await self.db.commit()
        await self.db.refresh(pr)
        return pr

    # ===== Read =====

    async def get_by_id(self, pr_id: UUID) -> ProductRequest | None:
        """Get product request by ID."""
        result = await self.db.execute(
            select(ProductRequest)
            .where(ProductRequest.id == pr_id)
            .options(selectinload(ProductRequest.quotations))
        )
        return result.scalar_one_or_none()

    async def get_by_code(self, pr_code: str) -> ProductRequest | None:
        """Get product request by PR code."""
        result = await self.db.execute(
            select(ProductRequest).where(ProductRequest.pr_number == pr_code)
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        skip: int = 0,
        limit: int = 50,
        program_family: str | None = None,
        customer: str | None = None,
        status: str | None = None,
    ) -> list[ProductRequest]:
        """List product requests with optional filtering."""
        query = select(ProductRequest)

        conditions = []
        if program_family:
            conditions.append(ProductRequest.platform == program_family)
        # Note: customer field not in model, skip filtering
        if status:
            conditions.append(ProductRequest.status == status)

        if conditions:
            query = query.where(and_(*conditions))

        query = (
            query.order_by(ProductRequest.created_at.desc()).offset(skip).limit(limit)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def count(
        self,
        program_family: str | None = None,
        customer: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count product requests with optional filtering."""
        query = select(func.count(ProductRequest.id))

        conditions = []
        if program_family:
            conditions.append(ProductRequest.platform == program_family)
        # Note: customer field not in model, skip filtering
        if status:
            conditions.append(ProductRequest.status == status)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.db.execute(query)
        return result.scalar() or 0

    async def search(
        self,
        query: str,
        limit: int = 20,
    ) -> list[ProductRequest]:
        """Search product requests by title, code, or description."""
        search_term = f"%{query}%"
        stmt = (
            select(ProductRequest)
            .where(
                (ProductRequest.title.ilike(search_term))
                | (ProductRequest.pr_number.ilike(search_term))
                | (ProductRequest.description.ilike(search_term))
            )
            .order_by(ProductRequest.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_by_program_family(
        self,
        program_family: str,
        limit: int = 100,
    ) -> list[ProductRequest]:
        """Get all PRs for a program family (platform)."""
        result = await self.db.execute(
            select(ProductRequest)
            .where(ProductRequest.platform == program_family)
            .order_by(ProductRequest.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent(self, limit: int = 10) -> list[ProductRequest]:
        """Get most recent product requests."""
        result = await self.db.execute(
            select(ProductRequest)
            .where(ProductRequest.status != "draft")
            .order_by(ProductRequest.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def list_with_quotations(
        self,
        limit: int = 500,
        status: str | None = None,
    ) -> list[ProductRequest]:
        """List product requests with their quotations eagerly loaded."""
        query = (
            select(ProductRequest)
            .options(selectinload(ProductRequest.quotations))
            .where(ProductRequest.status != "draft")
            .where(ProductRequest.status != "deleted")
        )

        if status:
            query = query.where(ProductRequest.status == status)

        query = query.order_by(ProductRequest.created_at.desc()).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    # ===== Update =====

    async def update(
        self,
        pr_id: UUID,
        **kwargs: Any,
    ) -> ProductRequest | None:
        """Update product request fields."""
        pr = await self.get_by_id(pr_id)
        if pr is None:
            return None

        for key, value in kwargs.items():
            if hasattr(pr, key):
                setattr(pr, key, value)

        # Use timezone-naive datetime for TIMESTAMP WITHOUT TIME ZONE column
        pr.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        await self.db.refresh(pr)
        return pr

    async def update_status(
        self,
        pr_id: UUID,
        status: str,
    ) -> ProductRequest | None:
        """Update product request status."""
        return await self.update(pr_id, status=status)

    async def set_embedding_id(
        self,
        pr_id: UUID,
        embedding_id: str,
    ) -> ProductRequest | None:
        """Set the vector embedding ID for a PR."""
        return await self.update(pr_id, embedding_id=embedding_id)

    # ===== Delete =====

    async def delete(self, pr_id: UUID) -> bool:
        """Delete a product request (soft delete by setting status)."""
        pr = await self.get_by_id(pr_id)
        if pr is None:
            return False

        pr.status = "deleted"
        # Use timezone-naive datetime for TIMESTAMP WITHOUT TIME ZONE column
        pr.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        return True

    async def hard_delete(self, pr_id: UUID) -> bool:
        """Permanently delete a product request."""
        pr = await self.get_by_id(pr_id)
        if pr is None:
            return False

        await self.db.delete(pr)
        await self.db.commit()
        return True

    # ===== Analytics =====

    async def get_statistics(self) -> dict[str, Any]:
        """Get PR statistics for dashboard."""
        total = await self.count()
        by_status = await self.db.execute(
            select(ProductRequest.status, func.count(ProductRequest.id)).group_by(
                ProductRequest.status
            )
        )
        by_family = await self.db.execute(
            select(ProductRequest.platform, func.count(ProductRequest.id))
            .where(ProductRequest.platform.isnot(None))
            .group_by(ProductRequest.platform)
            .order_by(func.count(ProductRequest.id).desc())
            .limit(10)
        )

        return {
            "total": total,
            "by_status": dict(by_status.all()),
            "by_family": dict(by_family.all()),
        }

"""
FPT Cost Brain 2.0 - Audit Repository
CRUD operations for audit logging
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AuditLog


class AuditRepository:
    """Repository for Audit Log operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Create =====

    async def log(
        self,
        action: str,
        entity_type: str,
        entity_id: UUID | str | None = None,
        user_id: UUID | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditLog:
        """Create an audit log entry."""
        entry = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=UUID(str(entity_id)) if entity_id else None,
            user_id=user_id,
            new_value=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)
        return entry

    async def log_action(
        self,
        action: str,
        entity_type: str,
        entity_id: UUID | str | None = None,
        user_id: UUID | None = None,
        old_value: Any = None,
        new_value: Any = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        """Log an action with before/after values."""
        details = {}
        if old_value is not None:
            details["old_value"] = old_value
        if new_value is not None:
            details["new_value"] = new_value

        return await self.log(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            details=details,
            ip_address=ip_address,
        )

    # ===== Convenience Methods =====

    async def log_create(
        self,
        entity_type: str,
        entity_id: UUID | str,
        user_id: UUID | None = None,
        details: dict | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        """Log a create action."""
        return await self.log(
            action="create",
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            details=details,
            ip_address=ip_address,
        )

    async def log_update(
        self,
        entity_type: str,
        entity_id: UUID | str,
        user_id: UUID | None = None,
        changes: dict | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        """Log an update action."""
        return await self.log(
            action="update",
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            details={"changes": changes} if changes else None,
            ip_address=ip_address,
        )

    async def log_delete(
        self,
        entity_type: str,
        entity_id: UUID | str,
        user_id: UUID | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        """Log a delete action."""
        return await self.log(
            action="delete",
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            ip_address=ip_address,
        )

    async def log_login(
        self,
        user_id: UUID,
        ip_address: str | None = None,
        user_agent: str | None = None,
        success: bool = True,
    ) -> AuditLog:
        """Log a login attempt."""
        return await self.log(
            action="login_success" if success else "login_failed",
            entity_type="user",
            entity_id=user_id,
            user_id=user_id if success else None,
            details={"success": success},
            ip_address=ip_address,
            user_agent=user_agent,
        )

    async def log_export(
        self,
        export_type: str,
        entity_id: UUID | str,
        user_id: UUID | None = None,
        format: str = "unknown",
        ip_address: str | None = None,
    ) -> AuditLog:
        """Log an export action."""
        return await self.log(
            action="export",
            entity_type=export_type,
            entity_id=entity_id,
            user_id=user_id,
            details={"format": format},
            ip_address=ip_address,
        )

    # ===== Read =====

    async def get_by_id(self, log_id: UUID) -> AuditLog | None:
        """Get audit log by ID."""
        result = await self.db.execute(select(AuditLog).where(AuditLog.id == log_id))
        return result.scalar_one_or_none()

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        action: str | None = None,
        entity_type: str | None = None,
        user_id: UUID | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> list[AuditLog]:
        """List audit logs with filtering."""
        query = select(AuditLog)

        conditions = []
        if action:
            conditions.append(AuditLog.action == action)
        if entity_type:
            conditions.append(AuditLog.entity_type == entity_type)
        if user_id:
            conditions.append(AuditLog.user_id == user_id)
        if from_date:
            conditions.append(AuditLog.created_at >= from_date)
        if to_date:
            conditions.append(AuditLog.created_at <= to_date)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_for_entity(
        self,
        entity_type: str,
        entity_id: UUID | str,
        limit: int = 50,
    ) -> list[AuditLog]:
        """Get all audit logs for a specific entity."""
        result = await self.db.execute(
            select(AuditLog)
            .where(
                and_(
                    AuditLog.entity_type == entity_type,
                    AuditLog.entity_id == str(entity_id),
                )
            )
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_for_user(
        self,
        user_id: UUID,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get all audit logs for a specific user."""
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent(
        self,
        hours: int = 24,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Get recent audit logs."""
        cutoff = datetime.now(timezone.utc) - timezone.timedelta(hours=hours)
        result = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.created_at >= cutoff)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def search(
        self,
        query: str,
        limit: int = 50,
    ) -> list[AuditLog]:
        """Search audit logs by action or entity type."""
        search_term = f"%{query}%"
        result = await self.db.execute(
            select(AuditLog)
            .where(
                (AuditLog.action.ilike(search_term))
                | (AuditLog.entity_type.ilike(search_term))
            )
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ===== Analytics =====

    async def count(
        self,
        action: str | None = None,
        entity_type: str | None = None,
        from_date: datetime | None = None,
    ) -> int:
        """Count audit logs with optional filtering."""
        query = select(func.count(AuditLog.id))

        conditions = []
        if action:
            conditions.append(AuditLog.action == action)
        if entity_type:
            conditions.append(AuditLog.entity_type == entity_type)
        if from_date:
            conditions.append(AuditLog.created_at >= from_date)

        if conditions:
            query = query.where(and_(*conditions))

        result = await self.db.execute(query)
        return result.scalar() or 0

    async def get_statistics(
        self,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get audit statistics."""
        cutoff = datetime.now(timezone.utc) - timezone.timedelta(days=days)

        # Total count
        total = await self.count(from_date=cutoff)

        # By action
        by_action = await self.db.execute(
            select(AuditLog.action, func.count(AuditLog.id))
            .where(AuditLog.created_at >= cutoff)
            .group_by(AuditLog.action)
            .order_by(func.count(AuditLog.id).desc())
        )

        # By entity type
        by_entity = await self.db.execute(
            select(AuditLog.entity_type, func.count(AuditLog.id))
            .where(AuditLog.created_at >= cutoff)
            .group_by(AuditLog.entity_type)
            .order_by(func.count(AuditLog.id).desc())
        )

        # Daily trend
        daily = await self.db.execute(
            select(
                func.date(AuditLog.created_at),
                func.count(AuditLog.id),
            )
            .where(AuditLog.created_at >= cutoff)
            .group_by(func.date(AuditLog.created_at))
            .order_by(func.date(AuditLog.created_at))
        )

        return {
            "total": total,
            "period_days": days,
            "by_action": dict(by_action.all()),
            "by_entity_type": dict(by_entity.all()),
            "daily_trend": [
                {"date": row[0].isoformat() if row[0] else None, "count": row[1]}
                for row in daily.all()
            ],
        }

    async def get_user_activity(
        self,
        user_id: UUID,
        days: int = 30,
    ) -> dict[str, Any]:
        """Get activity summary for a specific user."""
        cutoff = datetime.now(timezone.utc) - timezone.timedelta(days=days)

        total = await self.db.execute(
            select(func.count(AuditLog.id)).where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= cutoff,
                )
            )
        )

        by_action = await self.db.execute(
            select(AuditLog.action, func.count(AuditLog.id))
            .where(
                and_(
                    AuditLog.user_id == user_id,
                    AuditLog.created_at >= cutoff,
                )
            )
            .group_by(AuditLog.action)
        )

        last_activity = await self.db.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .order_by(AuditLog.created_at.desc())
            .limit(1)
        )
        last = last_activity.scalar_one_or_none()

        return {
            "user_id": str(user_id),
            "total_actions": total.scalar() or 0,
            "period_days": days,
            "by_action": dict(by_action.all()),
            "last_activity": last.created_at.isoformat() if last else None,
        }

    # ===== Cleanup =====

    async def delete_old_logs(self, days: int = 365) -> int:
        """Delete audit logs older than specified days."""
        cutoff = datetime.now(timezone.utc) - timezone.timedelta(days=days)

        result = await self.db.execute(
            select(AuditLog).where(AuditLog.created_at < cutoff)
        )
        old_logs = result.scalars().all()

        count = 0
        for log in old_logs:
            await self.db.delete(log)
            count += 1

        await self.db.commit()
        return count

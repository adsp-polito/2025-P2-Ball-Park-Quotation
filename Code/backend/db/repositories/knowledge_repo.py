"""
FPT Cost Brain 2.0 - Knowledge Repository
CRUD operations for knowledge documents and acronyms
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Acronym, KnowledgeDocument


class KnowledgeRepository:
    """Repository for Knowledge Documents and Acronyms."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Knowledge Documents =====

    async def create_document(
        self,
        title: str,
        content: str,
        doc_type: str = "general",
        source_file_path: str | None = None,
        metadata: dict | None = None,
        uploaded_by: UUID | None = None,
    ) -> KnowledgeDocument:
        """Create a new knowledge document."""
        doc = KnowledgeDocument(
            title=title,
            content=content,
            doc_type=doc_type,
            source_file_path=source_file_path,
            uploaded_by=uploaded_by,
        )
        self.db.add(doc)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def get_document(self, doc_id: UUID) -> KnowledgeDocument | None:
        """Get document by ID."""
        result = await self.db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 50,
        doc_type: str | None = None,
        is_indexed: bool | None = None,
    ) -> list[KnowledgeDocument]:
        """List documents with optional filtering."""
        query = select(KnowledgeDocument)

        conditions = []
        if doc_type:
            conditions.append(KnowledgeDocument.doc_type == doc_type)
        if is_indexed is not None:
            conditions.append(KnowledgeDocument.is_indexed == is_indexed)

        if conditions:
            query = query.where(and_(*conditions))

        query = (
            query.order_by(KnowledgeDocument.created_at.desc())
            .offset(skip)
            .limit(limit)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def search_documents(
        self,
        query: str,
        limit: int = 20,
    ) -> list[KnowledgeDocument]:
        """Search documents by title or content."""
        search_term = f"%{query}%"
        result = await self.db.execute(
            select(KnowledgeDocument)
            .where(
                (KnowledgeDocument.title.ilike(search_term))
                | (KnowledgeDocument.content.ilike(search_term))
            )
            .order_by(KnowledgeDocument.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_unindexed_documents(
        self,
        limit: int = 100,
    ) -> list[KnowledgeDocument]:
        """Get documents that need to be indexed."""
        result = await self.db.execute(
            select(KnowledgeDocument)
            .where(KnowledgeDocument.is_indexed == False)
            .order_by(KnowledgeDocument.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_document(
        self,
        doc_id: UUID,
        **kwargs: Any,
    ) -> KnowledgeDocument | None:
        """Update document fields."""
        doc = await self.get_document(doc_id)
        if doc is None:
            return None

        for key, value in kwargs.items():
            if hasattr(doc, key):
                setattr(doc, key, value)

        # Use timezone-naive datetime for TIMESTAMP WITHOUT TIME ZONE column
        doc.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        await self.db.refresh(doc)
        return doc

    async def mark_as_indexed(
        self,
        doc_id: UUID,
        chunk_count: int = 0,
    ) -> KnowledgeDocument | None:
        """Mark document as indexed."""
        return await self.update_document(
            doc_id,
            is_indexed=True,
            chunk_count=chunk_count,
            indexed_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    async def delete_document(self, doc_id: UUID) -> bool:
        """Delete a document."""
        doc = await self.get_document(doc_id)
        if doc is None:
            return False

        await self.db.delete(doc)
        await self.db.commit()
        return True

    # ===== Acronyms =====

    async def create_acronym(
        self,
        acronym: str,
        full_form: str,
        description: str | None = None,
        category: str = "general",
    ) -> Acronym:
        """Create a new acronym."""
        acr = Acronym(
            acronym=acronym.upper(),
            full_form=full_form,
            description=description,
            category=category,
        )
        self.db.add(acr)
        await self.db.commit()
        await self.db.refresh(acr)
        return acr

    async def get_acronym(self, acronym_id: UUID) -> Acronym | None:
        """Get acronym by ID."""
        result = await self.db.execute(select(Acronym).where(Acronym.id == acronym_id))
        return result.scalar_one_or_none()

    async def get_acronym_by_text(
        self,
        text: str,
    ) -> Acronym | None:
        """Get acronym by its text."""
        result = await self.db.execute(
            select(Acronym).where(Acronym.acronym == text.upper())
        )
        return result.scalar_one_or_none()

    async def list_acronyms(
        self,
        skip: int = 0,
        limit: int = 100,
        category: str | None = None,
    ) -> list[Acronym]:
        """List acronyms with optional filtering."""
        query = select(Acronym)

        if category:
            query = query.where(Acronym.category == category)

        query = query.order_by(Acronym.acronym).offset(skip).limit(limit)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def search_acronyms(
        self,
        query: str,
        limit: int = 20,
    ) -> list[Acronym]:
        """Search acronyms by text or full form."""
        search_term = f"%{query}%"
        result = await self.db.execute(
            select(Acronym)
            .where(
                (Acronym.acronym.ilike(search_term))
                | (Acronym.full_form.ilike(search_term))
                | (Acronym.description.ilike(search_term))
            )
            .order_by(Acronym.acronym)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_acronym(
        self,
        acronym_id: UUID,
        **kwargs: Any,
    ) -> Acronym | None:
        """Update acronym fields."""
        acr = await self.get_acronym(acronym_id)
        if acr is None:
            return None

        for key, value in kwargs.items():
            if hasattr(acr, key):
                if key == "acronym":
                    value = value.upper()
                setattr(acr, key, value)

        # Use timezone-naive datetime for TIMESTAMP WITHOUT TIME ZONE column
        acr.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await self.db.commit()
        await self.db.refresh(acr)
        return acr

    async def delete_acronym(self, acronym_id: UUID) -> bool:
        """Delete an acronym."""
        acr = await self.get_acronym(acronym_id)
        if acr is None:
            return False

        await self.db.delete(acr)
        await self.db.commit()
        return True

    async def bulk_create_acronyms(
        self,
        acronyms: list[dict],
    ) -> int:
        """Create multiple acronyms at once."""
        count = 0
        for data in acronyms:
            existing = await self.get_acronym_by_text(data["acronym"])
            if existing:
                continue

            acr = Acronym(
                acronym=data["acronym"].upper(),
                full_form=data["full_form"],
                description=data.get("description"),
                category=data.get("category", "general"),
            )
            self.db.add(acr)
            count += 1

        await self.db.commit()
        return count

    # ===== Analytics =====

    async def get_statistics(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        doc_count = await self.db.execute(select(func.count(KnowledgeDocument.id)))
        acr_count = await self.db.execute(select(func.count(Acronym.id)))

        docs_by_type = await self.db.execute(
            select(
                KnowledgeDocument.doc_type,
                func.count(KnowledgeDocument.id),
            ).group_by(KnowledgeDocument.doc_type)
        )

        acrs_by_category = await self.db.execute(
            select(Acronym.category, func.count(Acronym.id)).group_by(Acronym.category)
        )

        unindexed = await self.db.execute(
            select(func.count(KnowledgeDocument.id)).where(
                KnowledgeDocument.is_indexed == False
            )
        )

        return {
            "total_documents": doc_count.scalar() or 0,
            "total_acronyms": acr_count.scalar() or 0,
            "unindexed_documents": unindexed.scalar() or 0,
            "documents_by_type": dict(docs_by_type.all()),
            "acronyms_by_category": dict(acrs_by_category.all()),
        }

"""
FPT Cost Brain 2.0 - Knowledge API
Endpoints for knowledge base management (documents and acronyms)
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from db.models import User
from db.repositories.knowledge_repo import KnowledgeRepository

router = APIRouter(tags=["Knowledge"])


# ===== Schemas =====


class AcronymCreate(BaseModel):
    """Schema for creating an acronym."""

    acronym: str
    full_form: str
    description: str | None = None
    category: str = "general"


class AcronymUpdate(BaseModel):
    """Schema for updating an acronym."""

    acronym: str | None = None
    full_form: str | None = None
    description: str | None = None
    category: str | None = None


class AcronymResponse(BaseModel):
    """Schema for acronym response."""

    id: str
    acronym: str
    full_form: str
    description: str | None
    category: str | None

    class Config:
        from_attributes = True


class DocumentCreate(BaseModel):
    """Schema for creating a document."""

    title: str
    content: str
    document_type: str = "general"
    source: str | None = None
    metadata: dict | None = None


class DocumentResponse(BaseModel):
    """Schema for document response."""

    id: str
    title: str
    content: str
    doc_type: str | None
    category: str | None
    source_file_path: str | None
    is_indexed: bool
    chunk_count: int | None
    created_at: str


# ===== Acronym Endpoints =====


@router.get("/acronyms")
async def list_acronyms(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    category: str | None = None,
    search: str | None = None,
):
    """List acronyms with optional filtering."""
    knowledge_repo = KnowledgeRepository(db)

    if search:
        acronyms = await knowledge_repo.search_acronyms(search, limit=limit)
    else:
        acronyms = await knowledge_repo.list_acronyms(
            skip=skip,
            limit=limit,
            category=category,
        )

    return {
        "items": [
            AcronymResponse(
                id=str(a.id),
                acronym=a.acronym,
                full_form=a.full_form,
                description=a.description,
                category=a.category,
            ).model_dump()
            for a in acronyms
        ],
        "total": len(acronyms),
    }


@router.get("/acronyms/{acronym_id}", response_model=AcronymResponse)
async def get_acronym(
    acronym_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get a specific acronym by ID."""
    knowledge_repo = KnowledgeRepository(db)

    acronym = await knowledge_repo.get_acronym(acronym_id)
    if not acronym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Acronym not found",
        )

    return AcronymResponse(
        id=str(acronym.id),
        acronym=acronym.acronym,
        full_form=acronym.full_form,
        description=acronym.description,
        category=acronym.category,
    )


@router.get("/acronyms/lookup/{text}")
async def lookup_acronym(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    text: str,
):
    """Look up an acronym by its text."""
    knowledge_repo = KnowledgeRepository(db)

    acronym = await knowledge_repo.get_acronym_by_text(text)
    if not acronym:
        return {"found": False, "acronym": text, "full_form": None}

    return {
        "found": True,
        "acronym": acronym.acronym,
        "full_form": acronym.full_form,
        "description": acronym.description,
        "category": acronym.category,
    }


@router.post(
    "/acronyms", response_model=AcronymResponse, status_code=status.HTTP_201_CREATED
)
async def create_acronym(
    data: AcronymCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new acronym."""
    knowledge_repo = KnowledgeRepository(db)

    # Check if already exists
    existing = await knowledge_repo.get_acronym_by_text(data.acronym)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Acronym '{data.acronym}' already exists",
        )

    acronym = await knowledge_repo.create_acronym(
        acronym=data.acronym,
        full_form=data.full_form,
        description=data.description,
        category=data.category,
    )

    return AcronymResponse(
        id=str(acronym.id),
        acronym=acronym.acronym,
        full_form=acronym.full_form,
        description=acronym.description,
        category=acronym.category,
    )


@router.patch("/acronyms/{acronym_id}", response_model=AcronymResponse)
async def update_acronym(
    acronym_id: UUID,
    data: AcronymUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update an existing acronym."""
    knowledge_repo = KnowledgeRepository(db)

    update_data = data.model_dump(exclude_unset=True)
    acronym = await knowledge_repo.update_acronym(acronym_id, **update_data)

    if not acronym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Acronym not found",
        )

    return AcronymResponse(
        id=str(acronym.id),
        acronym=acronym.acronym,
        full_form=acronym.full_form,
        description=acronym.description,
        category=acronym.category,
    )


@router.delete("/acronyms/{acronym_id}")
async def delete_acronym(
    acronym_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete an acronym."""
    knowledge_repo = KnowledgeRepository(db)

    success = await knowledge_repo.delete_acronym(acronym_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Acronym not found",
        )

    return {"message": "Acronym deleted"}


@router.post("/acronyms/bulk")
async def bulk_create_acronyms(
    acronyms: list[AcronymCreate],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create multiple acronyms at once."""
    knowledge_repo = KnowledgeRepository(db)

    count = await knowledge_repo.bulk_create_acronyms(
        [a.model_dump() for a in acronyms]
    )

    return {"created": count, "total_submitted": len(acronyms)}


# ===== Document Endpoints =====


@router.get("/documents")
async def list_documents(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    doc_type: str | None = None,
    is_indexed: bool | None = None,
    search: str | None = None,
):
    """List knowledge documents."""
    knowledge_repo = KnowledgeRepository(db)

    if search:
        documents = await knowledge_repo.search_documents(search, limit=limit)
    else:
        documents = await knowledge_repo.list_documents(
            skip=skip,
            limit=limit,
            doc_type=doc_type,
            is_indexed=is_indexed,
        )

    return {
        "items": [
            DocumentResponse(
                id=str(d.id),
                title=d.title,
                content=d.content[:500] + "..." if len(d.content) > 500 else d.content,
                doc_type=d.doc_type,
                category=d.category,
                source_file_path=d.source_file_path,
                is_indexed=d.is_indexed,
                chunk_count=d.chunk_count,
                created_at=d.created_at.isoformat(),
            ).model_dump()
            for d in documents
        ],
        "total": len(documents),
    }


@router.get("/documents/{doc_id}", response_model=DocumentResponse)
async def get_document(
    doc_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get a specific document."""
    knowledge_repo = KnowledgeRepository(db)

    document = await knowledge_repo.get_document(doc_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse(
        id=str(document.id),
        title=document.title,
        content=document.content,
        doc_type=document.doc_type,
        category=document.category,
        source_file_path=document.source_file_path,
        is_indexed=document.is_indexed,
        chunk_count=document.chunk_count,
        created_at=document.created_at.isoformat(),
    )


@router.post(
    "/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED
)
async def create_document(
    data: DocumentCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Create a new knowledge document."""
    knowledge_repo = KnowledgeRepository(db)

    document = await knowledge_repo.create_document(
        title=data.title,
        content=data.content,
        doc_type=data.document_type,
        source_file_path=data.source,
        metadata=data.metadata,
        uploaded_by=current_user.id,
    )

    return DocumentResponse(
        id=str(document.id),
        title=document.title,
        content=document.content,
        doc_type=document.doc_type,
        category=document.category,
        source_file_path=document.source_file_path,
        is_indexed=document.is_indexed,
        chunk_count=document.chunk_count,
        created_at=document.created_at.isoformat(),
    )


@router.post("/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
    document_type: str = "uploaded",
):
    """Upload a document file (txt, pdf, docx)."""
    allowed_types = [".txt", ".pdf", ".docx", ".md"]
    file_ext = (
        "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    )

    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {allowed_types}",
        )

    content = await file.read()

    # TODO: Parse different file types
    # For now, assume text content
    try:
        text_content = content.decode("utf-8")
    except UnicodeDecodeError:
        text_content = f"[Binary content from {file.filename}]"

    knowledge_repo = KnowledgeRepository(db)

    document = await knowledge_repo.create_document(
        title=file.filename,
        content=text_content,
        doc_type=document_type,
        source_file_path=f"upload:{file.filename}",
        uploaded_by=current_user.id,
    )

    return {
        "id": str(document.id),
        "title": document.title,
        "size_bytes": len(content),
        "is_indexed": False,
        "message": "Document uploaded. Indexing will happen in background.",
    }


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete a knowledge document."""
    knowledge_repo = KnowledgeRepository(db)

    success = await knowledge_repo.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # TODO: Also delete from vector store

    return {"message": "Document deleted"}


@router.get("/statistics")
async def get_knowledge_statistics(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get knowledge base statistics."""
    knowledge_repo = KnowledgeRepository(db)

    stats = await knowledge_repo.get_statistics()

    return stats

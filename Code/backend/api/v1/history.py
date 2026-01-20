"""
FPT Cost Brain 2.0 - History API
Endpoints for browsing historical PRs and quotations
"""

from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from db.models import User
from db.repositories.pr_repo import ProductRequestRepository
from db.repositories.quotation_repo import QuotationRepository

router = APIRouter(tags=["History"])


# ===== Schemas =====


class PRListItem(BaseModel):
    """Schema for PR list item."""

    id: str
    pr_code: str
    title: str
    program_family: str | None
    customer: str | None
    status: str
    created_at: str
    total_hours: float | None
    total_cost_eur: float | None


class PRDetailResponse(BaseModel):
    """Schema for PR detail response."""

    id: str
    pr_code: str
    title: str
    description: str | None
    program_family: str | None
    customer: str | None
    status: str
    created_at: str
    updated_at: str | None
    raw_data: dict[str, Any]
    quotations: list[dict[str, Any]]


class ComparisonRequest(BaseModel):
    """Schema for PR comparison request."""

    pr_ids: list[str]


class ComparisonResponse(BaseModel):
    """Schema for PR comparison response."""

    prs: list[dict[str, Any]]
    differences: list[dict[str, Any]]


# ===== Endpoints =====


@router.get("/prs")
async def list_prs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    program_family: str | None = None,
    customer: str | None = None,
    status: str | None = None,
    search: str | None = None,
):
    """
    List historical product requests.

    Supports filtering by program family, customer, and status.
    """
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)

    if search:
        prs = await pr_repo.search(search, limit=limit)
    else:
        prs = await pr_repo.list(
            skip=skip,
            limit=limit,
            program_family=program_family,
            customer=customer,
            status=status,
        )

    total = await pr_repo.count(
        program_family=program_family,
        customer=customer,
        status=status,
    )

    # Get quotation info for each PR
    items = []
    for pr in prs:
        quotation = await quotation_repo.get_latest_for_pr(pr.id)
        items.append(
            PRListItem(
                id=str(pr.id),
                pr_code=pr.pr_number,  # Model uses pr_number
                title=pr.title,
                program_family=pr.platform,  # Model uses platform
                customer=pr.plant,  # Model uses plant as customer proxy
                status=pr.status,
                created_at=pr.created_at.isoformat(),
                total_hours=quotation.total_hours if quotation else None,
                total_cost_eur=quotation.total_cost_eur if quotation else None,
            )
        )

    return {
        "items": [item.model_dump() for item in items],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/prs/{pr_id}", response_model=PRDetailResponse)
async def get_pr_detail(
    pr_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get detailed information about a specific PR."""
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)

    pr = await pr_repo.get_by_id(pr_id)
    if not pr:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product request not found",
        )

    # Get all quotation versions
    quotations = await quotation_repo.get_all_for_pr(pr_id)

    return PRDetailResponse(
        id=str(pr.id),
        pr_code=pr.pr_number,  # Model uses pr_number
        title=pr.title,
        description=pr.description,
        program_family=pr.platform,  # Model uses platform
        customer=pr.plant,  # Model uses plant as customer proxy
        status=pr.status,
        created_at=pr.created_at.isoformat(),
        updated_at=pr.updated_at.isoformat() if pr.updated_at else None,
        raw_data={},  # raw_data not in model, return empty dict
        quotations=[
            {
                "id": str(q.id),
                "version": q.version,
                "total_hours": q.total_hours,
                "total_cost_eur": q.total_cost_eur,
                "confidence_score": q.confidence_score,
                "is_finalized": q.is_finalized,
                "created_at": q.created_at.isoformat(),
            }
            for q in quotations
        ],
    )


@router.get("/prs/{pr_id}/quotation/{version}")
async def get_quotation_version(
    pr_id: UUID,
    version: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get a specific quotation version for a PR."""
    quotation_repo = QuotationRepository(db)

    quotations = await quotation_repo.get_all_for_pr(pr_id)
    quotation = next((q for q in quotations if q.version == version), None)

    if not quotation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quotation version {version} not found",
        )

    breakdowns = await quotation_repo.get_breakdowns_for_quotation(quotation.id)

    return {
        "id": str(quotation.id),
        "pr_id": str(pr_id),
        "version": quotation.version,
        "total_hours": quotation.total_hours,
        "total_cost_eur": quotation.total_cost_eur,
        "confidence_score": quotation.confidence_score,
        "estimation_method": quotation.estimation_method,
        "is_finalized": quotation.is_finalized,
        "created_at": quotation.created_at.isoformat(),
        "breakdowns": [
            {
                "id": str(bd.id),
                "activity_code": bd.activity_code,
                "activity_name": bd.activity_name,
                "hours": bd.hours,
                "hourly_rate_eur": bd.hourly_rate_eur,
                "cost_eur": bd.cost_eur,
                "confidence_score": bd.confidence_score,
                "reasoning": bd.reasoning,
                "source": bd.source,
                "user_edited": bd.user_edited,
            }
            for bd in breakdowns
        ],
    }


@router.post("/compare", response_model=ComparisonResponse)
async def compare_prs(
    comparison: ComparisonRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Compare multiple PRs side by side.

    Returns breakdown comparison and highlights differences.
    """
    if len(comparison.pr_ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 PRs to compare",
        )

    if len(comparison.pr_ids) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot compare more than 5 PRs at once",
        )

    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)

    prs_data = []
    all_activities = set()

    for pr_id in comparison.pr_ids:
        pr = await pr_repo.get_by_id(UUID(pr_id))
        if not pr:
            continue

        quotation = await quotation_repo.get_latest_for_pr(UUID(pr_id))
        breakdowns = []
        if quotation:
            breakdowns = await quotation_repo.get_breakdowns_for_quotation(quotation.id)

        activity_map = {bd.activity_code: bd for bd in breakdowns}
        all_activities.update(activity_map.keys())

        prs_data.append(
            {
                "id": str(pr.id),
                "pr_code": pr.pr_code,
                "title": pr.title,
                "program_family": pr.program_family,
                "total_hours": quotation.total_hours if quotation else 0,
                "total_cost_eur": quotation.total_cost_eur if quotation else 0,
                "activities": activity_map,
            }
        )

    # Calculate differences
    differences = []
    for activity in sorted(all_activities):
        activity_diff = {
            "activity_code": activity,
            "values": [],
            "max_diff_percent": 0,
        }

        hours_values = []
        for pr_data in prs_data:
            bd = pr_data["activities"].get(activity)
            hours = bd.hours if bd else 0
            activity_diff["values"].append(
                {
                    "pr_id": pr_data["id"],
                    "pr_code": pr_data["pr_code"],
                    "hours": hours,
                    "cost_eur": bd.cost_eur if bd else 0,
                }
            )
            hours_values.append(hours)

        # Calculate max difference percentage
        if hours_values and max(hours_values) > 0:
            min_val = min(h for h in hours_values if h > 0) if any(hours_values) else 0
            max_val = max(hours_values)
            if min_val > 0:
                activity_diff["max_diff_percent"] = (
                    (max_val - min_val) / min_val
                ) * 100

        differences.append(activity_diff)

    # Sort by max difference
    differences.sort(key=lambda x: x["max_diff_percent"], reverse=True)

    return ComparisonResponse(
        prs=[
            {
                "id": p["id"],
                "pr_code": p["pr_code"],
                "title": p["title"],
                "program_family": p["program_family"],
                "total_hours": p["total_hours"],
                "total_cost_eur": p["total_cost_eur"],
            }
            for p in prs_data
        ],
        differences=differences,
    )


@router.get("/statistics")
async def get_history_statistics(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get statistics about historical PRs and quotations."""
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)

    pr_stats = await pr_repo.get_statistics()
    quotation_stats = await quotation_repo.get_statistics()
    activity_stats = await quotation_repo.get_activity_statistics()

    return {
        "product_requests": pr_stats,
        "quotations": quotation_stats,
        "top_activities": activity_stats[:10],
    }


@router.get("/similar/{pr_id}")
async def get_similar_prs(
    pr_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(5, ge=1, le=20),
):
    """
    Find PRs similar to the given one.

    Uses vector similarity search on PR embeddings.
    """
    pr_repo = ProductRequestRepository(db)

    pr = await pr_repo.get_by_id(pr_id)
    if not pr:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product request not found",
        )

    # If PR has same program family, get others from same family
    similar_prs = []
    if pr.program_family:
        family_prs = await pr_repo.get_by_program_family(
            pr.program_family,
            limit=limit + 1,
        )
        similar_prs = [p for p in family_prs if p.id != pr_id][:limit]

    # TODO: Add vector similarity search using Qdrant

    return {
        "source_pr": {
            "id": str(pr.id),
            "pr_code": pr.pr_code,
            "program_family": pr.program_family,
        },
        "similar": [
            {
                "id": str(p.id),
                "pr_code": p.pr_code,
                "title": p.title,
                "program_family": p.program_family,
                "similarity_reason": "same_program_family",
            }
            for p in similar_prs
        ],
    }

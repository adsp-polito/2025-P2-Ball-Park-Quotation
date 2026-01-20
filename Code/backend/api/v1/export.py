"""
FPT Cost Brain 2.0 - Export API
Endpoints for exporting PE02 documents with real generation
"""

import json
import logging
from io import BytesIO
from typing import Annotated
from uuid import UUID

from app.dependencies import get_current_user, get_db
from db.models import User
from db.repositories.audit_repo import AuditRepository
from db.repositories.pr_repo import ProductRequestRepository
from db.repositories.quotation_repo import QuotationRepository
from export.pe02_generator import ExportOptions, PE02Data, PE02Generator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["Export"])
logger = logging.getLogger(__name__)


# ===== Schemas =====


class ExportRequest(BaseModel):
    """Schema for export request."""

    session_id: str
    format: str = "pptx"  # pptx, xlsx, or bundle
    include_breakdown: bool = True
    include_summary: bool = True
    include_confidence: bool = True
    include_reasoning: bool = False
    include_similar_prs: bool = True
    language: str = "en"


class ExportResponse(BaseModel):
    """Schema for export response."""

    download_url: str
    filename: str
    format: str
    size_bytes: int


# ===== Helper Functions =====


async def build_pe02_data(
    session_id: str,
    pr_repo: ProductRequestRepository,
    quotation_repo: QuotationRepository,
) -> PE02Data:
    """
    Build PE02Data from database records and Redis state.
    """
    from services.estimation_service import load_state_from_redis

    # Load state from Redis to get the actual pr_id
    state = await load_state_from_redis(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found in cache",
        )

    # Get the pr_id from state (session_id != pr_id)
    pr_id_str = state.get("pr_id")
    if not pr_id_str:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product request ID not found in session",
        )

    try:
        pr_id = UUID(pr_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PR ID format in session",
        )

    # Get PR from database using the correct pr_id
    pr = await pr_repo.get_by_id(pr_id)
    if not pr:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product request not found",
        )

    # Get quotation and breakdowns using pr_id (not session_id)
    quotation = await quotation_repo.get_latest_for_pr(pr_id)
    breakdowns = []
    if quotation:
        breakdown_items = await quotation_repo.get_breakdowns_for_quotation(
            quotation.id
        )
        breakdowns = [
            {
                "activity_code": item.pe_function or "",
                "activity_name": item.activity_description
                or item.sub_function
                or item.pe_function
                or "",
                "hours": float(item.hours_manpower) if item.hours_manpower else 0,
                "hourly_rate_eur": 75.0,  # Default hourly rate
                "cost_eur": float(item.cost_eur) if item.cost_eur else 0,
                "confidence_score": float(item.confidence) if item.confidence else 0.7,
                "reasoning": item.basis or "",
                "source": "ai" if item.ai_generated else "manual",
            }
            for item in breakdown_items
        ]

    # Use state that was already loaded above
    pr_summary = state.get("pr_summary", {})
    similar_prs_raw = state.get("similar_prs", [])

    # Format similar PRs
    similar_prs = []
    for spr in similar_prs_raw[:5]:
        if isinstance(spr, dict):
            similar_prs.append(
                {
                    "pr_code": spr.get("pr_code", spr.get("pr_number", "Unknown")),
                    "title": spr.get("title", ""),
                    "total_hours": spr.get("total_hours", 0),
                    "similarity_score": spr.get(
                        "similarity_score", spr.get("score", 0)
                    ),
                }
            )

    # Build PE02Data
    return PE02Data(
        pr_code=pr.pr_number or f"PR-{str(pr_id)[:8]}",
        pr_title=pr.title or "Untitled",
        program_family=pr.platform or "",
        customer=pr.plant or "",
        project_phase=pr.tier or "",
        breakdown=breakdowns,
        total_hours=float(quotation.total_hours_manpower)
        if quotation and quotation.total_hours_manpower
        else sum(b["hours"] for b in breakdowns),
        total_cost_eur=float(quotation.total_cost_eur)
        if quotation and quotation.total_cost_eur
        else sum(b["cost_eur"] for b in breakdowns),
        program_size=pr.program_size or pr_summary.get("program_size", ""),
        complexity_score=float(pr.program_size_confidence)
        if pr.program_size_confidence
        else pr_summary.get("complexity_score", 0.5),
        key_features=pr_summary.get("key_features", []),
        similar_prs=similar_prs,
        estimation_method=state.get("estimation_method", "hybrid")
        if state
        else "hybrid",
        overall_confidence=float(quotation.confidence_score)
        if quotation and quotation.confidence_score
        else 0.7,
    )


# ===== Endpoints =====


@router.post("/pptx")
async def export_pptx(
    export_request: ExportRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Export estimation as PE02 PowerPoint presentation.

    The PPTX includes:
    - Slide 1: Title with Summary Information
    - Slide 2: Program Sizing (Functional Classification)
    - Slide 3: Engineering Activity Summary (PE.02 table)
    - Slide 4: Similar Projects (optional)
    """
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)
    audit_repo = AuditRepository(db)

    session_id = export_request.session_id
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required",
        )

    # Build PE02 data from database (uses session_id to look up pr_id from Redis)
    pe02_data = await build_pe02_data(session_id, pr_repo, quotation_repo)

    # Configure export options
    options = ExportOptions(
        language=export_request.language,
        include_confidence=export_request.include_confidence,
        include_reasoning=export_request.include_reasoning,
        include_similar_prs=export_request.include_similar_prs,
    )

    # Generate PPTX using PE02Generator
    generator = PE02Generator(options=options)

    try:
        pptx_bytes = generator.generate_pptx(pe02_data)
    except Exception as e:
        logger.error(f"PPTX generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PPTX: {str(e)}",
        )

    # Log the export
    quotation = await quotation_repo.get_latest_for_pr(session_id)
    if quotation:
        await audit_repo.log_export(
            export_type="quotation",
            entity_id=quotation.id,
            user_id=current_user.id,
            format="pptx",
        )

    # Return the file
    filename = f"PE02_{pe02_data.pr_code}_v{quotation.version if quotation else 1}.pptx"
    pptx_buffer = BytesIO(pptx_bytes)

    logger.info(f"Generated PPTX export: {filename} ({len(pptx_bytes)} bytes)")

    return StreamingResponse(
        pptx_buffer,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pptx_bytes)),
        },
    )


@router.post("/xlsx")
async def export_xlsx(
    export_request: ExportRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Export estimation as Excel spreadsheet.

    The Excel file includes:
    - Summary sheet with PR information
    - Breakdown sheet with all activity details
    - Similar PRs sheet (optional)
    """
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)
    audit_repo = AuditRepository(db)

    session_id = export_request.session_id
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required",
        )

    # Build PE02 data from database (uses session_id to look up pr_id from Redis)
    pe02_data = await build_pe02_data(session_id, pr_repo, quotation_repo)

    # Configure export options
    options = ExportOptions(
        language=export_request.language,
        include_confidence=export_request.include_confidence,
        include_reasoning=export_request.include_reasoning,
        include_similar_prs=export_request.include_similar_prs,
    )

    # Generate XLSX using PE02Generator
    generator = PE02Generator(options=options)

    try:
        xlsx_bytes = generator.generate_xlsx(pe02_data)
    except Exception as e:
        logger.error(f"XLSX generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate XLSX: {str(e)}",
        )

    # Log the export
    quotation = await quotation_repo.get_latest_for_pr(session_id)
    if quotation:
        await audit_repo.log_export(
            export_type="quotation",
            entity_id=quotation.id,
            user_id=current_user.id,
            format="xlsx",
        )

    # Return the file
    filename = (
        f"Quotation_{pe02_data.pr_code}_v{quotation.version if quotation else 1}.xlsx"
    )
    xlsx_buffer = BytesIO(xlsx_bytes)

    logger.info(f"Generated XLSX export: {filename} ({len(xlsx_bytes)} bytes)")

    return StreamingResponse(
        xlsx_buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(xlsx_bytes)),
        },
    )


@router.post("/bundle")
async def export_bundle(
    export_request: ExportRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Export both PPTX and XLSX as a ZIP bundle.

    Includes:
    - PE02 PowerPoint presentation
    - Detailed Excel breakdown
    - Summary JSON file with metadata
    """
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)
    audit_repo = AuditRepository(db)

    session_id = export_request.session_id
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session ID is required",
        )

    # Build PE02 data from database (uses session_id to look up pr_id from Redis)
    pe02_data = await build_pe02_data(session_id, pr_repo, quotation_repo)

    # Configure export options
    options = ExportOptions(
        language=export_request.language,
        include_confidence=export_request.include_confidence,
        include_reasoning=export_request.include_reasoning,
        include_similar_prs=export_request.include_similar_prs,
    )

    # Generate bundle using PE02Generator
    generator = PE02Generator(options=options)

    try:
        # Generate individual files
        pptx_bytes = generator.generate_pptx(pe02_data)
        xlsx_bytes = generator.generate_xlsx(pe02_data)

        # Create summary JSON
        summary_data = {
            "pr_code": pe02_data.pr_code,
            "pr_title": pe02_data.pr_title,
            "program_family": pe02_data.program_family,
            "customer": pe02_data.customer,
            "total_hours": pe02_data.total_hours,
            "total_cost_eur": pe02_data.total_cost_eur,
            "program_size": pe02_data.program_size,
            "overall_confidence": pe02_data.overall_confidence,
            "estimation_method": pe02_data.estimation_method,
            "activity_count": len(pe02_data.breakdown),
            "similar_prs_count": len(pe02_data.similar_prs),
            "key_features": pe02_data.key_features,
            "generated_at": pe02_data.created_at,
            "language": export_request.language,
        }

        # Create ZIP bundle
        import zipfile

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"PE02_{pe02_data.pr_code}.pptx", pptx_bytes)
            zf.writestr(f"Quotation_{pe02_data.pr_code}.xlsx", xlsx_bytes)
            zf.writestr(
                "summary.json",
                json.dumps(summary_data, indent=2, ensure_ascii=False),
            )

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()

    except Exception as e:
        logger.error(f"Bundle generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate bundle: {str(e)}",
        )

    # Log the export
    quotation = await quotation_repo.get_latest_for_pr(session_id)
    if quotation:
        await audit_repo.log_export(
            export_type="quotation",
            entity_id=quotation.id,
            user_id=current_user.id,
            format="bundle",
        )

    # Return the file
    filename = (
        f"Export_{pe02_data.pr_code}_v{quotation.version if quotation else 1}.zip"
    )

    logger.info(f"Generated bundle export: {filename} ({len(zip_bytes)} bytes)")

    return StreamingResponse(
        BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


@router.get("/preview/{session_id}")
async def preview_export(
    session_id: UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = "pptx",
):
    """
    Get a preview of what will be exported.

    Returns metadata and structure without generating the actual file.
    """
    pr_repo = ProductRequestRepository(db)
    quotation_repo = QuotationRepository(db)

    pr = await pr_repo.get_by_id(session_id)
    if not pr:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Estimation session not found",
        )

    quotation = await quotation_repo.get_latest_for_pr(session_id)
    breakdowns = []
    if quotation:
        breakdowns = await quotation_repo.get_breakdowns_for_quotation(quotation.id)

    # Get Redis state for additional preview info
    from services.estimation_service import load_state_from_redis

    state = await load_state_from_redis(str(session_id))
    pr_summary = state.get("pr_summary", {}) if state else {}
    similar_prs = state.get("similar_prs", []) if state else []

    return {
        "session_id": str(session_id),
        "pr_code": pr.pr_number,
        "title": pr.title,
        "format": format,
        "quotation": {
            "version": quotation.version if quotation else None,
            "total_hours": float(quotation.total_hours_manpower)
            if quotation and quotation.total_hours_manpower
            else 0,
            "total_cost_eur": float(quotation.total_cost_eur)
            if quotation and quotation.total_cost_eur
            else 0,
            "confidence_score": float(quotation.confidence_score)
            if quotation and quotation.confidence_score
            else 0,
            "item_count": len(breakdowns),
        },
        "summary": {
            "program_size": pr.program_size or pr_summary.get("program_size", ""),
            "key_features": pr_summary.get("key_features", [])[:5],
            "similar_prs_count": len(similar_prs),
        },
        "preview_content": {
            "slides" if format == "pptx" else "sheets": [
                {
                    "name": "Title & Summary" if format == "pptx" else "Summary",
                    "content_type": "info",
                },
                {
                    "name": "Program Sizing" if format == "pptx" else "Breakdown",
                    "content_type": "metrics",
                },
                {
                    "name": "PE.02 Summary" if format == "pptx" else "Details",
                    "content_type": "table",
                    "row_count": len(breakdowns),
                },
            ]
            + (
                [
                    {
                        "name": "Similar Projects",
                        "content_type": "table",
                        "row_count": len(similar_prs),
                    }
                ]
                if similar_prs
                else []
            ),
        },
        "available_formats": ["pptx", "xlsx", "bundle"],
    }


@router.get("/templates")
async def list_export_templates(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """List available export templates."""
    return {
        "templates": [
            {
                "id": "pe02_standard",
                "name": "PE02 Standard",
                "description": "Standard PE02 format with FPT branding",
                "formats": ["pptx", "xlsx"],
                "options": {
                    "include_confidence": True,
                    "include_reasoning": False,
                    "include_similar_prs": True,
                },
            },
            {
                "id": "pe02_detailed",
                "name": "PE02 Detailed",
                "description": "Detailed version with confidence scores and reasoning",
                "formats": ["pptx", "xlsx"],
                "options": {
                    "include_confidence": True,
                    "include_reasoning": True,
                    "include_similar_prs": True,
                },
            },
            {
                "id": "pe02_minimal",
                "name": "PE02 Minimal",
                "description": "Clean version without confidence scores",
                "formats": ["pptx", "xlsx"],
                "options": {
                    "include_confidence": False,
                    "include_reasoning": False,
                    "include_similar_prs": False,
                },
            },
        ],
        "languages": [
            {"code": "en", "name": "English"},
            {"code": "it", "name": "Italiano"},
        ],
    }

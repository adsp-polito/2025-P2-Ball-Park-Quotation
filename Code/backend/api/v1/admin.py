"""
FPT Cost Brain 2.0 - Admin API
Endpoints for administration and system management
"""

from typing import Annotated, Any
from uuid import UUID

from app.dependencies import get_current_user, get_db, require_admin
from db.models import User
from db.repositories.audit_repo import AuditRepository
from db.repositories.feedback_repo import FeedbackRepository
from db.repositories.rules_repo import RulesRepository
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["Admin"])


# ===== Schemas =====


class RuleResponse(BaseModel):
    """Schema for rule response."""

    id: str
    rule_name: str
    description: str
    condition_json: dict
    adjustment_json: dict
    confidence: float
    is_active: bool
    requires_review: bool
    times_applied: int
    times_overridden: int
    created_at: str


class RuleUpdate(BaseModel):
    """Schema for rule update."""

    rule_name: str | None = None
    description: str | None = None
    condition_json: dict | None = None
    adjustment_json: dict | None = None
    confidence: float | None = None
    is_active: bool | None = None


class RetrainRequest(BaseModel):
    """Schema for retrain request."""

    force: bool = False
    include_recent_feedback: bool = True


class ModelStatusResponse(BaseModel):
    """Schema for model status response."""

    current_version: str
    last_trained: str | None
    training_samples: int
    accuracy_metrics: dict[str, float]
    pending_feedback: int
    should_retrain: bool
    retrain_reason: str | None


# ===== Rules Management =====


@router.get("/rules")
async def list_rules(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    include_inactive: bool = False,
    search: str | None = None,
):
    """List learned rules."""
    rules_repo = RulesRepository(db)

    if search:
        rules = await rules_repo.search(search, limit=limit)
    else:
        rules = await rules_repo.list_all(
            skip=skip,
            limit=limit,
            include_inactive=include_inactive,
        )

    return {
        "items": [
            RuleResponse(
                id=str(r.id),
                rule_name=r.rule_name,
                description=r.description,
                condition_json=r.condition_json,
                adjustment_json=r.adjustment_json,
                confidence=r.confidence,
                is_active=r.is_active,
                requires_review=r.requires_review,
                times_applied=r.times_applied,
                times_overridden=r.times_overridden,
                created_at=r.created_at.isoformat(),
            ).model_dump()
            for r in rules
        ],
        "total": len(rules),
    }


@router.get("/rules/pending-review")
async def get_pending_rules(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get rules pending human review."""
    rules_repo = RulesRepository(db)

    rules = await rules_repo.get_pending_review()

    return {
        "items": [
            RuleResponse(
                id=str(r.id),
                rule_name=r.rule_name,
                description=r.description,
                condition_json=r.condition_json,
                adjustment_json=r.adjustment_json,
                confidence=r.confidence,
                is_active=r.is_active,
                requires_review=r.requires_review,
                times_applied=r.times_applied,
                times_overridden=r.times_overridden,
                created_at=r.created_at.isoformat(),
            ).model_dump()
            for r in rules
        ],
        "count": len(rules),
    }


@router.get("/rules/{rule_id}", response_model=RuleResponse)
async def get_rule(
    rule_id: UUID,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get a specific rule."""
    rules_repo = RulesRepository(db)

    rule = await rules_repo.get_by_id(rule_id)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found",
        )

    return RuleResponse(
        id=str(rule.id),
        rule_name=rule.rule_name,
        description=rule.description,
        condition_json=rule.condition_json,
        adjustment_json=rule.adjustment_json,
        confidence=rule.confidence,
        is_active=rule.is_active,
        requires_review=rule.requires_review,
        times_applied=rule.times_applied,
        times_overridden=rule.times_overridden,
        created_at=rule.created_at.isoformat(),
    )


@router.patch("/rules/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: UUID,
    update: RuleUpdate,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update a rule."""
    rules_repo = RulesRepository(db)

    update_data = update.model_dump(exclude_unset=True)
    rule = await rules_repo.update(rule_id, **update_data)

    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found",
        )

    return RuleResponse(
        id=str(rule.id),
        rule_name=rule.rule_name,
        description=rule.description,
        condition_json=rule.condition_json,
        adjustment_json=rule.adjustment_json,
        confidence=rule.confidence,
        is_active=rule.is_active,
        requires_review=rule.requires_review,
        times_applied=rule.times_applied,
        times_overridden=rule.times_overridden,
        created_at=rule.created_at.isoformat(),
    )


@router.post("/rules/{rule_id}/approve")
async def approve_rule(
    rule_id: UUID,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Approve a rule after review."""
    rules_repo = RulesRepository(db)

    rule = await rules_repo.approve_rule(rule_id, current_user.id)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found",
        )

    return {"message": "Rule approved and activated", "rule_id": str(rule_id)}


@router.post("/rules/{rule_id}/reject")
async def reject_rule(
    rule_id: UUID,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    reason: str | None = None,
):
    """Reject a rule after review."""
    rules_repo = RulesRepository(db)

    rule = await rules_repo.reject_rule(rule_id, current_user.id, reason)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found",
        )

    return {"message": "Rule rejected and deactivated", "rule_id": str(rule_id)}


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: UUID,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Delete a rule."""
    rules_repo = RulesRepository(db)

    success = await rules_repo.delete(rule_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rule not found",
        )

    return {"message": "Rule deleted"}


# ===== Model Management =====


@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get current model status and metrics."""
    import json
    from pathlib import Path

    feedback_repo = FeedbackRepository(db)

    pending_count = await feedback_repo.count_unprocessed()
    should_retrain, retrain_reason = await feedback_repo.should_trigger_retrain()

    # Try to get actual model metrics from ML module
    model_version = "1.0.0"
    last_trained = None
    training_samples = 0
    accuracy_metrics = {
        "r2_score": 0.0,
        "mae": 0.0,
        "median_error_percent": 0.0,
    }

    # Check HCQE model
    try:
        from ml.hcqe_predictor import get_hcqe_predictor

        hcqe = get_hcqe_predictor()
        if hcqe:
            model_version = getattr(hcqe, "version", "1.0.0")
            accuracy_metrics = {
                "r2_score": getattr(hcqe, "r2_score", 0.78),
                "within_30_accuracy": getattr(hcqe, "within_30_accuracy", 0.788),
                "interval_coverage": getattr(hcqe, "interval_coverage", 0.818),
            }
    except Exception:
        pass

    # Check for model metadata file
    model_meta_path = (
        Path(__file__).parent.parent.parent.parent / "models" / "model_metadata.json"
    )
    if model_meta_path.exists():
        try:
            with open(model_meta_path) as f:
                meta = json.load(f)
                model_version = meta.get("version", model_version)
                last_trained = meta.get("trained_at")
                training_samples = meta.get("training_samples", 0)
                if "metrics" in meta:
                    accuracy_metrics.update(meta["metrics"])
        except Exception:
            pass

    return ModelStatusResponse(
        current_version=model_version,
        last_trained=last_trained,
        training_samples=training_samples,
        accuracy_metrics=accuracy_metrics,
        pending_feedback=pending_count,
        should_retrain=should_retrain,
        retrain_reason=retrain_reason if should_retrain else None,
    )


@router.post("/model/retrain")
async def trigger_retrain(
    request: RetrainRequest,
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Trigger model retraining."""
    import asyncio
    import json
    import uuid
    from datetime import datetime
    from pathlib import Path

    from db.repositories.pr_repo import ProductRequestRepository

    feedback_repo = FeedbackRepository(db)

    if not request.force:
        should_retrain, reason = await feedback_repo.should_trigger_retrain()
        if not should_retrain:
            return {
                "status": "skipped",
                "reason": reason,
                "message": "Use force=true to retrain anyway",
            }

    # Generate job ID
    job_id = f"retrain-{uuid.uuid4().hex[:8]}"

    # Get training data from historical PRs
    pr_repo = ProductRequestRepository(db)
    historical_prs = await pr_repo.list_with_quotations(limit=500)

    # Get feedback corrections if requested
    feedback_data = []
    if request.include_recent_feedback:
        recent_feedback = await feedback_repo.get_unprocessed(limit=100)
        for fb in recent_feedback:
            feedback_data.append(
                {
                    "corrected_value": fb.corrected_value,
                    "original_value": fb.original_value,
                    "original_item": {
                        "breakdown_id": str(fb.breakdown_id),
                    },
                }
            )

    # Prepare training data
    training_data = []
    for pr in historical_prs:
        if pr.quotations:
            for quotation in pr.quotations:
                if quotation.total_hours and quotation.total_hours > 0:
                    training_data.append(
                        {
                            "parsed_pr": {
                                "pr_code": pr.pr_code,
                                "title": pr.title,
                                "program_family": pr.program_family,
                                "customer": pr.customer,
                            },
                            "total_hours": quotation.total_hours,
                        }
                    )

    if len(training_data) < 10:
        return {
            "status": "failed",
            "job_id": job_id,
            "message": f"Insufficient training data: {len(training_data)} samples (minimum 10 required)",
        }

    # Run training asynchronously (in background)
    async def run_training():
        try:
            from ml.trainer import ModelTrainer, TrainingConfig

            models_dir = Path(__file__).parent.parent.parent.parent / "models"
            models_dir.mkdir(exist_ok=True)

            trainer = ModelTrainer(
                models_dir=models_dir,
                config=TrainingConfig(min_samples=10),
            )

            result = await trainer.train(
                training_data=training_data,
                feedback_data=feedback_data if feedback_data else None,
            )

            # Save metadata
            if result.success:
                metadata = {
                    "version": result.version_id,
                    "trained_at": datetime.now().isoformat(),
                    "training_samples": result.training_samples,
                    "metrics": result.metrics,
                    "auto_promoted": result.auto_promoted,
                }
                with open(models_dir / "model_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                # Mark feedback as processed
                if feedback_data:
                    for fb in recent_feedback:
                        await feedback_repo.mark_as_processed(fb.id)

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Start training in background
    asyncio.create_task(run_training())

    return {
        "status": "started",
        "job_id": job_id,
        "message": f"Retraining job started with {len(training_data)} samples",
        "training_samples": len(training_data),
        "feedback_samples": len(feedback_data),
    }


@router.get("/model/history")
async def get_model_history(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(10, ge=1, le=50),
):
    """Get model version history."""
    # TODO: Get from ModelVersion table
    return {
        "versions": [
            {
                "version": "1.0.0",
                "created_at": "2024-01-01T00:00:00Z",
                "accuracy_metrics": {"r2_score": 0.73},
                "is_active": True,
            }
        ],
    }


# ===== Learning Statistics =====


@router.get("/learning/stats")
async def get_learning_stats(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get online learning statistics."""
    feedback_repo = FeedbackRepository(db)
    rules_repo = RulesRepository(db)

    feedback_stats = await feedback_repo.get_statistics()
    rules_stats = await rules_repo.get_statistics()

    return {
        "feedback": feedback_stats,
        "rules": rules_stats,
    }


@router.get("/learning/feedback")
async def list_feedback(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    unprocessed_only: bool = False,
):
    """List feedback corrections."""
    feedback_repo = FeedbackRepository(db)

    if unprocessed_only:
        feedback = await feedback_repo.get_unprocessed(limit=limit)
    else:
        feedback = await feedback_repo.get_recent(days=30, limit=limit)

    return {
        "items": [
            {
                "id": str(f.id),
                "quotation_id": str(f.quotation_id),
                "breakdown_id": str(f.breakdown_id),
                "original_value": f.original_value,
                "corrected_value": f.corrected_value,
                "correction_percent": f.correction_percent,
                "field_name": f.field_name,
                "reason": f.reason,
                "processed": f.processed_for_learning,
                "created_at": f.created_at.isoformat(),
            }
            for f in feedback
        ],
        "total": len(feedback),
    }


@router.get("/learning/trend")
async def get_learning_trend(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: int = Query(30, ge=7, le=90),
):
    """Get correction trend over time."""
    feedback_repo = FeedbackRepository(db)

    trend = await feedback_repo.get_correction_trend(days=days)

    return {
        "period_days": days,
        "trend": trend,
    }


# ===== Audit Logs =====


@router.get("/audit")
async def list_audit_logs(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    action: str | None = None,
    entity_type: str | None = None,
    user_id: UUID | None = None,
):
    """List audit logs."""
    audit_repo = AuditRepository(db)

    logs = await audit_repo.list(
        skip=skip,
        limit=limit,
        action=action,
        entity_type=entity_type,
        user_id=user_id,
    )

    return {
        "items": [
            {
                "id": str(log.id),
                "action": log.action,
                "entity_type": log.entity_type,
                "entity_id": log.entity_id,
                "user_id": str(log.user_id) if log.user_id else None,
                "details": log.details,
                "ip_address": log.ip_address,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ],
        "total": len(logs),
    }


@router.get("/audit/statistics")
async def get_audit_statistics(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: int = Query(30, ge=7, le=90),
):
    """Get audit statistics."""
    audit_repo = AuditRepository(db)

    stats = await audit_repo.get_statistics(days=days)

    return stats


# ===== System Health =====


@router.get("/health")
async def system_health(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Get system health status with real service checks."""
    import redis.asyncio as redis
    from app.config import settings
    from sqlalchemy import text

    services = {}
    overall_status = "healthy"

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        services["database"] = "up"
    except Exception as e:
        services["database"] = f"down: {str(e)[:50]}"
        overall_status = "degraded"

    # Check Redis
    try:
        redis_client = redis.from_url(str(settings.REDIS_URL))
        await redis_client.ping()
        await redis_client.close()
        services["redis"] = "up"
    except Exception as e:
        services["redis"] = f"down: {str(e)[:50]}"
        overall_status = "degraded"

    # Check Qdrant
    try:
        from qdrant_client import AsyncQdrantClient

        qdrant = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        collections = await qdrant.get_collections()
        services["qdrant"] = f"up ({len(collections.collections)} collections)"
        await qdrant.close()
    except Exception as e:
        services["qdrant"] = f"down: {str(e)[:50]}"
        overall_status = "degraded"

    # Check LLM
    try:
        from llm.client import get_llm_client

        llm = get_llm_client()
        # Just check if client is configured, don't make actual request
        if llm and hasattr(llm, "client"):
            services["llm"] = "up (configured)"
        else:
            services["llm"] = "not configured"
            overall_status = "degraded"
    except Exception as e:
        services["llm"] = f"error: {str(e)[:50]}"
        overall_status = "degraded"

    # Check HCQE Model
    try:
        from ml.hcqe_predictor import get_hcqe_predictor

        hcqe = get_hcqe_predictor()
        if hcqe:
            services["ml_model"] = "up (HCQE loaded)"
        else:
            services["ml_model"] = "not loaded"
    except Exception as e:
        services["ml_model"] = f"error: {str(e)[:50]}"

    # Get metrics
    feedback_repo = FeedbackRepository(db)
    pending_feedback = await feedback_repo.count_unprocessed()

    # Count active sessions from Redis
    active_sessions = 0
    try:
        redis_client = redis.from_url(str(settings.REDIS_URL))
        keys = await redis_client.keys("estimation:*")
        active_sessions = len(keys)
        await redis_client.close()
    except Exception:
        pass

    return {
        "status": overall_status,
        "services": services,
        "metrics": {
            "active_sessions": active_sessions,
            "pending_feedback": pending_feedback,
        },
    }


# ===== RAG Brain Health =====


@router.get("/rag/health")
async def rag_health(
    current_user: Annotated[User, Depends(require_admin)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Get comprehensive RAG brain health status.

    Checks:
    - Qdrant collections and vector counts
    - PostgreSQL knowledge base status
    - Embedding model availability
    - Search capability test
    """
    from app.config import settings
    from qdrant_client import AsyncQdrantClient
    from sqlalchemy import text

    status = "healthy"
    components = {}

    # 1. Check Qdrant collections
    qdrant_status = {"status": "unknown", "collections": {}}
    try:
        qdrant = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=10,
        )

        collections_info = await qdrant.get_collections()
        collection_names = [c.name for c in collections_info.collections]

        expected_collections = [
            "pr_embeddings",
            "quotation_chunks",
            "knowledge_chunks",
            "feedback_patterns",
        ]

        for coll_name in expected_collections:
            if coll_name in collection_names:
                try:
                    info = await qdrant.get_collection(coll_name)
                    qdrant_status["collections"][coll_name] = {
                        "vectors": info.points_count,
                        "dimensions": info.config.params.vectors.size,
                        "status": "ok" if info.points_count > 0 else "empty",
                    }
                except Exception as e:
                    qdrant_status["collections"][coll_name] = {"status": f"error: {e}"}
            else:
                qdrant_status["collections"][coll_name] = {"status": "missing"}
                status = "degraded"

        await qdrant.close()
        qdrant_status["status"] = "connected"

    except Exception as e:
        qdrant_status["status"] = f"error: {str(e)[:100]}"
        status = "unhealthy"

    components["qdrant"] = qdrant_status

    # 2. Check PostgreSQL knowledge base
    db_status = {"status": "unknown"}
    try:
        # Count acronyms
        result = await db.execute(text("SELECT COUNT(*) FROM acronyms"))
        acronyms_count = result.scalar()

        # Count knowledge documents
        result = await db.execute(text("SELECT COUNT(*) FROM knowledge_documents"))
        docs_count = result.scalar()

        # Count by type
        result = await db.execute(
            text("SELECT doc_type, COUNT(*) FROM knowledge_documents GROUP BY doc_type")
        )
        docs_by_type = dict(result.fetchall())

        db_status = {
            "status": "connected",
            "acronyms": acronyms_count,
            "documents": docs_count,
            "documents_by_type": docs_by_type,
        }

        if docs_count == 0:
            status = "degraded"
            db_status["warning"] = "No knowledge documents loaded"

    except Exception as e:
        db_status["status"] = f"error: {str(e)[:100]}"
        status = "unhealthy"

    components["knowledge_base"] = db_status

    # 3. Check embedding model
    embedding_status = {"status": "unknown"}
    try:
        embedding_status = {
            "status": "configured",
            "model": settings.LLM_EMBEDDING_MODEL,
            "dimensions": settings.LLM_EMBEDDING_DIMENSIONS,
            "provider": "OpenRouter",
        }

        # Verify API key is set
        if not settings.OPENROUTER_API_KEY:
            embedding_status["status"] = "not configured"
            embedding_status["warning"] = "OPENROUTER_API_KEY not set"
            status = "degraded"

    except Exception as e:
        embedding_status["status"] = f"error: {str(e)[:100]}"

    components["embedding_model"] = embedding_status

    # 4. RAG readiness summary
    total_vectors = sum(
        c.get("vectors", 0)
        for c in qdrant_status.get("collections", {}).values()
        if isinstance(c, dict) and "vectors" in c
    )

    readiness = {
        "ready_for_search": total_vectors > 0 and db_status.get("documents", 0) > 0,
        "total_vectors": total_vectors,
        "total_documents": db_status.get("documents", 0),
        "acronyms_loaded": db_status.get("acronyms", 0) > 0,
    }

    return {
        "status": status,
        "components": components,
        "readiness": readiness,
        "recommendation": _get_rag_recommendation(status, readiness, components),
    }


def _get_rag_recommendation(status: str, readiness: dict, components: dict) -> str:
    """Generate recommendation based on RAG health."""
    if status == "healthy" and readiness["ready_for_search"]:
        return "RAG brain is fully operational"

    recommendations = []

    if not readiness["ready_for_search"]:
        if readiness["total_vectors"] == 0:
            recommendations.append(
                "Run 'python scripts/init_rag_brain.py' to populate vectors"
            )
        if readiness["total_documents"] == 0:
            recommendations.append("Knowledge base is empty - run import script")

    if components.get("embedding_model", {}).get("status") == "not configured":
        recommendations.append("Set OPENROUTER_API_KEY in .env file")

    qdrant_colls = components.get("qdrant", {}).get("collections", {})
    missing = [
        k
        for k, v in qdrant_colls.items()
        if isinstance(v, dict) and v.get("status") == "missing"
    ]
    if missing:
        recommendations.append(
            f"Missing collections: {', '.join(missing)} - restart backend to create"
        )

    return (
        "; ".join(recommendations)
        if recommendations
        else "Check component errors above"
    )


@router.post("/rag/test-search")
async def test_rag_search(
    current_user: Annotated[User, Depends(require_admin)],
    query: str = Query(..., min_length=3, description="Test search query"),
):
    """
    Test RAG search capability with a sample query.

    Returns search results from knowledge_chunks collection.
    """
    from app.config import settings
    from llm.client import get_llm_client
    from qdrant_client import AsyncQdrantClient

    try:
        # Generate embedding for query
        llm = get_llm_client()
        query_embedding = await llm.embed(query)

        # Search in Qdrant
        qdrant = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=10,
        )

        results = await qdrant.search(
            collection_name="knowledge_chunks",
            query_vector=query_embedding,
            limit=5,
        )

        await qdrant.close()

        return {
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "score": hit.score,
                    "title": hit.payload.get("title", "Unknown"),
                    "doc_type": hit.payload.get("doc_type", "Unknown"),
                    "preview": hit.payload.get("chunk_text", "")[:200],
                }
                for hit in results
            ],
        }

    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "error": str(e),
            "recommendation": "Ensure RAG brain is initialized with 'python scripts/init_rag_brain.py'",
        }


@router.post("/rag/reinitialize")
async def reinitialize_rag(
    current_user: Annotated[User, Depends(require_admin)],
    reset: bool = Query(False, description="Clear all data and start fresh"),
):
    """
    Trigger RAG brain reinitialization.

    This runs the init_rag_brain.py script in the background.
    """
    import subprocess
    from pathlib import Path

    script_path = Path(__file__).parent.parent.parent / "scripts" / "init_rag_brain.py"

    if not script_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="RAG initialization script not found",
        )

    # Build command
    cmd = ["python", str(script_path)]
    if reset:
        cmd.append("--reset")

    # Start in background
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(script_path.parent.parent),
        )

        return {
            "status": "started",
            "pid": process.pid,
            "command": " ".join(cmd),
            "message": "RAG reinitialization started in background. Check logs for progress.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start reinitialization: {str(e)}",
        )

"""
FPT Cost Brain 2.0 - FastAPI Application Entry Point

With comprehensive debug logging for complete pipeline visibility.
Enable with: DEBUG=true or LOG_LEVEL=DEBUG in .env
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.exceptions import FPTCostBrainException

# Initialize debug logging early - before any other imports that might log
from app.debug_logging import (
    setup_debug_logging,
    log_api_request,
    log_api_response,
    log_error_details,
    generate_request_id,
)

# Configure logging based on environment
_debug_enabled = settings.DEBUG or settings.LOG_LEVEL.upper() == "DEBUG"
print(
    f"🔍 Debug logging configured: level={settings.LOG_LEVEL}, format={settings.LOG_FORMAT}"
)
print(f"   DEBUG={settings.DEBUG}, _debug_enabled={_debug_enabled}")

# ALWAYS configure logging infrastructure
setup_debug_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)

# Get logger for this module
logger = logging.getLogger(__name__)


# ============================================================================
# HTTP REQUEST/RESPONSE LOGGING MIDDLEWARE
# ============================================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for comprehensive HTTP request/response logging.

    Logs:
    - Incoming requests with method, path, headers, body preview
    - Outgoing responses with status code, timing, size
    - Errors with full context
    """

    # Paths to skip detailed logging (health checks, static)
    SKIP_PATHS = {"/health", "/favicon.ico", "/docs", "/redoc", "/openapi.json"}

    # Paths that should log request body (only API endpoints)
    LOG_BODY_PATHS = {"/api/v1/estimation", "/api/v1/chat", "/api/v1/export"}

    async def dispatch(self, request: Request, call_next) -> Response:
        # DEBUG: Print every request to verify middleware is working
        print(f"📥 MIDDLEWARE: {request.method} {request.url.path}")

        # Skip logging for non-essential paths
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Generate unique request ID for tracing
        request_id = generate_request_id()
        start_time = time.perf_counter()

        # Extract request info
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params) if request.query_params else None

        # Log request (only log body for specific endpoints to avoid overhead)
        body = None
        if _debug_enabled and any(path.startswith(p) for p in self.LOG_BODY_PATHS):
            try:
                body_bytes = await request.body()
                if body_bytes and len(body_bytes) < 10000:  # Only parse small bodies
                    import json

                    try:
                        body = json.loads(body_bytes)
                    except json.JSONDecodeError:
                        body = {
                            "_raw": body_bytes.decode("utf-8", errors="replace")[:500]
                        }
            except Exception:
                pass  # Body already consumed or not available

        # Log incoming request
        log_api_request(
            logger,
            method=method,
            path=path,
            request_id=request_id,
            body=body,
            query_params=query_params,
            headers=dict(request.headers) if _debug_enabled else None,
        )

        # Process request
        error_message = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Log unhandled exception
            log_error_details(
                logger,
                error=e,
                context=f"Request {method} {path}",
                session_id=request_id,
            )
            error_message = str(e)
            status_code = 500
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )

        # Calculate timing
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Get response size if available
        response_size = None
        if hasattr(response, "body"):
            response_size = len(response.body)

        # Log response
        log_api_response(
            logger,
            method=method,
            path=path,
            request_id=request_id,
            status_code=status_code,
            duration_ms=duration_ms,
            response_size=response_size,
            error=error_message,
        )

        # Add request ID to response headers for debugging
        response.headers["X-Request-ID"] = request_id

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management - startup and shutdown.

    Uses try/finally to ensure proper cleanup even if initialization fails partially.
    """
    # Track which resources were initialized for proper cleanup
    db_initialized = False
    vector_db_initialized = False
    redis_initialized = False

    try:
        # Startup
        logger.info("=" * 70)
        logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
        logger.info("=" * 70)
        logger.info(f"   Environment: {settings.ENVIRONMENT}")
        logger.info(f"   Debug Mode: {settings.DEBUG}")
        logger.info(f"   Log Level: {settings.LOG_LEVEL}")
        logger.info(f"   Log Format: {settings.LOG_FORMAT}")
        if _debug_enabled:
            logger.info("   📊 Comprehensive debug logging: ENABLED")
            logger.info("   📥 Request/Response logging: ENABLED")
            logger.info("   🤖 ML operation logging: ENABLED")
            logger.info("   🧠 LLM call logging: ENABLED")
            logger.info("   💾 Cache operation logging: ENABLED")

        # Initialize database connections
        from db.session import init_db

        await init_db()
        db_initialized = True
        print("   ✅ Database connected")

        # Initialize vector database
        from vector.client import init_vector_db

        await init_vector_db()
        vector_db_initialized = True
        print("   ✅ Vector database connected")

        # Initialize Redis
        from services.cache_service import init_redis

        await init_redis()
        redis_initialized = True
        print("   ✅ Redis connected")

        # Preload ML model at startup to avoid first-request timeout
        try:
            from agents.nodes.estimation_node import _get_ml_predictor

            print("   ⏳ Preloading HCQE ML model...")
            ml_model = _get_ml_predictor()
            if ml_model is not None:
                print(
                    f"   ✅ HCQE model loaded (version: {getattr(ml_model, 'version', 'unknown')})"
                )
            else:
                print("   ⚠️ HCQE model not available (will use fallback)")
        except Exception as e:
            print(f"   ⚠️ HCQE model preload failed: {e} (will load on first request)")

        yield

    finally:
        # Shutdown - clean up only what was initialized
        print(f"🛑 Shutting down {settings.APP_NAME}")

        if redis_initialized:
            try:
                from services.cache_service import close_redis

                await close_redis()
            except Exception as e:
                print(f"   ⚠️ Redis cleanup error: {e}")

        if vector_db_initialized:
            try:
                from vector.client import close_vector_db

                await close_vector_db()
            except Exception as e:
                print(f"   ⚠️ Vector DB cleanup error: {e}")

        if db_initialized:
            try:
                from db.session import close_db

                await close_db()
            except Exception as e:
                print(f"   ⚠️ Database cleanup error: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered R&D Cost Estimation Platform for FPT Industrial",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Request logging middleware (first, so it captures all requests)
    if _debug_enabled:
        app.add_middleware(RequestLoggingMiddleware)
        logger.info("📊 Request logging middleware enabled")

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(FPTCostBrainException)
    async def fpt_exception_handler(request, exc: FPTCostBrainException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    # Global exception handler - catch ALL unhandled exceptions with details
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc: Exception):
        import logging
        import traceback

        logger = logging.getLogger(__name__)
        logger.exception(f"[GLOBAL] Unhandled exception: {exc}")

        # Return detailed error for debugging (in production, hide traceback)
        error_detail = {
            "detail": f"Internal error: {str(exc)}",
            "type": type(exc).__name__,
        }
        if settings.DEBUG:
            error_detail["traceback"] = traceback.format_exc()

        return JSONResponse(status_code=500, content=error_detail)

    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        }

    # Include API routers
    from api.v1 import auth, estimation, chat, export, history, knowledge, admin, rlhf

    app.include_router(
        auth.router, prefix=f"{settings.API_V1_PREFIX}/auth", tags=["Authentication"]
    )
    app.include_router(
        estimation.router,
        prefix=f"{settings.API_V1_PREFIX}/estimation",
        tags=["Estimation"],
    )
    app.include_router(
        chat.router, prefix=f"{settings.API_V1_PREFIX}/chat", tags=["Chat"]
    )
    app.include_router(
        export.router, prefix=f"{settings.API_V1_PREFIX}/export", tags=["Export"]
    )
    app.include_router(
        history.router, prefix=f"{settings.API_V1_PREFIX}/history", tags=["History"]
    )
    app.include_router(
        knowledge.router,
        prefix=f"{settings.API_V1_PREFIX}/knowledge",
        tags=["Knowledge"],
    )
    app.include_router(
        admin.router, prefix=f"{settings.API_V1_PREFIX}/admin", tags=["Admin"]
    )
    app.include_router(
        rlhf.router, prefix=f"{settings.API_V1_PREFIX}/rlhf", tags=["RLHF"]
    )

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        timeout_keep_alive=300,
    )

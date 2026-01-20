"""
FPT Cost Brain 2.0 - Comprehensive Debug Logging Configuration

Enable with: LOG_LEVEL=DEBUG or DEBUG=true in .env
Provides complete visibility into the estimation pipeline:
- API Request/Response flow
- Estimation pipeline steps
- ML model operations (HCQE)
- LLM interactions (OpenRouter)
- Database operations (PostgreSQL, Redis, Qdrant)
- Error handling with full context

Usage:
    from app.debug_logging import (
        setup_debug_logging,
        DebugLogger,
        log_api_request,
        log_api_response,
        log_step,
        log_ml_operation,
        log_llm_call,
        log_db_operation,
        log_vector_operation,
        log_error_details,
    )
"""

import asyncio
import functools
import hashlib
import json
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar
from uuid import uuid4

# Type vars for decorator typing
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs when LOG_FORMAT=json,
    or pretty console logs otherwise.
    """

    def __init__(self, use_json: bool = False):
        super().__init__()
        self.use_json = use_json

    def format(self, record: logging.LogRecord) -> str:
        if self.use_json:
            return self._format_json(record)
        return self._format_console(record)

    def _format_console(self, record: logging.LogRecord) -> str:
        """Human-readable console format with colors and structure."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        level_colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        reset = "\033[0m"
        color = level_colors.get(record.levelname, "")

        # Build base message
        base = f"{timestamp} {color}[{record.levelname:7}]{reset} {record.name} → {record.getMessage()}"

        # Add extra fields if present
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields and isinstance(extra_fields, dict):
            for key, value in extra_fields.items():
                base += f"\n    {key}: {self._truncate(value)}"

        return base

    def _format_json(self, record: logging.LogRecord) -> str:
        """Structured JSON format for log aggregation."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        extra_fields = getattr(record, "extra_fields", None)
        if extra_fields and isinstance(extra_fields, dict):
            log_entry["data"] = extra_fields

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str, ensure_ascii=False)

    def _truncate(self, value: Any, max_len: int = 500) -> str:
        """Truncate long values for readable logs."""
        str_val = str(value)
        if len(str_val) > max_len:
            return str_val[:max_len] + f"... [{len(str_val)} chars]"
        return str_val


class ExtraFieldsAdapter(logging.LoggerAdapter):
    """Logger adapter that supports adding extra fields to log records."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if "extra_fields" not in extra:
            extra["extra_fields"] = {}
        kwargs["extra"] = extra
        return msg, kwargs

    def with_fields(self, **fields) -> "ExtraFieldsAdapter":
        """Create a new adapter with additional fields merged."""
        new_extra = {**self.extra, **fields}
        return ExtraFieldsAdapter(self.logger, new_extra)


def setup_debug_logging(
    log_level: str = "DEBUG",
    log_format: str = "console",
    force_reinit: bool = False,
) -> None:
    """
    Configure comprehensive debug logging for all estimation components.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Output format ("console" or "json")
        force_reinit: Force re-initialization even if already configured
    """
    # Check if already initialized (avoid duplicate handlers)
    root_logger = logging.getLogger()
    if not force_reinit and hasattr(root_logger, "_fpt_debug_initialized"):
        return

    # Create formatter
    use_json = log_format.lower() == "json"
    formatter = StructuredFormatter(use_json=use_json)

    # Console handler - force to stdout for Docker
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

    # All modules to configure
    modules = [
        # Root
        "",
        # API layer
        "api",
        "api.v1",
        "api.v1.estimation",
        "api.v1.chat",
        "api.v1.export",
        "api.v1.auth",
        # Services
        "services",
        "services.estimation_service",
        "services.cache_service",
        "services.lookup_estimator",
        "services.sizing_service",
        "services.cost_calculator",
        # Agents & Nodes
        "agents",
        "agents.graph",
        "agents.state",
        "agents.nodes",
        "agents.nodes.intake_node",
        "agents.nodes.qa_node",
        "agents.nodes.summary_node",
        "agents.nodes.estimation_node",
        "agents.nodes.export_node",
        "agents.nodes.learning_node",
        "agents.agentic",
        "agents.agentic.pipeline",
        "agents.agentic.cluster_agents",
        "agents.agentic.arbitrator",
        "agents.fpt_engineer_agent",
        # ML
        "ml",
        "ml.hcqe_predictor",
        "ml.hcqe_production_model",
        "ml.hcqe_feature_extractor",
        "ml.pe_function_distributor",
        "ml.features",
        "ml.model",
        # LLM
        "llm",
        "llm.client",
        # Vector DB
        "vector",
        "vector.client",
        # Database
        "db",
        "db.session",
        # App
        "app",
        "app.main",
        "__main__",
        # Third-party (controlled)
        "uvicorn",
        "uvicorn.access",
        "fastapi",
        "sqlalchemy.engine",
    ]

    level_int = getattr(logging, log_level.upper(), logging.DEBUG)

    for module in modules:
        logger = logging.getLogger(module)
        logger.setLevel(level_int)
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        logger.addHandler(console_handler)
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

    # Mark as initialized
    root_logger._fpt_debug_initialized = True

    # Flush and announce
    sys.stdout.flush()
    print(
        f"🔍 Debug logging configured: level={log_level}, format={log_format}",
        flush=True,
    )


def get_logger(name: str) -> ExtraFieldsAdapter:
    """Get a logger with extra fields support."""
    return ExtraFieldsAdapter(logging.getLogger(name), {})


# ============================================================================
# LOGGING UTILITIES
# ============================================================================


class DebugLogger:
    """
    Centralized debug logger with structured output and timing.

    Usage:
        debug = DebugLogger("my_module")
        debug.info("Processing started", session_id="abc123", step="intake")

        with debug.timed_operation("ML prediction"):
            result = model.predict(features)
    """

    def __init__(self, name: str, session_id: str | None = None):
        self.logger = logging.getLogger(name)
        self.session_id = session_id
        self._operation_stack: list[tuple[str, float]] = []

    def _log(self, level: int, msg: str, **extra_fields):
        """Internal log method with extra fields."""
        if self.session_id:
            extra_fields["session_id"] = self.session_id[:8] + "..."

        # Add operation context if in a timed block
        if self._operation_stack:
            extra_fields["operation"] = self._operation_stack[-1][0]

        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "(debug_logging)",
            0,
            msg,
            (),
            None,
        )
        record.extra_fields = extra_fields
        self.logger.handle(record)

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def exception(self, msg: str, exc: Exception | None = None, **kwargs):
        """Log error with full traceback."""
        if exc:
            kwargs["exception_type"] = type(exc).__name__
            kwargs["exception_message"] = str(exc)
            kwargs["traceback"] = traceback.format_exc()
        self._log(logging.ERROR, msg, **kwargs)

    @contextmanager
    def timed_operation(self, name: str, **context):
        """Context manager for timing operations."""
        start = time.perf_counter()
        self._operation_stack.append((name, start))

        self.info(f"▶ {name} started", **context)

        try:
            yield
            elapsed = time.perf_counter() - start
            self.info(f"✅ {name} completed", duration_ms=f"{elapsed * 1000:.2f}")
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.exception(
                f"❌ {name} failed", exc=e, duration_ms=f"{elapsed * 1000:.2f}"
            )
            raise
        finally:
            self._operation_stack.pop()

    @asynccontextmanager
    async def async_timed_operation(self, name: str, **context):
        """Async context manager for timing operations."""
        start = time.perf_counter()
        self._operation_stack.append((name, start))

        self.info(f"▶ {name} started", **context)

        try:
            yield
            elapsed = time.perf_counter() - start
            self.info(f"✅ {name} completed", duration_ms=f"{elapsed * 1000:.2f}")
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.exception(
                f"❌ {name} failed", exc=e, duration_ms=f"{elapsed * 1000:.2f}"
            )
            raise
        finally:
            self._operation_stack.pop()


# ============================================================================
# API REQUEST/RESPONSE LOGGING
# ============================================================================


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    request_id: str,
    body: dict | None = None,
    query_params: dict | None = None,
    headers: dict | None = None,
) -> None:
    """Log incoming API request with details."""
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "(api)",
        0,
        f"📥 {method} {path}",
        (),
        None,
    )
    record.extra_fields = {
        "request_id": request_id,
        "method": method,
        "path": path,
    }

    if body:
        # Truncate large bodies
        body_str = json.dumps(body, default=str)
        if len(body_str) > 1000:
            record.extra_fields["body"] = body_str[:1000] + "... [truncated]"
        else:
            record.extra_fields["body"] = body

    if query_params:
        record.extra_fields["query"] = query_params

    if headers:
        # Only log safe headers
        safe_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("authorization", "cookie", "x-api-key")
        }
        if safe_headers:
            record.extra_fields["headers"] = safe_headers

    logger.handle(record)


def log_api_response(
    logger: logging.Logger,
    method: str,
    path: str,
    request_id: str,
    status_code: int,
    duration_ms: float,
    response_size: int | None = None,
    error: str | None = None,
) -> None:
    """Log outgoing API response with timing."""
    level = logging.INFO if status_code < 400 else logging.ERROR

    status_emoji = "✅" if status_code < 400 else "❌"
    msg = f"📤 {status_emoji} {method} {path} → {status_code}"

    record = logger.makeRecord(
        logger.name,
        level,
        "(api)",
        0,
        msg,
        (),
        None,
    )
    record.extra_fields = {
        "request_id": request_id,
        "status_code": status_code,
        "duration_ms": f"{duration_ms:.2f}",
    }

    if response_size:
        record.extra_fields["response_size_bytes"] = response_size

    if error:
        record.extra_fields["error"] = error

    logger.handle(record)


# ============================================================================
# ESTIMATION PIPELINE LOGGING
# ============================================================================


@contextmanager
def log_step(
    logger: logging.Logger,
    step_name: str,
    session_id: str = "",
    **context,
):
    """
    Context manager to log estimation step entry/exit with timing.

    Usage:
        with log_step(logger, "QA Question Generation", session_id):
            questions = generate_questions(parsed_pr)
    """
    start = time.perf_counter()
    session_short = session_id[:8] + "..." if session_id else "N/A"

    logger.info("=" * 70)
    logger.info(f"▶ STEP: {step_name} [{session_short}]")
    for key, value in context.items():
        logger.info(f"    {key}: {_truncate(value)}")
    logger.info("=" * 70)

    try:
        yield
        elapsed = time.perf_counter() - start
        logger.info(f"✅ STEP COMPLETED: {step_name} in {elapsed * 1000:.2f}ms")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"❌ STEP FAILED: {step_name} after {elapsed * 1000:.2f}ms")
        logger.error(f"    Error: {type(e).__name__}: {e}")
        raise


@asynccontextmanager
async def async_log_step(
    logger: logging.Logger,
    step_name: str,
    session_id: str = "",
    **context,
):
    """Async version of log_step."""
    start = time.perf_counter()
    session_short = session_id[:8] + "..." if session_id else "N/A"

    logger.info("=" * 70)
    logger.info(f"▶ STEP: {step_name} [{session_short}]")
    for key, value in context.items():
        logger.info(f"    {key}: {_truncate(value)}")
    logger.info("=" * 70)

    try:
        yield
        elapsed = time.perf_counter() - start
        logger.info(f"✅ STEP COMPLETED: {step_name} in {elapsed * 1000:.2f}ms")
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error(f"❌ STEP FAILED: {step_name} after {elapsed * 1000:.2f}ms")
        logger.error(f"    Error: {type(e).__name__}: {e}")
        raise


def log_state_transition(
    logger: logging.Logger,
    state: dict,
    label: str = "STATE",
) -> None:
    """Log key state information during transitions."""
    logger.debug(f"📊 {label}:")
    logger.debug(f"    session_id: {state.get('session_id', 'N/A')}")
    logger.debug(f"    current_step: {state.get('current_step', 'N/A')}")
    logger.debug(f"    step_status: {state.get('step_status', {})}")

    if state.get("parsed_pr"):
        pr = state["parsed_pr"]
        logger.debug(f"    pr_code: {pr.get('pr_code', 'N/A')}")
        logger.debug(f"    program_family: {pr.get('program_family', 'N/A')}")

    if state.get("questions"):
        logger.debug(f"    questions: {len(state['questions'])} generated")

    if state.get("breakdown"):
        total_hours = sum(item.get("hours", 0) for item in state["breakdown"])
        logger.debug(
            f"    breakdown: {len(state['breakdown'])} items, {total_hours:.0f}h total"
        )


# ============================================================================
# ML MODEL LOGGING
# ============================================================================


def log_ml_operation(
    logger: logging.Logger,
    operation: str,
    model_name: str = "HCQE",
    **details,
) -> None:
    """Log ML model operations with details."""
    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        "(ml)",
        0,
        f"🤖 ML {operation}: {model_name}",
        (),
        None,
    )
    record.extra_fields = {"model": model_name, **details}
    logger.handle(record)


def log_ml_prediction(
    logger: logging.Logger,
    prediction: dict,
    features_count: int = 0,
    duration_ms: float = 0,
) -> None:
    """Log detailed ML prediction results."""
    logger.info("🤖 HCQE PREDICTION RESULT:")
    logger.info(
        f"    Predicted Cost: €{prediction.get('predicted_cost_keur', 0) * 1000:,.0f}"
    )
    logger.info(f"    Predicted Hours: {prediction.get('predicted_hours', 0):,.0f}")
    logger.info(f"    Confidence: {prediction.get('confidence', 0) * 100:.1f}%")
    logger.info(f"    Method: {prediction.get('method', 'N/A')}")

    if prediction.get("sizing"):
        sizing = prediction["sizing"]
        logger.info(
            f"    Sizing: {sizing.get('predicted', 'N/A')} ({sizing.get('confidence', 0) * 100:.0f}%)"
        )

    if prediction.get("prediction_interval"):
        interval = prediction["prediction_interval"]
        logger.info(
            f"    Interval: €{interval.get('lower_keur', 0) * 1000:,.0f} - "
            f"€{interval.get('upper_keur', 0) * 1000:,.0f}"
        )

    if features_count:
        logger.info(f"    Features used: {features_count}")

    if duration_ms:
        logger.info(f"    Prediction time: {duration_ms:.2f}ms")


def log_feature_extraction(
    logger: logging.Logger,
    features: dict,
    source: str = "PR",
) -> None:
    """Log feature extraction details."""
    true_features = [k for k, v in features.items() if v is True]
    false_features = [k for k, v in features.items() if v is False]
    other_features = {k: v for k, v in features.items() if v not in (True, False)}

    logger.debug(f"📊 FEATURES EXTRACTED from {source}:")
    logger.debug(f"    Total: {len(features)} features")
    logger.debug(f"    Active (True): {len(true_features)}")

    if true_features:
        logger.debug(f"    Active list: {', '.join(true_features[:15])}")
        if len(true_features) > 15:
            logger.debug(f"        ... and {len(true_features) - 15} more")

    if other_features:
        logger.debug(f"    Non-boolean: {other_features}")


# ============================================================================
# LLM INTERACTION LOGGING
# ============================================================================


def log_llm_call(
    logger: logging.Logger,
    purpose: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    duration_ms: float = 0,
    prompt_preview: str = "",
    response_preview: str = "",
    success: bool = True,
    error: str | None = None,
) -> None:
    """Log LLM API calls with token usage and timing."""
    status = "✅" if success else "❌"

    record = logger.makeRecord(
        logger.name,
        logging.INFO if success else logging.ERROR,
        "(llm)",
        0,
        f"🧠 LLM {status}: {purpose}",
        (),
        None,
    )

    record.extra_fields = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "duration_ms": f"{duration_ms:.2f}",
    }

    if prompt_preview:
        record.extra_fields["prompt_preview"] = _truncate(prompt_preview, 200)

    if response_preview:
        record.extra_fields["response_preview"] = _truncate(response_preview, 200)

    if error:
        record.extra_fields["error"] = error

    logger.handle(record)


def log_llm_streaming(
    logger: logging.Logger,
    purpose: str,
    model: str,
    chunks_received: int,
    total_chars: int,
    duration_ms: float,
) -> None:
    """Log LLM streaming response summary."""
    logger.info(f"🧠 LLM STREAM: {purpose}")
    logger.info(f"    Model: {model}")
    logger.info(f"    Chunks: {chunks_received}")
    logger.info(f"    Total chars: {total_chars}")
    logger.info(f"    Duration: {duration_ms:.2f}ms")


# ============================================================================
# DATABASE OPERATION LOGGING
# ============================================================================


def log_redis_operation(
    logger: logging.Logger,
    operation: str,
    key: str,
    success: bool = True,
    hit: bool | None = None,
    duration_ms: float = 0,
    **extra,
) -> None:
    """Log Redis cache operations."""
    status = "✓" if success else "✗"
    hit_info = ""
    if hit is not None:
        hit_info = " [HIT]" if hit else " [MISS]"

    record = logger.makeRecord(
        logger.name,
        logging.DEBUG,
        "(redis)",
        0,
        f"💾 REDIS {status}{hit_info}: {operation}",
        (),
        None,
    )

    record.extra_fields = {
        "key": _truncate(key, 80),
        "success": success,
        "duration_ms": f"{duration_ms:.2f}",
        **extra,
    }

    if hit is not None:
        record.extra_fields["cache_hit"] = hit

    logger.handle(record)


def log_db_query(
    logger: logging.Logger,
    operation: str,
    table: str = "",
    rows_affected: int = 0,
    duration_ms: float = 0,
    query_preview: str = "",
) -> None:
    """Log PostgreSQL database operations."""
    record = logger.makeRecord(
        logger.name,
        logging.DEBUG,
        "(db)",
        0,
        f"🗄️ DB: {operation}" + (f" on {table}" if table else ""),
        (),
        None,
    )

    record.extra_fields = {
        "operation": operation,
        "duration_ms": f"{duration_ms:.2f}",
    }

    if table:
        record.extra_fields["table"] = table

    if rows_affected:
        record.extra_fields["rows_affected"] = rows_affected

    if query_preview:
        record.extra_fields["query"] = _truncate(query_preview, 200)

    logger.handle(record)


def log_db_operation(
    logger: logging.Logger,
    operation: str,
    **details,
) -> None:
    """Generic database operation logging."""
    logger.debug(f"🗄️ DB {operation}:", extra={"extra_fields": details})


# ============================================================================
# VECTOR DATABASE LOGGING
# ============================================================================


def log_vector_operation(
    logger: logging.Logger,
    operation: str,
    collection: str,
    vectors_count: int = 0,
    results_count: int = 0,
    duration_ms: float = 0,
    **extra,
) -> None:
    """Log Qdrant vector database operations."""
    record = logger.makeRecord(
        logger.name,
        logging.DEBUG,
        "(qdrant)",
        0,
        f"🔍 QDRANT: {operation} on {collection}",
        (),
        None,
    )

    record.extra_fields = {
        "collection": collection,
        "duration_ms": f"{duration_ms:.2f}",
        **extra,
    }

    if vectors_count:
        record.extra_fields["vectors"] = vectors_count

    if results_count:
        record.extra_fields["results"] = results_count

    logger.handle(record)


def log_embedding_generation(
    logger: logging.Logger,
    text_length: int,
    dimensions: int,
    model: str,
    duration_ms: float,
    cached: bool = False,
) -> None:
    """Log embedding generation."""
    cache_info = " [CACHED]" if cached else ""
    logger.debug(f"🔢 EMBEDDING{cache_info}: {text_length} chars → {dimensions}d")
    logger.debug(f"    Model: {model}")
    logger.debug(f"    Duration: {duration_ms:.2f}ms")


def log_similarity_search(
    logger: logging.Logger,
    collection: str,
    query_length: int,
    results_count: int,
    top_score: float = 0,
    duration_ms: float = 0,
) -> None:
    """Log vector similarity search."""
    logger.info(f"🔍 SIMILARITY SEARCH: {collection}")
    logger.info(f"    Query length: {query_length} chars")
    logger.info(f"    Results: {results_count}")
    logger.info(f"    Top score: {top_score:.4f}")
    logger.info(f"    Duration: {duration_ms:.2f}ms")


# ============================================================================
# ERROR HANDLING LOGGING
# ============================================================================


def log_error_details(
    logger: logging.Logger,
    error: Exception,
    context: str = "",
    session_id: str = "",
    **extra,
) -> None:
    """Log detailed error information with full traceback."""
    logger.error("=" * 70)
    logger.error(f"💥 ERROR: {context}" if context else "💥 ERROR OCCURRED")
    logger.error("=" * 70)
    logger.error(f"    Type: {type(error).__name__}")
    logger.error(f"    Message: {str(error)}")

    if session_id:
        logger.error(f"    Session: {session_id[:8]}...")

    for key, value in extra.items():
        logger.error(f"    {key}: {_truncate(value)}")

    # Full traceback
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    logger.error("    Traceback:")
    for line in tb:
        for subline in line.rstrip().split("\n"):
            logger.error(f"      {subline}")


def log_recovery_attempt(
    logger: logging.Logger,
    error: Exception,
    recovery_action: str,
    success: bool,
) -> None:
    """Log error recovery attempts."""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    logger.warning(f"🔄 RECOVERY {status}: {recovery_action}")
    logger.warning(f"    Original error: {type(error).__name__}: {error}")


# ============================================================================
# HELPER UTILITIES
# ============================================================================


def _truncate(value: Any, max_len: int = 500) -> str:
    """Truncate long values for readable logging."""
    str_val = str(value)
    if len(str_val) > max_len:
        return str_val[:max_len] + f"... [{len(str_val)} chars]"
    return str_val


def log_dict(d: dict, prefix: str = "", max_keys: int = 20) -> str:
    """Format dict for logging, showing key info."""
    if not d:
        return "{}"

    lines = []
    for i, (k, v) in enumerate(d.items()):
        if i >= max_keys:
            lines.append(f"  ... and {len(d) - max_keys} more keys")
            break
        lines.append(f"  {prefix}{k}: {_truncate(v)}")
    return "\n".join(lines)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid4())[:8]


def hash_for_cache(data: Any) -> str:
    """Generate a hash for cache keys."""
    content = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(content.encode()).hexdigest()[:16]


# ============================================================================
# DECORATORS FOR AUTOMATIC LOGGING
# ============================================================================


def logged_operation(operation_name: str, logger_name: str = __name__):
    """Decorator to automatically log function entry/exit with timing."""

    def decorator(func: F) -> F:
        logger = logging.getLogger(logger_name)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.debug(f"▶ {operation_name} started")

            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"✅ {operation_name} completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(f"❌ {operation_name} failed after {elapsed:.2f}ms: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            logger.debug(f"▶ {operation_name} started")

            try:
                result = await func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(f"✅ {operation_name} completed in {elapsed:.2f}ms")
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(f"❌ {operation_name} failed after {elapsed:.2f}ms: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# These functions maintain backward compatibility with existing code


def truncate_value(value: Any, max_len: int = 200) -> str:
    """Legacy: Truncate long values."""
    return _truncate(value, max_len)


def log_parsed_pr(logger: logging.Logger, parsed_pr: dict) -> None:
    """Log detailed parsed PR information."""
    logger.info("📄 PARSED PR DETAILS:")
    logger.info(f"    PR Code: {parsed_pr.get('pr_code', 'N/A')}")
    logger.info(f"    Title: {parsed_pr.get('title', 'N/A')}")
    logger.info(f"    Program Family: {parsed_pr.get('program_family', 'N/A')}")
    logger.info(f"    Customer: {parsed_pr.get('customer', 'N/A')}")
    logger.info(f"    Sector: {parsed_pr.get('sector', 'N/A')}")

    features = parsed_pr.get("detected_features", {})
    if features:
        true_features = [k for k, v in features.items() if v]
        logger.info(
            f"    Features ({len(features)} total, {len(true_features)} active):"
        )
        logger.info(f"        Active: {', '.join(true_features[:10])}")


def log_questions(logger: logging.Logger, questions: list) -> None:
    """Log generated questions."""
    logger.info(f"❓ GENERATED QUESTIONS ({len(questions)} total):")
    for i, q in enumerate(questions[:5], 1):
        text = q.get("question", q.get("text", "N/A"))
        logger.info(f"    {i}. [{q.get('category', 'N/A')}] {_truncate(text, 80)}")
    if len(questions) > 5:
        logger.info(f"    ... and {len(questions) - 5} more questions")


def log_breakdown(logger: logging.Logger, breakdown: list) -> None:
    """Log estimation breakdown."""
    logger.info(f"📊 ESTIMATION BREAKDOWN ({len(breakdown)} items):")
    total_hours = 0
    total_cost = 0
    for item in breakdown[:8]:
        hours = item.get("hours", 0)
        cost = item.get("cost_eur", 0)
        total_hours += hours
        total_cost += cost
        name = item.get("activity_name", item.get("function", "N/A"))
        logger.info(f"    • {name}: {hours}h, €{cost:,.0f}")
    if len(breakdown) > 8:
        logger.info(f"    ... and {len(breakdown) - 8} more items")
    logger.info(f"    TOTAL: {total_hours}h, €{total_cost:,.0f}")

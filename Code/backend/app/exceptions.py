"""
FPT Cost Brain 2.0 - Custom Exceptions
"""

from typing import Any


class FPTCostBrainException(Exception):
    """Base exception for FPT Cost Brain application."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


# ===== Authentication Exceptions =====


class AuthenticationError(FPTCostBrainException):
    """Authentication failed."""

    def __init__(
        self, message: str = "Authentication failed", details: dict | None = None
    ):
        super().__init__(
            message=message,
            error_code="AUTH_FAILED",
            status_code=401,
            details=details,
        )


class InvalidCredentialsError(AuthenticationError):
    """Invalid username or password."""

    def __init__(self):
        super().__init__(
            message="Invalid email or password",
            details={"hint": "Check your credentials and try again"},
        )


class TokenExpiredError(AuthenticationError):
    """JWT token has expired."""

    def __init__(self):
        super().__init__(
            message="Token has expired",
            details={"hint": "Please log in again"},
        )


class InsufficientPermissionsError(FPTCostBrainException):
    """User lacks required permissions."""

    def __init__(self, required_role: str | None = None):
        details = {}
        if required_role:
            details["required_role"] = required_role
        super().__init__(
            message="Insufficient permissions for this action",
            error_code="FORBIDDEN",
            status_code=403,
            details=details,
        )


# ===== Validation Exceptions =====


class ValidationError(FPTCostBrainException):
    """Data validation failed."""

    def __init__(
        self, message: str, field: str | None = None, details: dict | None = None
    ):
        error_details = details or {}
        if field:
            error_details["field"] = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=422,
            details=error_details,
        )


class InvalidFileFormatError(ValidationError):
    """Invalid file format uploaded."""

    def __init__(self, expected: list[str], received: str):
        super().__init__(
            message=f"Invalid file format. Expected: {', '.join(expected)}, received: {received}",
            details={"expected_formats": expected, "received_format": received},
        )


class MissingFieldsError(ValidationError):
    """Required fields are missing from the document."""

    def __init__(self, missing_fields: list[str]):
        super().__init__(
            message=f"Missing required fields: {', '.join(missing_fields)}",
            details={"missing_fields": missing_fields},
        )


# ===== Resource Exceptions =====


class NotFoundError(FPTCostBrainException):
    """Resource not found."""

    def __init__(self, resource: str, identifier: str | None = None):
        details = {"resource": resource}
        if identifier:
            details["identifier"] = identifier
        super().__init__(
            message=f"{resource} not found",
            error_code="NOT_FOUND",
            status_code=404,
            details=details,
        )


class PRNotFoundError(NotFoundError):
    """Product Request not found."""

    def __init__(self, pr_id: str):
        super().__init__(resource="Product Request", identifier=pr_id)


class QuotationNotFoundError(NotFoundError):
    """Quotation not found."""

    def __init__(self, quotation_id: str):
        super().__init__(resource="Quotation", identifier=quotation_id)


class SessionNotFoundError(NotFoundError):
    """Estimation session not found."""

    def __init__(self, session_id: str):
        super().__init__(resource="Estimation Session", identifier=session_id)


# ===== Estimation Exceptions =====


class EstimationError(FPTCostBrainException):
    """Error during estimation process."""

    def __init__(
        self, message: str, step: str | None = None, details: dict | None = None
    ):
        error_details = details or {}
        if step:
            error_details["step"] = step
        super().__init__(
            message=message,
            error_code="ESTIMATION_ERROR",
            status_code=400,
            details=error_details,
        )


class StepTransitionError(EstimationError):
    """Invalid step transition in estimation workflow."""

    def __init__(self, current_step: str, requested_step: str):
        super().__init__(
            message=f"Cannot transition from '{current_step}' to '{requested_step}'",
            details={"current_step": current_step, "requested_step": requested_step},
        )


class ModelPredictionError(EstimationError):
    """ML model prediction failed."""

    def __init__(self, message: str = "Model prediction failed"):
        super().__init__(
            message=message,
            step="estimation",
            details={"hint": "Try adjusting input features or contact support"},
        )


# ===== LLM Exceptions =====


class LLMError(FPTCostBrainException):
    """Error communicating with LLM."""

    def __init__(
        self, message: str, provider: str = "OpenRouter", details: dict | None = None
    ):
        error_details = details or {}
        error_details["provider"] = provider
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=502,
            details=error_details,
        )


class LLMRateLimitError(LLMError):
    """LLM rate limit exceeded."""

    def __init__(self, retry_after: int | None = None):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(
            message="LLM rate limit exceeded. Please try again later.",
            details=details,
        )


# ===== Database Exceptions =====


class DatabaseError(FPTCostBrainException):
    """Database operation failed."""

    def __init__(
        self, message: str = "Database operation failed", details: dict | None = None
    ):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=500,
            details=details,
        )


class DuplicateEntryError(DatabaseError):
    """Duplicate entry in database."""

    def __init__(self, field: str, value: str):
        super().__init__(
            message=f"Duplicate entry for {field}: {value}",
            details={"field": field, "value": value},
        )


# ===== Export Exceptions =====


class ExportError(FPTCostBrainException):
    """Error during export generation."""

    def __init__(
        self, message: str, format: str | None = None, details: dict | None = None
    ):
        error_details = details or {}
        if format:
            error_details["format"] = format
        super().__init__(
            message=message,
            error_code="EXPORT_ERROR",
            status_code=500,
            details=error_details,
        )

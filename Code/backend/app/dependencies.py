"""
FPT Cost Brain 2.0 - Dependency Injection
FastAPI dependencies for database sessions, authentication, etc.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import (
    AuthenticationError,
    InsufficientPermissionsError,
    TokenExpiredError,
)
from db.models import User
from db.session import async_session_maker

# Security scheme
security = HTTPBearer()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()


# Type alias for database dependency
DatabaseSession = Annotated[AsyncSession, Depends(get_db)]


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: DatabaseSession,
) -> User:
    """
    Get current authenticated user from JWT token.

    Returns User model instance.
    """
    from services.auth_service import AuthService

    auth_service = AuthService(db)

    try:
        token = credentials.credentials
        user = await auth_service.verify_token(token)
        return user
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Type alias for current user dependency
CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_active_user(current_user: CurrentUser) -> User:
    """Ensure user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return current_user


ActiveUser = Annotated[User, Depends(get_current_active_user)]


def require_role(required_roles: list[str]):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(["admin", "head"]))])
    """

    async def role_checker(current_user: CurrentUser) -> User:
        if current_user.role not in required_roles:
            raise InsufficientPermissionsError(required_role=", ".join(required_roles))
        return current_user

    return role_checker


# Pre-defined role checkers
RequireManager = Depends(require_role(["manager", "head", "executive"]))
RequireHead = Depends(require_role(["head", "executive"]))
RequireExecutive = Depends(require_role(["executive"]))


async def require_admin(current_user: CurrentUser) -> User:
    """Require admin role for access."""
    if current_user.role not in ["admin", "executive"]:
        raise InsufficientPermissionsError(required_role="admin")
    return current_user


RequireAdmin = Depends(require_admin)


async def get_optional_user(
    db: DatabaseSession,
    authorization: str | None = Header(default=None),
) -> User | None:
    """
    Get current user if token provided, otherwise return None.
    Useful for endpoints that work both authenticated and anonymous.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    from services.auth_service import AuthService

    try:
        token = authorization.replace("Bearer ", "")
        auth_service = AuthService(db)
        return await auth_service.verify_token(token)
    except Exception:
        return None


OptionalUser = Annotated[User | None, Depends(get_optional_user)]


async def get_redis():
    """Get Redis client dependency."""
    from services.cache_service import get_redis_client

    return await get_redis_client()


RedisClient = Annotated[object, Depends(get_redis)]


async def get_vector_db():
    """Get Qdrant client dependency."""
    from vector.client import get_qdrant_client

    return await get_qdrant_client()


VectorDB = Annotated[object, Depends(get_vector_db)]


async def get_llm_client():
    """Get LLM client dependency."""
    from llm.client import get_llm_client as _get_llm_client

    return _get_llm_client()


LLMClient = Annotated[object, Depends(get_llm_client)]

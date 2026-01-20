"""
FPT Cost Brain 2.0 - Authentication API
Endpoints for user authentication and management
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_current_user, get_db
from app.exceptions import AuthenticationError, InvalidCredentialsError
from db.models import User
from db.repositories.audit_repo import AuditRepository
from services.auth_service import AuthService, create_token_for_user

router = APIRouter(tags=["Authentication"])


# ===== Schemas =====


class UserCreate(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str
    full_name: str
    role: str = "engineer"
    language: str = "en"


class UserResponse(BaseModel):
    """Schema for user response."""

    id: str
    email: str
    full_name: str
    role: str
    language: str
    is_active: bool

    class Config:
        from_attributes = True


class TokenUser(BaseModel):
    """User info in token response."""

    id: str
    email: str
    full_name: str
    role: str


class TokenResponse(BaseModel):
    """Schema for token response."""

    access_token: str
    token_type: str
    user: TokenUser


class PasswordChange(BaseModel):
    """Schema for password change."""

    current_password: str
    new_password: str


class LanguageUpdate(BaseModel):
    """Schema for language preference update."""

    language: str


class LoginRequest(BaseModel):
    """Schema for JSON login request."""

    email: EmailStr
    password: str


# ===== Endpoints =====


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Authenticate user and return access token.

    - **username**: User's email address
    - **password**: User's password
    """
    auth_service = AuthService(db)
    audit_repo = AuditRepository(db)

    try:
        user = await auth_service.authenticate_user(
            email=form_data.username,
            password=form_data.password,
        )

        # Log successful login
        await audit_repo.log_login(
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            success=True,
        )

        return create_token_for_user(user)

    except InvalidCredentialsError:
        # Log failed login attempt
        existing_user = await auth_service.get_user_by_email(form_data.username)
        if existing_user:
            await audit_repo.log_login(
                user_id=existing_user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                success=False,
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/login/json", response_model=TokenResponse)
async def login_json(
    request: Request,
    login_data: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Authenticate user with JSON body and return access token.

    - **email**: User's email address
    - **password**: User's password
    """
    auth_service = AuthService(db)
    audit_repo = AuditRepository(db)

    try:
        user = await auth_service.authenticate_user(
            email=login_data.email,
            password=login_data.password,
        )

        # Log successful login
        await audit_repo.log_login(
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            success=True,
        )

        return create_token_for_user(user)

    except InvalidCredentialsError:
        # Log failed login attempt
        existing_user = await auth_service.get_user_by_email(login_data.email)
        if existing_user:
            await audit_repo.log_login(
                user_id=existing_user.id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                success=False,
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/logout")
async def logout(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    Logout current user.

    Note: JWT tokens are stateless, so this endpoint is mainly for
    client-side cleanup and audit logging.
    """
    # In a stateful session implementation, you would invalidate the session here
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get current authenticated user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        language=current_user.language,
        is_active=current_user.is_active,
    )


@router.post(
    "/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED
)
async def register_user(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Register a new user.

    Note: In production, this endpoint should be restricted to admins only.
    """
    auth_service = AuthService(db)

    try:
        user = await auth_service.create_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            role=user_data.role,
            language=user_data.language,
        )

        return UserResponse(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            language=user.language,
            is_active=user.is_active,
        )

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Change current user's password."""
    auth_service = AuthService(db)

    # Verify current password
    if not auth_service.verify_password(
        password_data.current_password,
        current_user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    await auth_service.update_user_password(
        user_id=str(current_user.id),
        new_password=password_data.new_password,
    )

    return {"message": "Password changed successfully"}


@router.patch("/language", response_model=UserResponse)
async def update_language(
    language_data: LanguageUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Update current user's language preference."""
    if language_data.language not in ["en", "it"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid language. Supported: en, it",
        )

    auth_service = AuthService(db)
    await auth_service.update_user_language(
        user_id=str(current_user.id),
        language=language_data.language,
    )

    # Refresh user data
    updated_user = await auth_service.get_user_by_id(str(current_user.id))

    return UserResponse(
        id=str(updated_user.id),
        email=updated_user.email,
        full_name=updated_user.full_name,
        role=updated_user.role,
        language=updated_user.language,
        is_active=updated_user.is_active,
    )


@router.get("/verify")
async def verify_token(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Verify that the current token is valid."""
    return {
        "valid": True,
        "user_id": str(current_user.id),
        "email": current_user.email,
    }

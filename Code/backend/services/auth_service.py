"""
FPT Cost Brain 2.0 - Authentication Service
JWT-based authentication with password hashing
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    TokenExpiredError,
)
from db.models import User


class AuthService:
    """Authentication service for user management and JWT handling."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ===== Password Operations =====

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        password_bytes = password.encode("utf-8")
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        return hashed.decode("utf-8")

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        password_bytes = plain_password.encode("utf-8")
        hashed_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_bytes)

    # ===== Token Operations =====

    @staticmethod
    def create_access_token(
        data: dict[str, Any],
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
            )

        to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})

        return jwt.encode(
            to_encode,
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

    async def verify_token(self, token: str) -> User:
        """Verify a JWT token and return the User model."""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )

            user_id: str = payload.get("sub")
            if user_id is None:
                raise AuthenticationError("Invalid token: missing subject")

            # Get user from database
            user = await self.get_user_by_id(user_id)
            if user is None:
                raise AuthenticationError("User not found")

            return user

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError()
        except JWTError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")

    # ===== User Operations =====

    async def authenticate_user(self, email: str, password: str) -> User:
        """
        Authenticate a user with email and password.

        Uses constant-time comparison to prevent timing attacks that could
        enumerate valid email addresses.
        """
        user = await self.get_user_by_email(email)

        if user is None:
            # Perform dummy password check to prevent timing attacks
            # This ensures the response time is similar whether user exists or not
            self.verify_password(password, self.hash_password("dummy_password_check"))
            raise InvalidCredentialsError()

        if not self.verify_password(password, user.password_hash):
            raise InvalidCredentialsError()

        if not user.is_active:
            raise AuthenticationError("User account is disabled")

        # Update last login (use UTC datetime, timezone-naive for TIMESTAMP WITHOUT TIME ZONE column)
        user.last_login = datetime.utcnow()
        await self.db.commit()

        return user

    async def get_user_by_email(self, email: str) -> User | None:
        """Get a user by email address."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get a user by ID."""
        result = await self.db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str,
        role: str = "engineer",
        language: str = "en",
    ) -> User:
        """Create a new user."""
        # Check if user already exists
        existing = await self.get_user_by_email(email)
        if existing:
            raise AuthenticationError(f"User with email {email} already exists")

        user = User(
            email=email,
            password_hash=self.hash_password(password),
            full_name=full_name,
            role=role,
            language=language,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def update_user_password(self, user_id: str, new_password: str) -> None:
        """Update a user's password."""
        user = await self.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationError("User not found")

        user.password_hash = self.hash_password(new_password)
        await self.db.commit()

    async def update_user_language(self, user_id: str, language: str) -> None:
        """Update a user's preferred language."""
        user = await self.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationError("User not found")

        user.language = language
        await self.db.commit()


def create_token_for_user(user: User) -> dict:
    """Create access token for a user."""
    access_token = AuthService.create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "role": user.role,
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }

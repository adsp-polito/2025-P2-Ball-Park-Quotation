"""
FPT Cost Brain 2.0 - Configuration
Centralized settings management using pydantic-settings
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Application =====
    APP_NAME: str = "FPT Cost Brain"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ===== API =====
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ===== Authentication =====
    JWT_SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # ===== Database =====
    DATABASE_URL: PostgresDsn = Field(
        default="postgresql+asyncpg://fpt:fpt@localhost:5432/fpt_cost_brain",
        description="PostgreSQL connection string",
    )
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # ===== Redis =====
    REDIS_URL: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string",
    )
    DISABLE_CACHE: bool = Field(
        default=False,
        description="Disable Redis caching (useful for development/debugging)",
    )

    # ===== Vector Database (Qdrant) =====
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: str | None = None

    # ===== LLM (OpenRouter) =====
    OPENROUTER_API_KEY: str = Field(..., description="OpenRouter API key")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # LLM Models (all via OpenRouter)
    LLM_REASONING_MODEL: str = "deepseek/deepseek-chat"
    LLM_REASONING_ALT_MODEL: str = "google/gemini-2.5-flash-preview"
    LLM_FAST_MODEL: str = "deepseek/deepseek-chat"

    # Embedding model via OpenRouter (Qwen3 Embedding - cost-effective, high quality)
    LLM_EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"
    LLM_EMBEDDING_DIMENSIONS: int = (
        4096  # Qwen3 supports 32-4096, using max for best quality
    )

    # ===== ML Model =====
    ML_MODEL_PATH: str = "models/production_predictor.pkl"
    ML_MIN_CONFIDENCE: float = 0.3
    ML_HIGH_CONFIDENCE: float = 0.8

    # ===== Online Learning =====
    RETRAIN_MIN_CORRECTIONS: int = 5
    RETRAIN_MAX_CORRECTIONS: int = 20
    RETRAIN_DRIFT_THRESHOLD: float = 0.15
    RETRAIN_MAX_DAYS: int = 7
    RETRAIN_IMPROVEMENT_THRESHOLD: float = 0.05

    # ===== Export =====
    PE02_TEMPLATE_PATH: str = "templates/pe02_template.pptx"

    # ===== Validation =====
    MIN_COST_EUR: float = 5_000
    MAX_COST_EUR: float = 10_000_000
    MIN_HOURS: int = 100
    MAX_HOURS: int = 200_000

    # ===== Logging =====
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "console"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export singleton
settings = get_settings()

"""
FPT Cost Brain 2.0 - Redis Cache Service
Caching layer for sessions, embeddings, and frequent queries

With comprehensive debug logging for cache hits/misses visibility.
Enable with: DEBUG=true or LOG_LEVEL=DEBUG in .env
"""

import asyncio
import json
import logging
import time
from datetime import timedelta
from typing import Any

import redis.asyncio as redis

from app.config import settings

# Initialize logger for cache operations
logger = logging.getLogger(__name__)


class CacheService:
    """Redis cache service for application-wide caching."""

    def __init__(self, redis_client: redis.Redis, disabled: bool = False):
        self.redis = redis_client
        self.disabled = disabled  # Skip caching when True (for dev/debug)

    # ===== Basic Operations =====

    async def get(self, key: str) -> Any | None:
        """Get a value from cache."""
        start_time = time.perf_counter()
        key_type = key.split(":")[0] if ":" in key else "unknown"

        if self.disabled:
            logger.debug(f"💾 CACHE SKIP (disabled): {key_type}:{key[-16:]}")
            return None  # Cache miss when disabled

        value = await self.redis.get(key)
        duration_ms = (time.perf_counter() - start_time) * 1000

        if value is None:
            logger.debug(f"❌ CACHE MISS: {key_type}:{key[-20:]} [{duration_ms:.2f}ms]")
            return None

        # Parse value
        try:
            result = json.loads(value)
            value_size = len(value) if isinstance(value, (str, bytes)) else 0
            logger.debug(
                f"✅ CACHE HIT: {key_type}:{key[-20:]} "
                f"[{duration_ms:.2f}ms, {value_size} bytes]"
            )
            return result
        except json.JSONDecodeError:
            result = value.decode("utf-8") if isinstance(value, bytes) else value
            logger.debug(
                f"✅ CACHE HIT (raw): {key_type}:{key[-20:]} [{duration_ms:.2f}ms]"
            )
            return result

    async def set(
        self,
        key: str,
        value: Any,
        expire: int | timedelta | None = None,
    ) -> bool:
        """Set a value in cache with optional expiration."""
        start_time = time.perf_counter()
        key_type = key.split(":")[0] if ":" in key else "unknown"

        if self.disabled:
            logger.debug(f"💾 CACHE SET SKIP (disabled): {key_type}:{key[-16:]}")
            return True  # Pretend success when disabled

        # Serialize value
        original_value = value
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        elif not isinstance(value, (str, bytes)):
            value = str(value)

        value_size = len(value) if isinstance(value, (str, bytes)) else 0

        if isinstance(expire, timedelta):
            expire = int(expire.total_seconds())

        result = await self.redis.set(key, value, ex=expire)
        duration_ms = (time.perf_counter() - start_time) * 1000

        ttl_str = f", TTL={expire}s" if expire else ", no TTL"
        logger.debug(
            f"📝 CACHE SET: {key_type}:{key[-20:]} "
            f"[{duration_ms:.2f}ms, {value_size} bytes{ttl_str}]"
        )

        return result

    async def delete(self, key: str) -> int:
        """Delete a key from cache."""
        start_time = time.perf_counter()
        key_type = key.split(":")[0] if ":" in key else "unknown"

        result = await self.redis.delete(key)
        duration_ms = (time.perf_counter() - start_time) * 1000

        if result > 0:
            logger.debug(
                f"🗑️ CACHE DELETE: {key_type}:{key[-20:]} [{duration_ms:.2f}ms]"
            )
        else:
            logger.debug(
                f"🗑️ CACHE DELETE (not found): {key_type}:{key[-20:]} [{duration_ms:.2f}ms]"
            )

        return result

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return await self.redis.exists(key) > 0

    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on a key."""
        return await self.redis.expire(key, seconds)

    async def ttl(self, key: str) -> int:
        """Get remaining TTL for a key."""
        return await self.redis.ttl(key)

    # ===== Session Caching =====

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def get_session(self, session_id: str) -> dict | None:
        """Get estimation session state from cache."""
        result = await self.get(self._session_key(session_id))
        if result:
            current_step = result.get("current_step", "unknown")
            logger.info(
                f"📋 SESSION LOADED: {session_id[:8]}... (step: {current_step})"
            )
        return result

    async def set_session(
        self,
        session_id: str,
        state: dict,
        expire: int = 86400,  # 24 hours default
    ) -> bool:
        """Cache estimation session state."""
        current_step = state.get("current_step", "unknown")
        steps_count = len(state.get("steps", {}))
        logger.info(
            f"📋 SESSION SAVED: {session_id[:8]}... "
            f"(step: {current_step}, {steps_count} steps cached, TTL={expire}s)"
        )
        return await self.set(self._session_key(session_id), state, expire)

    async def delete_session(self, session_id: str) -> int:
        """Delete session from cache."""
        logger.info(f"📋 SESSION DELETED: {session_id[:8]}...")
        return await self.delete(self._session_key(session_id))

    async def update_session_step(
        self,
        session_id: str,
        step: str,
        step_data: dict,
    ) -> bool:
        """Update specific step data in session."""
        session = await self.get_session(session_id)
        if session is None:
            logger.warning(
                f"⚠️ SESSION UPDATE FAILED: {session_id[:8]}... "
                f"(step: {step}) - session not found"
            )
            return False

        old_step = session.get("current_step", "unknown")
        session["current_step"] = step
        session["steps"] = session.get("steps", {})
        session["steps"][step] = step_data

        logger.info(
            f"📋 SESSION STEP UPDATE: {session_id[:8]}... ({old_step} → {step})"
        )

        return await self.set_session(session_id, session)

    # ===== Embedding Caching =====

    def _embedding_key(self, text_hash: str) -> str:
        return f"embedding:{text_hash}"

    async def get_embedding(self, text_hash: str) -> list[float] | None:
        """Get cached embedding by text hash."""
        result = await self.get(self._embedding_key(text_hash))
        if result:
            dims = len(result) if isinstance(result, list) else 0
            logger.debug(f"🧬 EMBEDDING HIT: {text_hash[:12]}... ({dims} dims)")
        else:
            logger.debug(f"🧬 EMBEDDING MISS: {text_hash[:12]}...")
        return result

    async def set_embedding(
        self,
        text_hash: str,
        embedding: list[float],
        expire: int = 604800,  # 7 days default
    ) -> bool:
        """Cache embedding vector."""
        dims = len(embedding)
        logger.debug(
            f"🧬 EMBEDDING CACHED: {text_hash[:12]}... "
            f"({dims} dims, TTL={expire // 86400} days)"
        )
        return await self.set(self._embedding_key(text_hash), embedding, expire)

    # ===== Similar PRs Caching =====

    def _similar_prs_key(self, pr_id: str) -> str:
        return f"similar_prs:{pr_id}"

    async def get_similar_prs(self, pr_id: str) -> list[dict] | None:
        """Get cached similar PRs for a product request."""
        result = await self.get(self._similar_prs_key(pr_id))
        if result:
            count = len(result) if isinstance(result, list) else 0
            logger.info(f"🔍 SIMILAR PRs HIT: PR {pr_id[:8]}... ({count} matches)")
        else:
            logger.debug(f"🔍 SIMILAR PRs MISS: PR {pr_id[:8]}...")
        return result

    async def set_similar_prs(
        self,
        pr_id: str,
        similar_prs: list[dict],
        expire: int = 3600,  # 1 hour default
    ) -> bool:
        """Cache similar PRs results."""
        count = len(similar_prs)
        logger.info(
            f"🔍 SIMILAR PRs CACHED: PR {pr_id[:8]}... "
            f"({count} matches, TTL={expire // 60} min)"
        )
        return await self.set(self._similar_prs_key(pr_id), similar_prs, expire)

    # ===== ML Prediction Caching =====

    def _prediction_key(self, features_hash: str) -> str:
        return f"prediction:{features_hash}"

    async def get_prediction(self, features_hash: str) -> dict | None:
        """Get cached ML prediction."""
        result = await self.get(self._prediction_key(features_hash))
        if result:
            point_est = result.get("point_estimate", 0)
            confidence = result.get("confidence", 0)
            logger.info(
                f"🤖 ML PREDICTION HIT: {features_hash[:12]}... "
                f"(€{point_est * 1000:,.0f}, conf={confidence:.1%})"
            )
        else:
            logger.debug(f"🤖 ML PREDICTION MISS: {features_hash[:12]}...")
        return result

    async def set_prediction(
        self,
        features_hash: str,
        prediction: dict,
        expire: int = 3600,  # 1 hour default
    ) -> bool:
        """Cache ML prediction result."""
        point_est = prediction.get("point_estimate", 0)
        confidence = prediction.get("confidence", 0)
        logger.info(
            f"🤖 ML PREDICTION CACHED: {features_hash[:12]}... "
            f"(€{point_est * 1000:,.0f}, conf={confidence:.1%}, TTL={expire // 60} min)"
        )
        return await self.set(self._prediction_key(features_hash), prediction, expire)

    # ===== Rules Caching =====

    def _rules_key(self) -> str:
        return "learned_rules:active"

    async def get_rules(self) -> list[dict] | None:
        """Get cached active rules."""
        result = await self.get(self._rules_key())
        if result:
            count = len(result) if isinstance(result, list) else 0
            logger.debug(f"📜 RULES HIT: {count} active rules loaded from cache")
        else:
            logger.debug("📜 RULES MISS: no cached rules")
        return result

    async def set_rules(
        self,
        rules: list[dict],
        expire: int = 300,  # 5 minutes default
    ) -> bool:
        """Cache active rules."""
        count = len(rules)
        logger.debug(f"📜 RULES CACHED: {count} rules (TTL={expire}s)")
        return await self.set(self._rules_key(), rules, expire)

    async def invalidate_rules(self) -> int:
        """Invalidate rules cache."""
        logger.info("📜 RULES INVALIDATED: forcing reload from database")
        return await self.delete(self._rules_key())

    # ===== Rate Limiting =====

    def _rate_limit_key(self, user_id: str, action: str) -> str:
        return f"rate_limit:{user_id}:{action}"

    async def check_rate_limit(
        self,
        user_id: str,
        action: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check rate limit for a user action using atomic INCR.
        Returns (allowed, remaining_requests).

        Uses Redis INCR which is atomic, then checks the result.
        If this is the first request, we set the expiry.
        """
        key = self._rate_limit_key(user_id, action)

        # INCR is atomic - increment first, check after
        # This prevents race conditions between check and increment
        current_count = await self.redis.incr(key)

        # If this is the first request (count == 1), set expiration
        if current_count == 1:
            await self.redis.expire(key, window_seconds)

        # Check if over limit
        if current_count > max_requests:
            # Over limit - decrement back (optional, for accuracy)
            # We're already over, so this request is denied
            logger.warning(
                f"🚫 RATE LIMIT EXCEEDED: user={user_id[:8]}... "
                f"action={action} ({current_count}/{max_requests})"
            )
            return False, 0

        remaining = max_requests - current_count
        logger.debug(
            f"⏱️ RATE LIMIT OK: user={user_id[:8]}... "
            f"action={action} ({current_count}/{max_requests}, {remaining} remaining)"
        )
        return True, remaining

    # ===== Pub/Sub for Real-time Updates =====

    async def publish(self, channel: str, message: dict) -> int:
        """Publish message to channel."""
        return await self.redis.publish(channel, json.dumps(message, default=str))

    async def subscribe(self, *channels: str):
        """Subscribe to channels. Returns pubsub object."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(*channels)
        return pubsub

    # ===== Bulk Operations =====

    async def mget(self, *keys: str) -> list[Any]:
        """Get multiple values at once."""
        values = await self.redis.mget(*keys)
        results = []
        for v in values:
            if v is None:
                results.append(None)
            else:
                try:
                    results.append(json.loads(v))
                except json.JSONDecodeError:
                    results.append(v.decode("utf-8") if isinstance(v, bytes) else v)
        return results

    async def mset(self, mapping: dict[str, Any]) -> bool:
        """Set multiple values at once."""
        encoded = {}
        for k, v in mapping.items():
            if isinstance(v, (dict, list)):
                encoded[k] = json.dumps(v, default=str)
            elif not isinstance(v, (str, bytes)):
                encoded[k] = str(v)
            else:
                encoded[k] = v
        return await self.redis.mset(encoded)

    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                deleted += await self.redis.delete(*keys)
            if cursor == 0:
                break
        return deleted

    # ===== Health Check =====

    async def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return await self.redis.ping()
        except Exception:
            return False


async def get_cache_service() -> CacheService:
    """Create cache service instance."""
    client = redis.from_url(
        str(settings.REDIS_URL),
        encoding="utf-8",
        decode_responses=False,
    )
    return CacheService(client, disabled=settings.DISABLE_CACHE)


# Global Redis client (thread-safe singleton)
_redis_client: redis.Redis | None = None
_redis_init_lock: "asyncio.Lock | None" = None


def _get_redis_lock() -> "asyncio.Lock":
    """Get or create the Redis initialization lock."""
    global _redis_init_lock
    if _redis_init_lock is None:
        _redis_init_lock = asyncio.Lock()
    return _redis_init_lock


async def init_redis() -> None:
    """Initialize Redis connection (thread-safe)."""
    global _redis_client
    async with _get_redis_lock():
        if _redis_client is None:
            start_time = time.perf_counter()
            logger.info("💾 REDIS: Initializing connection...")

            _redis_client = redis.from_url(
                str(settings.REDIS_URL),
                encoding="utf-8",
                decode_responses=False,
            )
            # Test connection
            await _redis_client.ping()

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"💾 REDIS: Connected successfully "
                f"[{duration_ms:.2f}ms, url={settings.REDIS_URL}]"
            )


async def close_redis() -> None:
    """Close Redis connection (thread-safe)."""
    global _redis_client
    async with _get_redis_lock():
        if _redis_client is not None:
            logger.info("💾 REDIS: Closing connection...")
            await _redis_client.aclose()
            _redis_client = None
            logger.info("💾 REDIS: Connection closed")


async def get_redis_client() -> redis.Redis:
    """Get the global Redis client (thread-safe)."""
    global _redis_client
    if _redis_client is None:
        await init_redis()
    return _redis_client

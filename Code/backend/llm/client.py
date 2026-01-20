"""
FPT Cost Brain 2.0 - LLM Client
OpenRouter integration with streaming support

With comprehensive debug logging for LLM operations visibility.
Enable with: DEBUG=true or LOG_LEVEL=DEBUG in .env
"""

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from app.config import settings
from openai import AsyncOpenAI

# Initialize logger for LLM operations
logger = logging.getLogger(__name__)


class LLMClient:
    """OpenRouter LLM client with model switching and streaming."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=str(settings.OPENROUTER_BASE_URL),
            default_headers={
                "HTTP-Referer": "https://fpt-costbrain.com",
                "X-Title": "FPT Cost Brain",
            },
        )
        self._http_client = httpx.AsyncClient(timeout=60.0)

    # ===== Chat Completion =====

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Send chat completion request and return response text."""
        start_time = time.perf_counter()
        model = model or settings.LLM_REASONING_MODEL
        model_short = model.split("/")[-1] if "/" in model else model

        # Estimate input tokens
        total_input_chars = sum(len(m.get("content", "")) for m in messages)
        est_input_tokens = total_input_chars // 4

        # Log request details
        logger.info(
            f"🧠 LLM REQUEST: model={model_short}, temp={temperature}, "
            f"max_tokens={max_tokens}, messages={len(messages)}, "
            f"~{est_input_tokens} input tokens"
        )

        # Log prompt preview (first message truncated)
        if messages and logger.isEnabledFor(logging.DEBUG):
            last_msg = messages[-1].get("content", "")[:200]
            logger.debug(f"   📝 Prompt preview: {last_msg}...")

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        content = response.choices[0].message.content or ""
        est_output_tokens = len(content) // 4

        # Extract usage if available
        usage_info = ""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            usage_info = (
                f", tokens: {usage.prompt_tokens}→{usage.completion_tokens} "
                f"(total: {usage.total_tokens})"
            )

        logger.info(
            f"✅ LLM RESPONSE: model={model_short}, "
            f"output={len(content)} chars (~{est_output_tokens} tokens), "
            f"duration={duration_ms:.0f}ms{usage_info}"
        )

        # Log response preview
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"   📤 Response preview: {content[:200]}...")

        return content

    async def chat_with_retry(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """Chat with automatic retry and model fallback."""
        primary_model = model or settings.LLM_REASONING_MODEL
        fallback_model = settings.LLM_REASONING_ALT_MODEL
        primary_short = (
            primary_model.split("/")[-1] if "/" in primary_model else primary_model
        )
        fallback_short = (
            fallback_model.split("/")[-1] if "/" in fallback_model else fallback_model
        )

        for attempt in range(max_retries):
            try:
                current_model = primary_model if attempt < 2 else fallback_model
                current_short = (
                    current_model.split("/")[-1]
                    if "/" in current_model
                    else current_model
                )

                if attempt > 0:
                    logger.warning(
                        f"🔄 LLM RETRY: attempt {attempt + 1}/{max_retries}, "
                        f"using {current_short}"
                    )

                return await self.chat(messages, model=current_model, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                logger.error(
                    f"❌ LLM ERROR: {error_type}: {str(e)[:100]}, "
                    f"attempt {attempt + 1}/{max_retries}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"💀 LLM FAILED: all {max_retries} attempts exhausted, "
                        f"tried {primary_short} and {fallback_short}"
                    )
                    raise
                backoff = 2**attempt
                logger.info(f"⏳ LLM BACKOFF: waiting {backoff}s before retry...")
                await asyncio.sleep(backoff)

        return ""  # Should not reach here

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat completion response."""
        model = model or settings.LLM_REASONING_MODEL

        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ===== Specialized Methods =====

    async def reason(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """
        Use reasoning model for complex analysis.
        Best for: cost estimation, feature extraction, rule generation.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
            prompt = f"Context:\n{context_str}\n\n{prompt}"

        messages.append({"role": "user", "content": prompt})

        return await self.chat_with_retry(
            messages,
            model=settings.LLM_REASONING_MODEL,
            temperature=0.3,  # Lower for reasoning tasks
        )

    async def fast_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Use fast model for quick responses.
        Best for: Q&A, summaries, simple classification.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages,
            model=settings.LLM_FAST_MODEL,
            temperature=0.5,
            max_tokens=2048,
        )

    async def extract_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Extract structured JSON from LLM response."""
        import json

        full_system = (system_prompt or "") + (
            "\n\nRespond ONLY with valid JSON. No explanations, no markdown."
        )

        response = await self.reason(prompt, full_system)

        # Try to parse JSON from response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]

        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            # Try to find JSON in response
            import re

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")

    # ===== Embeddings =====

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for text using OpenRouter API.

        Uses Qwen3 Embedding 8B model with configurable dimensions.
        Reuses the shared HTTP client for connection pooling.
        """
        start_time = time.perf_counter()
        text_len = len(text)
        model_short = settings.LLM_EMBEDDING_MODEL.split("/")[-1]

        logger.debug(
            f"🧬 EMBED REQUEST: {text_len} chars, "
            f"dims={settings.LLM_EMBEDDING_DIMENSIONS}, model={model_short}"
        )

        response = await self._http_client.post(
            f"{settings.OPENROUTER_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://fpt-costbrain.com",
                "X-Title": "FPT Cost Brain",
            },
            json={
                "model": settings.LLM_EMBEDDING_MODEL,
                "input": text,
                "dimensions": settings.LLM_EMBEDDING_DIMENSIONS,  # Qwen3 supports 32-4096
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        duration_ms = (time.perf_counter() - start_time) * 1000
        embedding = data["data"][0]["embedding"]

        logger.debug(f"✅ EMBED COMPLETE: {len(embedding)} dims, {duration_ms:.0f}ms")

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 20,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts using OpenRouter.

        Reuses the shared HTTP client for connection pooling.
        """
        start_time = time.perf_counter()
        total_texts = len(texts)
        num_batches = (total_texts + batch_size - 1) // batch_size
        model_short = settings.LLM_EMBEDDING_MODEL.split("/")[-1]

        logger.info(
            f"🧬 EMBED BATCH START: {total_texts} texts, "
            f"{num_batches} batches, model={model_short}"
        )

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            batch = texts[i : i + batch_size]
            batch_start = time.perf_counter()

            logger.debug(
                f"   🧬 Batch {batch_num}/{num_batches}: {len(batch)} texts..."
            )

            response = await self._http_client.post(
                f"{settings.OPENROUTER_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://fpt-costbrain.com",
                    "X-Title": "FPT Cost Brain",
                },
                json={
                    "model": settings.LLM_EMBEDDING_MODEL,
                    "input": batch,
                    "dimensions": settings.LLM_EMBEDDING_DIMENSIONS,  # Qwen3 supports 32-4096
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            embeddings.extend([d["embedding"] for d in sorted_data])

            batch_ms = (time.perf_counter() - batch_start) * 1000
            logger.debug(f"   ✅ Batch {batch_num} complete: {batch_ms:.0f}ms")

        total_ms = (time.perf_counter() - start_time) * 1000
        dims = len(embeddings[0]) if embeddings else 0
        logger.info(
            f"✅ EMBED BATCH COMPLETE: {total_texts} embeddings, "
            f"{dims} dims each, {total_ms:.0f}ms total"
        )

        return embeddings

    def get_text_hash(self, text: str) -> str:
        """Generate hash for text (for caching embeddings)."""
        return hashlib.md5(text.encode()).hexdigest()

    # ===== Utilities =====

    def count_tokens(self, text: str) -> int:
        """Estimate token count (approximate)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4

    async def summarize(self, text: str, max_length: int = 500) -> str:
        """Summarize text to specified length."""
        return await self.fast_response(
            f"Summarize the following text in {max_length} characters or less:\n\n{text}",
            system_prompt="You are a concise summarizer. Preserve key technical details.",
        )

    async def classify(
        self,
        text: str,
        categories: list[str],
        allow_multiple: bool = False,
    ) -> list[str]:
        """Classify text into one or more categories."""
        mode = "one or more" if allow_multiple else "exactly one"
        categories_str = ", ".join(categories)

        response = await self.fast_response(
            f"Classify the following text into {mode} of these categories: {categories_str}\n\nText: {text}\n\nReturn only the category name(s), comma-separated if multiple.",
        )

        result = [c.strip() for c in response.split(",")]
        return [c for c in result if c in categories]

    # ===== Cleanup =====

    async def close(self):
        """Close HTTP client."""
        await self._http_client.aclose()


# Thread-safe singleton instance
_llm_client: LLMClient | None = None
_llm_client_lock = asyncio.Lock()


def get_llm_client() -> LLMClient:
    """
    Get or create LLM client singleton (thread-safe).

    Uses a simple check since LLMClient creation is fast and idempotent.
    For truly thread-safe lazy initialization in async context, use
    get_llm_client_async().
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


async def get_llm_client_async() -> LLMClient:
    """Get or create LLM client singleton (async thread-safe)."""
    global _llm_client
    if _llm_client is None:
        async with _llm_client_lock:
            # Double-check pattern
            if _llm_client is None:
                _llm_client = LLMClient()
    return _llm_client

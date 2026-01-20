"""
FPT Cost Brain 2.0 - Qdrant Vector Database Client
Async client for vector similarity search
"""

import asyncio
from typing import Any

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from vector.collections import COLLECTIONS, CollectionConfig

# Thread-safe global client instance
_client: AsyncQdrantClient | None = None
_client_lock = asyncio.Lock()
_initialized = False


async def get_qdrant_client() -> AsyncQdrantClient:
    """Get the Qdrant client instance."""
    global _client
    if _client is None:
        raise RuntimeError(
            "Qdrant client not initialized. Call init_vector_db() first."
        )
    return _client


async def init_vector_db() -> None:
    """Initialize Qdrant client and create collections (thread-safe)."""
    global _client, _initialized

    async with _client_lock:
        # Double-check pattern
        if _initialized:
            return

        _client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )

        # Create collections if they don't exist
        for collection_name, config in COLLECTIONS.items():
            await _ensure_collection_exists(collection_name, config)

        _initialized = True


async def close_vector_db() -> None:
    """Close Qdrant client connection (thread-safe)."""
    global _client, _initialized
    async with _client_lock:
        if _client is not None:
            await _client.close()
            _client = None
            _initialized = False


async def _ensure_collection_exists(name: str, config: CollectionConfig) -> None:
    """Create collection if it doesn't exist."""
    try:
        await _client.get_collection(name)
    except (UnexpectedResponse, Exception):
        # Collection doesn't exist, create it
        await _client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=config.vector_size,
                distance=models.Distance.COSINE,
            ),
        )


class VectorStore:
    """High-level vector store operations."""

    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def upsert(
        self,
        collection: str,
        id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Insert or update a vector."""
        await self.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    async def upsert_batch(
        self,
        collection: str,
        points: list[dict],
    ) -> None:
        """Batch upsert vectors."""
        qdrant_points = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        await self.client.upsert(
            collection_name=collection,
            points=qdrant_points,
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 5,
        filter_conditions: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        Search for similar vectors.

        Returns list of dicts with: id, score, payload
        """
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            for field, value in filter_conditions.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value),
                        )
                    )
            qdrant_filter = models.Filter(must=must_conditions)

        # qdrant-client 1.16.2+ uses query_points instead of search
        results = await self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    async def get_by_id(self, collection: str, id: str) -> dict | None:
        """Get a specific point by ID."""
        results = await self.client.retrieve(
            collection_name=collection,
            ids=[id],
            with_vectors=True,
        )
        if results:
            point = results[0]
            return {
                "id": str(point.id),
                "vector": point.vector,
                "payload": point.payload,
            }
        return None

    async def delete(self, collection: str, ids: list[str]) -> None:
        """Delete vectors by IDs."""
        await self.client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=ids),
        )

    async def count(self, collection: str) -> int:
        """Get collection point count."""
        info = await self.client.get_collection(collection)
        return info.points_count

    async def ensure_collection(self, collection: str, dimension: int = 4096) -> None:
        """Ensure collection exists, create if not."""
        try:
            await self.client.get_collection(collection)
        except (UnexpectedResponse, Exception):
            await self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

    async def upsert_points(
        self,
        collection: str,
        points: list[dict],
    ) -> None:
        """Batch upsert vectors (alias for upsert_batch)."""
        await self.upsert_batch(collection, points)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using the LLM client."""
        from llm.client import get_llm_client

        llm = get_llm_client()
        embeddings = []

        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = await llm.embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings


async def get_vector_store() -> VectorStore:
    """Get VectorStore instance."""
    client = await get_qdrant_client()
    return VectorStore(client)

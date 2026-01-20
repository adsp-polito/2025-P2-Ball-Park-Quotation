"""
FPT Cost Brain 2.0 - Qdrant Collection Definitions
4 vector collections for different embedding types
"""

from dataclasses import dataclass
from typing import Literal

from app.config import settings


@dataclass
class CollectionConfig:
    """Configuration for a Qdrant collection."""

    name: str
    description: str
    vector_size: int
    distance: Literal["Cosine", "Euclid", "Dot"] = "Cosine"


# Collection names as constants
PR_EMBEDDINGS = "pr_embeddings"
QUOTATION_CHUNKS = "quotation_chunks"
KNOWLEDGE_CHUNKS = "knowledge_chunks"
FEEDBACK_PATTERNS = "feedback_patterns"


# Collection configurations
COLLECTIONS: dict[str, CollectionConfig] = {
    PR_EMBEDDINGS: CollectionConfig(
        name=PR_EMBEDDINGS,
        description="Product Request embeddings for similarity search",
        vector_size=settings.LLM_EMBEDDING_DIMENSIONS,  # 4096 for qwen3-embedding-8b
    ),
    QUOTATION_CHUNKS: CollectionConfig(
        name=QUOTATION_CHUNKS,
        description="Quotation breakdown embeddings for activity matching",
        vector_size=settings.LLM_EMBEDDING_DIMENSIONS,
    ),
    KNOWLEDGE_CHUNKS: CollectionConfig(
        name=KNOWLEDGE_CHUNKS,
        description="Enterprise knowledge document chunks for RAG",
        vector_size=settings.LLM_EMBEDDING_DIMENSIONS,
    ),
    FEEDBACK_PATTERNS: CollectionConfig(
        name=FEEDBACK_PATTERNS,
        description="User correction patterns for learning",
        vector_size=settings.LLM_EMBEDDING_DIMENSIONS,
    ),
}


# Payload schemas for reference (Qdrant doesn't enforce these, but useful for documentation)
PAYLOAD_SCHEMAS = {
    PR_EMBEDDINGS: {
        "pr_id": "uuid",
        "pr_number": "keyword",
        "title": "text",
        "platform": "keyword",
        "program_size": "keyword",
        "total_cost": "float",
        "created_at": "datetime",
    },
    QUOTATION_CHUNKS: {
        "quotation_id": "uuid",
        "pr_id": "uuid",
        "pe_function": "keyword",
        "sub_function": "keyword",
        "activity_description": "text",
        "hours": "integer",
        "cost": "float",
    },
    KNOWLEDGE_CHUNKS: {
        "doc_id": "uuid",
        "doc_type": "keyword",
        "category": "keyword",
        "chunk_index": "integer",
        "chunk_text": "text",
        "title": "text",
    },
    FEEDBACK_PATTERNS: {
        "feedback_id": "uuid",
        "reason_text": "text",
        "reason_category": "keyword",
        "change_percentage": "float",
        "field_path": "keyword",
    },
}

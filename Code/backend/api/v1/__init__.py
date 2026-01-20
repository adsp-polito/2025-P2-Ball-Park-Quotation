"""
FPT Cost Brain 2.0 - API v1 Package
"""

from api.v1.admin import router as admin_router
from api.v1.auth import router as auth_router
from api.v1.chat import router as chat_router
from api.v1.estimation import router as estimation_router
from api.v1.export import router as export_router
from api.v1.history import router as history_router
from api.v1.knowledge import router as knowledge_router
from api.v1.rlhf import router as rlhf_router

__all__ = [
    "auth_router",
    "estimation_router",
    "chat_router",
    "export_router",
    "history_router",
    "knowledge_router",
    "admin_router",
    "rlhf_router",
]

"""
FPT Cost Brain 2.0 - Database Repositories
Repository pattern for data access layer
"""

from db.repositories.audit_repo import AuditRepository
from db.repositories.feedback_repo import FeedbackRepository
from db.repositories.knowledge_repo import KnowledgeRepository
from db.repositories.pr_repo import ProductRequestRepository
from db.repositories.quotation_repo import QuotationRepository
from db.repositories.rules_repo import RulesRepository

__all__ = [
    "ProductRequestRepository",
    "QuotationRepository",
    "FeedbackRepository",
    "RulesRepository",
    "KnowledgeRepository",
    "AuditRepository",
]

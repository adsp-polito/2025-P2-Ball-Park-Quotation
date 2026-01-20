"""
FPT Cost Brain 2.0 - Agentic Estimation Types
Data structures for multi-agent estimation pipeline
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Decision(Enum):
    """Arbitrator decision outcomes."""

    USE_HCQE = "use_hcqe"
    USE_LLM = "use_llm"
    ESCALATE_TO_USER = "escalate"


class EstimationMethod(Enum):
    """Final estimation method used."""

    HCQE_ONLY = "hcqe_only"
    LLM_ONLY = "llm_only"
    HCQE_ACCEPTED = "hcqe_accepted"
    LLM_ACCEPTED = "llm_accepted"
    USER_DECIDED = "user_decided"


# Sizing-specific deviation thresholds
SIZING_THRESHOLDS = {
    "X-Small": 0.20,  # 20%
    "Small": 0.25,  # 25%
    "Medium": 0.30,  # 30%
    "Large": 0.35,  # 35%
    "Full": 0.40,  # 40%
}


@dataclass
class ClusterEstimate:
    """Estimate for a single activity cluster."""

    cluster: str
    activities: list[dict]
    total_hours: float
    total_cost_eur: float
    confidence: float
    reasoning: str
    agent: str


@dataclass
class AgentLog:
    """Log entry for agent execution trace."""

    timestamp: str
    agent: str
    action: str
    input_summary: str
    output_summary: str
    latency_ms: int


@dataclass
class ArbitrationScores:
    """Scores from multi-factor arbitration."""

    hcqe: int
    llm: int
    historical: dict[str, Any]
    confidence: dict[str, Any]
    domain_rules: dict[str, Any]


@dataclass
class ArbitrationDecision:
    """Complete arbitration result."""

    decision: Decision
    scores: ArbitrationScores
    deviation_pct: float
    critique: str | None
    escalation_reason: str | None
    analysis_summary: str


@dataclass
class EscalationData:
    """Data for user escalation."""

    reason: str
    hcqe_total: float
    llm_total: float
    deviation_pct: float
    arbitrator_analysis: str


@dataclass
class EstimationSource:
    """Tracking of estimation source."""

    method: EstimationMethod
    arbitration_scores: ArbitrationScores | None
    retries_used: int


@dataclass
class FinalEstimate:
    """Final estimation result."""

    total_hours: float
    total_cost_eur: float
    breakdown: list[dict]
    confidence: float
    global_justification: str | None = None


@dataclass
class ExecutionTrace:
    """Full execution trace for audit."""

    hcqe_latency_ms: int
    llm_latency_ms: int
    arbitration_latency_ms: int
    total_latency_ms: int
    agent_logs: list[AgentLog] = field(default_factory=list)


@dataclass
class AgenticEstimationResult:
    """Complete result from agentic estimation pipeline."""

    session_id: str
    status: str  # "completed" | "escalated" | "error"

    final_estimate: FinalEstimate
    estimation_source: EstimationSource

    ml_prediction: dict[str, Any]
    llm_breakdown: list[dict] | None

    escalation: EscalationData | None
    trace: ExecutionTrace

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "session_id": self.session_id,
            "status": self.status,
            "final_estimate": {
                "total_hours": self.final_estimate.total_hours,
                "total_cost_eur": self.final_estimate.total_cost_eur,
                "breakdown": self.final_estimate.breakdown,
                "confidence": self.final_estimate.confidence,
                "global_justification": self.final_estimate.global_justification,
            },
            "estimation_source": {
                "method": self.estimation_source.method.value,
                "arbitration_scores": (
                    {
                        "hcqe": self.estimation_source.arbitration_scores.hcqe,
                        "llm": self.estimation_source.arbitration_scores.llm,
                    }
                    if self.estimation_source.arbitration_scores
                    else None
                ),
                "retries_used": self.estimation_source.retries_used,
            },
            "ml_prediction": self.ml_prediction,
            "llm_breakdown": self.llm_breakdown,
            "escalation": (
                {
                    "reason": self.escalation.reason,
                    "hcqe_total": self.escalation.hcqe_total,
                    "llm_total": self.escalation.llm_total,
                    "deviation_pct": self.escalation.deviation_pct,
                    "arbitrator_analysis": self.escalation.arbitrator_analysis,
                }
                if self.escalation
                else None
            ),
            "trace": {
                "hcqe_latency_ms": self.trace.hcqe_latency_ms,
                "llm_latency_ms": self.trace.llm_latency_ms,
                "arbitration_latency_ms": self.trace.arbitration_latency_ms,
                "total_latency_ms": self.trace.total_latency_ms,
                "agent_logs": [
                    {
                        "timestamp": log.timestamp,
                        "agent": log.agent,
                        "action": log.action,
                        "input_summary": log.input_summary,
                        "output_summary": log.output_summary,
                        "latency_ms": log.latency_ms,
                    }
                    for log in self.trace.agent_logs
                ],
            },
        }

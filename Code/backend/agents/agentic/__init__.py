"""
FPT Cost Brain 2.0 - Agentic Estimation Module
Multi-agent system with arbitration and self-correction
"""

from agents.agentic.types import (
    AgenticEstimationResult,
    ArbitrationDecision,
    ArbitrationScores,
    ClusterEstimate,
    Decision,
    EscalationData,
    EstimationMethod,
    EstimationSource,
    ExecutionTrace,
    FinalEstimate,
    SIZING_THRESHOLDS,
)
from agents.agentic.pipeline import run_agentic_estimation
from agents.agentic.arbitrator import ArbitratorAgent
from agents.agentic.cluster_agents import (
    HardwareAgent,
    CalibrationAgent,
    TestingAgent,
    DependentAgent,
    run_cluster_agents,
)

__all__ = [
    # Main entry point
    "run_agentic_estimation",
    # Agents
    "ArbitratorAgent",
    "HardwareAgent",
    "CalibrationAgent",
    "TestingAgent",
    "DependentAgent",
    "run_cluster_agents",
    # Types
    "AgenticEstimationResult",
    "ArbitrationDecision",
    "ArbitrationScores",
    "ClusterEstimate",
    "Decision",
    "EscalationData",
    "EstimationMethod",
    "EstimationSource",
    "ExecutionTrace",
    "FinalEstimate",
    "SIZING_THRESHOLDS",
]

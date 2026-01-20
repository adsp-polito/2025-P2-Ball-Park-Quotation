"""
FPT Cost Brain 2.0 - Agentic Estimation Pipeline
Main orchestration for multi-agent estimation with arbitration and self-correction
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from agents.agentic.arbitrator import ArbitratorAgent
from agents.agentic.cluster_agents import (
    ClusterResult,
    aggregate_cluster_results,
    run_cluster_agents,
)
from agents.agentic.types import (
    AgentLog,
    AgenticEstimationResult,
    ArbitrationDecision,
    Decision,
    EscalationData,
    EstimationMethod,
    EstimationSource,
    ExecutionTrace,
    FinalEstimate,
    SIZING_THRESHOLDS,
)
from llm.client import get_llm_client
from services.cost_calculator import get_cost_calculator
from llm.prompts import SELF_CORRECTION_CRITIQUE

# Import lookup estimator for reference-based estimation
from services.lookup_estimator import LookupEstimator, Sector

# Import rule-based sizing service for accurate sizing classification
from services.sizing_service import SizingService, create_sizing_service

logger = logging.getLogger(__name__)

# Sizing level to numeric score mapping (for HCQE v7 features)
SIZING_LEVEL_TO_SCORE = {
    "X-small": 0,
    "X-Small": 0,
    "Small": 1,
    "Medium": 2,
    "Mid": 2,
    "Large": 3,
    "Full": 4,
}

MAX_SELF_CORRECTION_RETRIES = 2
# PE function rates from price_rate_db.json
# Used to convert K€ prediction to hours using function-specific rates
# The weighted average rate depends on which functions are affected
from ml.pe_function_distributor import (
    PE_FUNCTION_RATES,
    PE_FUNCTION_BASE_WEIGHTS,
    get_affected_pe_functions,
    distribute_hours_to_pe_functions,
    calculate_effective_rate,
)


def _calculate_weighted_average_rate(features: dict | None = None) -> float:
    """
    Calculate weighted average hourly rate based on affected PE functions.

    Uses function-specific rates from price_rate_db.json weighted by
    function involvement. Falls back to 65 €/h (average of common rates).
    """
    if not features:
        # Default weighted average of all function rates
        # (59 * 12 + 89 * 2 + 44 * 1 + 106 * 1 + 107 * 1) / 17 ≈ 65
        return 65.0

    affected = get_affected_pe_functions(features)
    if not affected:
        return 65.0

    # Calculate weighted average rate for affected functions
    total_weight = 0.0
    weighted_rate = 0.0

    for func in affected:
        rate = PE_FUNCTION_RATES.get(func, 59.0)
        weight = PE_FUNCTION_BASE_WEIGHTS.get(func, 0.05)
        weighted_rate += rate * weight
        total_weight += weight

    if total_weight > 0:
        return round(weighted_rate / total_weight, 2)

    return 65.0


async def run_agentic_estimation(
    session_id: str,
    pr_context: dict[str, Any],
    hcqe_predictor,
    ml_features: dict[str, Any],
    historical_accuracy_db: dict[str, dict] | None = None,
) -> AgenticEstimationResult:
    """
    Run the complete agentic estimation pipeline.

    Flow:
    1. HCQE Prediction (ML baseline)
    2. Cluster Agents (parallel LLM breakdown)
    3. Arbitration (multi-factor decision)
    4. Self-Correction Loop (if needed, max 2 retries)
    5. Final Result Assembly

    Args:
        session_id: Unique estimation session ID
        pr_context: Parsed PR data with features
        hcqe_predictor: Trained HCQE model instance
        ml_features: Extracted ML features dictionary
        historical_accuracy_db: Optional historical accuracy data

    Returns:
        AgenticEstimationResult with full trace
    """
    start_time = time.time()
    agent_logs: list[AgentLog] = []
    llm_client = get_llm_client()

    # === STAGE 0: LOOKUP ESTIMATE (Reference from ref_Sizing) ===
    # This provides a reliable baseline from historical cost tables
    lookup_estimate = None
    try:
        lookup_estimator = LookupEstimator()
        pr_text = (
            f"{pr_context.get('pr_title', '')} {pr_context.get('pr_description', '')}"
        )
        sector, sector_conf = lookup_estimator.detect_sector(pr_text)

        # Get sizing hint from ml_features or default to Mid
        # Ensure sizing_hint is string (ml_features might contain int)
        sizing_hint_raw = ml_features.get("sizing_program", "Mid")
        sizing_hint = str(sizing_hint_raw) if sizing_hint_raw is not None else "Mid"
        lookup_result = lookup_estimator.quick_estimate(pr_text, sizing_hint)

        lookup_estimate = {
            "sector": sector.value,
            "sizing": lookup_result.sizing_level.value,
            "cost_keur": lookup_result.point_estimate_keur,
            "range_low": lookup_result.low_estimate_keur,
            "range_high": lookup_result.high_estimate_keur,
            "confidence": lookup_result.confidence,
        }

        agent_logs.append(
            AgentLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent="lookup_estimator",
                action="estimate",
                input_summary=f"Sector: {sector.value}, Sizing: {sizing_hint}",
                output_summary=f"ref_Sizing: {lookup_result.point_estimate_keur:.0f} K EUR [{lookup_result.low_estimate_keur:.0f}-{lookup_result.high_estimate_keur:.0f}]",
                latency_ms=1,  # Instant lookup
            )
        )
        logger.info(
            f"Lookup estimate: {lookup_result.point_estimate_keur:.0f} K EUR ({sector.value}, {lookup_result.sizing_level.value})"
        )

        # Add lookup estimate to pr_context for cluster agents
        pr_context["lookup_estimate"] = lookup_estimate
    except Exception as e:
        logger.warning(f"Lookup estimation failed (continuing without): {e}")

    # === STAGE 0.5: RULE-BASED SIZING CLASSIFICATION ===
    # CRITICAL: This must happen BEFORE HCQE to provide accurate sizing features!
    # Without this, ml_features defaults to "Medium" sizing causing massive overestimation.
    sizing_result = None
    try:
        sizing_service = create_sizing_service()
        pr_text_for_sizing = (
            f"{pr_context.get('pr_title', '')} {pr_context.get('pr_description', '')} "
            f"{pr_context.get('pr_type', '')} {pr_context.get('product_family', '')}"
        )

        # Build parsed_pr dict for SizingService
        parsed_pr_for_sizing = {
            "title": pr_context.get("pr_title", ""),
            "description": pr_context.get("pr_description", ""),
            "pr_type": pr_context.get("pr_type", ""),
            "product_family": pr_context.get("product_family", ""),
            "sector": pr_context.get("sector", ""),
            "is_homologation": pr_context.get("is_homologation", False),
            "is_bom": pr_context.get("is_bom", False),
            "is_new_engine": pr_context.get("is_new_engine", False),
        }

        # Classify sizing using 45 rules from ref_sizing.json
        sizing_result = await sizing_service.classify_sizing(
            pr_text=pr_text_for_sizing,
            parsed_pr=parsed_pr_for_sizing,
            llm=llm_client,
        )

        # Update ml_features with CORRECT sizing scores (BEFORE HCQE!)
        if sizing_result:
            # ProgramSizingResult is a dataclass - access attributes directly
            # Each attribute is a SizingResult with .sizing, .confidence, etc.

            # Map SizingService attributes to HCQE v7 feature names
            sizing_mapping = [
                ("pe_base_powertrain", "sizing_PE_base_score"),
                ("pe_system_assembly", "sizing_PE_system_score"),
                ("pe_installation_application", "sizing_PE_install_score"),
                ("program_manager_overall", "sizing_program_score"),
            ]

            for attr_name, feature_name in sizing_mapping:
                if hasattr(sizing_result, attr_name):
                    result_obj = getattr(sizing_result, attr_name)
                    sizing_level = result_obj.sizing if result_obj else "Medium"
                    ml_features[feature_name] = SIZING_LEVEL_TO_SCORE.get(
                        sizing_level, 2
                    )

            # Get overall sizing from program_manager_overall
            overall_sizing = "Medium"
            if (
                hasattr(sizing_result, "program_manager_overall")
                and sizing_result.program_manager_overall
            ):
                overall_sizing = sizing_result.program_manager_overall.sizing
            ml_features["sizing_program"] = overall_sizing
            ml_features["sizing_program_score"] = SIZING_LEVEL_TO_SCORE.get(
                overall_sizing, 2
            )

            agent_logs.append(
                AgentLog(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent="sizing_service",
                    action="classify",
                    input_summary=f"PR Type: {pr_context.get('pr_type', 'unknown')}",
                    output_summary=f"Sizing: {overall_sizing} (scores: base={ml_features.get('sizing_PE_base_score', 2)}, prog={ml_features.get('sizing_program_score', 2)})",
                    latency_ms=50,
                )
            )
            logger.info(
                f"SizingService: {overall_sizing} "
                f"(base={ml_features.get('sizing_PE_base_score')}, "
                f"sys={ml_features.get('sizing_PE_system_score')}, "
                f"install={ml_features.get('sizing_PE_install_score')}, "
                f"program={ml_features.get('sizing_program_score')})"
            )
    except Exception as e:
        logger.warning(f"SizingService failed, using defaults: {e}")

    # === STAGE 1: HCQE Prediction ===
    hcqe_start = time.time()
    try:
        hcqe_result = _run_hcqe_prediction(hcqe_predictor, ml_features)
        hcqe_latency = int((time.time() - hcqe_start) * 1000)
        agent_logs.append(
            AgentLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent="hcqe",
                action="predict",
                input_summary=f"Features: {len(ml_features)} fields",
                output_summary=f"Total: {hcqe_result['predicted_total_hours']:.0f}h, Sizing: {hcqe_result['sizing']}",
                latency_ms=hcqe_latency,
            )
        )
        logger.info(f"HCQE prediction: {hcqe_result['predicted_total_hours']:.0f}h")
    except Exception as e:
        logger.error(f"HCQE prediction failed: {e}")
        # Fallback to LLM-only mode
        return await _run_llm_only_estimation(
            session_id, pr_context, llm_client, start_time, agent_logs
        )

    # === STAGE 2: Cluster Agents (Parallel LLM) ===
    llm_start = time.time()
    try:
        # CRITICAL: Pass sizing to cluster agents for HARD CEILING enforcement
        # This prevents LLM hallucinations from producing 17000h for X-Small projects
        sizing_for_agents = hcqe_result.get("sizing", "Medium")
        cluster_results = await run_cluster_agents(
            pr_context, hcqe_result, llm_client, sizing=sizing_for_agents
        )
        llm_latency = int((time.time() - llm_start) * 1000)

        breakdown, llm_total, avg_confidence = aggregate_cluster_results(
            cluster_results
        )

        agent_logs.append(
            AgentLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent="cluster_agents",
                action="estimate",
                input_summary=f"HCQE baseline: {hcqe_result['predicted_total_hours']:.0f}h",
                output_summary=f"LLM total: {llm_total:.0f}h, Clusters: {len(cluster_results)}",
                latency_ms=llm_latency,
            )
        )
        logger.info(
            f"Cluster agents: {llm_total:.0f}h across {len(cluster_results)} clusters"
        )
    except Exception as e:
        logger.error(f"Cluster agents failed: {e}")
        # Fallback to HCQE-only mode
        return _create_hcqe_only_result(
            session_id, hcqe_result, start_time, hcqe_latency, agent_logs
        )

    # === STAGE 3: Arbitration ===
    arb_start = time.time()
    arbitrator = ArbitratorAgent(historical_accuracy_db)

    llm_estimate = {
        "breakdown": breakdown,
        "total": llm_total,
        "avg_confidence": avg_confidence,
    }

    arbitration = arbitrator.arbitrate(hcqe_result, llm_estimate, pr_context)
    arb_latency = int((time.time() - arb_start) * 1000)

    agent_logs.append(
        AgentLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent="arbitrator",
            action="decide",
            input_summary=f"HCQE: {hcqe_result['predicted_total_hours']:.0f}h vs LLM: {llm_total:.0f}h",
            output_summary=f"Decision: {arbitration.decision.value}, Deviation: {arbitration.deviation_pct:.1%}",
            latency_ms=arb_latency,
        )
    )

    logger.info(
        f"Arbitration: {arbitration.decision.value}, "
        f"scores HCQE={arbitration.scores.hcqe} LLM={arbitration.scores.llm}"
    )

    # === STAGE 4: Self-Correction Loop (if needed) ===
    retries_used = 0
    final_breakdown = breakdown
    final_total = llm_total
    final_confidence = avg_confidence
    global_justification = None

    # CRITICAL FIX v2.1: Disable "justification escape" for small projects
    # For X-Small and Small projects, HCQE is more reliable than LLM
    # LLM tends to hallucinate large estimates due to training data bias
    sizing_for_correction = hcqe_result.get("sizing", "Medium")
    allow_justification_escape = sizing_for_correction not in [
        "X-Small",
        "X-small",
        "Small",
    ]

    if not allow_justification_escape:
        logger.info(
            f"Self-correction disabled for {sizing_for_correction} project - "
            "HCQE is more reliable for small projects"
        )

    if (
        arbitration.decision == Decision.USE_HCQE
        and arbitration.critique
        and allow_justification_escape
    ):
        # LLM deviated beyond threshold and lost arbitration
        # Give LLM chance to self-correct or justify (ONLY for Medium+ projects)
        for retry in range(MAX_SELF_CORRECTION_RETRIES):
            retries_used += 1
            logger.info(f"Self-correction attempt {retries_used}")

            correction_start = time.time()
            correction_result = await _run_self_correction(
                llm_client,
                arbitration.critique,
                hcqe_result,
                breakdown,
                llm_total,
                arbitration.deviation_pct,
            )
            correction_latency = int((time.time() - correction_start) * 1000)

            agent_logs.append(
                AgentLog(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent="self_correction",
                    action=f"retry_{retries_used}",
                    input_summary=f"Critique delivered, deviation: {arbitration.deviation_pct:.1%}",
                    output_summary=f"Revised: {correction_result.get('revised', False)}, New total: {correction_result.get('total_hours', 0):.0f}h",
                    latency_ms=correction_latency,
                )
            )

            if correction_result.get("revised"):
                # LLM revised its estimate
                final_breakdown = correction_result.get("activities", breakdown)
                final_total = correction_result.get("total_hours", llm_total)
                final_confidence = correction_result.get(
                    "confidence_score", avg_confidence
                )

                # Check if now within threshold
                new_deviation = (
                    abs(final_total - hcqe_result["predicted_total_hours"])
                    / hcqe_result["predicted_total_hours"]
                )
                sizing = hcqe_result.get("sizing", "Medium")
                threshold = SIZING_THRESHOLDS.get(sizing, 0.30)

                if new_deviation <= threshold:
                    logger.info(
                        f"Self-correction successful: deviation now {new_deviation:.1%}"
                    )
                    break
            else:
                # LLM justified its deviation - accept escape hatch
                global_justification = correction_result.get("justification")
                final_breakdown = correction_result.get("activities", breakdown)
                final_total = correction_result.get("total_hours", llm_total)
                final_confidence = correction_result.get(
                    "confidence_score", avg_confidence
                )
                logger.info(f"LLM justified deviation: {global_justification[:100]}...")
                break

    # === STAGE 5: Final Result Assembly ===
    total_latency = int((time.time() - start_time) * 1000)

    # Get CostCalculator for function-specific rates from price_rate_db.json
    cost_calculator = get_cost_calculator()

    # Get product family from PR context for bench rates
    product_family = pr_context.get("product_family", "E5F0")

    # Determine final values based on arbitration decision
    if arbitration.decision == Decision.USE_HCQE and not global_justification:
        # Use HCQE baseline (LLM didn't convince us)
        hcqe_breakdown = _convert_hcqe_to_breakdown(hcqe_result)

        # Calculate cost using function-specific rates
        cost_result = cost_calculator.calculate_cost(hcqe_breakdown, product_family)

        final_estimate = FinalEstimate(
            total_hours=hcqe_result["predicted_total_hours"],
            total_cost_eur=cost_result.total_cost_eur,
            breakdown=hcqe_breakdown,
            confidence=hcqe_result["confidence"],
            global_justification=None,
        )
        method = EstimationMethod.HCQE_ACCEPTED
        logger.info(
            f"HCQE cost (via CostCalculator): {cost_result.total_cost_keur:.0f} K€"
        )
    elif arbitration.decision == Decision.ESCALATE_TO_USER:
        # Escalate - return LLM breakdown with escalation data
        cost_result = cost_calculator.calculate_cost(final_breakdown, product_family)

        final_estimate = FinalEstimate(
            total_hours=final_total,
            total_cost_eur=cost_result.total_cost_eur,
            breakdown=final_breakdown,
            confidence=final_confidence,
            global_justification=global_justification,
        )
        method = EstimationMethod.USER_DECIDED
        logger.info(
            f"Escalated cost (via CostCalculator): {cost_result.total_cost_keur:.0f} K€"
        )
    else:
        # Use LLM breakdown (either won arbitration or justified deviation)
        cost_result = cost_calculator.calculate_cost(final_breakdown, product_family)

        final_estimate = FinalEstimate(
            total_hours=final_total,
            total_cost_eur=cost_result.total_cost_eur,
            breakdown=final_breakdown,
            confidence=final_confidence,
            global_justification=global_justification,
        )
        method = EstimationMethod.LLM_ACCEPTED
        logger.info(
            f"LLM cost (via CostCalculator): {cost_result.total_cost_keur:.0f} K€"
        )

    # Build escalation data if needed
    escalation = None
    if arbitration.decision == Decision.ESCALATE_TO_USER:
        escalation = EscalationData(
            reason=arbitration.escalation_reason or "Arbitration inconclusive",
            hcqe_total=hcqe_result["predicted_total_hours"],
            llm_total=final_total,
            deviation_pct=arbitration.deviation_pct,
            arbitrator_analysis=arbitration.analysis_summary,
        )

    return AgenticEstimationResult(
        session_id=session_id,
        status="escalated" if escalation else "completed",
        final_estimate=final_estimate,
        estimation_source=EstimationSource(
            method=method,
            arbitration_scores=arbitration.scores,
            retries_used=retries_used,
        ),
        ml_prediction={
            "point_estimate": hcqe_result["predicted_cost_keur"],
            "sizing": hcqe_result["sizing"],
            "confidence": hcqe_result["confidence"],
            "interval": {
                "low": hcqe_result["interval_low"],
                "high": hcqe_result["interval_high"],
            },
            "cluster_estimates": hcqe_result.get("cluster_estimates", {}),
            "recommendations": hcqe_result.get("recommendations", []),
        },
        llm_breakdown=final_breakdown,
        escalation=escalation,
        trace=ExecutionTrace(
            hcqe_latency_ms=hcqe_latency,
            llm_latency_ms=llm_latency,
            arbitration_latency_ms=arb_latency,
            total_latency_ms=total_latency,
            agent_logs=agent_logs,
        ),
    )


def _run_hcqe_prediction(predictor, features: dict) -> dict:
    """
    Run HCQE prediction and format result.

    Supports both:
    - HCQEProductionModelV7: predict_single(dict) -> dict (predicts K€)
    - HCQEPredictor (legacy): predict(dict) -> HCQEPrediction dataclass

    IMPORTANT: Both models predict COST (K€), then we convert to hours using
    function-specific rates from price_rate_db.json, not a constant rate.
    """
    # Calculate effective hourly rate based on affected PE functions
    effective_rate = _calculate_weighted_average_rate(features)

    # Detect model interface
    if hasattr(predictor, "predict_single"):
        # HCQEProductionModelV7.2 - returns dict with BOTH hours and K€
        result = predictor.predict_single(features)
        point_estimate = result.get("point_estimate", 500)  # K€
        confidence = result.get("confidence", 0.75)
        lower_bound = result.get("lower_bound", point_estimate * 0.7)
        upper_bound = result.get("upper_bound", point_estimate * 1.4)
        method_used = f"hcqe_{result.get('model_version', 'v7.2')}"

        # v7.2: Use hours directly from model (not converted from K€)
        predicted_hours = result.get(
            "point_estimate_hours", point_estimate * 1000 / effective_rate
        )
        predicted_cost_keur = point_estimate

        # Use implied rate from model if available (sizing-specific)
        implied_rate = result.get("implied_rate", effective_rate)

        # v7 model: Use sizing from features (set by SizingService)
        sizing_score = result.get(
            "sizing_score", features.get("sizing_program_score", 0)
        )
        sizing_map = {0: "X-Small", 1: "Small", 2: "Medium", 3: "Large", 4: "Full"}
        predicted_sizing = sizing_map.get(sizing_score, "X-Small")

        if "sizing_program" in features and isinstance(features["sizing_program"], str):
            predicted_sizing = features["sizing_program"]

        cluster_estimates = {}
        recommendations = [
            f"HCQE v7.2: {predicted_hours:.0f}h, {point_estimate:.0f} K€"
        ]

        logger.info(
            f"HCQE v7.2 prediction: {predicted_hours:.0f}h, {point_estimate:.0f} K€, "
            f"{predicted_sizing}, {confidence:.0%} conf, implied rate: {implied_rate:.0f} €/h"
        )

    else:
        # Legacy HCQEPredictor - returns dataclass with K€
        result = predictor.predict(features)
        point_estimate = result.point_estimate
        confidence = result.calibrated_confidence
        lower_bound = result.prediction_interval[0]
        upper_bound = result.prediction_interval[1]
        method_used = result.method_used
        predicted_sizing = result.predicted_sizing
        cluster_estimates = result.cluster_estimates or {}
        recommendations = result.recommendations

        logger.info(
            f"HCQE legacy prediction: {point_estimate:.0f} K€, {predicted_sizing}, "
            f"effective rate: {effective_rate:.0f} €/h"
        )

        # Legacy model returns K€, convert to hours using effective rate
        predicted_hours = point_estimate * 1000 / effective_rate
        predicted_cost_keur = point_estimate

    # Convert cluster_estimates from K€ to hours using cluster-specific rates
    cluster_hours = {}
    if cluster_estimates:
        # Use cluster-specific rates for conversion
        cluster_rate_mapping = {
            "hardware": 59.0,  # Design rate
            "calibration": 59.0,  # CP&E rate
            "testing": 44.0,  # Testing rate (lower)
            "ats": 59.0,  # ATS rate
            "software": 59.0,  # CS&SW rate
            "documentation": 89.0,  # Tech Doc rate (higher)
            "installation": 59.0,  # Standard
            "dataset": 59.0,  # Standard
        }
        for cluster, cost_keur in cluster_estimates.items():
            rate = cluster_rate_mapping.get(cluster, 59.0)
            cluster_hours[cluster] = cost_keur * 1000 / rate
    else:
        # Generate default cluster hours based on sizing percentages
        default_cluster_pcts = {
            "hardware": 0.25,
            "calibration": 0.20,
            "testing": 0.20,
            "ats": 0.10,
            "software": 0.08,
            "documentation": 0.05,
            "installation": 0.05,
            "dataset": 0.07,
        }
        for cluster, pct in default_cluster_pcts.items():
            cluster_hours[cluster] = predicted_hours * pct

    return {
        "predicted_total_hours": predicted_hours,
        "predicted_cost_keur": predicted_cost_keur,
        "effective_rate": effective_rate,  # Include for transparency
        "confidence": confidence,
        "method": method_used,
        "sizing": predicted_sizing,
        "interval_low": lower_bound,
        "interval_high": upper_bound,
        "cluster_estimates": cluster_hours,  # Hours per cluster
        "recommendations": recommendations,
    }


async def _run_self_correction(
    llm_client,
    critique: str,
    hcqe_result: dict,
    current_breakdown: list,
    current_total: float,
    deviation_pct: float,
) -> dict:
    """Run self-correction with natural language critique."""
    import json

    # Build cluster-specific critique
    cluster_critique = ""
    cluster_estimates = hcqe_result.get("cluster_estimates", {})
    for cluster, hcqe_value in cluster_estimates.items():
        llm_value = sum(
            b.get("hours", 0) for b in current_breakdown if b.get("cluster") == cluster
        )
        if hcqe_value > 0 and llm_value > 0:
            cluster_dev = (llm_value - hcqe_value) / hcqe_value
            if abs(cluster_dev) > 0.25:
                direction = "reduce" if cluster_dev > 0 else "increase"
                cluster_critique += f"• {cluster.title()}: You allocated {llm_value:,.0f}h but HCQE suggests ~{hcqe_value:,.0f}h. Please {direction} by ~{abs(cluster_dev):.0%}.\n"

    prompt = SELF_CORRECTION_CRITIQUE.format(
        llm_total=current_total,
        hcqe_total=hcqe_result["predicted_total_hours"],
        deviation_pct=deviation_pct,
        cluster_critique=cluster_critique
        or "No specific cluster adjustments suggested.",
    )

    try:
        response = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        # chat() returns string directly
        content = response if isinstance(response, str) else str(response)
        return json.loads(content)
    except Exception as e:
        logger.error(f"Self-correction failed: {e}")
        return {
            "revised": False,
            "activities": current_breakdown,
            "total_hours": current_total,
        }


async def _run_llm_only_estimation(
    session_id: str,
    pr_context: dict,
    llm_client,
    start_time: float,
    agent_logs: list[AgentLog],
) -> AgenticEstimationResult:
    """Fallback to LLM-only estimation when HCQE fails."""
    llm_start = time.time()

    # Provide reasonable default baseline for LLM when HCQE unavailable
    # Based on historical FPT program averages (Medium sizing = ~3000h total)
    DEFAULT_TOTAL_HOURS = 3000
    default_cluster_estimates = {
        "hardware": 750,  # 25%
        "calibration": 600,  # 20%
        "testing": 600,  # 20%
        "ats": 300,  # 10%
        "software": 240,  # 8%
        "documentation": 150,  # 5%
        "installation": 150,  # 5%
        "dataset": 210,  # 7%
    }

    fallback_hcqe = {
        "predicted_total_hours": DEFAULT_TOTAL_HOURS,
        "cluster_estimates": default_cluster_estimates,
        "confidence": 0.4,  # Low confidence for fallback
        "sizing": "Medium",  # Default sizing for fallback
    }

    try:
        # Use Medium sizing in fallback mode (no ML sizing available)
        cluster_results = await run_cluster_agents(
            pr_context, fallback_hcqe, llm_client, sizing="Medium"
        )
        breakdown, total_hours, confidence = aggregate_cluster_results(cluster_results)
    except Exception as e:
        logger.error(f"LLM-only estimation also failed: {e}")
        # CRITICAL FIX: Use HCQE breakdown instead of empty list
        # Empty breakdown causes hardcoded defaults in _convert_agentic_to_breakdown
        breakdown = _convert_hcqe_to_breakdown(fallback_hcqe)
        total_hours = DEFAULT_TOTAL_HOURS
        confidence = 0.3
        logger.warning(f"Using HCQE fallback breakdown with {len(breakdown)} items")

    llm_latency = int((time.time() - llm_start) * 1000)
    total_latency = int((time.time() - start_time) * 1000)

    agent_logs.append(
        AgentLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent="llm_only",
            action="estimate",
            input_summary="HCQE failed, fallback mode",
            output_summary=f"Total: {total_hours:.0f}h",
            latency_ms=llm_latency,
        )
    )

    # Use default weighted average rate when HCQE is unavailable
    fallback_rate = 65.0  # Weighted average of PE function rates

    return AgenticEstimationResult(
        session_id=session_id,
        status="completed",
        final_estimate=FinalEstimate(
            total_hours=total_hours,
            total_cost_eur=total_hours * fallback_rate,
            breakdown=breakdown,
            confidence=confidence * 0.7,  # Lower confidence for LLM-only
            global_justification="ML model unavailable, LLM-only estimation",
        ),
        estimation_source=EstimationSource(
            method=EstimationMethod.LLM_ONLY,
            arbitration_scores=None,
            retries_used=0,
        ),
        ml_prediction={},
        llm_breakdown=breakdown,
        escalation=None,
        trace=ExecutionTrace(
            hcqe_latency_ms=0,
            llm_latency_ms=llm_latency,
            arbitration_latency_ms=0,
            total_latency_ms=total_latency,
            agent_logs=agent_logs,
        ),
    )


def _create_hcqe_only_result(
    session_id: str,
    hcqe_result: dict,
    start_time: float,
    hcqe_latency: int,
    agent_logs: list[AgentLog],
) -> AgenticEstimationResult:
    """Create result using HCQE only when LLM fails."""
    total_latency = int((time.time() - start_time) * 1000)

    # Use predicted_cost_keur directly (already calculated with proper rates)
    return AgenticEstimationResult(
        session_id=session_id,
        status="completed",
        final_estimate=FinalEstimate(
            total_hours=hcqe_result["predicted_total_hours"],
            total_cost_eur=hcqe_result["predicted_cost_keur"] * 1000,
            breakdown=_convert_hcqe_to_breakdown(hcqe_result),
            confidence=hcqe_result["confidence"],
            global_justification="LLM agents unavailable, HCQE-only estimation",
        ),
        estimation_source=EstimationSource(
            method=EstimationMethod.HCQE_ONLY,
            arbitration_scores=None,
            retries_used=0,
        ),
        ml_prediction={
            "point_estimate": hcqe_result["predicted_cost_keur"],
            "sizing": hcqe_result["sizing"],
            "confidence": hcqe_result["confidence"],
            "interval": {
                "low": hcqe_result["interval_low"],
                "high": hcqe_result["interval_high"],
            },
            "cluster_estimates": hcqe_result.get("cluster_estimates", {}),
            "recommendations": hcqe_result.get("recommendations", []),
        },
        # CRITICAL FIX: Use HCQE breakdown instead of None
        # estimation_node.py uses result.llm_breakdown for conversion
        llm_breakdown=_convert_hcqe_to_breakdown(hcqe_result),
        escalation=None,
        trace=ExecutionTrace(
            hcqe_latency_ms=hcqe_latency,
            llm_latency_ms=0,
            arbitration_latency_ms=0,
            total_latency_ms=total_latency,
            agent_logs=agent_logs,
        ),
    )


def _convert_hcqe_to_breakdown(hcqe_result: dict) -> list[dict]:
    """Convert HCQE cluster estimates to activity breakdown format."""
    breakdown = []
    cluster_estimates = hcqe_result.get("cluster_estimates", {})

    for cluster, hours in cluster_estimates.items():
        if hours > 0:
            breakdown.append(
                {
                    "cluster": cluster,
                    "activity": f"{cluster.title()} Activities",
                    "hours": hours,
                    "description": f"HCQE baseline estimate for {cluster}",
                    "confidence_score": hcqe_result.get("confidence", 0.7),
                }
            )

    return breakdown

"""
FPT Cost Brain 2.0 - Arbitrator Agent
Multi-factor decision between HCQE and LLM estimates
"""

import logging
from typing import Any

from agents.agentic.types import (
    ArbitrationDecision,
    ArbitrationScores,
    Decision,
    SIZING_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# Known program families with good HCQE training data
KNOWN_FAMILIES = {"MK08", "CURSOR", "NEF", "E0C0", "E9C0", "F1C", "F1A"}


class ArbitratorAgent:
    """
    Arbitrator Agent for deciding between HCQE and LLM estimates.

    Uses multi-factor scoring:
    - Historical accuracy (40 points)
    - Confidence comparison (30 points)
    - Domain rules (30 points)
    """

    def __init__(self, historical_accuracy_db: dict[str, dict] | None = None):
        self.historical_accuracy = historical_accuracy_db or {}

    def arbitrate(
        self,
        hcqe_estimate: dict[str, Any],
        llm_estimate: dict[str, Any],
        context: dict[str, Any],
    ) -> ArbitrationDecision:
        """
        Decide between HCQE and LLM estimates.

        Args:
            hcqe_estimate: HCQE prediction with total, confidence, sizing
            llm_estimate: LLM breakdown with total, avg_confidence
            context: PR context with features, program_family, etc.
        """
        hcqe_total = hcqe_estimate.get("predicted_total_hours", 0)
        llm_total = sum(
            item.get("hours", 0) for item in llm_estimate.get("breakdown", [])
        )

        if hcqe_total == 0:
            return self._create_decision(
                Decision.USE_LLM,
                ArbitrationScores(0, 100, {}, {}, {}),
                0.0,
                None,
                None,
                "HCQE returned zero estimate, using LLM",
            )

        deviation_pct = abs(llm_total - hcqe_total) / hcqe_total

        # Sanity check: If deviation < 5%, use HCQE
        if deviation_pct < 0.05:
            return self._create_decision(
                Decision.USE_HCQE,
                ArbitrationScores(100, 0, {}, {}, {}),
                deviation_pct,
                None,
                None,
                f"Deviation {deviation_pct:.1%} < 5%, using HCQE baseline",
            )

        # Multi-factor scoring
        scores = {"hcqe": 0, "llm": 0}
        score_details = {"historical": {}, "confidence": {}, "domain_rules": {}}

        # Factor 1: Historical Accuracy (40 points)
        historical = self._score_historical_accuracy(context, scores)
        score_details["historical"] = historical

        # Factor 2: Confidence Comparison (30 points)
        confidence = self._score_confidence(hcqe_estimate, llm_estimate, scores)
        score_details["confidence"] = confidence

        # Factor 3: Domain Rules (30 points)
        domain = self._score_domain_rules(context, scores)
        score_details["domain_rules"] = domain

        # Decision logic
        score_diff = abs(scores["hcqe"] - scores["llm"])
        sizing = hcqe_estimate.get("sizing", "Medium")
        threshold = SIZING_THRESHOLDS.get(sizing, 0.30)

        if score_diff < 15 and deviation_pct > 0.20:
            # Ambiguous scores + high deviation → escalate
            return self._create_decision(
                Decision.ESCALATE_TO_USER,
                ArbitrationScores(
                    scores["hcqe"],
                    scores["llm"],
                    score_details["historical"],
                    score_details["confidence"],
                    score_details["domain_rules"],
                ),
                deviation_pct,
                None,
                f"Scores too close ({scores['hcqe']} vs {scores['llm']}) with {deviation_pct:.0%} deviation",
                f"Arbitration inconclusive. HCQE: {hcqe_total:.0f}h, LLM: {llm_total:.0f}h",
            )

        winner = (
            Decision.USE_HCQE if scores["hcqe"] > scores["llm"] else Decision.USE_LLM
        )

        # Generate critique if LLM deviates beyond threshold
        critique = None
        if deviation_pct > threshold and winner == Decision.USE_HCQE:
            critique = self._generate_critique(
                hcqe_estimate, llm_estimate, deviation_pct
            )

        return self._create_decision(
            winner,
            ArbitrationScores(
                scores["hcqe"],
                scores["llm"],
                score_details["historical"],
                score_details["confidence"],
                score_details["domain_rules"],
            ),
            deviation_pct,
            critique,
            None,
            f"Decision: {winner.value}. HCQE={scores['hcqe']}, LLM={scores['llm']}",
        )

    def _score_historical_accuracy(
        self, context: dict, scores: dict[str, int]
    ) -> dict[str, Any]:
        """Score based on historical accuracy for this program family."""
        family = context.get("program_family", "")

        if family in self.historical_accuracy:
            acc = self.historical_accuracy[family]
            if acc.get("hcqe_mape", 100) < acc.get("llm_mape", 100):
                scores["hcqe"] += 40
                return {
                    "hcqe": 40,
                    "llm": 0,
                    "reason": f"HCQE more accurate for {family}",
                }
            else:
                scores["llm"] += 40
                return {
                    "hcqe": 0,
                    "llm": 40,
                    "reason": f"LLM more accurate for {family}",
                }

        # No historical data - split evenly
        scores["hcqe"] += 20
        scores["llm"] += 20
        return {"hcqe": 20, "llm": 20, "reason": "No historical data, split evenly"}

    def _score_confidence(
        self, hcqe: dict, llm: dict, scores: dict[str, int]
    ) -> dict[str, Any]:
        """Score based on confidence comparison."""
        hcqe_conf = hcqe.get("confidence", 0.5)

        breakdown = llm.get("breakdown", [])
        if breakdown:
            llm_conf = sum(b.get("confidence_score", 0.5) for b in breakdown) / len(
                breakdown
            )
        else:
            llm_conf = 0.5

        if hcqe_conf > llm_conf:
            scores["hcqe"] += 30
            return {
                "hcqe": 30,
                "llm": 0,
                "reason": f"HCQE conf {hcqe_conf:.0%} > LLM {llm_conf:.0%}",
            }
        else:
            scores["llm"] += 30
            return {
                "hcqe": 0,
                "llm": 30,
                "reason": f"LLM conf {llm_conf:.0%} > HCQE {hcqe_conf:.0%}",
            }

    def _score_domain_rules(
        self, context: dict, scores: dict[str, int]
    ) -> dict[str, Any]:
        """
        Score based on FPT domain rules.

        CRITICAL v7 UPDATE:
        - Added is_new_engine rule: LLM weighted higher (ML has no history for new platforms)
        - Added is_bom rule: HCQE weighted higher (routine BOM changes are well-modeled)

        These rules address the asymmetric knowledge problem:
        - HCQE excels at routine changes with historical precedent
        - LLM excels at novel situations requiring reasoning
        """
        rules_applied = []
        features = context.get("features", {})

        # === NEW v7 RULES ===

        # Rule 0a: New Engine → trust LLM (ML has no training data for new platforms!)
        is_new_engine = features.get("is_new_engine", 0) or context.get(
            "is_new_engine", 0
        )
        if is_new_engine == 1:
            scores["llm"] += 30
            rules_applied.append("is_new_engine=1 → LLM (no ML history)")
            logger.info("Arbitrator: NEW ENGINE detected - weighting LLM higher")
            return {"hcqe": 0, "llm": 30, "rules_applied": rules_applied}

        # Rule 0b: BOM change → trust HCQE (routine change, well-modeled)
        is_bom = features.get("is_bom", 0) or context.get("is_bom", 0)
        if is_bom == 1:
            scores["hcqe"] += 30
            rules_applied.append("is_bom=1 → HCQE (routine BOM)")
            logger.info("Arbitrator: BOM CHANGE detected - weighting HCQE higher")
            return {"hcqe": 30, "llm": 0, "rules_applied": rules_applied}

        # === EXISTING RULES ===

        # Rule 1: Calibration-heavy → trust HCQE
        calibration_change = features.get("calibration_change", 0) or context.get(
            "calibration_change", 0
        )
        emissions_level = features.get("emissions_level", 0) or context.get(
            "emission_level", 0
        )
        if calibration_change and emissions_level > 1:
            scores["hcqe"] += 30
            rules_applied.append("calibration_heavy → HCQE")
            return {"hcqe": 30, "llm": 0, "rules_applied": rules_applied}

        # Rule 2: Novel hardware combo → trust LLM
        turbo = features.get("hardware_change", 0) or context.get("turbo_related", 0)
        injectors = context.get("injectors_related", 0)
        if turbo and injectors:
            scores["llm"] += 30
            rules_applied.append("novel_hardware → LLM")
            return {"hcqe": 0, "llm": 30, "rules_applied": rules_applied}

        # Rule 3: Known family → trust HCQE
        family = context.get("program_family", "")
        if family in KNOWN_FAMILIES:
            scores["hcqe"] += 30
            rules_applied.append(f"known_family({family}) → HCQE")
            return {"hcqe": 30, "llm": 0, "rules_applied": rules_applied}

        # Rule 4: Unknown family → trust LLM
        if family and family not in KNOWN_FAMILIES:
            scores["llm"] += 30
            rules_applied.append(f"unknown_family({family}) → LLM")
            return {"hcqe": 0, "llm": 30, "rules_applied": rules_applied}

        # No rules matched - split
        scores["hcqe"] += 15
        scores["llm"] += 15
        return {"hcqe": 15, "llm": 15, "rules_applied": ["no_rules_matched"]}

    def _generate_critique(self, hcqe: dict, llm: dict, deviation_pct: float) -> str:
        """Generate natural language critique for self-correction."""
        hcqe_total = hcqe.get("predicted_total_hours", 0)
        breakdown = llm.get("breakdown", [])
        llm_total = sum(b.get("hours", 0) for b in breakdown)

        direction = "exceeds" if llm_total > hcqe_total else "is below"

        critique = f"""Your estimate of {llm_total:,.0f}h {direction} the ML prediction of {hcqe_total:,.0f}h by {deviation_pct:.0%}.

Based on analysis:
"""
        # Identify clusters that deviate most
        cluster_estimates = hcqe.get("cluster_estimates", {})
        for cluster, hcqe_value in cluster_estimates.items():
            llm_value = sum(
                b.get("hours", 0) for b in breakdown if b.get("cluster") == cluster
            )
            if hcqe_value > 0 and llm_value > 0:
                cluster_dev = (llm_value - hcqe_value) / hcqe_value
                if abs(cluster_dev) > 0.25:
                    adj_direction = "reduce" if cluster_dev > 0 else "increase"
                    critique += f"• {cluster.title()}: You allocated {llm_value:,.0f}h but HCQE suggests ~{hcqe_value:,.0f}h. Please {adj_direction} by ~{abs(cluster_dev):.0%}.\n"

        critique += f"""
Please revise to align closer to {hcqe_total:,.0f}h total.

IMPORTANT: If you believe your estimate is correct due to specific novelties in this PR that historical data would miss, DO NOT lower the estimate. Instead, maintain your numbers and provide a "Justification" section explaining the specific technical reasons."""

        return critique

    def _create_decision(
        self,
        decision: Decision,
        scores: ArbitrationScores,
        deviation_pct: float,
        critique: str | None,
        escalation_reason: str | None,
        analysis_summary: str,
    ) -> ArbitrationDecision:
        """Create ArbitrationDecision object."""
        return ArbitrationDecision(
            decision=decision,
            scores=scores,
            deviation_pct=deviation_pct,
            critique=critique,
            escalation_reason=escalation_reason,
            analysis_summary=analysis_summary,
        )

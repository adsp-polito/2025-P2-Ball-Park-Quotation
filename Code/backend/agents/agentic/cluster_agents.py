"""
FPT Cost Brain 2.0 - Cluster Agents
Specialized agents for activity cluster estimation
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from llm.client import get_llm_client
from llm.prompts import CLUSTER_AGENT_PROMPTS

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result from a cluster agent."""

    cluster: str
    activities: list[dict]
    total_hours: float
    confidence_score: float
    reasoning: str


class BaseClusterAgent:
    """
    Base class for cluster-specific estimation agents.

    Uses "Blind then Refine" approach to prevent anchoring bias:
    1. BLIND: LLM estimates WITHOUT seeing HCQE reference
    2. REFINE: If deviation > 30%, show HCQE and ask to reconcile/justify

    CRITICAL FIX v2.1: Added hard ceilings based on sizing to prevent
    LLM hallucinations from overriding ML predictions for small projects.
    """

    cluster_name: str = "base"
    handled_clusters: list[str] = []

    # Threshold for triggering reconciliation
    DEVIATION_THRESHOLD = 0.30  # 30%

    # HARD CEILINGS by sizing level (total hours across ALL clusters)
    # These are absolute maximums that CANNOT be exceeded, even with "justify"
    SIZING_CEILINGS = {
        "X-Small": 2500,  # Max ~400 K€ at 60 €/h
        "X-small": 2500,
        "Small": 5000,  # Max ~800 K€ at 60 €/h
        "Medium": 12000,  # Max ~1.9 M€ at 60 €/h
        "Mid": 12000,
        "Large": 25000,  # Max ~4 M€ at 60 €/h
        "Full": 50000,  # Max ~8 M€ at 60 €/h
    }

    # Cluster share of total (approximate, for ceiling calculation)
    CLUSTER_SHARE = {
        "hardware": 0.25,
        "calibration": 0.20,
        "testing": 0.35,
        "dependent": 0.20,
    }

    def __init__(self, llm_client=None, sizing: str = "Medium"):
        self.llm_client = llm_client or get_llm_client()
        self.sizing = sizing
        self._total_ceiling = self.SIZING_CEILINGS.get(sizing, 12000)
        self._cluster_ceiling = self._total_ceiling * self.CLUSTER_SHARE.get(
            self.cluster_name, 0.25
        )

    async def estimate(
        self,
        pr_context: dict[str, Any],
        hcqe_cluster_estimate: float,
        hcqe_total: float,
        prior_estimates: dict[str, ClusterResult] | None = None,
    ) -> ClusterResult:
        """
        Generate activity breakdown using Blind then Refine approach.

        CRITICAL: Prevents anchoring bias by NOT showing HCQE reference initially.

        Args:
            pr_context: Parsed PR features and metadata
            hcqe_cluster_estimate: HCQE's estimate for this cluster (hours)
            hcqe_total: HCQE's total estimate (for reference)
            prior_estimates: Results from prior agents (for dependent agent)
        """
        # Store for constraint checking in _parse_response
        self._current_hcqe_total = hcqe_total
        self._current_hcqe_cluster = hcqe_cluster_estimate

        try:
            # === STEP 1: BLIND ESTIMATION (no HCQE reference!) ===
            blind_prompt = self._build_blind_prompt(pr_context, prior_estimates)
            blind_response = await self.llm_client.chat(
                messages=[{"role": "user", "content": blind_prompt}],
                response_format={"type": "json_object"},
            )

            # Parse blind estimate (no clamping yet)
            blind_result = self._parse_blind_response(blind_response)
            blind_hours = blind_result.total_hours

            logger.info(
                f"{self.cluster_name} BLIND estimate: {blind_hours:.0f}h "
                f"(HCQE: {hcqe_cluster_estimate:.0f}h)"
            )

            # === STEP 2: CHECK DEVIATION ===
            if hcqe_cluster_estimate > 0:
                deviation = (
                    abs(blind_hours - hcqe_cluster_estimate) / hcqe_cluster_estimate
                )
            else:
                deviation = 0.0

            # === STEP 3: RECONCILE if deviation > threshold ===
            if deviation > self.DEVIATION_THRESHOLD:
                logger.info(
                    f"{self.cluster_name}: Deviation {deviation:.0%} > {self.DEVIATION_THRESHOLD:.0%}, "
                    f"triggering reconciliation"
                )
                result = await self._reconcile_estimate(
                    pr_context,
                    blind_result,
                    hcqe_cluster_estimate,
                    deviation,
                )
            else:
                # Accept blind estimate (within threshold)
                result = self._apply_soft_constraints(
                    blind_result, hcqe_cluster_estimate
                )

            logger.info(
                f"{self.cluster_name} FINAL: {result.total_hours:.0f}h "
                f"(HCQE: {hcqe_cluster_estimate:.0f}h, deviation: {deviation:.0%})"
            )
            return result

        except Exception as e:
            logger.error(f"{self.cluster_name} agent failed: {e}")
            return self._fallback_result(hcqe_cluster_estimate)

    def _build_blind_prompt(
        self,
        pr_context: dict,
        prior_estimates: dict | None,
    ) -> str:
        """
        Build BLIND estimation prompt - NO HCQE reference!

        This prevents anchoring bias by letting LLM form independent estimate.
        """
        base_prompt = CLUSTER_AGENT_PROMPTS.get(
            self.cluster_name, CLUSTER_AGENT_PROMPTS["default"]
        )

        context_str = self._format_context(pr_context)
        prior_str = self._format_prior_estimates(prior_estimates)

        return f"""{base_prompt}

## PR Context
{context_str}

{prior_str}

## Your Task
Based ONLY on the PR technical description above, estimate the hours needed for {self.cluster_name} activities.

IMPORTANT:
- Use your domain knowledge and experience from similar projects
- Do NOT assume any reference numbers - form your own independent estimate
- Consider the complexity factors visible in the PR description
- Be specific about which activities are needed and why

## Output Format
Return JSON with:
{{
    "activities": [
        {{
            "name": "Activity name",
            "hours": <number>,
            "description": "Brief description",
            "confidence": <0.0-1.0>
        }}
    ],
    "total_hours": <sum of activity hours>,
    "confidence_score": <overall 0.0-1.0>,
    "reasoning": "Brief explanation of estimate rationale"
}}
"""

    async def _reconcile_estimate(
        self,
        pr_context: dict,
        blind_result: ClusterResult,
        hcqe_estimate: float,
        deviation: float,
    ) -> ClusterResult:
        """
        Reconciliation step: show HCQE reference and ask LLM to reconcile or justify.

        This is the "Refine" part of "Blind then Refine".
        """
        direction = "higher" if blind_result.total_hours > hcqe_estimate else "lower"

        reconcile_prompt = f"""## Reconciliation Required

You estimated {blind_result.total_hours:,.0f} hours for {self.cluster_name} activities.
The statistical model (HCQE) predicts {hcqe_estimate:,.0f} hours for this cluster.

Your estimate is {deviation:.0%} {direction} than the HCQE baseline.

## Your Initial Reasoning
{blind_result.reasoning}

## Options
Choose ONE:

1. **ADJUST** your estimate closer to HCQE if you believe the statistical model captures typical patterns better:
   - Revise your activities and hours
   - Explain why the adjustment is appropriate

2. **JUSTIFY** your original estimate if you believe this PR has unique factors the statistical model would miss:
   - Keep your original numbers
   - Provide specific technical justification for why this PR is different

## Output Format
Return JSON with:
{{
    "decision": "adjust" or "justify",
    "activities": [
        {{
            "name": "Activity name",
            "hours": <number>,
            "description": "Brief description",
            "confidence": <0.0-1.0>
        }}
    ],
    "total_hours": <final sum>,
    "confidence_score": <0.0-1.0>,
    "reasoning": "Explanation of your decision"
}}
"""

        try:
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": reconcile_prompt}],
                response_format={"type": "json_object"},
            )

            result = self._parse_reconciliation_response(
                response, blind_result, hcqe_estimate
            )
            return result

        except Exception as e:
            logger.warning(f"{self.cluster_name} reconciliation failed: {e}")
            # Fallback to soft-constrained blind result
            return self._apply_soft_constraints(blind_result, hcqe_estimate)

    def _parse_blind_response(self, response: str | dict) -> ClusterResult:
        """Parse blind estimation response (no hard constraints)."""
        import json

        try:
            if isinstance(response, str):
                content = response.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
            elif isinstance(response, dict):
                content = response.get("content", "{}")
                data = json.loads(content) if isinstance(content, str) else content
            else:
                return self._fallback_result(0)

            activities = data.get("activities", [])
            total_hours = data.get(
                "total_hours", sum(a.get("hours", 0) for a in activities)
            )

            return ClusterResult(
                cluster=self.cluster_name,
                activities=activities,
                total_hours=float(total_hours),
                confidence_score=float(data.get("confidence_score", 0.7)),
                reasoning=data.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse {self.cluster_name} blind response: {e}")
            return self._fallback_result(0)

    def _parse_reconciliation_response(
        self,
        response: str | dict,
        blind_result: ClusterResult,
        hcqe_estimate: float,
    ) -> ClusterResult:
        """Parse reconciliation response."""
        import json

        try:
            if isinstance(response, str):
                content = response.strip()
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
            elif isinstance(response, dict):
                content = response.get("content", "{}")
                data = json.loads(content) if isinstance(content, str) else content
            else:
                return self._apply_soft_constraints(blind_result, hcqe_estimate)

            decision = data.get("decision", "adjust")
            activities = data.get("activities", blind_result.activities)
            total_hours = data.get(
                "total_hours", sum(a.get("hours", 0) for a in activities)
            )

            result = ClusterResult(
                cluster=self.cluster_name,
                activities=activities,
                total_hours=float(total_hours),
                confidence_score=float(data.get("confidence_score", 0.7)),
                reasoning=f"[{decision.upper()}] {data.get('reasoning', '')}",
            )

            # CRITICAL FIX v2.1: Apply hard ceiling EVEN for "justify"
            # LLM cannot exceed sizing-based limits regardless of justification
            result = self._apply_hard_ceiling(result)

            if decision == "justify":
                logger.info(
                    f"{self.cluster_name}: LLM justified deviation - accepting "
                    f"(capped to {self._cluster_ceiling:.0f}h ceiling for {self.sizing})"
                )
                return result
            else:
                return self._apply_soft_constraints(result, hcqe_estimate)

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse {self.cluster_name} reconciliation: {e}")
            return self._apply_soft_constraints(blind_result, hcqe_estimate)

    def _apply_soft_constraints(
        self,
        result: ClusterResult,
        hcqe_estimate: float,
    ) -> ClusterResult:
        """Apply soft constraints (±50% of HCQE) to prevent wild estimates."""
        if hcqe_estimate <= 0:
            return result

        MAX_DEVIATION = 0.50  # Allow up to 50% deviation after reconciliation
        min_allowed = hcqe_estimate * (1 - MAX_DEVIATION)
        max_allowed = hcqe_estimate * (1 + MAX_DEVIATION)

        total_hours = result.total_hours
        activities = result.activities

        if total_hours > max_allowed:
            scale_factor = max_allowed / total_hours
            total_hours = max_allowed
            for act in activities:
                act["hours"] = act.get("hours", 0) * scale_factor
            logger.info(
                f"{self.cluster_name}: Clamped from {result.total_hours:.0f}h to {max_allowed:.0f}h"
            )
        elif total_hours < min_allowed:
            scale_factor = min_allowed / total_hours if total_hours > 0 else 1
            total_hours = min_allowed
            for act in activities:
                act["hours"] = act.get("hours", 0) * scale_factor
            logger.info(
                f"{self.cluster_name}: Clamped from {result.total_hours:.0f}h to {min_allowed:.0f}h"
            )

        return ClusterResult(
            cluster=self.cluster_name,
            activities=activities,
            total_hours=total_hours,
            confidence_score=result.confidence_score,
            reasoning=result.reasoning,
        )

    def _apply_hard_ceiling(self, result: ClusterResult) -> ClusterResult:
        """
        Apply HARD ceiling based on sizing level.

        CRITICAL: This is an absolute cap that CANNOT be bypassed by LLM justification.
        For X-Small projects, cluster hours are capped at ~625h (25% of 2500h total).
        This prevents LLM hallucinations from producing 17000h estimates for small PRs.
        """
        if result.total_hours <= self._cluster_ceiling:
            return result

        scale_factor = self._cluster_ceiling / result.total_hours
        activities = result.activities.copy()
        for act in activities:
            act["hours"] = act.get("hours", 0) * scale_factor

        logger.warning(
            f"{self.cluster_name}: HARD CEILING applied! "
            f"Clamped from {result.total_hours:.0f}h to {self._cluster_ceiling:.0f}h "
            f"(sizing: {self.sizing})"
        )

        return ClusterResult(
            cluster=self.cluster_name,
            activities=activities,
            total_hours=self._cluster_ceiling,
            confidence_score=result.confidence_score
            * 0.8,  # Reduce confidence when clamped
            reasoning=f"[HARD CEILING: {self.sizing}] {result.reasoning}",
        )

    def _build_prompt(
        self,
        pr_context: dict,
        hcqe_estimate: float,
        hcqe_total: float,
        prior_estimates: dict | None,
    ) -> str:
        """Build the estimation prompt for this cluster."""
        base_prompt = CLUSTER_AGENT_PROMPTS.get(
            self.cluster_name, CLUSTER_AGENT_PROMPTS["default"]
        )

        context_str = self._format_context(pr_context)
        prior_str = self._format_prior_estimates(prior_estimates)

        return f"""{base_prompt}

## PR Context
{context_str}

## HCQE Reference
- Cluster estimate: {hcqe_estimate:,.0f} hours
- Total project estimate: {hcqe_total:,.0f} hours
- Cluster weight: {(hcqe_estimate / hcqe_total * 100) if hcqe_total > 0 else 0:.1f}%

{prior_str}

## Output Format
Return JSON with:
{{
    "activities": [
        {{
            "name": "Activity name",
            "hours": <number>,
            "description": "Brief description",
            "confidence": <0.0-1.0>
        }}
    ],
    "total_hours": <sum of activity hours>,
    "confidence_score": <overall 0.0-1.0>,
    "reasoning": "Brief explanation of estimate rationale"
}}
"""

    def _format_context(self, pr_context: dict) -> str:
        """Format RICH PR context for CoT reasoning prompt."""
        lines = []

        # Basic PR info
        if pr_context.get("program_family"):
            lines.append(f"- Program Family: {pr_context['program_family']}")
        if pr_context.get("pr_title"):
            lines.append(f"- PR Title: {pr_context['pr_title']}")
        if pr_context.get("pr_description"):
            desc = pr_context["pr_description"][:500]
            lines.append(f"- Description: {desc}")

        # Cluster-specific features
        for feature, value in pr_context.get("features", {}).items():
            if self._is_relevant_feature(feature) and value:
                lines.append(f"- {feature}: {value}")

        # Q&A Answers (user clarifications - critical for CoT)
        qa_answers = pr_context.get("qa_answers", {})
        if qa_answers:
            lines.append("\n### User Clarifications (Q&A Answers)")
            for q_id, answer in list(qa_answers.items())[:5]:  # Top 5 answers
                if answer:
                    lines.append(f"- {q_id}: {answer[:200]}")

        # PR Summary insights
        pr_summary = pr_context.get("pr_summary", {})
        if pr_summary:
            lines.append("\n### PR Summary")
            if pr_summary.get("complexity"):
                lines.append(f"- Complexity: {pr_summary['complexity']}")
            if pr_summary.get("key_components"):
                lines.append(f"- Key Components: {pr_summary['key_components']}")
            if pr_summary.get("risk_factors"):
                lines.append(f"- Risk Factors: {pr_summary['risk_factors']}")

        # Similar PRs reference (case-based reasoning)
        similar_prs = pr_context.get("similar_prs", [])
        if similar_prs:
            lines.append("\n### Similar Historical PRs (Reference)")
            for sp in similar_prs[:3]:
                lines.append(
                    f"- {sp.get('pr_code', 'N/A')}: {sp.get('total_hours', 0):.0f}h "
                    f"(similarity: {sp.get('similarity_score', 0):.0%})"
                )

        return "\n".join(lines) if lines else "No specific context provided"

    def _is_relevant_feature(self, feature: str) -> bool:
        """Check if feature is relevant to this cluster."""
        feature_lower = feature.lower()
        for cluster in self.handled_clusters:
            if cluster.lower() in feature_lower:
                return True
        return False

    def _format_prior_estimates(self, prior_estimates: dict | None) -> str:
        """Format prior estimates for dependent agent context."""
        if not prior_estimates:
            return ""

        lines = ["## Prior Estimates (for context)"]
        for cluster_name, result in prior_estimates.items():
            lines.append(f"- {cluster_name}: {result.total_hours:,.0f}h")

        return "\n".join(lines)

    def _parse_response(
        self, response: str | dict, hcqe_estimate: float
    ) -> ClusterResult:
        """Parse LLM response into ClusterResult.

        Note: llm_client.chat() returns a string directly, not a dict.
        This method handles both cases for safety.
        """
        import json

        try:
            # Handle string response (the actual return type of llm_client.chat())
            if isinstance(response, str):
                content = response.strip()
                # Handle markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                data = json.loads(content)
            elif isinstance(response, dict):
                # Legacy handling for dict response
                content = response.get("content", "{}")
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                return self._fallback_result(hcqe_estimate)

            activities = data.get("activities", [])
            total_hours = data.get(
                "total_hours", sum(a.get("hours", 0) for a in activities)
            )
            total_hours = float(total_hours)

            # CRITICAL: Constrain LLM estimate to prevent wild deviations
            # Use cluster estimate if available, otherwise use % of total
            MAX_CLUSTER_SHARE = 0.25  # Max 25% of total for any single cluster
            MAX_DEVIATION = 0.30  # ±30% from HCQE baseline (tighter constraint)

            # Get hcqe_total from context (passed to _parse_response)
            hcqe_total = getattr(self, "_current_hcqe_total", 0)

            # Determine the constraint
            if hcqe_estimate > 0:
                # Use cluster-specific estimate
                baseline = hcqe_estimate
            elif hcqe_total > 0:
                # Use proportion of total (max 35% per cluster)
                baseline = hcqe_total * MAX_CLUSTER_SHARE
                logger.info(
                    f"{self.cluster_name}: No cluster estimate, using {MAX_CLUSTER_SHARE:.0%} "
                    f"of total = {baseline:.0f}h"
                )
            else:
                baseline = 0

            if baseline > 0:
                min_allowed = baseline * (1 - MAX_DEVIATION)
                max_allowed = baseline * (1 + MAX_DEVIATION)

                if total_hours > max_allowed:
                    logger.warning(
                        f"{self.cluster_name}: LLM={total_hours:.0f}h > max={max_allowed:.0f}h, "
                        f"clamping to baseline+50%"
                    )
                    scale_factor = max_allowed / total_hours
                    total_hours = max_allowed
                    # Scale activities proportionally
                    for act in activities:
                        act["hours"] = act.get("hours", 0) * scale_factor
                elif total_hours < min_allowed:
                    logger.warning(
                        f"{self.cluster_name}: LLM={total_hours:.0f}h < min={min_allowed:.0f}h, "
                        f"clamping to baseline-50%"
                    )
                    scale_factor = min_allowed / total_hours if total_hours > 0 else 1
                    total_hours = min_allowed
                    for act in activities:
                        act["hours"] = act.get("hours", 0) * scale_factor

            return ClusterResult(
                cluster=self.cluster_name,
                activities=activities,
                total_hours=total_hours,
                confidence_score=float(data.get("confidence_score", 0.7)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse {self.cluster_name} response: {e}")
            return self._fallback_result(hcqe_estimate)

    def _fallback_result(self, hcqe_estimate: float) -> ClusterResult:
        """Return fallback result using HCQE estimate."""
        return ClusterResult(
            cluster=self.cluster_name,
            activities=[
                {
                    "name": f"{self.cluster_name.title()} Activities",
                    "hours": hcqe_estimate,
                    "description": "Fallback to HCQE estimate",
                    "confidence": 0.5,
                }
            ],
            total_hours=hcqe_estimate,
            confidence_score=0.5,
            reasoning="LLM estimation failed, using HCQE baseline",
        )


class HardwareAgent(BaseClusterAgent):
    """Agent for hardware-related activity estimation."""

    cluster_name = "hardware"
    handled_clusters = ["hardware", "turbo", "injector", "cooling", "egr"]


class CalibrationAgent(BaseClusterAgent):
    """Agent for calibration and ATS activity estimation."""

    cluster_name = "calibration"
    handled_clusters = ["calibration", "ats", "emission", "regen"]


class TestingAgent(BaseClusterAgent):
    """Agent for testing and dataset activity estimation."""

    cluster_name = "testing"
    handled_clusters = ["testing", "bench", "vehicle", "field", "dataset"]


class DependentAgent(BaseClusterAgent):
    """
    Agent for dependent activities (software, docs, installation).

    Runs AFTER major agents to use their output as context.
    """

    cluster_name = "dependent"
    handled_clusters = ["software", "documentation", "installation"]

    def _build_prompt(
        self,
        pr_context: dict,
        hcqe_estimate: float,
        hcqe_total: float,
        prior_estimates: dict | None,
    ) -> str:
        """Build prompt with emphasis on prior estimates context."""
        base_prompt = CLUSTER_AGENT_PROMPTS.get(
            "dependent", CLUSTER_AGENT_PROMPTS["default"]
        )

        context_str = self._format_context(pr_context)
        prior_str = self._format_prior_estimates(prior_estimates)

        return f"""{base_prompt}

## PR Context
{context_str}

## HCQE Reference
- Combined estimate for software/docs/installation: {hcqe_estimate:,.0f} hours
- Total project estimate: {hcqe_total:,.0f} hours

{prior_str}

IMPORTANT: Use the prior estimates above to inform your estimation.
Software hours typically correlate with hardware complexity.
Documentation hours scale with overall project scope.
Installation hours depend on hardware and calibration scope.

## Output Format
Return JSON with:
{{
    "activities": [
        {{
            "name": "Activity name",
            "hours": <number>,
            "description": "Brief description",
            "confidence": <0.0-1.0>,
            "sub_cluster": "software|documentation|installation"
        }}
    ],
    "total_hours": <sum of activity hours>,
    "confidence_score": <overall 0.0-1.0>,
    "reasoning": "Brief explanation of estimate rationale"
}}
"""


async def run_cluster_agents(
    pr_context: dict[str, Any],
    hcqe_prediction: dict[str, Any],
    llm_client=None,
    sizing: str = "Medium",
) -> dict[str, ClusterResult]:
    """
    Run all cluster agents with hybrid parallel execution.

    Major agents (Hardware, Calibration, Testing) run in parallel.
    Dependent agent runs after, using major results as context.

    Args:
        pr_context: Parsed PR features and metadata
        hcqe_prediction: Full HCQE prediction with cluster_estimates
        llm_client: LLM client instance
        sizing: Project sizing level (X-Small, Small, Medium, Large, Full)
                Used to apply HARD CEILINGS to prevent LLM hallucinations

    Returns:
        Dict mapping cluster name to ClusterResult
    """
    cluster_estimates = hcqe_prediction.get("cluster_estimates", {})
    hcqe_total = hcqe_prediction.get("predicted_total_hours", 0)

    # Get sizing from hcqe_prediction if not provided explicitly
    if sizing == "Medium":
        sizing = hcqe_prediction.get("sizing", "Medium")

    logger.info(f"Cluster agents initialized with sizing={sizing} (ceiling enabled)")

    # Initialize agents WITH SIZING for hard ceiling constraints
    hardware_agent = HardwareAgent(llm_client, sizing=sizing)
    calibration_agent = CalibrationAgent(llm_client, sizing=sizing)
    testing_agent = TestingAgent(llm_client, sizing=sizing)
    dependent_agent = DependentAgent(llm_client, sizing=sizing)

    # Phase 1: Run major agents in parallel
    logger.info("Running major cluster agents in parallel...")

    major_tasks = [
        hardware_agent.estimate(
            pr_context,
            cluster_estimates.get("hardware", 0),
            hcqe_total,
        ),
        calibration_agent.estimate(
            pr_context,
            cluster_estimates.get("calibration", 0) + cluster_estimates.get("ats", 0),
            hcqe_total,
        ),
        testing_agent.estimate(
            pr_context,
            cluster_estimates.get("testing", 0) + cluster_estimates.get("dataset", 0),
            hcqe_total,
        ),
    ]

    major_results = await asyncio.gather(*major_tasks, return_exceptions=True)

    # Process results, handle any exceptions
    results: dict[str, ClusterResult] = {}
    for i, (agent, result) in enumerate(
        zip([hardware_agent, calibration_agent, testing_agent], major_results)
    ):
        if isinstance(result, Exception):
            logger.error(f"{agent.cluster_name} agent raised exception: {result}")
            results[agent.cluster_name] = agent._fallback_result(
                cluster_estimates.get(agent.cluster_name, 0)
            )
        else:
            results[agent.cluster_name] = result

    # Phase 2: Run dependent agent with major results as context
    logger.info("Running dependent agent with major results context...")

    dependent_estimate = (
        cluster_estimates.get("software", 0)
        + cluster_estimates.get("documentation", 0)
        + cluster_estimates.get("installation", 0)
    )

    dependent_result = await dependent_agent.estimate(
        pr_context,
        dependent_estimate,
        hcqe_total,
        prior_estimates=results,
    )

    results["dependent"] = dependent_result

    # Log summary
    total_llm = sum(r.total_hours for r in results.values())
    logger.info(
        f"Cluster agents complete. Total LLM: {total_llm:,.0f}h, HCQE: {hcqe_total:,.0f}h"
    )

    return results


def aggregate_cluster_results(
    results: dict[str, ClusterResult],
) -> tuple[list[dict], float, float]:
    """
    Aggregate cluster results into unified breakdown.

    Returns:
        Tuple of (breakdown list, total hours, average confidence)
    """
    breakdown = []
    total_hours = 0.0
    weighted_confidence = 0.0

    for cluster_name, result in results.items():
        for activity in result.activities:
            breakdown.append(
                {
                    "cluster": cluster_name,
                    "activity": activity.get("name", "Unknown"),
                    "hours": activity.get("hours", 0),
                    "description": activity.get("description", ""),
                    "confidence_score": activity.get(
                        "confidence", result.confidence_score
                    ),
                }
            )
            hours = activity.get("hours", 0)
            total_hours += hours
            weighted_confidence += hours * activity.get(
                "confidence", result.confidence_score
            )

    avg_confidence = weighted_confidence / total_hours if total_hours > 0 else 0.5

    return breakdown, total_hours, avg_confidence

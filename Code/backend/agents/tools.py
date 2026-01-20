"""
FPT Cost Brain 2.0 - Estimation Tools
Deterministic capability layer for the Legendary FPT Agent.

These are the "Hands" - hard logic tools that handle:
- Math calculations
- Table formatting
- Comparisons and diffs
- Statistical analysis

All methods are STATELESS - they accept data and return formatted strings.
No dependencies on agent state or async operations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EstimationTools:
    """
    Deterministic capability layer for the FPT Agent.
    Handles 'hard' tasks: math, comparisons, formatting, and diffs.

    Usage:
        tools = EstimationTools()
        result = tools.explain_estimate(breakdown, ml_prediction, applied_rules)
    """

    # ===== Estimation Analysis Tools =====

    @staticmethod
    def explain_estimate(
        breakdown: list[dict],
        ml_prediction: dict | None = None,
        applied_rules: list[dict] | None = None,
        historical_activities: list[dict] | None = None,
    ) -> str:
        """
        Explain the estimation breakdown with calculations.

        Args:
            breakdown: List of activity dicts with hours/cost
            ml_prediction: ML model prediction info
            applied_rules: Rules that were applied
            historical_activities: Similar activities from RAG (optional)

        Returns:
            Formatted explanation string
        """
        if not breakdown:
            return "No estimation available yet. Complete the estimation step first."

        # Calculate totals
        total_hours = sum(item.get("hours", 0) for item in breakdown)
        total_cost = sum(item.get("cost", 0) for item in breakdown)

        output_lines = [
            "## Estimation Breakdown\n",
            f"- **Total Hours**: {total_hours:,}",
            f"- **Total Cost**: €{total_cost:,.0f}",
            f"- **Activities**: {len(breakdown)}",
        ]

        # Add ML prediction info
        if ml_prediction:
            confidence = ml_prediction.get("confidence", 0)
            output_lines.append(f"- **ML Confidence**: {confidence:.0%}")

            if ml_prediction.get("model_version"):
                output_lines.append(
                    f"- **Model**: {ml_prediction.get('model_version')}"
                )

        # Add applied rules
        if applied_rules:
            output_lines.append("\n### Applied Rules")
            for rule in applied_rules[:5]:
                rule_name = rule.get("rule_name", "Unknown rule")
                effect = rule.get("effect_value", 0)
                confidence = rule.get("confidence", 0)
                output_lines.append(
                    f"- {rule_name}: {effect:+.0%} (conf: {confidence:.0%})"
                )

        # Add historical comparison if available
        if historical_activities:
            output_lines.append("\n### Similar Historical Activities")
            for act in historical_activities[:5]:
                func = act.get("pe_function", "Unknown")
                hours = act.get("hours", 0)
                output_lines.append(f"- {func}: {hours}h")

        # Add top activities by hours
        sorted_activities = sorted(
            breakdown, key=lambda x: x.get("hours", 0), reverse=True
        )
        if sorted_activities:
            output_lines.append("\n### Top Activities by Hours")
            for act in sorted_activities[:5]:
                name = act.get("activity_name", act.get("activity_code", "Unknown"))[
                    :40
                ]
                hours = act.get("hours", 0)
                pct = (hours / total_hours * 100) if total_hours > 0 else 0
                output_lines.append(f"- {name}: {hours}h ({pct:.1f}%)")

        return "\n".join(output_lines)

    @staticmethod
    def compare_breakdown(
        current_breakdown: list[dict],
        historical_data: list[dict],
    ) -> str:
        """
        Compare current breakdown with historical data found via RAG.

        Args:
            current_breakdown: Current estimation breakdown
            historical_data: Historical activities/PRs from vector search

        Returns:
            Formatted comparison table
        """
        if not current_breakdown:
            return "No current breakdown available for comparison."

        if not historical_data:
            return "No historical data available for comparison."

        output_lines = [
            "## Historical Comparison\n",
            "| Activity | Current | Historical | Diff |",
            "|----------|--------:|----------:|-----:|",
        ]

        # Build lookup of current activities by code/name
        current_lookup = {}
        for item in current_breakdown:
            key = item.get("activity_code", item.get("pe_function", ""))
            if key:
                current_lookup[key.lower()] = item.get("hours", 0)

        total_current = 0
        total_historical = 0
        matches_found = 0

        for hist in historical_data[:10]:
            func = hist.get("pe_function", hist.get("activity_code", "Unknown"))
            hist_hours = hist.get("hours", 0)

            # Find matching current activity
            current_hours = None
            for key, hours in current_lookup.items():
                if func.lower() in key or key in func.lower():
                    current_hours = hours
                    matches_found += 1
                    break

            if current_hours is not None:
                diff = current_hours - hist_hours
                diff_str = f"{diff:+.0f}h" if diff != 0 else "="
                output_lines.append(
                    f"| {func[:25]} | {current_hours}h | {hist_hours}h | {diff_str} |"
                )
                total_current += current_hours
                total_historical += hist_hours
            else:
                output_lines.append(f"| {func[:25]} | - | {hist_hours}h | N/A |")
                total_historical += hist_hours

        # Add totals
        if matches_found > 0:
            total_diff = total_current - total_historical
            output_lines.append("|----------|--------:|----------:|-----:|")
            output_lines.append(
                f"| **Total (matched)** | **{total_current}h** | **{total_historical}h** | **{total_diff:+.0f}h** |"
            )

            # Add summary
            if total_historical > 0:
                pct_diff = (total_diff / total_historical) * 100
                output_lines.append(
                    f"\n*Current estimate is {pct_diff:+.1f}% vs historical average*"
                )

        return "\n".join(output_lines)

    @staticmethod
    def format_rules(
        applied_rules: list[dict],
        potential_rules: list[dict] | None = None,
    ) -> str:
        """
        Format applied and potential rules into readable output.

        Args:
            applied_rules: Rules that were applied to this estimation
            potential_rules: Potential rules from RAG search (optional)

        Returns:
            Formatted rules string
        """
        output_lines = []

        if applied_rules:
            output_lines.append("## Applied Rules\n")
            for rule in applied_rules:
                rule_name = rule.get("rule_name", "Unknown")
                description = rule.get("rule_description", "")
                effect = rule.get("effect_value", 0)
                confidence = rule.get("confidence", 0)

                output_lines.append(f"### {rule_name}")
                if description:
                    output_lines.append(f"*{description}*")
                output_lines.append(f"- Effect: {effect:+.0%}")
                output_lines.append(f"- Confidence: {confidence:.0%}")
                output_lines.append("")
        else:
            output_lines.append("## Applied Rules\n")
            output_lines.append("*No specific rules were applied to this estimation.*")

        if potential_rules:
            output_lines.append("\n## Potentially Applicable Rules (from history)\n")
            for rule in potential_rules[:5]:
                category = rule.get("reason_category", "General")
                text = rule.get("reason_text", "")[:100]
                change = rule.get("change_percentage", 0)
                output_lines.append(f"- **{category}**: {text} ({change:+.0f}%)")

        return (
            "\n".join(output_lines)
            if output_lines
            else "No rules information available."
        )

    # ===== Learning & Feedback Tools =====

    @staticmethod
    def preview_learning(user_edits: list[dict]) -> str:
        """
        Analyze user corrections to show system learning impact.

        Args:
            user_edits: List of user corrections with original/corrected values

        Returns:
            Learning preview string
        """
        if not user_edits:
            return "No corrections made yet. Make changes in the review step to see what the system will learn."

        output_lines = [
            f"## Learning Preview ({len(user_edits)} corrections)\n",
            "The system will analyze these corrections to improve future estimates:\n",
        ]

        # Analyze edit patterns
        total_increase = 0
        total_decrease = 0
        corrections_by_activity = {}

        for edit in user_edits:
            original = edit.get("original_value", 0)
            corrected = edit.get("corrected_value", 0)
            activity = edit.get("activity_name", edit.get("activity_code", "Unknown"))
            diff = corrected - original

            corrections_by_activity[activity] = diff

            if diff > 0:
                total_increase += diff
            else:
                total_decrease += abs(diff)

        # Summary of changes
        output_lines.append("### Correction Summary")
        if total_increase > 0:
            output_lines.append(f"- Hours **increased**: +{total_increase:.0f}h")
        if total_decrease > 0:
            output_lines.append(f"- Hours **decreased**: -{total_decrease:.0f}h")

        net_change = total_increase - total_decrease
        output_lines.append(f"- **Net change**: {net_change:+.0f}h")

        # Show individual corrections
        if corrections_by_activity:
            output_lines.append("\n### By Activity")
            for activity, diff in sorted(
                corrections_by_activity.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]:
                output_lines.append(f"- {activity[:30]}: {diff:+.0f}h")

        # Learning outcomes
        output_lines.append("\n### What Happens Next")
        output_lines.append(
            "1. **Immediate**: Rules extracted for similar future projects"
        )
        output_lines.append(
            "2. **Batch**: Model retrained when enough corrections accumulate (≥5)"
        )
        output_lines.append("3. **Validation**: New model tested before deployment")

        return "\n".join(output_lines)

    @staticmethod
    def suggest_correction_reasons() -> str:
        """
        Provide standard correction reason suggestions.

        Returns:
            List of common correction reasons
        """
        return """## Suggested Correction Reasons

Choose a reason that best explains your correction:

- **Historical precedent** - Similar projects took more/less time
- **Team expertise** - Specialized skills affect estimation
- **Scope clarification** - Requirements were adjusted after initial estimate
- **Technical complexity** - Implementation harder/easier than expected
- **External dependencies** - Third-party constraints or changes
- **Regulatory requirements** - Compliance overhead not originally accounted for
- **Resource availability** - Team capacity differs from assumption
- **Tool/process changes** - New tools or processes affect effort"""

    # ===== Summary & Statistics Tools =====

    @staticmethod
    def generate_summary_stats(
        breakdown: list[dict],
        user_edits: list[dict] | None = None,
        applied_rules: list[dict] | None = None,
        pr_summary: dict | None = None,
    ) -> str:
        """
        Generate statistical summary of the estimation.

        Args:
            breakdown: Activity breakdown
            user_edits: User corrections
            applied_rules: Applied rules
            pr_summary: PR summary info

        Returns:
            Formatted statistics string
        """
        if not breakdown:
            return "No estimation data available for summary."

        # Calculate basic stats
        total_hours = sum(item.get("hours", 0) for item in breakdown)
        total_cost = sum(item.get("cost", 0) for item in breakdown)
        activity_count = len(breakdown)

        # Calculate hours distribution
        hours_list = [
            item.get("hours", 0) for item in breakdown if item.get("hours", 0) > 0
        ]
        avg_hours = sum(hours_list) / len(hours_list) if hours_list else 0
        max_hours = max(hours_list) if hours_list else 0
        min_hours = min(hours_list) if hours_list else 0

        edits_count = len(user_edits) if user_edits else 0
        rules_count = len(applied_rules) if applied_rules else 0

        output_lines = [
            "## Estimation Statistics\n",
            "### Totals",
            f"- **Total Hours**: {total_hours:,}h",
            f"- **Total Cost**: €{total_cost:,.0f}",
            f"- **Activities**: {activity_count}",
            "",
            "### Hours Distribution",
            f"- Average per activity: {avg_hours:.0f}h",
            f"- Maximum: {max_hours}h",
            f"- Minimum: {min_hours}h",
            "",
            "### Process",
            f"- Rules applied: {rules_count}",
            f"- Manual corrections: {edits_count}",
        ]

        # Add PR info if available
        if pr_summary:
            output_lines.insert(2, f"\n**Project**: {pr_summary.get('title', 'N/A')}")
            output_lines.insert(3, f"**Platform**: {pr_summary.get('platform', 'N/A')}")
            output_lines.insert(
                4, f"**Program Size**: {pr_summary.get('program_size', 'N/A')}"
            )

        # Add confidence metrics if corrections made
        if edits_count > 0 and user_edits:
            total_correction = sum(
                abs(e.get("corrected_value", 0) - e.get("original_value", 0))
                for e in user_edits
            )
            correction_pct = (
                (total_correction / total_hours * 100) if total_hours > 0 else 0
            )
            output_lines.append(
                f"- Correction magnitude: {correction_pct:.1f}% of total"
            )

        return "\n".join(output_lines)

    @staticmethod
    def format_breakdown_table(
        breakdown: list[dict],
        show_confidence: bool = True,
    ) -> str:
        """
        Format breakdown as a clean markdown table.

        Args:
            breakdown: Activity breakdown list
            show_confidence: Whether to show confidence column

        Returns:
            Formatted markdown table
        """
        if not breakdown:
            return "No breakdown data available."

        # Build header
        if show_confidence:
            header = "| Code | Activity | Hours | Cost (€) | Conf |"
            separator = "|------|----------|------:|--------:|-----:|"
        else:
            header = "| Code | Activity | Hours | Cost (€) |"
            separator = "|------|----------|------:|--------:|"

        output_lines = [header, separator]

        total_hours = 0
        total_cost = 0

        for item in breakdown:
            code = item.get("activity_code", "")[:6]
            name = item.get("activity_name", "")[:30]
            hours = item.get("hours", 0)
            cost = item.get("cost", 0)
            conf = item.get("confidence_score", 0)

            total_hours += hours
            total_cost += cost

            if show_confidence:
                output_lines.append(
                    f"| {code} | {name} | {hours} | {cost:,.0f} | {conf:.0%} |"
                )
            else:
                output_lines.append(f"| {code} | {name} | {hours} | {cost:,.0f} |")

        # Add totals row
        output_lines.append(separator)
        if show_confidence:
            output_lines.append(
                f"| | **TOTAL** | **{total_hours}** | **{total_cost:,.0f}** | |"
            )
        else:
            output_lines.append(
                f"| | **TOTAL** | **{total_hours}** | **{total_cost:,.0f}** |"
            )

        return "\n".join(output_lines)

    # ===== PR Analysis Tools =====

    @staticmethod
    def format_pr_comparison(
        current_pr: dict,
        similar_prs: list[dict],
    ) -> str:
        """
        Format comparison between current PR and similar historical PRs.

        Args:
            current_pr: Current PR data
            similar_prs: Similar PRs from RAG search

        Returns:
            Formatted comparison string
        """
        if not similar_prs:
            return "No similar PRs found for comparison."

        output_lines = [
            "## Similar Historical Projects\n",
            "| PR # | Title | Platform | Cost | Similarity |",
            "|------|-------|----------|-----:|----------:|",
        ]

        costs = []
        for pr in similar_prs[:7]:
            pr_num = pr.get("pr_number", pr.get("pr_code", "?"))
            title = pr.get("title", "Unknown")[:35]
            platform = pr.get("platform", "?")
            cost = pr.get("total_cost", 0)
            score = pr.get("score", 0)

            if cost > 0:
                costs.append(cost)

            output_lines.append(
                f"| {pr_num} | {title} | {platform} | €{cost:,.0f} | {score:.0%} |"
            )

        # Add statistics
        if costs:
            avg_cost = sum(costs) / len(costs)
            output_lines.append("")
            output_lines.append(f"**Average cost of similar PRs**: €{avg_cost:,.0f}")

            current_cost = current_pr.get("total_cost", 0)
            if current_cost > 0:
                diff_pct = ((current_cost - avg_cost) / avg_cost) * 100
                output_lines.append(f"**Your estimate vs average**: {diff_pct:+.1f}%")

        return "\n".join(output_lines)

    @staticmethod
    def format_questions_status(questions: list[dict]) -> str:
        """
        Format Q&A status summary.

        Args:
            questions: List of questions with answers

        Returns:
            Formatted status string
        """
        if not questions:
            return "No questions generated yet."

        answered = sum(1 for q in questions if q.get("answer"))
        total = len(questions)

        output_lines = [
            f"## Q&A Status: {answered}/{total} answered\n",
        ]

        for i, q in enumerate(questions, 1):
            question_text = q.get("question_text", q.get("question", ""))[:60]
            has_answer = "✅" if q.get("answer") else "⬜"
            priority = q.get("priority", "medium")

            output_lines.append(f"{has_answer} **Q{i}** [{priority}]: {question_text}")

        if answered < total:
            output_lines.append(f"\n*{total - answered} questions remaining*")
        else:
            output_lines.append("\n*All questions answered - ready to proceed*")

        return "\n".join(output_lines)


# Singleton instance for easy access
estimation_tools = EstimationTools()

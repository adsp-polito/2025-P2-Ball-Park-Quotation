"""
FPT Cost Brain 2.0 - Learning Node
Extract rules from user corrections for online learning
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from llm.client import get_llm_client
from llm.prompts import RULE_EXTRACTION

from agents.state import EstimationState, LearningResult, StepStatus

logger = logging.getLogger(__name__)

# Database session holder - set by the estimation service before running
_db_session = None


def set_db_session(db):
    """Set the database session for learning operations."""
    global _db_session
    _db_session = db


async def process_learning(state: EstimationState) -> EstimationState:
    """
    Process the learning step: extract rules from user corrections.

    This step:
    1. Analyzes user edits to find patterns
    2. Uses LLM to extract generalizable rules
    3. Stores feedback for batch retraining
    4. Updates rule confidence for applied rules
    """
    state["step_status"]["learning"] = StepStatus.IN_PROGRESS
    state["current_step"] = "learning"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    user_edits = state.get("user_edits", [])
    feedback_reasons = state.get("feedback_reasons", {})
    breakdown = state.get("breakdown", [])
    applied_rules = state.get("applied_rules", [])

    learning_result: LearningResult = {
        "extracted_rules": [],
        "feedback_ids": [],
        "queued_for_retrain": False,
    }

    try:
        # Step 1: Process user edits
        if user_edits:
            llm = get_llm_client()

            for edit in user_edits:
                # Find the breakdown item that was edited
                edited_item = next(
                    (item for item in breakdown if item["id"] == edit["breakdown_id"]),
                    None,
                )

                if not edited_item:
                    continue

                # Get feedback reason if provided
                reason = feedback_reasons.get(
                    edit["breakdown_id"], edit.get("reason", "")
                )

                # Calculate correction magnitude
                original = edit["original_hours"]
                corrected = edit["new_hours"]
                correction_percent = (
                    ((corrected - original) / original * 100) if original != 0 else 0
                )

                # Store feedback in database
                feedback_id = await store_feedback(
                    quotation_id=state.get("quotation_id", edit.get("breakdown_id")),
                    breakdown_id=edit["breakdown_id"],
                    original_value=original,
                    corrected_value=corrected,
                    reason=reason,
                    user_id=state.get("user_id", ""),
                )
                learning_result["feedback_ids"].append(feedback_id)

                # Only extract rules for significant corrections
                if abs(correction_percent) > 10:
                    rule = await extract_rule_from_correction(
                        edited_item=edited_item,
                        original_hours=original,
                        corrected_hours=corrected,
                        reason=reason,
                        correction_percent=correction_percent,
                        state=state,
                        llm=llm,
                    )

                    if rule:
                        # Store rule in database
                        rule_id = await store_extracted_rule(rule, feedback_id)
                        if rule_id:
                            rule["id"] = rule_id
                        learning_result["extracted_rules"].append(rule)

        # Step 2: Update confidence for applied rules
        for applied_rule in applied_rules:
            was_overridden = any(
                edit["breakdown_id"]
                in [item["id"] for item in breakdown if item.get("user_edited")]
                for edit in user_edits
            )

            if was_overridden:
                # Rule was overridden - decrease confidence
                await update_rule_confidence(
                    rule_id=applied_rule["rule_id"],
                    increase=False,
                )
            else:
                # Rule was kept - increase confidence
                await update_rule_confidence(
                    rule_id=applied_rule["rule_id"],
                    increase=True,
                )

        # Step 3: Check if batch retrain should be triggered
        should_retrain, retrain_reason = await check_retrain_trigger()

        if should_retrain:
            learning_result["queued_for_retrain"] = True
            learning_result["retrain_reason"] = retrain_reason
            logger.info(f"Queued for retrain: {retrain_reason}")

        state["learning_result"] = learning_result
        state["step_status"]["learning"] = StepStatus.COMPLETED
        state["current_step"] = "complete"

    except Exception as e:
        # Learning failures should not block the workflow
        state["learning_result"] = learning_result
        state["step_status"]["learning"] = StepStatus.COMPLETED
        state["current_step"] = "complete"

    return state


async def extract_rule_from_correction(
    edited_item: dict,
    original_hours: float,
    corrected_hours: float,
    reason: str,
    correction_percent: float,
    state: EstimationState,
    llm,
) -> dict[str, Any] | None:
    """
    Use LLM to extract a generalizable rule from a correction.

    Returns a rule dict or None if no rule could be extracted.
    """
    try:
        # Build context for rule extraction
        parsed_pr = state.get("parsed_pr", {})
        pr_summary = state.get("pr_summary", {})

        context = {
            "activity_code": edited_item.get("activity_code"),
            "activity_name": edited_item.get("activity_name"),
            "program_family": parsed_pr.get("program_family", "Unknown"),
            "program_size": pr_summary.get("program_size", "Unknown")
            if pr_summary
            else "Unknown",
        }

        prompt = RULE_EXTRACTION.format(
            original=f"{original_hours} hours",
            corrected=f"{corrected_hours} hours",
            reason=reason or "No reason provided",
        )

        # Add context
        prompt = f"""Context:
- Activity: {context["activity_name"]} ({context["activity_code"]})
- Program Family: {context["program_family"]}
- Program Size: {context["program_size"]}
- Correction: {original_hours}h -> {corrected_hours}h ({correction_percent:+.1f}%)

{prompt}"""

        result = await llm.extract_json(
            prompt=prompt,
            system_prompt="You are an expert at extracting patterns from cost estimation corrections.",
        )

        if result and result.get("rule_name"):
            # Validate the extracted rule
            if validate_rule(result):
                return result

        return None

    except Exception:
        return None


def validate_rule(rule: dict) -> bool:
    """Validate that an extracted rule has required fields."""
    required_fields = ["rule_name", "conditions", "adjustment"]

    for field in required_fields:
        if field not in rule:
            return False

    # Validate conditions
    conditions = rule.get("conditions", {})
    if not conditions.get("field") or not conditions.get("operator"):
        return False

    # Validate adjustment
    adjustment = rule.get("adjustment", {})
    if not adjustment.get("type") or "value" not in adjustment:
        return False

    return True


async def update_rule_confidence(
    rule_id: str,
    increase: bool,
) -> None:
    """
    Update confidence score for a rule in the database.
    """
    global _db_session

    if _db_session is None:
        logger.warning("No database session available for rule confidence update")
        return

    try:
        from db.repositories.rules_repo import RulesRepository

        rules_repo = RulesRepository(_db_session)
        rule_uuid = UUID(rule_id)

        if increase:
            await rules_repo.increase_confidence(rule_uuid)
            logger.info(f"Increased confidence for rule {rule_id}")
        else:
            await rules_repo.decrease_confidence(rule_uuid)
            logger.info(f"Decreased confidence for rule {rule_id}")
    except Exception as e:
        logger.error(f"Failed to update rule confidence: {e}")


async def store_feedback(
    quotation_id: str,
    breakdown_id: str,
    original_value: float,
    corrected_value: float,
    reason: str,
    user_id: str,
) -> str:
    """
    Store feedback correction in the database.

    Returns the feedback ID.
    """
    global _db_session

    if _db_session is None:
        logger.warning("No database session available for feedback storage")
        return f"fb_{breakdown_id}_{datetime.now().timestamp()}"

    try:
        from db.repositories.feedback_repo import FeedbackRepository

        feedback_repo = FeedbackRepository(_db_session)
        feedback = await feedback_repo.create(
            quotation_id=UUID(quotation_id),
            breakdown_id=UUID(breakdown_id),
            original_value=original_value,
            corrected_value=corrected_value,
            field_name="hours",
            reason=reason,
            created_by=UUID(user_id) if user_id else None,
        )
        logger.info(f"Stored feedback {feedback.id} for breakdown {breakdown_id}")
        return str(feedback.id)
    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        return f"fb_{breakdown_id}_{datetime.now().timestamp()}"


async def store_extracted_rule(
    rule_data: dict[str, Any],
    source_feedback_id: str | None = None,
) -> str | None:
    """
    Store an extracted rule in the database.

    Returns the rule ID or None if storage failed.
    """
    global _db_session

    if _db_session is None:
        logger.warning("No database session available for rule storage")
        return None

    try:
        from db.repositories.rules_repo import RulesRepository

        rules_repo = RulesRepository(_db_session)

        # Check for conflicting rules
        conflicts = await rules_repo.find_conflicting_rules(
            rule_data.get("conditions", {})
        )

        if conflicts:
            logger.info(f"Found {len(conflicts)} potentially conflicting rules")
            # Flag new rule for review if conflicts exist
            rule_data["requires_review"] = True

        rule = await rules_repo.create_from_extraction(
            extraction_result=rule_data,
            source_feedback_id=UUID(source_feedback_id) if source_feedback_id else None,
        )
        logger.info(f"Stored extracted rule: {rule.rule_name} (id={rule.id})")
        return str(rule.id)
    except Exception as e:
        logger.error(f"Failed to store extracted rule: {e}")
        return None


async def check_retrain_trigger() -> tuple[bool, str]:
    """
    Check if model retraining should be triggered.

    Returns (should_retrain, reason).
    """
    global _db_session

    if _db_session is None:
        logger.warning("No database session available for retrain check")
        return False, "No database session"

    try:
        from db.repositories.feedback_repo import FeedbackRepository

        feedback_repo = FeedbackRepository(_db_session)
        return await feedback_repo.should_trigger_retrain()
    except Exception as e:
        logger.error(f"Failed to check retrain trigger: {e}")
        return False, f"Error: {e}"

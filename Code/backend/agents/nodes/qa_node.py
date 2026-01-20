"""
FPT Cost Brain 2.0 - Q&A Node
Generate and validate clarifying questions

Dynamic question generation based on:
- Parsed PR features (product_family, emissions, sector, etc.)
- Detected change types (hardware, calibration, ATS, software)
- Similar projects context
- Chat interactions (for real-time question updates)
- **Feature extraction confidence** (low confidence triggers additional questions)
- **Missing features** (targeted questions to complete ML pipeline input)
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from llm.client import get_llm_client

from llm.prompts import QA_ANSWER_VALIDATION, QA_QUESTION_GENERATION

from agents.state import EstimationState, ParsedPR, Question, StepStatus

from ml.hcqe_feature_extractor import MISSING_FEATURE_QUESTIONS
from app.debug_logging import log_questions, log_llm_call, log_error_details

logger = logging.getLogger(__name__)

# Confidence threshold below which additional questions are generated
CONFIDENCE_THRESHOLD = 0.7

# Maximum number of questions to add from missing features
MAX_MISSING_FEATURE_QUESTIONS = 4

# Features with highest impact on cost estimation accuracy
HIGH_IMPACT_FEATURES = [
    "sector",
    "hardware_change",
    "calibration_change",
    "ATS_change",
    "emissions",
    "sizing_program",
    "power_increase_kw",
    "product_family",
]


def build_pr_context(parsed_pr: ParsedPR) -> dict[str, Any]:
    """
    Build comprehensive context from parsed PR for question generation.

    Includes all detected features from v1 parser port.
    Prioritizes structured_text (LLM-optimized) over raw_text.
    """
    return {
        # Basic info
        "pr_code": parsed_pr.get("pr_code", "Unknown"),
        "revision": parsed_pr.get("revision", ""),
        "title": parsed_pr.get("title", "Unknown"),
        "description": parsed_pr.get("description", ""),
        # Platform/Engine/Tier
        "platform": parsed_pr.get("platform", ""),
        "engine": parsed_pr.get("engine", ""),
        "tier": parsed_pr.get("tier", ""),
        "customer": parsed_pr.get("customer", ""),
        # Detected features
        "product_family": parsed_pr.get("product_family", ""),
        "emissions": parsed_pr.get("emissions", ""),
        "sector": parsed_pr.get("sector", ""),
        "ats_tech": parsed_pr.get("ats_tech", ""),
        # Change type flags
        "hardware_change": parsed_pr.get("hardware_change", False),
        "calibration_change": parsed_pr.get("calibration_change", False),
        "ats_change": parsed_pr.get("ats_change", False),
        "software_vcu_change": parsed_pr.get("software_vcu_change", False),
        # Activities
        "activities": parsed_pr.get("raw_activities", []),
        # LLM-optimized structured text (preferred for prompts)
        "structured_text": parsed_pr.get("structured_text", ""),
        "llm_context": parsed_pr.get("llm_context", ""),
    }


def build_llm_prompt(context: dict[str, Any], similar_prs: list = None) -> str:
    """Build detailed LLM prompt for FPT-specific question generation."""

    # Build change types summary
    changes = []
    if context.get("hardware_change"):
        changes.append("Hardware modifications")
    if context.get("calibration_change"):
        changes.append("Calibration changes")
    if context.get("ats_change"):
        changes.append("ATS (Aftertreatment System) changes")
    if context.get("software_vcu_change"):
        changes.append("Software/VCU updates")

    changes_text = ", ".join(changes) if changes else "None detected"

    # Build similar projects text
    similar_text = ""
    if similar_prs:
        similar_text = "\n\nSimilar Historical Projects (for reference):\n" + "\n".join(
            [
                f"- {sp['pr_code']}: {sp['title'][:50]}... ({sp['total_hours']} hours)"
                for sp in similar_prs[:3]
            ]
        )

    # Include structured text if available (much better for LLM understanding)
    structured_section = ""
    if context.get("structured_text"):
        structured_section = f"""

=== FULL PR DOCUMENT (Structured) ===
{context["structured_text"]}
"""

    return f"""**FPT Cost Brain - Question Generation**

IMPORTANT CONTEXT:
FPT Cost Brain PREDICTS engineering effort (hours, cost, activity breakdown) from Product Requests.
- The PR describes WHAT the customer wants - it does NOT contain a pre-defined breakdown
- Our ML system will GENERATE the detailed activity breakdown and cost estimates
- Your questions should gather context that helps us PREDICT MORE ACCURATELY
- Do NOT ask for breakdown details - we predict those. Ask about factors that AFFECT the prediction.

=== PR Summary ===
PR Code: {context["pr_code"]} Rev {context.get("revision", "A")}
Title: {context["title"]}
{structured_section}
=== Detected Features (Auto-Extracted) ===
Product Family: {context.get("product_family") or "Not detected"}
Emissions Standard: {context.get("emissions") or "Not detected"}
Sector: {context.get("sector") or "Not detected"} (AG=Agriculture, CE=Construction, PT=Powertrain)
ATS Technology: {context.get("ats_tech") or "Not detected"}

=== Detected Change Types ===
{changes_text}
{similar_text}

Generate 4-6 specific questions that would help us PREDICT R&D costs more accurately.
Focus on questions about:
1. Technical complexity factors that affect effort (not asking for breakdown - we predict that)
2. Scope boundaries and constraints
3. Timeline and resource constraints
4. Risk factors and dependencies

Return JSON format with questions array."""


async def generate_questions(state: EstimationState) -> EstimationState:
    """
    Generate clarifying questions based on the parsed PR.

    Uses LLM to analyze the PR with all detected features
    to generate context-aware questions for cost estimation.
    """
    qa_start = time.time()
    logger.info("=" * 70)
    logger.info("❓ Q&A NODE STARTED")
    logger.info("=" * 70)

    state["step_status"]["qa"] = StepStatus.IN_PROGRESS
    state["current_step"] = "qa"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    parsed_pr = state.get("parsed_pr")
    if not parsed_pr:
        logger.warning("  ⚠️ No parsed PR, skipping Q&A")
        state["questions"] = []
        state["qa_complete"] = True
        state["step_status"]["qa"] = StepStatus.SKIPPED
        return state

    try:
        llm = get_llm_client()
        logger.info(f"  🤖 LLM client initialized")

        # Build comprehensive context from parsed PR
        context = build_pr_context(parsed_pr)

        # DEBUG: Log what context we're sending to LLM
        logger.info(f"  📄 PR context for question generation:")
        logger.info(f"    pr_code: {context.get('pr_code')}")
        logger.info(f"    title: {context.get('title', 'NO TITLE')[:50]}")
        logger.info(f"    hardware_change: {context.get('hardware_change')}")
        logger.info(f"    calibration_change: {context.get('calibration_change')}")
        logger.info(f"    ats_change: {context.get('ats_change')}")
        logger.info(f"    software_vcu_change: {context.get('software_vcu_change')}")
        logger.info(f"    description length: {len(context.get('description', ''))}")

        # Get similar PRs if available
        similar_prs = state.get("similar_prs", [])
        logger.info(f"  🔍 Similar PRs: {len(similar_prs)}")

        # Generate questions using LLM with full context
        prompt = build_llm_prompt(context, similar_prs)
        logger.info(f"  📝 Prompt built ({len(prompt)} chars)")

        logger.info("  ⏳ Calling LLM for question generation...")
        llm_start = time.time()
        log_llm_call(logger, "question_generation", prompt[:200])
        result = await llm.extract_json(
            prompt=prompt,
            system_prompt=QA_QUESTION_GENERATION,
        )
        logger.info(f"  ✅ LLM responded in {time.time() - llm_start:.2f}s")

        # Convert to Question format
        questions: list[Question] = []
        for q in result.get("questions", []):
            question: Question = {
                "id": q.get("id", f"q{len(questions) + 1}"),
                "question": q.get("question", ""),
                "reason": q.get("reason", ""),
                "category": q.get("category", "general"),
                "priority": q.get("priority", "medium"),
                "suggested_answers": q.get("suggested_answers", []),
                "answer": None,
                "is_answered": False,
            }
            questions.append(question)

        logger.info(f"  📋 Generated {len(questions)} questions from LLM")

        # If no questions generated, use smart defaults based on context
        if not questions:
            logger.info("  ⚠️ No LLM questions, using defaults")
            questions = get_default_questions(parsed_pr)

        # === FEATURE COMPLETION INTEGRATION ===
        # Check if we have feature extraction results from intake/summary step
        feature_extraction_result = state.get("feature_extraction_result")
        if feature_extraction_result:
            # Add feature-completion questions based on missing features
            existing_ids = {q["id"] for q in questions}
            feature_questions = generate_questions_from_missing_features(
                missing_features=feature_extraction_result.get("missing_features", []),
                feature_confidence=feature_extraction_result.get("confidence", 1.0),
                existing_question_ids=existing_ids,
            )
            if feature_questions:
                logger.info(
                    f"  ➕ Adding {len(feature_questions)} "
                    "feature-completion questions from extraction results"
                )
                questions.extend(feature_questions)

        state["questions"] = questions
        state["qa_complete"] = False
        state["step_status"]["qa"] = StepStatus.WAITING_INPUT

    except Exception as e:
        logger.error(
            f"[QA_NODE] LLM question generation failed: {type(e).__name__}: {e}"
        )

        # PRESERVE existing questions if they exist
        existing_questions = state.get("questions", [])
        if existing_questions and len(existing_questions) > 0:
            logger.info(
                f"[QA_NODE] Preserving {len(existing_questions)} existing questions after LLM error"
            )
        else:
            # For PRODUCTION: Do NOT use hardcoded fallback - propagate the error
            # The user needs to know that LLM generation failed so they can fix it
            logger.error(
                "[QA_NODE] No existing questions and LLM failed - raising error"
            )
            state["error_message"] = f"Failed to generate questions: {str(e)}"
            # Keep questions empty - frontend will show error

        state["qa_complete"] = False
        state["step_status"]["qa"] = StepStatus.WAITING_INPUT

    return state


async def regenerate_questions_from_chat(
    state: EstimationState,
    user_message: str,
) -> list[Question]:
    """
    Regenerate or refine questions based on chat interaction.

    Called when user provides context through chat that should
    update the question list dynamically.

    IMPORTANT: Preserves existing question fields (priority, suggested_answers)
    when possible to avoid breaking the UI.
    """
    parsed_pr = state.get("parsed_pr")
    if not parsed_pr:
        return state.get("questions", [])

    try:
        llm = get_llm_client()

        context = build_pr_context(parsed_pr)
        existing_questions = state.get("questions", [])
        existing_answers = state.get("answers", {})

        # Create a map of existing questions by ID for preserving fields
        existing_q_map = {q.get("id"): q for q in existing_questions}

        # Get question text (support both field names)
        def get_q_text(q: dict) -> str:
            return q.get("question") or q.get("question_text") or ""

        # Build prompt for question refinement with FULL schema
        prompt = f"""Based on user's chat message, refine the clarifying questions.

=== PR Context ===
PR Code: {context["pr_code"]}
Title: {context["title"]}
Product Family: {context.get("product_family") or "Unknown"}
Sector: {context.get("sector") or "Unknown"}

=== User's Message ===
{user_message}

=== Current Questions ===
{chr(10).join([f"- Q{i + 1} (id={q.get('id')}): {get_q_text(q)} [priority={q.get('priority', 'medium')}, answered={q.get('id') in existing_answers}]" for i, q in enumerate(existing_questions)])}

Based on the user's message:
1. Keep questions that are still relevant (preserve their ID, priority, and suggested_answers)
2. Remove questions that are now answered by the message
3. Add new relevant questions based on information revealed
4. Only change priority if clearly warranted

IMPORTANT: Return questions with the EXACT JSON schema:
[{{
  "id": "existing_id_or_new",
  "question": "the question text",
  "reason": "why this question helps",
  "category": "scope|complexity|technical|timeline|testing",
  "priority": "high|medium|low",
  "suggested_answers": ["option 1", "option 2", "option 3"]
}}]

For EXISTING questions you want to keep, use their original ID (e.g., "q1", "q_1") to preserve their data."""

        result = await llm.extract_json(
            prompt=prompt,
            system_prompt=QA_QUESTION_GENERATION,
        )

        # Convert to Question format, PRESERVING existing fields where possible
        questions: list[Question] = []
        for q in result.get("questions", []):
            qid = q.get("id", f"q{len(questions) + 1}")

            # Check if this is an existing question we should preserve fields from
            existing_q = existing_q_map.get(qid, {})

            question: Question = {
                "id": qid,
                # Use new question text if provided, else keep existing
                "question": q.get("question")
                or q.get("question_text")
                or get_q_text(existing_q),
                "reason": q.get("reason") or existing_q.get("reason", ""),
                "category": q.get("category") or existing_q.get("category", "general"),
                # Preserve priority unless explicitly changed
                "priority": q.get("priority") or existing_q.get("priority", "medium"),
                # Preserve suggested_answers if LLM didn't provide new ones
                "suggested_answers": q.get("suggested_answers")
                or existing_q.get("suggested_answers", []),
                "answer": existing_answers.get(qid) or existing_q.get("answer"),
                "is_answered": qid in existing_answers
                or existing_q.get("is_answered", False),
            }
            questions.append(question)

        return questions if questions else existing_questions

    except Exception as e:
        logger.error(f"[QA_NODE] regenerate_questions_from_chat failed: {e}")
        return state.get("questions", [])


def get_default_questions(parsed_pr: dict) -> list[Question]:
    """
    Generate smart default questions based on parsed PR context.

    Questions are dynamically selected based on:
    - Detected change types (hardware, calibration, ATS, software)
    - Product family
    - Sector
    - Missing information
    """
    questions: list[Question] = []
    q_count = 1

    # === Change Type Specific Questions ===

    # Hardware change questions
    if parsed_pr.get("hardware_change"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": "What specific hardware components are being modified or added?",
                "reason": "Hardware changes significantly impact material and labor costs",
                "category": "hardware",
                "priority": "high",
                "suggested_answers": [
                    "Injector modifications",
                    "Turbocharger changes",
                    "New sensors/actuators",
                    "Engine block modifications",
                    "Multiple components",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # Calibration change questions
    if parsed_pr.get("calibration_change"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": "What is the scope of calibration work required?",
                "reason": "Calibration scope affects testing and validation effort",
                "category": "calibration",
                "priority": "high",
                "suggested_answers": [
                    "Minor parameter adjustments",
                    "Full engine calibration",
                    "Emissions calibration only",
                    "Performance + emissions calibration",
                    "Multi-application calibration",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # ATS change questions
    if parsed_pr.get("ats_change"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": "What aftertreatment system modifications are needed?",
                "reason": "ATS changes have major cost implications due to emissions compliance",
                "category": "ats",
                "priority": "high",
                "suggested_answers": [
                    "DOC optimization",
                    "SCR system changes",
                    "DPF modifications",
                    "Complete ATS redesign",
                    "Integration with existing ATS",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # Software/VCU change questions
    if parsed_pr.get("software_vcu_change"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": "What is the extent of software/ECU changes required?",
                "reason": "Software complexity directly affects development and validation time",
                "category": "software",
                "priority": "high",
                "suggested_answers": [
                    "Parameter updates only",
                    "New control strategies",
                    "ECU hardware change",
                    "Multi-ECU coordination",
                    "New diagnostics implementation",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # === Missing Information Questions ===

    # Product family unknown
    if not parsed_pr.get("product_family"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": "Which engine product family does this project target?",
                "reason": "Product family determines baseline complexity and historical data",
                "category": "technical",
                "priority": "high",
                "suggested_answers": [
                    "NEF (N45/N67/E0N6)",
                    "CURSOR (C87/C9/C11/C13)",
                    "E0C0/E9C0",
                    "F1 series",
                    "E5F0 series",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # Emissions unknown but tier detected
    if not parsed_pr.get("emissions") and parsed_pr.get("tier"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": f"Please confirm the target emissions standard for '{parsed_pr.get('tier')}'",
                "reason": "Emissions standard affects certification and testing requirements",
                "category": "emissions",
                "priority": "high",
                "suggested_answers": [
                    "Stage V",
                    "Tier 4B/Final",
                    "China NRIV",
                    "Euro VI",
                    "Multiple standards",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # Sector unknown
    if not parsed_pr.get("sector"):
        questions.append(
            {
                "id": f"q{q_count}",
                "question": "Which sector/application is this project for?",
                "reason": "Sector determines application-specific testing requirements",
                "category": "application",
                "priority": "medium",
                "suggested_answers": [
                    "AG - Agriculture (tractors, harvesters)",
                    "CE - Construction (excavators, loaders)",
                    "PT - Powertrain/Trucks",
                    "Marine applications",
                    "Power generation",
                ],
                "answer": None,
                "is_answered": False,
            }
        )
        q_count += 1

    # === Standard Questions (always relevant) ===

    # Scope/complexity question
    questions.append(
        {
            "id": f"q{q_count}",
            "question": "How would you rate the overall complexity of this project?",
            "reason": "Complexity assessment helps calibrate estimation confidence",
            "category": "complexity",
            "priority": "medium",
            "suggested_answers": [
                "Low - Minor changes, similar to previous projects",
                "Medium - Some new development, mostly standard work",
                "High - Significant new development required",
                "Very High - First-of-kind, high uncertainty",
            ],
            "answer": None,
            "is_answered": False,
        }
    )
    q_count += 1

    # Timeline question
    questions.append(
        {
            "id": f"q{q_count}",
            "question": "What is the target delivery timeline?",
            "reason": "Timeline affects resource allocation and potential overtime costs",
            "category": "timeline",
            "priority": "medium",
            "suggested_answers": [
                "Standard timeline (6+ months)",
                "Accelerated (3-6 months)",
                "Urgent (< 3 months)",
                "Flexible / No fixed deadline",
            ],
            "answer": None,
            "is_answered": False,
        }
    )
    q_count += 1

    # Testing requirements
    questions.append(
        {
            "id": f"q{q_count}",
            "question": "What testing and validation activities are expected?",
            "reason": "Testing scope is often a major cost driver in R&D projects",
            "category": "testing",
            "priority": "medium",
            "suggested_answers": [
                "Bench testing only",
                "Bench + limited vehicle testing",
                "Full vehicle validation program",
                "Certification testing required",
                "Customer-specific validation",
            ],
            "answer": None,
            "is_answered": False,
        }
    )

    return questions


def generate_questions_from_missing_features(
    missing_features: list[str],
    feature_confidence: float,
    existing_question_ids: set[str],
) -> list[Question]:
    """
    Generate targeted questions based on missing features from extraction.

    This function acts as the "feature completion" mechanism, generating
    questions specifically for features that could not be extracted and
    would improve ML prediction accuracy.

    Args:
        missing_features: List of feature names that were not extracted
        feature_confidence: Overall confidence score from feature extraction (0-1)
        existing_question_ids: Set of question IDs already in the question list

    Returns:
        List of Question dicts for missing high-impact features
    """
    questions: list[Question] = []

    # Prioritize high-impact features first
    prioritized_missing = []
    for feat in HIGH_IMPACT_FEATURES:
        if feat in missing_features:
            prioritized_missing.append(feat)
    # Add remaining missing features
    for feat in missing_features:
        if feat not in prioritized_missing:
            prioritized_missing.append(feat)

    # Generate questions for missing features
    for feat in prioritized_missing[:MAX_MISSING_FEATURE_QUESTIONS]:
        if feat not in MISSING_FEATURE_QUESTIONS:
            continue

        q_template = MISSING_FEATURE_QUESTIONS[feat]
        q_id = f"mf_{feat}"

        # Skip if similar question already exists
        if q_id in existing_question_ids:
            continue

        question: Question = {
            "id": q_id,
            "question": q_template["question"],
            "reason": f"[Feature Completion] {q_template['reason']}",
            "category": q_template["category"],
            "priority": q_template["priority"],
            "suggested_answers": q_template["suggested_answers"],
            "answer": None,
            "is_answered": False,
        }
        questions.append(question)

    # If confidence is low, add a general complexity question
    if feature_confidence < CONFIDENCE_THRESHOLD and len(questions) < 2:
        complexity_q_id = "mf_low_confidence"
        if complexity_q_id not in existing_question_ids:
            questions.append(
                {
                    "id": complexity_q_id,
                    "question": "Please describe the main technical challenges "
                    "or uncertainties in this project.",
                    "reason": "[Low Confidence] Feature extraction had low confidence. "
                    "Additional context will improve estimation accuracy.",
                    "category": "complexity",
                    "priority": "high",
                    "suggested_answers": [
                        "First-of-kind development with high uncertainty",
                        "Integration challenges with existing systems",
                        "Tight emissions margin requiring extensive testing",
                        "Complex multi-application calibration",
                        "Standard project with well-understood scope",
                    ],
                    "answer": None,
                    "is_answered": False,
                }
            )

    return questions


def get_questions_with_feature_completion(
    parsed_pr: dict,
    feature_extraction_result: dict | None = None,
) -> list[Question]:
    """
    Generate questions combining default logic and feature completion.

    This is the main entry point for question generation that integrates:
    1. Default change-type-specific questions
    2. Missing information questions
    3. Feature-completion questions based on extraction confidence

    Args:
        parsed_pr: Parsed Product Request data
        feature_extraction_result: Optional dict with keys:
            - 'confidence': float (0-1) extraction confidence
            - 'missing_features': list[str] features not extracted
            - 'extraction_method': str ('llm' or 'rule_based')

    Returns:
        Complete list of questions for the Q&A step
    """
    # Start with default questions based on parsed PR
    questions = get_default_questions(parsed_pr)
    existing_ids = {q["id"] for q in questions}

    # If we have feature extraction results, add feature completion questions
    if feature_extraction_result:
        confidence = feature_extraction_result.get("confidence", 1.0)
        missing = feature_extraction_result.get("missing_features", [])
        method = feature_extraction_result.get("extraction_method", "unknown")

        logger.info(
            f"[QA_NODE] Feature extraction: method={method}, "
            f"confidence={confidence:.0%}, missing={len(missing)} features"
        )

        # Generate targeted questions for missing features
        if missing or confidence < CONFIDENCE_THRESHOLD:
            feature_questions = generate_questions_from_missing_features(
                missing_features=missing,
                feature_confidence=confidence,
                existing_question_ids=existing_ids,
            )

            if feature_questions:
                logger.info(
                    f"[QA_NODE] Adding {len(feature_questions)} "
                    "feature-completion questions"
                )
                questions.extend(feature_questions)

    return questions


async def validate_answers(state: EstimationState) -> EstimationState:
    """
    Validate user answers and check if Q&A is complete.

    May generate follow-up questions if answers are insufficient.
    """
    questions = state.get("questions", [])
    answers = state.get("answers", {})

    # Check which questions have been answered
    unanswered_high_priority = []
    for q in questions:
        if q["id"] in answers:
            q["answer"] = answers[q["id"]]
            q["is_answered"] = True
        else:
            if q["priority"] == "high":
                unanswered_high_priority.append(q)

    state["questions"] = questions

    # Q&A is complete if all high-priority questions are answered
    if not unanswered_high_priority:
        state["qa_complete"] = True
        state["step_status"]["qa"] = StepStatus.COMPLETED
    else:
        state["qa_complete"] = False
        state["step_status"]["qa"] = StepStatus.WAITING_INPUT

    return state


async def validate_single_answer(
    question: Question,
    answer: str,
) -> dict:
    """
    Validate a single answer using LLM.

    Returns validation result with potential follow-up.
    """
    try:
        llm = get_llm_client()

        prompt = f"""Question: {question["question"]}
Reason this matters: {question["reason"]}
User's answer: {answer}

Validate if this answer adequately addresses the question."""

        result = await llm.extract_json(
            prompt=prompt,
            system_prompt=QA_ANSWER_VALIDATION,
        )

        return result

    except Exception:
        # Default to accepting the answer on error
        return {
            "is_valid": True,
            "confidence": 0.5,
            "follow_up_needed": False,
            "follow_up_question": None,
        }

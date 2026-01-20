"""
FPT Cost Brain 2.0 - Summary Node
Analyze PR and generate comprehensive summary
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from llm.client import get_llm_client
from llm.prompts import PR_FEATURE_EXTRACTION, PR_SUMMARY_GENERATION

from agents.state import (
    EstimationState,
    MLFeature,
    PRSummary,
    SimilarPR,
    StepStatus,
)
from app.debug_logging import log_error_details, log_llm_call

logger = logging.getLogger(__name__)


async def process_summary(state: EstimationState) -> EstimationState:
    """
    Process the summary step: analyze PR and extract features.

    This step:
    1. Generates embedding for the PR
    2. Finds similar historical PRs
    3. Extracts ML features
    4. Generates comprehensive summary
    5. Classifies program size
    """
    import asyncio

    summary_start = time.time()
    logger.info("=" * 70)
    logger.info("📋 SUMMARY NODE STARTED")
    logger.info("=" * 70)

    # DEMO MODE - Skip LLM calls for stable demo
    DEMO_MODE = False  # Disabled for production testing

    # Skip if summary was already completed (avoid re-processing)
    summary_status = state.get("step_status", {}).get("summary")
    if summary_status in (StepStatus.COMPLETED, "completed"):
        logger.info("  ⏭️ Summary already completed, skipping")
        return state

    # Skip if we already have pr_summary (state was loaded from Redis)
    if state.get("pr_summary") and state.get("ml_features"):
        logger.info("  ⏭️ PR summary already exists, skipping")
        state["step_status"]["summary"] = StepStatus.COMPLETED
        return state

    state["step_status"]["summary"] = StepStatus.IN_PROGRESS
    state["current_step"] = "summary"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    parsed_pr = state.get("parsed_pr")
    answers = state.get("answers", {})
    logger.info(f"  📄 Parsed PR: {parsed_pr.get('pr_code', 'N/A') if parsed_pr else 'NONE'}")
    logger.info(f"  ❓ Q&A Answers: {len(answers)}")

    if not parsed_pr:
        logger.error("  ❌ No parsed PR data available")
        state["error_message"] = "No parsed PR data available"
        state["error_step"] = "summary"
        state["step_status"]["summary"] = StepStatus.ERROR
        return state

    # DEMO MODE: Return hardcoded demo data instantly
    if DEMO_MODE:
        logger.info("DEMO MODE: Generating demo summary...")
        await asyncio.sleep(2)  # Brief loading animation

        state["embedding"] = [0.1] * 100  # Dummy embedding
        state["similar_prs"] = [
            SimilarPR(
                id="demo-1",
                pr_code="PR_23045",
                title="Similar Engine Calibration Project",
                program_family=parsed_pr.get("program_family", "F1C"),
                similarity_score=0.89,
                total_hours=4200,
                total_cost_eur=520000,
            ),
            SimilarPR(
                id="demo-2",
                pr_code="PR_22087",
                title="Emission System Upgrade",
                program_family="NEF",
                similarity_score=0.82,
                total_hours=3800,
                total_cost_eur=480000,
            ),
        ]
        state["ml_features"] = [
            MLFeature(name="turbo_change", value=1, confidence=0.9, source="parsed"),
            MLFeature(
                name="calibration_change", value=1, confidence=0.85, source="parsed"
            ),
            MLFeature(
                name="emission_compliance", value=1, confidence=0.95, source="parsed"
            ),
            MLFeature(
                name="ats_modification", value=0, confidence=0.8, source="inferred"
            ),
        ]
        # Build executive summary based on parsed PR context
        pr_code = parsed_pr.get("pr_code", "PR_DEMO")
        pr_title = parsed_pr.get("title", "Demo Project")
        customer = parsed_pr.get("customer", "Demo Customer")
        program_family = parsed_pr.get("program_family", "F1C")

        # Include Q&A answers in summary if available
        qa_context = ""
        if answers:
            qa_context = "\n\nBased on clarifications provided during Q&A, "
            qa_context += "the project scope has been refined to address specific technical requirements."

        state["pr_summary"] = PRSummary(
            summary_text=f"""This Product Request ({pr_code}) covers {pr_title} for {customer}.

The project involves engine calibration and emission compliance activities with turbo modifications for the {program_family} program family.{qa_context}

Based on similar historical projects, we estimate this as a medium-complexity effort requiring coordination across calibration, testing, and software teams.""",
            program_size="medium",
            complexity_score=0.65,
            activity_count=5,  # Default for demo mode
            key_features=[
                f"Program: {program_family}",
                "Turbo calibration required",
                "Emission compliance testing",
                "Software/VCU updates",
                "Vehicle validation phase",
            ],
            dependencies=[
                "Base engine calibration data",
                "Emission test facility availability",
            ],
            risk_factors=[
                "Timeline constraints",
                "New emission standards compliance",
            ],
            special_requirements=[
                "Customer approval for turbo modifications",
            ],
        )
        state["step_status"]["summary"] = StepStatus.COMPLETED
        logger.info("DEMO MODE: Summary complete")
        return state

    try:
        llm = get_llm_client()
        logger.info("  🤖 LLM client initialized")

        # Step 1: Generate PR text for embedding
        logger.info("-" * 50)
        logger.info("  📝 Step 1: Generate PR text for embedding")
        pr_text = generate_pr_text(parsed_pr, answers)
        logger.info(f"    PR text length: {len(pr_text)} chars")

        # Step 2: Generate embedding
        logger.info("-" * 50)
        logger.info("  🔢 Step 2: Generate embedding")
        embed_start = time.time()
        embedding = await llm.embed(pr_text)
        state["embedding"] = embedding
        logger.info(f"    ✅ Embedding generated in {time.time() - embed_start:.2f}s ({len(embedding)} dims)")

        # Step 3: Find similar PRs using Qdrant
        logger.info("-" * 50)
        logger.info("  🔍 Step 3: Find similar PRs")
        similar_start = time.time()
        similar_prs = await find_similar_prs(embedding, parsed_pr)
        state["similar_prs"] = similar_prs
        logger.info(f"    ✅ Found {len(similar_prs)} similar PRs in {time.time() - similar_start:.2f}s")
        for i, spr in enumerate(similar_prs[:3]):
            logger.info(f"      {i+1}. {spr.get('pr_code', 'N/A')} (score: {spr.get('similarity_score', 0):.2f})")

        # Step 3b: Store this PR's embedding for future similarity searches
        logger.info("  💾 Storing PR embedding for future searches...")
        await store_pr_embedding(
            session_id=state.get("session_id", ""),
            embedding=embedding,
            parsed_pr=parsed_pr,
        )

        # Step 4: Extract ML features
        logger.info("-" * 50)
        logger.info("  🔧 Step 4: Extract ML features")
        feature_start = time.time()
        ml_features, extraction_result = await extract_ml_features(
            parsed_pr, answers, similar_prs, llm
        )
        state["ml_features"] = ml_features
        logger.info(f"    ✅ Extracted {len(ml_features)} features in {time.time() - feature_start:.2f}s")

        # Store extraction result for Q&A feature completion mechanism
        state["feature_extraction_result"] = extraction_result
        logger.info(
            f"    Confidence: {extraction_result['confidence']:.0%}, "
            f"Missing: {len(extraction_result['missing_features'])} features"
        )

        # Step 5: Generate comprehensive summary
        logger.info("-" * 50)
        logger.info("  📋 Step 5: Generate comprehensive summary")
        summary_gen_start = time.time()
        pr_summary = await generate_summary(
            parsed_pr, answers, similar_prs, ml_features, llm
        )
        state["pr_summary"] = pr_summary
        logger.info(f"    ✅ Summary generated in {time.time() - summary_gen_start:.2f}s")
        logger.info(f"    Program size: {pr_summary.get('program_size', 'N/A')}")
        logger.info(f"    Complexity: {pr_summary.get('complexity_score', 0):.2f}")

        state["step_status"]["summary"] = StepStatus.COMPLETED
        logger.info(f"🏁 SUMMARY NODE COMPLETED in {time.time() - summary_start:.2f}s")

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        logger.error(f"❌ Summary generation failed: {str(e)}\n{error_detail}")
        log_error_details(logger, e, "summary_node")
        state["error_message"] = f"Summary generation failed: {str(e)}"
        state["error_step"] = "summary"
        state["step_status"]["summary"] = StepStatus.ERROR

    return state


async def reextract_features_with_qa_answers(state: EstimationState) -> EstimationState:
    """
    Re-run feature extraction with Q&A answers for improved accuracy.

    This function is called after the Q&A step completes to:
    1. Re-extract features using Q&A answers as additional context
    2. Update confidence scores based on answered questions
    3. Reduce the list of missing features

    This creates a feedback loop where user answers directly improve
    the ML pipeline input quality.
    """
    import logging

    logger = logging.getLogger(__name__)

    parsed_pr = state.get("parsed_pr")
    answers = state.get("answers", {})
    similar_prs = state.get("similar_prs", [])

    if not parsed_pr or not answers:
        logger.info("[REEXTRACT] No PR or answers, skipping re-extraction")
        return state

    # Check if we already have high confidence
    prev_extraction = state.get("feature_extraction_result", {})
    prev_confidence = prev_extraction.get("confidence", 0)

    if prev_confidence >= 0.85:
        logger.info(
            f"[REEXTRACT] Previous confidence {prev_confidence:.0%} >= 85%, "
            "skipping re-extraction"
        )
        return state

    try:
        llm = get_llm_client()

        logger.info(
            f"[REEXTRACT] Re-extracting features with {len(answers)} Q&A answers"
        )

        # Re-run feature extraction with Q&A answers
        new_features, new_extraction_result = await extract_ml_features(
            parsed_pr, answers, similar_prs, llm
        )

        new_confidence = new_extraction_result.get("confidence", 0)
        new_missing = len(new_extraction_result.get("missing_features", []))

        logger.info(
            f"[REEXTRACT] New extraction: confidence={new_confidence:.0%} "
            f"(was {prev_confidence:.0%}), missing={new_missing}"
        )

        # Only update if confidence improved
        if new_confidence > prev_confidence:
            state["ml_features"] = new_features
            state["feature_extraction_result"] = new_extraction_result
            logger.info("[REEXTRACT] Features updated with improved extraction")
        else:
            logger.info("[REEXTRACT] Keeping previous extraction (no improvement)")

    except Exception as e:
        logger.warning(f"[REEXTRACT] Feature re-extraction failed: {e}")

    return state


def generate_pr_text(parsed_pr: dict, answers: dict) -> str:
    """
    Generate text representation of PR for embedding.

    Uses UNIFIED format from pr_embedding_text module to ensure
    identical text construction for both indexing and querying.
    This solves the query-document mismatch problem.
    """
    from utils.pr_embedding_text import build_pr_embedding_text

    # Use the unified embedding text builder
    return build_pr_embedding_text(parsed_pr)


async def store_pr_embedding(
    session_id: str,
    embedding: list[float],
    parsed_pr: dict,
) -> None:
    """
    Store PR embedding in Qdrant for future similarity searches.
    """
    try:
        from vector.client import get_vector_store

        vector_store = await get_vector_store()

        payload = {
            "pr_code": parsed_pr.get("pr_code", "Unknown"),
            "title": parsed_pr.get("title", "Unknown"),
            "program_family": parsed_pr.get("program_family"),
            "customer": parsed_pr.get("customer"),
            "total_hours": 0,  # Will be updated after estimation
            "total_cost_eur": 0,  # Will be updated after estimation
        }

        await vector_store.upsert(
            collection="pr_embeddings",
            id=session_id,
            vector=embedding,
            payload=payload,
        )

    except Exception as e:
        import logging

        logging.warning(f"Failed to store PR embedding: {e}")


async def find_similar_prs(
    embedding: list[float],
    parsed_pr: dict,
    k: int = 5,
) -> list[SimilarPR]:
    """
    Find similar PRs using HYBRID search: metadata filters + vector similarity + reranking.

    Implements cascading filter strategy:
    - Level 1: MUST sector + SHOULD sizing proximity
    - Level 2: MUST sector only (if Level 1 < k results)
    - Level 3: Pure vector search (if Level 2 < k results)

    Final ranking uses ensemble: 0.6 * feature_score + 0.4 * vector_score
    """
    import logging

    from utils.pr_embedding_text import (
        calculate_ensemble_score,
        calculate_feature_similarity,
        get_sizing_filter_values,
    )

    logger = logging.getLogger(__name__)
    similar: list[SimilarPR] = []

    try:
        from vector.client import get_vector_store

        vector_store = await get_vector_store()

        # Extract filter values from parsed PR
        query_sector = parsed_pr.get("sector") or parsed_pr.get("Sector")
        query_sizing = (
            parsed_pr.get("sizing")
            or parsed_pr.get("Sizing")
            or parsed_pr.get("sizing_program")
        )

        results = []

        # Level 1: Sector + Sizing proximity filter
        if query_sector:
            filter_conditions = {"Sector": query_sector}
            if query_sizing:
                sizing_values = get_sizing_filter_values(query_sizing)
                filter_conditions["Sizing"] = sizing_values

            results = await vector_store.search(
                collection="pr_embeddings",
                query_vector=embedding,
                limit=k * 2,  # Get more for reranking
                filter_conditions=filter_conditions,
                score_threshold=0.4,
            )
            logger.debug(f"Level 1 (sector+sizing) returned {len(results)} results")

        # Level 2: Sector only filter (if Level 1 insufficient)
        if len(results) < k and query_sector:
            filter_conditions = {"Sector": query_sector}
            results = await vector_store.search(
                collection="pr_embeddings",
                query_vector=embedding,
                limit=k * 2,
                filter_conditions=filter_conditions,
                score_threshold=0.4,
            )
            logger.debug(f"Level 2 (sector only) returned {len(results)} results")

        # Level 3: Pure vector search (fallback)
        if len(results) < k:
            results = await vector_store.search(
                collection="pr_embeddings",
                query_vector=embedding,
                limit=k * 2,
                score_threshold=0.4,
            )
            logger.debug(f"Level 3 (pure vector) returned {len(results)} results")

        # Rerank using ensemble scoring
        ranked_results = []
        for result in results:
            payload = result.get("payload", {})
            vector_score = result["score"]

            # Calculate feature similarity
            feature_score = calculate_feature_similarity(parsed_pr, payload)

            # Ensemble score: 0.6 * feature + 0.4 * vector
            ensemble_score = calculate_ensemble_score(vector_score, feature_score)

            ranked_results.append(
                {
                    "result": result,
                    "payload": payload,
                    "vector_score": vector_score,
                    "feature_score": feature_score,
                    "ensemble_score": ensemble_score,
                }
            )

        # Sort by ensemble score (descending)
        ranked_results.sort(key=lambda x: x["ensemble_score"], reverse=True)

        # Build SimilarPR objects from top-k reranked results
        for item in ranked_results[:k]:
            payload = item["payload"]
            similar_pr: SimilarPR = {
                "id": item["result"]["id"],
                "pr_code": payload.get("pr_id") or payload.get("pr_code", "Unknown"),
                "title": payload.get("title")
                or payload.get("Customer_Platform", "Unknown"),
                "program_family": payload.get("Product_Family")
                or payload.get("program_family"),
                "similarity_score": item["ensemble_score"],  # Use ensemble score
                "total_hours": payload.get("Manpower") or payload.get("total_hours", 0),
                "total_cost_eur": payload.get("Cost")
                or payload.get("total_cost_eur", 0),
                # Enriched payload fields
                "customer_platform": payload.get("Customer_Platform"),
                "sector": payload.get("Sector") or payload.get("sector"),
                "sizing": payload.get("Sizing") or payload.get("program_size"),
                "emissions": payload.get("Emissions"),
                # R&D breakdown for CBR context (PE02 function-level hours)
                "rd_breakdown": payload.get("rd_breakdown"),
            }
            similar.append(similar_pr)

        logger.info(
            f"Hybrid search: {len(similar)} PRs found "
            f"(sector={query_sector}, sizing={query_sizing})"
        )

    except Exception as e:
        import logging

        logging.warning(f"Failed to find similar PRs: {e}")

    return similar


async def extract_ml_features(
    parsed_pr: dict,
    answers: dict,
    similar_prs: list[SimilarPR],
    llm,
) -> tuple[list[MLFeature], dict]:
    """
    Extract features for ML prediction using HCQE-compatible feature extractor.

    This uses the new hybrid approach:
    1. LLM extracts domain-specific features (turbo, injectors, ATS, etc.)
    2. Rule-based fallback for keyword matching
    3. Similar PRs provide context for sizing estimation

    Returns:
        Tuple of (features list, extraction_result dict with confidence/missing)
    """
    import logging
    from ml.hcqe_feature_extractor import extract_hcqe_features, features_to_list

    logger = logging.getLogger(__name__)

    # Default extraction result for feature completion mechanism
    extraction_result = {
        "confidence": 1.0,
        "missing_features": [],
        "extraction_method": "unknown",
    }

    # Convert similar_prs to dict format for feature extractor
    similar_prs_dict = (
        [
            {
                "pr_code": sp.get("pr_code", ""),
                "total_hours": sp.get("total_hours", 0),
                "total_cost_keur": sp.get("total_cost_keur", 0),
            }
            for sp in similar_prs
        ]
        if similar_prs
        else []
    )

    # Extract HCQE-compatible features using LLM + rules
    try:
        hcqe_result = await extract_hcqe_features(
            parsed_pr=parsed_pr,
            qa_answers=answers,
            similar_prs=similar_prs_dict,
            llm=llm,
        )

        logger.info(
            f"HCQE feature extraction: {hcqe_result.extraction_method}, "
            f"confidence={hcqe_result.confidence:.0%}, "
            f"{len(hcqe_result.features)} features, "
            f"missing={len(hcqe_result.missing_features)}"
        )

        # Store extraction result for Q&A feature completion
        extraction_result = {
            "confidence": hcqe_result.confidence,
            "missing_features": hcqe_result.missing_features,
            "extraction_method": hcqe_result.extraction_method,
        }

        # Convert to MLFeature list format
        features = [
            MLFeature(
                name=name,
                value=value,
                source=f"hcqe_{hcqe_result.extraction_method}",
            )
            for name, value in hcqe_result.features.items()
        ]

        # Calculate activity_count from HCQE features (for backward compatibility)
        # Count technical changes as "activities"
        activity_count = sum(
            [
                1 if hcqe_result.features.get("hardware_change", 0) else 0,
                1 if hcqe_result.features.get("calibration_change", 0) else 0,
                1 if hcqe_result.features.get("ATS_change", 0) else 0,
                1 if hcqe_result.features.get("software_VCU_change", 0) else 0,
                1 if hcqe_result.features.get("turbo_related", 0) else 0,
                1 if hcqe_result.features.get("injectors_related", 0) else 0,
                1 if hcqe_result.features.get("EGR_related", 0) else 0,
                1 if hcqe_result.features.get("cooling_related", 0) else 0,
            ]
        )
        # Add num_functions if available (better estimate)
        num_functions = hcqe_result.features.get("num_functions", 0)
        if num_functions > activity_count:
            activity_count = num_functions
        # Minimum of 1 if we have any features
        if activity_count == 0 and len(hcqe_result.features) > 0:
            activity_count = max(5, len(hcqe_result.features) // 3)

        features.append(
            MLFeature(
                name="activity_count",
                value=activity_count,
                source="hcqe_calculated",
            )
        )

        # Add extraction metadata
        features.append(
            MLFeature(
                name="_extraction_method",
                value=hcqe_result.extraction_method,
                source="metadata",
            )
        )
        features.append(
            MLFeature(
                name="_extraction_confidence",
                value=hcqe_result.confidence,
                source="metadata",
            )
        )
        features.append(
            MLFeature(
                name="_missing_features",
                value=hcqe_result.missing_features,
                source="metadata",
            )
        )

        return features, extraction_result

    except Exception as e:
        logger.warning(f"HCQE feature extraction failed: {e}, using legacy extraction")

        # Fallback extraction result with low confidence
        extraction_result = {
            "confidence": 0.3,
            "missing_features": [
                "hardware_change",
                "calibration_change",
                "ATS_change",
                "software_VCU_change",
                "emissions",
                "sector",
            ],
            "extraction_method": "fallback",
        }

        # Fallback to basic features if HCQE extraction fails
        features: list[MLFeature] = []

        features.append(
            MLFeature(
                name="activity_count",
                value=len(parsed_pr.get("raw_activities", [])),
                source="extracted",
            )
        )

        if similar_prs:
            avg_hours = sum(sp["total_hours"] for sp in similar_prs) / len(similar_prs)
            features.append(
                MLFeature(
                    name="similar_avg_hours",
                    value=avg_hours,
                    source="similar_prs",
                )
            )
            # Estimate sizing from similar PRs average
            avg_cost = sum(sp.get("total_cost_keur", 0) for sp in similar_prs) / len(
                similar_prs
            )
            if avg_cost < 100:
                sizing = 0
            elif avg_cost < 300:
                sizing = 1
            elif avg_cost < 800:
                sizing = 2
            elif avg_cost < 1500:
                sizing = 3
            else:
                sizing = 4
            features.append(
                MLFeature(
                    name="sizing_program",
                    value=sizing,
                    source="similar_prs",
                )
            )

        return features, extraction_result


async def generate_summary(
    parsed_pr: dict,
    answers: dict,
    similar_prs: list[SimilarPR],
    ml_features: list[MLFeature],
    llm,
) -> PRSummary:
    """Generate comprehensive PR summary using LLM."""
    activities = parsed_pr.get("raw_activities", [])

    # Build context for summary
    context = {
        "pr_code": parsed_pr.get("pr_code", "Unknown"),
        "title": parsed_pr.get("title", "Unknown"),
        "program_family": parsed_pr.get("program_family", "Unknown"),
        "customer": parsed_pr.get("customer", "Unknown"),
        "activities": activities,
        "qa_answers": answers,
        "similar_prs": [
            {"pr_code": sp["pr_code"], "total_hours": sp["total_hours"]}
            for sp in similar_prs
        ],
    }

    # Get activity count and complexity from ml_features first (used in all paths)
    feat_dict = {f["name"]: f["value"] for f in ml_features}
    activity_count = int(feat_dict.get("activity_count", 0))
    program_size = classify_program_size(ml_features, similar_prs)
    complexity_score = calculate_complexity_score(ml_features)

    try:
        # Build similar PRs context for prediction reference
        similar_prs_context = ""
        if similar_prs:
            similar_prs_context = "\n".join(
                [
                    f"  - {sp['pr_code']}: {sp['total_hours']} hours estimated"
                    for sp in similar_prs[:3]
                    if sp.get("total_hours", 0) > 0
                ]
            )
            if not similar_prs_context:
                similar_prs_context = (
                    "  (Similar PRs found but hours data not available)"
                )

        # Handle case when LLM is not available
        if llm is None:
            summary_text = f"PR {context.get('pr_code', 'Unknown')}: {context.get('title', 'No title')}"
        else:
            # Generate summary text - focused on PREDICTION purpose
            prompt = f"""You are an AI assistant for FPT Cost Brain, an R&D cost estimation tool.

Your task is to create an EXECUTIVE SUMMARY for a customer manager reviewing this Product Request (PR).

**IMPORTANT CONTEXT:**
- This tool PREDICTS engineering effort (hours, cost, breakdown) based on the PR description
- The breakdown of activities is GENERATED by our ML system, NOT expected to be in the PR
- Similar historical PRs help calibrate our predictions

**Product Request Details:**
- PR Code: {context.get("pr_code", "Unknown")}
- Title: {context.get("title", "Unknown")}
- Customer: {context.get("customer", "Unknown")}
- Program Family: {context.get("program_family", "Unknown")}

**Q&A Clarifications Provided:**
{context.get("qa_answers", "None provided")}

**Similar Historical PRs (for prediction reference):**
{similar_prs_context if similar_prs_context else "  No similar PRs found"}

**Write an executive summary that includes:**
1. **Project Overview**: What this PR is requesting (scope, objectives)
2. **Technical Scope**: What engineering work will likely be involved
3. **Prediction Basis**: How similar projects inform our estimate
4. **Key Considerations**: Important factors for the cost estimate

Keep it concise (2-3 paragraphs) and professional for management review.
Do NOT say activities are "missing" - our system will PREDICT them."""

            summary_text = await llm.reason(
                prompt=prompt,
                system_prompt=PR_SUMMARY_GENERATION,
            )

        summary: PRSummary = {
            "summary_text": summary_text,
            "program_size": program_size,
            "complexity_score": round(
                complexity_score, 2
            ),  # Round to avoid floating point issues
            "activity_count": activity_count,
            "key_features": extract_key_features(parsed_pr, ml_features),
            "dependencies": extract_dependencies(parsed_pr, answers),
            "risk_factors": extract_risk_factors(parsed_pr, answers),
            "special_requirements": extract_special_requirements(parsed_pr, answers),
        }

        return summary

    except Exception as e:
        # Return basic summary on error, but preserve calculated values
        import logging

        logging.warning(f"Summary generation failed: {e}")
        return PRSummary(
            summary_text=f"PR {parsed_pr.get('pr_code', 'Unknown')}: {parsed_pr.get('title', 'No title')}",
            program_size=program_size,  # Use pre-calculated value
            complexity_score=round(complexity_score, 2),  # Use pre-calculated value
            activity_count=activity_count,  # Use pre-calculated value
            key_features=[],
            dependencies=[],
            risk_factors=["Summary generation failed"],
            special_requirements=[],
        )


def classify_program_size(
    features: list[MLFeature],
    similar_prs: list[SimilarPR],
) -> str:
    """Classify program size based on HCQE features and similar PRs."""
    # Build feature dict for easier access
    feat_dict = {f["name"]: f["value"] for f in features}

    # Primary: use sizing_program from HCQE extraction if available
    sizing_program = feat_dict.get("sizing_program")
    if sizing_program is not None:
        sizing_val = sizing_program
        if isinstance(sizing_val, str):
            sizing_map = {
                "x-small": 0,
                "small": 1,
                "mid": 2,
                "medium": 2,
                "large": 3,
                "full": 4,
            }
            sizing_val = sizing_map.get(sizing_val.lower(), 2)
        if sizing_val == 0:
            return "x-small"
        elif sizing_val == 1:
            return "small"
        elif sizing_val == 2:
            return "medium"
        elif sizing_val == 3:
            return "large"
        else:
            return "xl"

    # Secondary: use num_functions or activity_count
    num_functions = feat_dict.get("num_functions", 0)
    activity_count = feat_dict.get("activity_count", 0)
    scope_indicator = max(num_functions, activity_count)

    # Tertiary: get average hours from similar PRs
    avg_similar_hours = 0
    if similar_prs:
        avg_similar_hours = sum(sp["total_hours"] for sp in similar_prs) / len(
            similar_prs
        )

    # Classification logic combining scope and similar hours
    if scope_indicator > 50 or avg_similar_hours > 5000:
        return "xl"
    elif scope_indicator > 30 or avg_similar_hours > 2000:
        return "large"
    elif scope_indicator > 15 or avg_similar_hours > 500:
        return "medium"
    elif scope_indicator > 5 or avg_similar_hours > 100:
        return "small"
    else:
        return "x-small"


def calculate_complexity_score(features: list[MLFeature]) -> float:
    """Calculate complexity score from HCQE features."""
    feat_dict = {f["name"]: f["value"] for f in features}

    # Base score from sizing_program
    sizing_program = feat_dict.get("sizing_program")
    if sizing_program is not None:
        sizing_val = sizing_program
        if isinstance(sizing_val, str):
            sizing_map = {
                "x-small": 0.2,
                "small": 0.35,
                "mid": 0.5,
                "medium": 0.5,
                "large": 0.7,
                "full": 0.9,
            }
            return sizing_map.get(sizing_val.lower(), 0.5)
        else:
            # Numeric 0-4 scale
            return min(1.0, 0.2 + (sizing_val * 0.2))

    # Calculate from component complexity scores
    complexity_components = []

    # Count active technical changes (each adds to complexity)
    for change_type in [
        "hardware_change",
        "calibration_change",
        "ATS_change",
        "software_VCU_change",
        "turbo_related",
        "injectors_related",
    ]:
        if feat_dict.get(change_type, 0):
            complexity_components.append(0.1)

    # Add component complexity scores
    for complexity_key in [
        "hardware_complexity",
        "calibration_complexity",
        "ats_complexity",
        "software_complexity",
    ]:
        val = feat_dict.get(complexity_key, 0)
        if val:
            complexity_components.append(val * 0.1)  # 1=0.1, 2=0.2, 3=0.3

    # Use num_functions as complexity indicator
    num_functions = feat_dict.get("num_functions", 0)
    if num_functions > 0:
        complexity_components.append(min(0.3, num_functions / 100))

    # Activity count fallback
    activity_count = feat_dict.get("activity_count", 0)
    if activity_count > 0 and num_functions == 0:
        complexity_components.append(min(0.3, activity_count / 50))

    if complexity_components:
        return min(1.0, max(0.2, sum(complexity_components)))

    return 0.5  # Default medium complexity


def extract_key_features(parsed_pr: dict, ml_features: list[MLFeature]) -> list[str]:
    """Extract key features as list of strings."""
    features = []

    if parsed_pr.get("program_family"):
        features.append(f"Program: {parsed_pr['program_family']}")

    activities = parsed_pr.get("raw_activities", [])
    if activities:
        features.append(f"{len(activities)} activities identified")

    for f in ml_features:
        if f["name"] == "mentioned_hours" and f["value"] > 0:
            features.append(f"Mentioned hours: {f['value']}")

    return features


def extract_dependencies(parsed_pr: dict, answers: dict) -> list[str]:
    """Extract dependencies from PR and answers."""
    dependencies = []

    # Check answers for dependency information
    for q_id, answer in answers.items():
        if "depend" in q_id.lower() or "depend" in answer.lower():
            dependencies.append(answer)

    return dependencies


def extract_risk_factors(parsed_pr: dict, answers: dict) -> list[str]:
    """Extract risk factors from PR and answers."""
    risks = []

    # Check for timeline risks
    for q_id, answer in answers.items():
        if "urgent" in answer.lower() or "deadline" in answer.lower():
            risks.append("Timeline pressure")
            break

    return risks


def extract_special_requirements(parsed_pr: dict, answers: dict) -> list[str]:
    """Extract special requirements from PR and answers."""
    requirements = []

    if parsed_pr.get("project_phase"):
        requirements.append(f"Phase: {parsed_pr['project_phase']}")

    return requirements

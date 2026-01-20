"""
FPT Cost Brain 2.0 - Estimation Service
Service layer for the estimation workflow, integrating LangGraph with API
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
from fastapi import HTTPException
from agents.graph import EstimationGraph, create_estimation_graph, create_simple_graph
from agents.state import BreakdownItem, EstimationState, Question, create_initial_state
from app.config import settings
from app.debug_logging import (
    log_step,
    log_state_transition,
    log_parsed_pr,
    log_questions,
    log_breakdown,
    log_error_details,
    log_redis_operation,
)
from db.repositories.pr_repo import ProductRequestRepository
from db.repositories.quotation_repo import QuotationRepository
from services.rlhf_service import PreferencePairService
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Redis client for state storage (thread-safe singleton)
_redis_client: redis.Redis | None = None
_redis_lock = asyncio.Lock()


async def get_redis() -> redis.Redis:
    """Get or create Redis client (thread-safe)."""
    global _redis_client
    if _redis_client is None:
        async with _redis_lock:
            # Double-check pattern to avoid race condition
            if _redis_client is None:
                _redis_client = redis.from_url(
                    str(settings.REDIS_URL),
                    encoding="utf-8",
                    decode_responses=True,
                )
    return _redis_client


def _sanitize_floats(obj: Any) -> Any:
    """Recursively sanitize floats to handle NaN/Inf values."""
    import math

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0  # Replace NaN/Inf with 0
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_floats(v) for v in obj)
    return obj


async def save_state_to_redis(session_id: str, state: dict[str, Any]) -> None:
    """Save estimation state to Redis."""
    try:
        client = await get_redis()

        # Create a copy and remove non-serializable fields
        # pr_file_bytes is only needed during intake and is large
        # embedding is very large (4096 floats) - exclude to save space
        exclude_keys = ("pr_file_bytes", "embedding")
        state_to_save = {k: v for k, v in state.items() if k not in exclude_keys}

        # Ensure step_status enum values are converted to strings properly
        # StepStatus enum's str() returns "StepStatus.COMPLETED", but we need "completed"
        if "step_status" in state_to_save and state_to_save["step_status"]:
            state_to_save["step_status"] = {
                step: (status.value if hasattr(status, "value") else str(status))
                for step, status in state_to_save["step_status"].items()
            }

        # Sanitize floats to handle NaN/Inf values
        state_to_save = _sanitize_floats(state_to_save)

        # Convert state to JSON-serializable format
        state_json = json.dumps(state_to_save, default=str)
        # Store with 24 hour expiry
        await client.setex(f"estimation:{session_id}", 86400, state_json)
        logger.debug(f"Saved state to Redis for session {session_id}")
    except Exception as e:
        logger.exception(f"Failed to save state to Redis for session {session_id}: {e}")
        # Don't raise - Redis failure shouldn't block the workflow
        # State will be lost on restart but user can continue


async def load_state_from_redis(session_id: str) -> dict[str, Any] | None:
    """Load estimation state from Redis."""
    client = await get_redis()
    state_json = await client.get(f"estimation:{session_id}")
    if state_json:
        state = json.loads(state_json)
        # Debug: Log step_status when loading
        step_status = state.get("step_status", {})
        logger.info(f"[REDIS_LOAD] session={session_id}, step_status={step_status}")
        return state
    return None


class EstimationService:
    """
    Service for managing estimation sessions.

    This service:
    - Creates and manages LangGraph sessions
    - Persists state to PostgreSQL
    - Provides API-friendly methods for the estimation workflow
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.pr_repo = ProductRequestRepository(db)
        self.quotation_repo = QuotationRepository(db)
        self._graph: EstimationGraph | None = None

    async def _get_graph(self) -> EstimationGraph:
        """Get or create the LangGraph instance."""
        if self._graph is None:
            # For now, use simple graph without PostgreSQL checkpointing
            # This avoids connection issues and allows testing the flow
            # TODO: Re-enable PostgreSQL checkpointing once flow is verified
            logger.info("Creating simple estimation graph (no persistence)")
            self._graph = create_simple_graph()

            # Future: Enable PostgreSQL checkpointing
            # db_url = str(settings.DATABASE_URL).replace("+asyncpg", "")
            # self._graph = await create_estimation_graph(db_url)

        # Set the database session for nodes that need it (learning, rules)
        self._graph.set_db_session(self.db)

        return self._graph

    async def start_session(
        self,
        user_id: str,
        file_bytes: bytes | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Start a new estimation session.

        Args:
            user_id: The ID of the user starting the session
            file_bytes: Optional PR Excel file content
            filename: Original filename of the uploaded file

        Returns:
            Session information including session_id and current step
        """
        session_id = str(uuid.uuid4())
        logger.info(f"Starting estimation session: {session_id} for user: {user_id}")

        # Create initial state
        initial_state = create_initial_state(session_id, user_id)
        logger.debug(f"Initial state created with keys: {list(initial_state.keys())}")

        # Add file data if provided
        if file_bytes and filename:
            initial_state["pr_file_bytes"] = file_bytes
            initial_state["pr_filename"] = filename
            logger.info(f"File attached: {filename} ({len(file_bytes)} bytes)")

        # Get the graph
        logger.info("Getting estimation graph...")
        graph = await self._get_graph()
        logger.info("Graph obtained successfully")

        # Run the graph - it will process intake and qa_generate, then pause at qa_wait
        config = {"configurable": {"thread_id": session_id}}

        try:
            logger.info("Running graph...")
            result_state = await graph.run(initial_state, config)

            # Handle None result from graph
            if result_state is None:
                logger.warning("Graph returned None result, using initial state")
                result_state = initial_state

            logger.info(
                f"Graph run completed. Current step: {result_state.get('current_step')}"
            )

            # Get parsed_pr safely
            parsed_pr = result_state.get("parsed_pr") or {}

            # Generate a unique PR code
            base_pr_code = parsed_pr.get("pr_code") or ""
            # Validate pr_code - must be non-empty and not look like a column header
            invalid_pr_codes = {"name", "title", "pr", "code", "number", "id", ""}
            if base_pr_code.lower().strip() in invalid_pr_codes:
                base_pr_code = ""

            if base_pr_code:
                # Check if this pr_code already exists, append timestamp if so
                existing = await self.pr_repo.get_by_code(base_pr_code)
                if existing:
                    import time

                    base_pr_code = f"{base_pr_code}_{int(time.time())}"
            else:
                # Generate from session_id if no valid pr_code
                base_pr_code = f"PR-{session_id[:8].upper()}"

            # Save PR to database
            logger.info(f"Saving PR to database with code: {base_pr_code}")
            pr = await self.pr_repo.create(
                pr_code=base_pr_code,
                title=parsed_pr.get("title") or filename or "Demo Estimation",
                raw_data=parsed_pr.get("raw_data") or {},
                uploaded_by=uuid.UUID(user_id),
            )
            logger.info(f"PR saved with ID: {pr.id}")

            # Update state with PR ID
            result_state["pr_id"] = str(pr.id)

            # Determine current step based on where graph paused
            current_step = result_state.get("current_step", "qa")

            # Update PR status
            await self.pr_repo.update_status(pr.id, current_step)

            # Build response with full state
            # LAZY LOADING: Questions are not generated yet at this point
            # They will be generated when user enters Q&A step
            questions = result_state.get("questions", [])
            questions_ready = len(questions) > 0

            # Ensure step_status has proper values after graph run
            step_status = result_state.get("step_status", {})
            # Convert any enum values to strings (in case nodes used enums)
            step_status = {
                step: (status.value if hasattr(status, "value") else str(status))
                for step, status in step_status.items()
            }
            logger.info(f"[START_SESSION] step_status after graph: {step_status}")

            response = {
                "session_id": session_id,
                "pr_id": str(pr.id),
                "current_step": current_step,
                "status": "active",
                "created_at": result_state.get("created_at"),
                "parsed_pr": result_state.get("parsed_pr"),
                "questions": questions,
                "questions_ready": questions_ready,  # LAZY LOADING flag
                "questions_generating": False,
                "answers": result_state.get("answers", {}),
                "pr_summary": result_state.get("pr_summary"),
                "similar_prs": result_state.get("similar_prs", []),
                "breakdown": result_state.get("breakdown", []),
                "total_hours": result_state.get("total_hours", 0),
                "total_cost_eur": result_state.get("total_cost_eur", 0),
                "overall_confidence": result_state.get("overall_confidence", 0),
                "is_valid": result_state.get("is_valid", False),
                "error_message": result_state.get("error_message"),
                "step_status": step_status,  # Include step_status in response
            }

            # Save full state to Redis for later retrieval
            result_state["pr_id"] = str(pr.id)
            await save_state_to_redis(session_id, result_state)
            logger.info(f"Session state saved to Redis: {session_id}")

            return response
        except Exception as e:
            logger.exception(f"Error during session start: {e}")
            # Handle graph execution errors
            return {
                "session_id": session_id,
                "pr_id": None,
                "current_step": "error",
                "status": "error",
                "error_message": str(e),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """
        Get the current state of an estimation session.

        Args:
            session_id: The session ID (thread_id for LangGraph)

        Returns:
            Current session state or None if not found
        """
        # Try to load from Redis first
        state = await load_state_from_redis(session_id)

        if not state:
            # Fallback to LangGraph checkpointer if available
            graph = await self._get_graph()
            state = await graph.get_state(session_id)

        if not state:
            logger.warning(f"Session not found: {session_id}")
            return None

        return self._state_to_response(state)

    async def generate_questions(self, session_id: str) -> dict[str, Any]:
        """
        Generate Q&A questions for a session (LAZY LOADING).

        This is called when user enters the Q&A step, not during upload.
        This enables fast upload experience while deferring expensive LLM calls.

        Args:
            session_id: The session ID

        Returns:
            Updated session state with generated questions
        """
        logger.info(f"[GENERATE_QUESTIONS] Starting for session {session_id}")

        # Load current state from Redis
        state = await load_state_from_redis(session_id)
        if not state:
            logger.error(
                f"[GENERATE_QUESTIONS] Session {session_id} not found in Redis"
            )
            raise ValueError(f"Session {session_id} not found")

        logger.info(f"[GENERATE_QUESTIONS] Loaded state, keys: {list(state.keys())}")

        # Ensure step_status dict exists (for sessions created before this feature)
        if "step_status" not in state or not state["step_status"]:
            state["step_status"] = {
                "intake": "completed",  # Must be completed to get here
                "qa": "waiting_input",
                "summary": "pending",
                "estimation": "pending",
                "review": "pending",
                "export": "pending",
            }
            logger.info(
                f"[GENERATE_QUESTIONS] Initialized step_status for session {session_id}"
            )

        # Check if questions already generated
        existing_questions = state.get("questions", [])
        if existing_questions and len(existing_questions) > 0:
            logger.info(
                f"[GENERATE_QUESTIONS] Questions already exist ({len(existing_questions)}), returning early"
            )
            return self._state_to_response(state)

        # Verify parsed_pr exists
        parsed_pr = state.get("parsed_pr")
        if not parsed_pr:
            logger.error(
                f"[GENERATE_QUESTIONS] No parsed_pr in state! This is required for question generation."
            )
            raise ValueError("No parsed PR data found. Please upload a PR file first.")

        logger.info(
            f"[GENERATE_QUESTIONS] parsed_pr title: {parsed_pr.get('title', 'Unknown')}"
        )

        # Check we're in the right step (qa_wait is the LAZY LOADING state after upload)
        current_step = state.get("current_step", "")
        if current_step not in ("qa", "qa_wait", "intake"):
            logger.warning(
                f"[GENERATE_QUESTIONS] Wrong step: {current_step}, but proceeding anyway"
            )

        try:
            # Import and run the question generation node directly
            from agents.nodes.qa_node import generate_questions as qa_generate

            logger.info(f"[GENERATE_QUESTIONS] Calling qa_generate...")

            # Store original question count for comparison
            original_count = len(state.get("questions", []))

            # Run question generation
            logger.info("[GENERATE_QUESTIONS] Calling qa_generate with state...")
            updated_state = await qa_generate(state)
            logger.info(
                f"[GENERATE_QUESTIONS] qa_generate returned, questions: {len(updated_state.get('questions', []))}"
            )

            # Ensure we're in Q&A step
            updated_state["current_step"] = "qa"
            updated_state["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Save updated state to Redis
            await save_state_to_redis(session_id, updated_state)
            logger.info("[GENERATE_QUESTIONS] Saved to Redis")

            questions_count = len(updated_state.get("questions", []))
            if original_count > 0 and questions_count != original_count:
                logger.warning(
                    f"[GENERATE_QUESTIONS] Question count changed from {original_count} to {questions_count}"
                )
            else:
                logger.info(
                    f"[GENERATE_QUESTIONS] SUCCESS: Generated {questions_count} questions"
                )

            return self._state_to_response(updated_state)

        except Exception as e:
            logger.exception(f"[GENERATE_QUESTIONS] ERROR: {type(e).__name__}: {e}")
            # Return current state with error - but DO NOT use hardcoded questions
            state["error_message"] = f"Failed to generate questions: {str(e)}"
            # Try to provide a helpful message
            if "API" in str(e) or "key" in str(e).lower():
                state["error_message"] = (
                    "LLM API error - please check API key configuration"
                )
            return self._state_to_response(state)

    async def submit_answers(
        self,
        session_id: str,
        answers: dict[str, str],
    ) -> dict[str, Any]:
        """
        Submit answers to Q&A questions and advance to next step.

        Args:
            session_id: The session ID
            answers: Dict of question_id -> answer

        Returns:
            Updated session state
        """
        # Load current state from Redis
        state = await load_state_from_redis(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Update state with answers
        state["answers"] = answers
        state["qa_complete"] = True

        try:
            # Run only summary node - estimation will run on next step advance
            from agents.nodes.summary_node import process_summary

            # Run summary node
            logger.info(f"Running summary node for session {session_id}")
            state = await process_summary(state)

            # Stop at summary step - user will see summary before estimation
            state["current_step"] = "summary"

            # Update PR status in database
            if state.get("pr_id"):
                await self.pr_repo.update_status(
                    uuid.UUID(state["pr_id"]),
                    "summary",
                )

            # Save updated state to Redis
            await save_state_to_redis(session_id, state)

            return self._state_to_response(state)
        except Exception as e:
            logger.exception(f"Error submitting answers: {e}")
            raise

    async def skip_qa(self, session_id: str) -> dict[str, Any]:
        """
        Skip Q&A and proceed to summary.

        Args:
            session_id: The session ID

        Returns:
            Updated session state
        """
        # Load current state from Redis
        state = await load_state_from_redis(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Mark Q&A as skipped
        state["qa_skipped"] = True
        state["qa_complete"] = True

        # Get the graph and continue execution
        graph = await self._get_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            result_state = await graph.run(state, config)

            if result_state is None:
                result_state = state

            # Update PR status
            if result_state.get("pr_id"):
                await self.pr_repo.update_status(
                    uuid.UUID(result_state["pr_id"]),
                    result_state.get("current_step", "summary"),
                )

            # Save updated state to Redis
            await save_state_to_redis(session_id, result_state)

            return self._state_to_response(result_state)
        except Exception as e:
            logger.exception(f"Error skipping Q&A: {e}")
            raise

    async def regenerate_questions(
        self,
        session_id: str,
        user_message: str,
    ) -> dict[str, Any]:
        """
        Regenerate questions dynamically based on chat message.

        This enables interactive Q&A where questions update in real-time
        as the user provides information through chat.

        Args:
            session_id: The session ID
            user_message: Chat message that triggers question update

        Returns:
            Updated session state with new questions
        """
        # Load current state from Redis
        state = await load_state_from_redis(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Only regenerate if we're in the Q&A step
        current_step = state.get("current_step", "")
        if current_step not in ("qa", "intake"):
            logger.warning(f"Cannot regenerate questions in step: {current_step}")
            return self._state_to_response(state)

        try:
            from agents.nodes.qa_node import regenerate_questions_from_chat

            # Call the question regeneration function
            logger.info(f"Regenerating questions for session {session_id}")
            new_questions = await regenerate_questions_from_chat(state, user_message)

            # Update state with new questions
            state["questions"] = new_questions
            state["updated_at"] = datetime.now(timezone.utc).isoformat()

            # Save updated state to Redis
            await save_state_to_redis(session_id, state)

            logger.info(f"Questions regenerated: {len(new_questions)} questions")
            return self._state_to_response(state)
        except Exception as e:
            logger.exception(f"Error regenerating questions: {e}")
            raise

    async def advance_step(
        self,
        session_id: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Advance to the next step in the workflow.

        Directly calls the appropriate node processing function for efficiency.

        Args:
            session_id: The session ID
            data: Optional data to include in state update

        Returns:
            Updated session state
        """
        step_start_time = time.time()
        # Use print for guaranteed visibility
        print("=" * 70, flush=True)
        print(f"🚀 ADVANCE_STEP called for session: {session_id[:8]}...", flush=True)
        print("=" * 70, flush=True)
        logger.info("=" * 70)
        logger.info(f"🚀 ADVANCE_STEP called for session: {session_id[:8]}...")
        logger.info("=" * 70)

        # Load current state from Redis
        print("📥 Loading state from Redis...", flush=True)
        logger.info("📥 Loading state from Redis...")
        state = await load_state_from_redis(session_id)
        if not state:
            print(f"❌ Session {session_id} not found in Redis!", flush=True)
            logger.error(f"❌ Session {session_id} not found in Redis!")
            raise ValueError(f"Session {session_id} not found")

        print(f"✅ State loaded from Redis in {time.time() - step_start_time:.2f}s", flush=True)
        logger.info(f"✅ State loaded from Redis in {time.time() - step_start_time:.2f}s")
        log_state_transition(logger, state, "CURRENT STATE")

        current_step = state.get("current_step", "intake")
        updates = data or {}
        print(f"📍 Current step: {current_step}", flush=True)
        print(f"📝 Updates provided: {list(updates.keys()) if updates else 'None'}", flush=True)
        logger.info(f"📍 Current step: {current_step}")
        logger.info(f"📝 Updates provided: {list(updates.keys()) if updates else 'None'}")

        # Apply any provided updates
        state.update(updates)

        # Ensure step_status dict exists (for sessions created before this feature)
        if "step_status" not in state or not state["step_status"]:
            state["step_status"] = {
                "intake": "completed",  # Must be completed to get here
                "qa": "pending",
                "summary": "pending",
                "estimation": "pending",
                "review": "pending",
                "export": "pending",
            }
            logger.info(f"[ADVANCE] Initialized step_status for session {session_id}")

        try:
            # Handle step transitions directly (more efficient than running graph)
            if current_step in ("qa", "qa_wait"):
                print("=" * 50, flush=True)
                print("▶ TRANSITION: Q&A → Summary", flush=True)
                print("=" * 50, flush=True)
                logger.info("=" * 50)
                logger.info("▶ TRANSITION: Q&A → Summary")
                logger.info("=" * 50)
                # Advancing from Q&A -> run summary
                from agents.nodes.summary_node import process_summary

                # Mark Q&A as completed before advancing
                state["step_status"]["qa"] = "completed"
                state["qa_complete"] = True
                print(f"⏳ Running summary node for session {session_id[:8]}...", flush=True)
                logger.info(f"[ADVANCE] Running summary node for session {session_id}")
                try:
                    summary_start = time.time()
                    state = await process_summary(state)
                    print(f"✅ Summary node completed in {time.time() - summary_start:.2f}s", flush=True)
                    logger.info(
                        f"[ADVANCE] Summary node completed for session {session_id}"
                    )
                except Exception as summary_error:
                    print(f"❌ Summary node FAILED: {summary_error}", flush=True)
                    logger.exception(f"[ADVANCE] Summary node FAILED: {summary_error}")
                    raise
                state["current_step"] = "summary"
                # summary node sets its own step_status

            elif current_step == "summary":
                print("=" * 50, flush=True)
                print("▶ TRANSITION: Summary → Estimation", flush=True)
                print("=" * 50, flush=True)
                logger.info("=" * 50)
                logger.info("▶ TRANSITION: Summary → Estimation")
                logger.info("=" * 50)

                # Advancing from summary -> run estimation
                from agents.nodes.estimation_node import (
                    process_estimation,
                    set_db_session,
                )

                # Mark summary as completed before advancing
                state["step_status"]["summary"] = "completed"

                # Set the database session for rule retrieval
                set_db_session(self.db)

                # Log what we're passing to estimation
                parsed_pr = state.get("parsed_pr", {})
                print(f"📄 PR for estimation: {parsed_pr.get('pr_code', 'N/A')} - {parsed_pr.get('title', 'N/A')[:50]}", flush=True)
                print(f"📊 ML Features: {len(state.get('ml_features', []))} features", flush=True)
                print(f"🔍 Similar PRs: {len(state.get('similar_prs', []))} found", flush=True)
                logger.info(f"📄 PR for estimation: {parsed_pr.get('pr_code', 'N/A')} - {parsed_pr.get('title', 'N/A')[:50]}")
                logger.info(f"📊 ML Features: {len(state.get('ml_features', []))} features")
                logger.info(f"🔍 Similar PRs: {len(state.get('similar_prs', []))} found")

                print(f"⏳ Starting estimation node for session {session_id[:8]}...", flush=True)
                logger.info(f"⏳ Starting estimation node for session {session_id[:8]}...")
                estimation_start = time.time()

                # Add timeout to prevent socket hang up on slow LLM/ML operations
                # Estimation can take 60-90s on first run (model loading + LLM calls)
                ESTIMATION_TIMEOUT = 180  # 3 minutes max
                try:
                    state = await asyncio.wait_for(
                        process_estimation(state),
                        timeout=ESTIMATION_TIMEOUT
                    )
                    estimation_elapsed = time.time() - estimation_start
                    print(f"✅ Estimation completed in {estimation_elapsed:.2f}s", flush=True)
                    logger.info(f"✅ Estimation completed in {estimation_elapsed:.2f}s")

                    # Log estimation results
                    if state.get("breakdown"):
                        log_breakdown(logger, state["breakdown"])
                    total_hours = state.get('total_hours', 'N/A')
                    total_cost = state.get('total_cost_eur', 0)
                    confidence = state.get('overall_confidence', 0)
                    print(f"📊 Total hours: {total_hours}", flush=True)
                    print(f"💰 Total cost: €{total_cost:,.0f}", flush=True)
                    print(f"📈 Confidence: {confidence*100:.1f}%", flush=True)
                    logger.info(f"📊 Total hours: {total_hours}")
                    logger.info(f"💰 Total cost: €{total_cost:,.0f}")
                    logger.info(f"📈 Confidence: {confidence*100:.1f}%")

                except asyncio.TimeoutError:
                    estimation_elapsed = time.time() - estimation_start
                    print(f"❌ Estimation TIMEOUT after {estimation_elapsed:.2f}s (limit: {ESTIMATION_TIMEOUT}s)", flush=True)
                    logger.error(f"❌ Estimation TIMEOUT after {estimation_elapsed:.2f}s (limit: {ESTIMATION_TIMEOUT}s)")
                    # Set error state but don't crash - allow retry
                    state["step_status"]["estimation"] = "error"
                    state["estimation_error"] = f"Estimation timed out after {ESTIMATION_TIMEOUT}s. Please try again."
                    raise HTTPException(
                        status_code=504,
                        detail=f"Estimation timed out. The ML model or LLM took too long to respond. Please refresh and try again."
                    )
                except Exception as est_error:
                    estimation_elapsed = time.time() - estimation_start
                    print(f"❌ Estimation FAILED after {estimation_elapsed:.2f}s: {est_error}", flush=True)
                    logger.error(f"❌ Estimation FAILED after {estimation_elapsed:.2f}s: {est_error}")
                    log_error_details(logger, est_error, "estimation_node")
                    raise

                # Pause at estimation step - user can interact before review
                state["current_step"] = "estimation"
                # estimation node sets its own step_status

            elif current_step == "estimation":
                # Advancing from estimation -> go to review (no processing needed)
                state["step_status"]["estimation"] = "completed"
                state["step_status"]["review"] = "waiting_input"
                state["current_step"] = "review"

            elif current_step == "review":
                # Advancing from review -> run export and learning
                from agents.nodes.export_node import process_export
                from agents.nodes.learning_node import (
                    process_learning,
                    set_db_session as set_learning_db,
                )

                state["step_status"]["review"] = "completed"
                state["is_finalized"] = True
                set_learning_db(self.db)

                logger.info(f"Running export node for session {session_id}")
                state = await process_export(state)

                logger.info(f"Running learning node for session {session_id}")
                state = await process_learning(state)
                state["current_step"] = "complete"

            else:
                logger.warning(f"Unknown step: {current_step}, using graph.run()")
                graph = await self._get_graph()
                config = {"configurable": {"thread_id": session_id}}
                result_state = await graph.run(state, config)
                if result_state:
                    state = result_state

            # Log step_status after transition
            logger.info(
                f"[ADVANCE] After transition: step_status={state.get('step_status')}"
            )

            # Update PR status
            logger.info(f"[ADVANCE] Updating PR status to {state.get('current_step')}")
            if state.get("pr_id"):
                await self.pr_repo.update_status(
                    uuid.UUID(state["pr_id"]),
                    state.get("current_step", current_step),
                )
            logger.info(f"[ADVANCE] PR status updated successfully")

            # Save updated state to Redis
            logger.info(f"[ADVANCE] Saving state to Redis...")
            await save_state_to_redis(session_id, state)
            logger.info(f"[ADVANCE] State saved to Redis successfully")

            # Build response
            logger.info(f"[ADVANCE] Building response...")
            response = self._state_to_response(state)
            logger.info(f"[ADVANCE] Response built, returning...")
            return response
        except Exception as e:
            logger.exception(f"[ADVANCE] Error advancing step: {e}")
            raise

    async def go_to_step(
        self,
        session_id: str,
        target_step: str,
    ) -> dict[str, Any]:
        """
        Navigate to a specific completed step (view only, no re-processing).

        Allows users to go back and review previous steps without restarting
        the estimation process. No LLM/ML nodes are re-executed.

        Args:
            session_id: The session ID
            target_step: The step to navigate to (upload, qa, summary, estimation, review)

        Returns:
            Current session state with updated current_step
        """
        # Load current state from Redis
        state = await load_state_from_redis(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Map frontend steps to backend steps
        step_map = {
            "upload": "intake",
            "qa": "qa",
            "summary": "summary",
            "estimation": "estimation",
            "review": "review",
        }
        backend_step = step_map.get(target_step, target_step)

        # Ensure step_status dict exists (for sessions created before this feature)
        if "step_status" not in state or not state["step_status"]:
            # Infer completed steps from current_step
            current = state.get("current_step", "intake")
            step_order = ["intake", "qa", "summary", "estimation", "review", "export"]
            current_idx = step_order.index(current) if current in step_order else 0
            state["step_status"] = {
                step: (
                    "completed"
                    if i < current_idx
                    else ("waiting_input" if i == current_idx else "pending")
                )
                for i, step in enumerate(step_order)
            }
            logger.info(f"[GO_TO_STEP] Initialized step_status: {state['step_status']}")
            # Save the initialized step_status
            await save_state_to_redis(session_id, state)

        # Get step completion status
        step_status = state.get("step_status", {})

        # Determine which steps are completed
        # NOTE: StepStatus enum values are lowercase: "completed", "waiting_input"
        completed_steps = [
            s
            for s, status in step_status.items()
            if status in ("completed", "waiting_input")
        ]

        # Also consider the current step as navigable
        current_step = state.get("current_step", "intake")

        # Validate: can only go to completed steps or current step
        if backend_step not in completed_steps and backend_step != current_step:
            # Build list of available steps for error message
            available = [s for s in completed_steps] + [current_step]
            logger.warning(
                f"Cannot navigate to step '{target_step}' - not completed. "
                f"Available: {available}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Cannot navigate to step '{target_step}' - not completed yet. "
                f"Available steps: {available}",
            )

        # Update current_step only (no re-running nodes)
        state["current_step"] = backend_step
        state["updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Session {session_id}: navigated to step '{backend_step}'")

        # Save updated state to Redis
        await save_state_to_redis(session_id, state)

        return self._state_to_response(state)

    async def update_breakdown_item(
        self,
        session_id: str,
        item_id: str,
        hours: float | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Update a breakdown item with user edits.

        Uses Redis WATCH for optimistic locking to prevent lost updates
        from concurrent modifications.

        Args:
            session_id: The session ID
            item_id: The breakdown item ID
            hours: New hours value
            reason: Reason for the edit

        Returns:
            Updated breakdown item
        """
        redis_client = await get_redis()
        redis_key = f"estimation:{session_id}"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Use Redis WATCH for optimistic locking
                await redis_client.watch(redis_key)

                # Load current state
                state_json = await redis_client.get(redis_key)
                if not state_json:
                    await redis_client.unwatch()
                    raise ValueError(f"Session {session_id} not found")

                state = json.loads(state_json)
                breakdown = state.get("breakdown", [])
                user_edits = state.get("user_edits", [])

                # Find and update the item
                updated_item = None
                for item in breakdown:
                    if item["id"] == item_id:
                        if hours is not None:
                            # Record the edit
                            user_edits.append(
                                {
                                    "breakdown_id": item_id,
                                    "original_hours": item["hours"],
                                    "new_hours": hours,
                                    "reason": reason or "",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                            )

                            # Update the item
                            item["hours"] = hours
                            item["cost_eur"] = hours * item.get("hourly_rate_eur", 85)
                            item["user_edited"] = True
                            item["edit_reason"] = reason

                        updated_item = item
                        break

                if not updated_item:
                    await redis_client.unwatch()
                    raise ValueError(f"Breakdown item {item_id} not found")

                # Recalculate totals
                total_hours = sum(item.get("hours", 0) for item in breakdown)
                total_cost = sum(item.get("cost_eur", 0) for item in breakdown)

                # Update state
                state["breakdown"] = breakdown
                state["user_edits"] = user_edits
                state["total_hours"] = total_hours
                state["total_cost_eur"] = total_cost

                # Atomic transaction - will fail if key was modified
                pipe = redis_client.pipeline()
                state_json = json.dumps(state, default=str)
                pipe.setex(redis_key, 86400, state_json)
                await pipe.execute()

                logger.debug(f"Updated breakdown item {item_id} atomically")
                return updated_item

            except redis.WatchError:
                # Key was modified by another request, retry
                logger.warning(
                    f"Concurrent modification detected for {session_id}, "
                    f"retry {attempt + 1}/{max_retries}"
                )
                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Failed to update after {max_retries} attempts due to "
                        "concurrent modifications"
                    )
                await asyncio.sleep(0.1 * (attempt + 1))  # Backoff

        raise ValueError("Failed to update breakdown item")

    async def _create_preference_pairs_from_session(
        self,
        session_id: str,
        state: dict[str, Any],
    ) -> int:
        """
        Create RLHF preference pairs from session edits.

        Called during finalization to capture user corrections as training data.
        Returns number of pairs created.
        """
        user_edits = state.get("user_edits", [])
        breakdown = state.get("breakdown", [])

        if not user_edits:
            # No edits - check if this is an approval
            if state.get("is_finalized") and breakdown:
                # User approved without edits - create synthetic negative
                try:
                    pair_service = PreferencePairService(self.db)
                    breakdown_dict = {
                        item.get("activity", f"item_{i}"): item.get("hours", 0)
                        for i, item in enumerate(breakdown)
                    }
                    context = {
                        "pr_id": state.get("pr_id"),
                        "program_family": state.get("parsed_pr", {}).get(
                            "program_family", ""
                        ),
                        "total_hours": state.get("total_hours", 0),
                    }
                    await pair_service.create_from_explicit_approval(
                        session_id=uuid.UUID(session_id),
                        approved_breakdown=breakdown_dict,
                        reasoning=state.get(
                            "estimation_reasoning", "HCQE prediction accepted"
                        ),
                        context=context,
                    )
                    await self.db.commit()
                    logger.info(
                        f"Created approval preference pair for session {session_id}"
                    )
                    return 1
                except Exception as e:
                    logger.warning(f"Failed to create approval pair: {e}")
            return 0

        # Build original breakdown from edits
        original_breakdown = {}
        edited_breakdown = {}

        for item in breakdown:
            item_id = item.get("id")
            activity = item.get("activity", f"item_{item_id}")
            edited_breakdown[activity] = item.get("hours", 0)

            # Find original value from edits
            for edit in user_edits:
                if edit.get("breakdown_id") == item_id:
                    original_breakdown[activity] = edit.get(
                        "original_hours", item.get("hours", 0)
                    )
                    break
            else:
                original_breakdown[activity] = item.get("hours", 0)

        # Create preference pair for the edit
        try:
            pair_service = PreferencePairService(self.db)
            context = {
                "pr_id": state.get("pr_id"),
                "program_family": state.get("parsed_pr", {}).get("program_family", ""),
                "total_hours": state.get("total_hours", 0),
                "edit_reasons": [e.get("reason", "") for e in user_edits],
            }

            # Aggregate edit reasons for reasoning
            edit_reasons = "; ".join(
                f"{e.get('reason', 'No reason')}" for e in user_edits if e.get("reason")
            )
            edited_reasoning = edit_reasons or "User corrections applied"
            original_reasoning = state.get(
                "estimation_reasoning", "Original HCQE prediction"
            )

            await pair_service.create_from_user_edit(
                session_id=uuid.UUID(session_id),
                correction=None,  # We don't have FeedbackCorrection model here
                original_breakdown=original_breakdown,
                edited_breakdown=edited_breakdown,
                original_reasoning=original_reasoning,
                edited_reasoning=edited_reasoning,
                context=context,
            )
            await self.db.commit()
            logger.info(
                f"Created {len(user_edits)} preference pairs for session {session_id}"
            )
            return len(user_edits)
        except Exception as e:
            logger.warning(f"Failed to create preference pairs: {e}")
            return 0

    async def finalize_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Finalize the estimation session and trigger export/learning.

        Args:
            session_id: The session ID
            user_id: ID of user finalizing

        Returns:
            Final session state with export results
        """
        # Load current state from Redis
        state = await load_state_from_redis(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Mark as finalized
        state["is_finalized"] = True
        state["finalized_by"] = user_id
        state["finalized_at"] = datetime.now(timezone.utc).isoformat()

        # Get the graph and run export/learning
        graph = await self._get_graph()
        config = {"configurable": {"thread_id": session_id}}

        try:
            result_state = await graph.run(state, config)

            if result_state is None:
                result_state = state

            # Update PR status
            if result_state.get("pr_id"):
                await self.pr_repo.update_status(
                    uuid.UUID(result_state["pr_id"]),
                    "complete",
                )

            # Save updated state to Redis
            await save_state_to_redis(session_id, result_state)

            # Create RLHF preference pairs from edits/approval
            try:
                pairs_created = await self._create_preference_pairs_from_session(
                    session_id, result_state
                )
                if pairs_created > 0:
                    logger.info(
                        f"RLHF: Created {pairs_created} preference pair(s) for session {session_id}"
                    )
            except Exception as rlhf_error:
                # Don't fail finalization if RLHF fails
                logger.warning(f"RLHF preference pair creation failed: {rlhf_error}")

            return self._state_to_response(result_state)
        except Exception as e:
            logger.exception(f"Error finalizing session: {e}")
            raise

    def _state_to_response(self, state: EstimationState) -> dict[str, Any]:
        """Convert LangGraph state to API response format."""
        # Determine status - only error if no breakdown and has error message
        has_breakdown = len(state.get("breakdown", [])) > 0
        has_fatal_error = state.get("error_message") and not has_breakdown
        status = "error" if has_fatal_error else "active"

        # LAZY LOADING: Check if questions are ready
        questions = state.get("questions", [])
        questions_ready = len(questions) > 0

        # Convert StepStatus enums to strings for JSON serialization
        raw_step_status = state.get("step_status", {})
        step_status = {
            k: str(v) if hasattr(v, "value") else v for k, v in raw_step_status.items()
        }
        logger.info(f"[DEBUG] _state_to_response step_status: {step_status}")

        return {
            "session_id": state.get("session_id"),
            "pr_id": state.get("pr_id"),
            "current_step": state.get("current_step", "intake"),
            "status": status,
            "created_at": state.get("created_at"),
            "updated_at": state.get("updated_at"),
            # Step completion status (for navigation)
            "step_status": step_status,
            # Step data
            "parsed_pr": state.get("parsed_pr"),
            "questions": questions,
            "questions_ready": questions_ready,  # LAZY LOADING: true when questions generated
            "questions_generating": False,  # Set by API layer during generation
            "answers": state.get("answers", {}),
            "pr_summary": state.get("pr_summary"),
            "similar_prs": state.get("similar_prs", []),
            "breakdown": state.get("breakdown", []),
            "total_hours": state.get("total_hours", 0),
            "total_cost_eur": state.get("total_cost_eur", 0),
            "overall_confidence": state.get("overall_confidence", 0),
            "applied_rules": state.get("applied_rules", []),
            "user_edits": state.get("user_edits", []),
            "is_finalized": state.get("is_finalized", False),
            "export_result": state.get("export_result"),
            # Estimation metadata
            "ml_sizing": state.get("ml_sizing"),
            "sizing_predictions": state.get("sizing_predictions"),
            "sizing_confidence": state.get("sizing_confidence"),
            "ml_prediction": state.get("ml_prediction"),
            # Convert ml_interval tuple to list for JSON serialization
            "ml_interval": list(state["ml_interval"])
            if state.get("ml_interval")
            else None,
            "ml_recommendations": state.get("ml_recommendations", []),
            "estimation_method": state.get("estimation_method"),
            # Errors
            "error_message": state.get("error_message"),
        }


# Singleton for dependency injection
_estimation_service: EstimationService | None = None


async def get_estimation_service(db: AsyncSession) -> EstimationService:
    """Get or create the estimation service."""
    return EstimationService(db)

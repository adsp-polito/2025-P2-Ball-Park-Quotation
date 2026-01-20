"""
FPT Cost Brain 2.0 - Chat API
Endpoints for the adaptive RAG chat system with database persistence and streaming
"""

import json
import logging
import uuid
from enum import Enum
from typing import Annotated, Any

from app.dependencies import get_current_user, get_db
from db.models import ChatMessage, ChatSession, User
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)


# ===== Schemas =====


class PageContext(BaseModel):
    """Schema for current page context - what's visible on screen."""

    questions: list[dict[str, Any]] | None = None  # Current questions with answers
    breakdown: list[dict[str, Any]] | None = None  # Current cost breakdown
    parsed_pr: dict[str, Any] | None = None  # Parsed PR data
    pr_summary: dict[str, Any] | None = None  # PR summary
    user_edits: list[dict[str, Any]] | None = None  # User corrections
    current_step: str | None = None  # Current wizard step


class ChatMode(str, Enum):
    """Chat mode - determines agent capabilities."""

    CHAT = "chat"  # Read-only assistant mode (default)
    AGENT = "agent"  # Full agent mode with write capabilities (GOD MODE)


class ChatMessageRequest(BaseModel):
    """Schema for chat message request."""

    message: str
    history: list[dict[str, str]] = []  # Previous messages for context
    page_context: PageContext | None = None  # Explicit page context from frontend
    mode: ChatMode = (
        ChatMode.CHAT
    )  # Chat mode toggle - AGENT enables write capabilities


class ActionResult(BaseModel):
    """Schema for GOD MODE action result."""

    status: str  # success, error, pending_reprocess, no_action
    action_type: str  # regenerate_questions, add_question, update_hours, etc.
    details: str  # Human-readable description


class ChatResponse(BaseModel):
    """Schema for chat response - matches frontend ChatResponse with GOD MODE support."""

    response: str
    suggestions: list[dict[str, str]]
    tool_calls: list[dict[str, Any]] | None = None
    step: str | None = None
    # GOD MODE fields
    intent: str | None = None  # Classified intent type
    action_executed: bool = False  # Whether a state modification was executed
    action_result: ActionResult | None = None  # Result of state modification
    updated_state: dict[str, Any] | None = None  # Updated state for frontend sync


class SuggestionResponse(BaseModel):
    """Schema for chat suggestions."""

    suggestions: list[dict[str, str]]


class ChatHistoryItem(BaseModel):
    """Schema for chat history item."""

    id: str
    content: str
    role: str
    created_at: str
    tool_calls: list[dict[str, Any]] | None = None
    sources: dict | None = None


# ===== Helper Functions =====


async def get_or_create_chat_session(
    db: AsyncSession,
    session_id: str,
    user_id: uuid.UUID,
    current_step: str = "qa",
) -> ChatSession:
    """Get existing chat session or create a new one."""
    # Try to find existing session by session_id (which maps to PR session)
    # We use session_id as the identifier to link chat with estimation session
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.user_id == user_id,
            ChatSession.id == uuid.UUID(session_id) if len(session_id) == 36 else None,
        )
    )
    chat_session = result.scalar_one_or_none()

    if not chat_session:
        # Try to find by looking for session with matching ID pattern
        # If session_id is a valid UUID, use it directly
        try:
            session_uuid = uuid.UUID(session_id)
            chat_session = ChatSession(
                id=session_uuid,
                user_id=user_id,
                current_step=current_step,
            )
        except ValueError:
            # If not a valid UUID, create new session
            chat_session = ChatSession(
                user_id=user_id,
                current_step=current_step,
            )

        db.add(chat_session)
        await db.commit()
        await db.refresh(chat_session)

    return chat_session


async def save_message(
    db: AsyncSession,
    session: ChatSession,
    role: str,
    content: str,
    step: str | None = None,
    sources: dict | None = None,
    tools_used: list[str] | None = None,
) -> ChatMessage:
    """Save a chat message to the database."""
    message = ChatMessage(
        session_id=session.id,
        role=role,
        content=content,
        step=step,
        sources=sources,
        tools_used=tools_used,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


async def get_chat_history_from_db(
    db: AsyncSession,
    session_id: uuid.UUID,
    limit: int = 50,
) -> list[ChatMessage]:
    """Get chat history from database."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()
    # Return in chronological order
    return list(reversed(messages))


# ===== Endpoints =====


@router.post("/{session_id}/message", response_model=ChatResponse)
async def send_message(
    session_id: str,
    request: ChatMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Send a message to the FPT Engineering Agent with RAG-first approach.

    The agent is a domain-expert AI companion that:
    - Understands FPT Industrial terminology and context deeply
    - Uses RAG-first architecture for contextual responses
    - Auto-detects and explains acronyms
    - Classifies intent for appropriate routing
    - Provides step-aware assistance

    Messages are persisted to the database for history.
    """
    from agents.fpt_engineer_agent import FPTEngineerAgent
    from services.estimation_service import load_state_from_redis

    # Load session state from Redis
    state = await load_state_from_redis(session_id)
    current_step = state.get("current_step", "qa") if state else "qa"

    if not state:
        state = {"current_step": current_step, "session_id": session_id}

    # Merge explicit page context from frontend (overrides Redis state)
    # This ensures chat sees exactly what's on screen, even if Redis is stale
    if request.page_context:
        page_ctx = request.page_context
        if page_ctx.questions is not None:
            state["questions"] = page_ctx.questions
            logger.debug(f"Using {len(page_ctx.questions)} questions from page context")
        if page_ctx.breakdown is not None:
            state["breakdown"] = page_ctx.breakdown
            logger.debug(
                f"Using {len(page_ctx.breakdown)} breakdown items from page context"
            )
        if page_ctx.parsed_pr is not None:
            state["parsed_pr"] = page_ctx.parsed_pr
        if page_ctx.pr_summary is not None:
            state["pr_summary"] = page_ctx.pr_summary
        if page_ctx.user_edits is not None:
            state["user_edits"] = page_ctx.user_edits
        if page_ctx.current_step is not None:
            current_step = page_ctx.current_step
            state["current_step"] = current_step

    try:
        # Get or create chat session in database
        chat_session = await get_or_create_chat_session(
            db, session_id, current_user.id, current_step
        )

        # Save user message to database
        await save_message(
            db,
            chat_session,
            role="user",
            content=request.message,
            step=current_step,
        )

        # Use FPTEngineerAgent for domain-expert response with RAG
        # Pass mode to agent - AGENT mode enables write capabilities (GOD MODE)
        agent = FPTEngineerAgent(state, agent_mode=(request.mode == ChatMode.AGENT))
        result = await agent.chat(
            user_message=request.message,
            history=request.history,
        )

        response_text = result.get("response", "I'm processing your request.")
        suggestions = result.get("suggestions", [])
        intent = result.get("intent", "general")
        rag_used = result.get("rag_context_used", False)

        # Log intent classification for debugging
        logger.info(f"Chat intent: {intent}, RAG used: {rag_used}")

        # Save assistant response to database with metadata
        await save_message(
            db,
            chat_session,
            role="assistant",
            content=response_text,
            step=current_step,
            tools_used=[f"intent:{intent}"] if intent else None,
            sources={
                "suggestions": suggestions,
                "intent": intent,
                "rag_context_used": rag_used,
            },
        )

        # Update chat session step if changed
        if chat_session.current_step != current_step:
            chat_session.current_step = current_step
            await db.commit()

        # Build response with GOD MODE fields
        response_data = ChatResponse(
            response=response_text,
            suggestions=suggestions,
            tool_calls=None,  # FPTEngineerAgent doesn't use explicit tool calls
            step=result.get("step", current_step),
            intent=intent,
            action_executed=result.get("action_executed", False),
            action_result=ActionResult(**result["action_result"])
            if result.get("action_result")
            else None,
            updated_state=result.get("updated_state"),
        )

        return response_data

    except Exception as e:
        logger.error(f"Chat LLM error: {e}", exc_info=True)

        # Fallback to simple response if LLM fails
        step_responses = {
            "qa": f"I can help you answer the clarifying questions. You asked: '{request.message}'. "
            "Based on similar projects, I'd suggest providing specific technical details.",
            "summary": f"Looking at your question about '{request.message}': "
            "I can explain features or compare with similar projects.",
            "estimation": f"Regarding '{request.message}': "
            "The cost estimation is based on historical data and ML predictions.",
            "review": f"About '{request.message}': "
            "When editing estimates, providing a reason helps the system learn.",
        }

        step_suggestions = {
            "qa": [
                {
                    "text": "Suggest an answer",
                    "action": "suggest_answer",
                    "icon": "lightbulb",
                },
                {
                    "text": "Show similar Q&A",
                    "action": "search_similar_qa",
                    "icon": "search",
                },
            ],
            "summary": [
                {
                    "text": "Explain features",
                    "action": "explain_features",
                    "icon": "list",
                },
                {"text": "Compare PRs", "action": "compare_prs", "icon": "git-compare"},
            ],
            "estimation": [
                {
                    "text": "Explain estimate",
                    "action": "explain_estimate",
                    "icon": "clock",
                },
                {"text": "Show rules", "action": "show_rules", "icon": "book"},
            ],
            "review": [
                {"text": "Suggest reason", "action": "suggest_reason", "icon": "edit"},
                {
                    "text": "Preview learning",
                    "action": "preview_learning",
                    "icon": "brain",
                },
            ],
        }

        return ChatResponse(
            response=step_responses.get(current_step, f"Processing: {request.message}"),
            suggestions=step_suggestions.get(current_step, []),
            step=current_step,
        )


@router.post("/{session_id}/message/stream")
async def send_message_stream(
    session_id: str,
    request: ChatMessageRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Send a message and receive a streaming response from FPT Engineering Agent.

    Returns Server-Sent Events (SSE) for real-time streaming.
    Uses RAG-first approach with domain expertise.
    Messages are persisted to the database after streaming completes.
    """
    from agents.fpt_engineer_agent import FPTEngineerAgent
    from services.estimation_service import load_state_from_redis

    # Load session state from Redis
    state = await load_state_from_redis(session_id)
    current_step = state.get("current_step", "qa") if state else "qa"

    if not state:
        state = {"current_step": current_step, "session_id": session_id}

    # Merge explicit page context from frontend (overrides Redis state)
    if request.page_context:
        page_ctx = request.page_context
        if page_ctx.questions is not None:
            state["questions"] = page_ctx.questions
        if page_ctx.breakdown is not None:
            state["breakdown"] = page_ctx.breakdown
        if page_ctx.parsed_pr is not None:
            state["parsed_pr"] = page_ctx.parsed_pr
        if page_ctx.pr_summary is not None:
            state["pr_summary"] = page_ctx.pr_summary
        if page_ctx.user_edits is not None:
            state["user_edits"] = page_ctx.user_edits
        if page_ctx.current_step is not None:
            current_step = page_ctx.current_step
            state["current_step"] = current_step

    async def generate_stream():
        full_response = ""

        try:
            # STEP 1: IMMEDIATELY send "thinking" status so frontend knows we're processing
            yield f"data: {json.dumps({'type': 'status', 'status': 'thinking', 'message': 'Analyzing your question...'})}\n\n"

            # Get or create chat session
            chat_session = await get_or_create_chat_session(
                db, session_id, current_user.id, current_step
            )

            # Save user message
            await save_message(
                db,
                chat_session,
                role="user",
                content=request.message,
                step=current_step,
            )

            # Initialize FPT Engineering Agent with mode
            agent = FPTEngineerAgent(state, agent_mode=(request.mode == ChatMode.AGENT))

            # STEP 2: Send "gathering context" status before RAG
            mode_label = "Agent" if request.mode == ChatMode.AGENT else "Chat"
            yield f"data: {json.dumps({'type': 'status', 'status': 'thinking', 'message': f'[{mode_label} Mode] Gathering relevant context...'})}\n\n"

            # Prepare stream context (this does the heavy RAG work)
            prepared_context = await agent.prepare_stream_context(
                user_message=request.message,
                history=request.history,
            )

            # Get intent from prepared context
            intent = prepared_context.get("intent")
            logger.info(f"Stream chat intent: {intent.value if intent else 'unknown'}")

            # GOD MODE: Execute action if intent is MODIFY_STATE
            action_result = None
            updated_state = None
            if request.mode == ChatMode.AGENT and intent:
                from agents.fpt_engineer_agent import IntentType

                if intent == IntentType.MODIFY_STATE:
                    logger.info("[GOD MODE] Executing state modification action")
                    yield f"data: {json.dumps({'type': 'status', 'status': 'thinking', 'message': '[Agent Mode] Executing modification...'})}\n\n"
                    action_result = await agent._execute_state_action(
                        intent, request.message
                    )
                    if action_result and action_result.get("status") == "success":
                        updated_state = action_result.get("updated_state")
                        # Persist to Redis
                        await agent._persist_state()
                        logger.info(
                            f"[GOD MODE] Action executed: {action_result.get('action_type')}"
                        )

            # STEP 3: Send "generating" status - now we're ready to stream!
            yield f"data: {json.dumps({'type': 'status', 'status': 'generating', 'message': 'Writing response...'})}\n\n"

            # Stream response chunks from LLM
            async for chunk in agent.stream_response(prepared_context):
                full_response += chunk
                # Send SSE formatted data
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

            # Save complete assistant response
            await save_message(
                db,
                chat_session,
                role="assistant",
                content=full_response,
                step=current_step,
                tools_used=[f"intent:{intent.value}"],
            )

            # Generate suggestions based on intent
            suggestions = await agent._generate_suggestions(
                request.message, full_response, intent
            )

            # Build done event with GOD MODE fields
            done_data = {
                "type": "done",
                "suggestions": suggestions,
                "step": current_step,
                "intent": intent.value,
                "action_executed": action_result is not None
                and action_result.get("status") == "success",
                "action_result": action_result,
                "updated_state": updated_state,
            }
            yield f"data: {json.dumps(done_data)}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_msg = (
                "I encountered an error processing your request. Please try again."
            )
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{session_id}/suggestions")
async def get_suggestions(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    step: str | None = None,
):
    """
    Get contextual suggestions for the chat.

    Suggestions are tailored to the current estimation step.
    """
    # Get step from Redis if not provided
    if not step:
        from services.estimation_service import load_state_from_redis

        state = await load_state_from_redis(session_id)
        step = state.get("current_step", "qa") if state else "qa"

    # Define step-specific suggestions
    step_suggestions = {
        "qa": [
            {
                "text": "Why is this question important?",
                "icon": "help-circle",
                "action": "explain_question",
            },
            {
                "text": "Show similar Q&A from past projects",
                "icon": "search",
                "action": "search_similar_qa",
            },
            {
                "text": "Suggest an answer",
                "icon": "lightbulb",
                "action": "suggest_answer",
            },
        ],
        "summary": [
            {
                "text": "Explain these features",
                "icon": "list",
                "action": "explain_features",
            },
            {
                "text": "Compare with similar PRs",
                "icon": "git-compare",
                "action": "compare_prs",
            },
            {
                "text": "What's the program size?",
                "icon": "bar-chart",
                "action": "explain_size",
            },
        ],
        "estimation": [
            {"text": "Why these hours?", "icon": "clock", "action": "explain_estimate"},
            {
                "text": "Show historical comparison",
                "icon": "trending-up",
                "action": "compare_breakdown",
            },
            {
                "text": "What rules were applied?",
                "icon": "book",
                "action": "show_rules",
            },
        ],
        "review": [
            {
                "text": "Suggest a reason for this change",
                "icon": "edit",
                "action": "suggest_reason",
            },
            {
                "text": "Preview what will be learned",
                "icon": "brain",
                "action": "preview_learning",
            },
            {
                "text": "Generate summary of changes",
                "icon": "file-text",
                "action": "generate_summary",
            },
        ],
    }

    suggestions = step_suggestions.get(
        step,
        [
            {"text": "How can I help?", "icon": "message-circle", "action": "help"},
            {
                "text": "Show estimation progress",
                "icon": "bar-chart",
                "action": "show_progress",
            },
        ],
    )

    return suggestions  # Return list directly to match frontend expectation


@router.get("/{session_id}/history", response_model=list[ChatHistoryItem])
async def get_chat_history(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
):
    """Get chat history for an estimation session from database."""
    try:
        # Try to get chat session
        session_uuid = uuid.UUID(session_id)

        # Query messages directly by session_id
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_uuid)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = result.scalars().all()

        # Convert to response format (chronological order)
        history = []
        for msg in reversed(list(messages)):
            history.append(
                ChatHistoryItem(
                    id=str(msg.id),
                    content=msg.content,
                    role=msg.role,
                    created_at=msg.created_at.isoformat(),
                    tool_calls=None,  # Could extract from sources if stored
                    sources=msg.sources,
                )
            )

        return history

    except ValueError:
        # Invalid UUID format
        return []
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        return []


@router.delete("/{session_id}/history")
async def clear_chat_history(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Clear chat history for an estimation session."""
    try:
        session_uuid = uuid.UUID(session_id)

        # Delete all messages for this session
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_uuid)
        )
        messages = result.scalars().all()

        for msg in messages:
            await db.delete(msg)

        await db.commit()

        return {
            "message": f"Cleared {len(messages)} messages",
            "session_id": session_id,
        }

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID format",
        )
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear chat history",
        )


@router.post("/tool/{tool_name}")
async def execute_chat_tool(
    tool_name: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    session_id: str | None = None,
    params: dict[str, Any] | None = None,
):
    """
    Execute a specific chat tool.

    Available tools vary by step:
    - search_similar_qa: Search Q&A from similar projects
    - explain_features: Explain extracted features
    - compare_prs: Compare with similar projects
    - explain_estimate: Explain cost estimate reasoning
    - show_rules: Show applied rules
    - suggest_reason: Suggest correction reasons
    """
    from agents.adaptive_rag import AdaptiveRAGChat
    from services.estimation_service import load_state_from_redis

    valid_tools = [
        "search_similar_qa",
        "generate_question",
        "rephrase_question",
        "explain_feature",
        "compare_prs",
        "search_knowledge",
        "explain_estimate",
        "compare_breakdown",
        "show_rules",
        "suggest_reason",
        "preview_learning",
        "generate_summary",
    ]

    if tool_name not in valid_tools:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tool. Available tools: {valid_tools}",
        )

    # Load state if session provided
    state = {}
    if session_id:
        state = await load_state_from_redis(session_id) or {}

    try:
        # Initialize chat and execute tool
        chat = AdaptiveRAGChat(state)
        tool_fn = chat._get_tool_function(tool_name)

        if tool_fn:
            user_context = params.get("context", "") if params else ""
            result = await tool_fn(user_context)
        else:
            result = f"Tool '{tool_name}' not available for current step"

        return {
            "tool": tool_name,
            "result": result,
            "data": params or {},
        }

    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool execution failed: {str(e)}",
        )

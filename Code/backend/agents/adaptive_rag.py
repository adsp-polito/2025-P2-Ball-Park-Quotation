"""
FPT Cost Brain 2.0 - Adaptive RAG Chat System
Step-aware conversational assistant with real vector search and tool access
"""

import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any, Callable

from llm.client import get_llm_client
from llm.prompts import (
    CHAT_SYSTEM_PROMPTS,
    CHAT_TOOL_DESCRIPTIONS,
    SUGGESTION_GENERATION,
)
from vector.client import get_vector_store
from vector.collections import (
    FEEDBACK_PATTERNS,
    KNOWLEDGE_CHUNKS,
    PR_EMBEDDINGS,
    QUOTATION_CHUNKS,
)

from agents.state import EstimationState

logger = logging.getLogger(__name__)


class ChatStep(str, Enum):
    """Chat context steps aligned with estimation workflow."""

    QA = "qa"
    SUMMARY = "summary"
    ESTIMATION = "estimation"
    REVIEW = "review"


# Step-specific tool configurations
STEP_TOOLS: dict[ChatStep, list[str]] = {
    ChatStep.QA: [
        "search_similar_qa",
        "generate_question",
        "rephrase_question",
    ],
    ChatStep.SUMMARY: [
        "explain_feature",
        "compare_prs",
        "search_knowledge",
    ],
    ChatStep.ESTIMATION: [
        "explain_estimate",
        "compare_breakdown",
        "show_rules",
    ],
    ChatStep.REVIEW: [
        "suggest_reason",
        "preview_learning",
        "generate_summary",
    ],
}

# Step-specific context keys from state
STEP_CONTEXT_KEYS: dict[ChatStep, list[str]] = {
    ChatStep.QA: ["parsed_pr", "questions", "similar_prs"],
    ChatStep.SUMMARY: ["pr_summary", "ml_features", "similar_prs"],
    ChatStep.ESTIMATION: [
        "pr_summary",
        "ml_features",
        "similar_prs",
        "breakdown",
        "applied_rules",
        "ml_prediction",
    ],
    ChatStep.REVIEW: [
        "pr_summary",
        "breakdown",
        "user_edits",
        "applied_rules",
    ],
}


class AdaptiveRAGChat:
    """
    Step-aware chat assistant with contextual tools and real vector search.

    Provides different capabilities based on the current
    step of the estimation workflow.
    """

    def __init__(self, state: EstimationState):
        self.state = state
        self.llm = get_llm_client()
        self.current_step = self._determine_step()
        self.tools = self._get_tools_for_step()
        self.system_prompt = self._get_system_prompt()

    def _determine_step(self) -> ChatStep:
        """Determine the current chat step from state."""
        current = self.state.get("current_step", "qa")

        step_mapping = {
            "intake": ChatStep.QA,
            "qa": ChatStep.QA,
            "summary": ChatStep.SUMMARY,
            "estimation": ChatStep.ESTIMATION,
            "review": ChatStep.REVIEW,
            "export": ChatStep.REVIEW,
        }

        return step_mapping.get(current, ChatStep.QA)

    def _get_tools_for_step(self) -> dict[str, Callable]:
        """Get available tools for current step."""
        tool_names = STEP_TOOLS.get(self.current_step, [])
        tools = {}

        for name in tool_names:
            tool_fn = self._get_tool_function(name)
            if tool_fn:
                tools[name] = tool_fn

        return tools

    def _get_tool_function(self, name: str) -> Callable | None:
        """Get the actual tool function by name."""
        tool_registry = {
            "search_similar_qa": self._tool_search_similar_qa,
            "generate_question": self._tool_generate_question,
            "rephrase_question": self._tool_rephrase_question,
            "explain_feature": self._tool_explain_feature,
            "compare_prs": self._tool_compare_prs,
            "search_knowledge": self._tool_search_knowledge,
            "explain_estimate": self._tool_explain_estimate,
            "compare_breakdown": self._tool_compare_breakdown,
            "show_rules": self._tool_show_rules,
            "suggest_reason": self._tool_suggest_reason,
            "preview_learning": self._tool_preview_learning,
            "generate_summary": self._tool_generate_summary,
        }

        return tool_registry.get(name)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for current step."""
        return CHAT_SYSTEM_PROMPTS.get(
            self.current_step.value,
            CHAT_SYSTEM_PROMPTS["qa"],
        )

    def _get_context_for_step(self) -> dict[str, Any]:
        """Get relevant context from state for current step."""
        keys = STEP_CONTEXT_KEYS.get(self.current_step, [])
        return {k: self.state.get(k) for k in keys if k in self.state}

    async def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        try:
            return await self.llm.embed(query)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def _search_vectors(
        self,
        collection: str,
        query: str,
        limit: int = 5,
        filter_conditions: dict | None = None,
        score_threshold: float = 0.5,
    ) -> list[dict]:
        """Search vectors in a collection using query embedding."""
        try:
            # Generate embedding for the query
            query_vector = await self._embed_query(query)

            # Get vector store
            vector_store = await get_vector_store()

            # Search
            results = await vector_store.search(
                collection=collection,
                query_vector=query_vector,
                limit=limit,
                filter_conditions=filter_conditions,
                score_threshold=score_threshold,
            )

            return results
        except Exception as e:
            logger.error(f"Vector search failed in {collection}: {e}")
            return []

    async def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Process a chat message and return response.

        Args:
            user_message: The user's message
            history: Optional conversation history

        Returns:
            Dict with response, suggestions, and tool calls
        """
        history = history or []

        # Build messages with context
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add context as system message
        context = self._get_context_for_step()
        if context:
            context_msg = f"Current context:\n{self._format_context(context)}"
            messages.append({"role": "system", "content": context_msg})

        # Add available tools info
        tools_info = self._format_tools_info()
        messages.append({"role": "system", "content": tools_info})

        # Add history
        for msg in history[-10:]:  # Last 10 messages
            messages.append(msg)

        # Add user message
        messages.append({"role": "user", "content": user_message})

        # Check if user is asking to use a tool
        tool_call = self._detect_tool_request(user_message)
        tool_result = None

        if tool_call:
            tool_result = await self._execute_tool(tool_call, user_message)
            if tool_result:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Tool '{tool_call}' result:\n{tool_result}",
                    }
                )

        # Get LLM response
        response = await self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )

        # Generate suggestions
        suggestions = await self._generate_suggestions(user_message, response)

        return {
            "response": response,
            "suggestions": suggestions,
            "tool_calls": [{"tool": tool_call, "result": tool_result}]
            if tool_call
            else None,
            "step": self.current_step.value,
        }

    async def chat_stream(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat response."""
        history = history or []

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]

        context = self._get_context_for_step()
        if context:
            context_msg = f"Current context:\n{self._format_context(context)}"
            messages.append({"role": "system", "content": context_msg})

        # Add available tools info
        tools_info = self._format_tools_info()
        messages.append({"role": "system", "content": tools_info})

        # Check if user is asking to use a tool and pre-execute it
        tool_call = self._detect_tool_request(user_message)
        if tool_call:
            tool_result = await self._execute_tool(tool_call, user_message)
            if tool_result:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Tool '{tool_call}' result:\n{tool_result}",
                    }
                )

        for msg in history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        # Stream response
        async for chunk in self.llm.chat_stream(messages):
            yield chunk

    def _format_context(self, context: dict) -> str:
        """Format context dict as readable string with full visibility into page content."""
        parts = []

        for key, value in context.items():
            if value is None:
                continue

            # Handle questions - show full question texts (this is what user sees on screen)
            if key == "questions" and isinstance(value, list):
                if value:
                    parts.append(
                        f"\n**CURRENT QUESTIONS ON SCREEN ({len(value)} questions):**"
                    )
                    for i, q in enumerate(value, 1):
                        q_text = q.get("question_text", q.get("text", str(q)))
                        q_answer = q.get("answer", "")
                        q_category = q.get("category", "")
                        answer_status = (
                            f" [ANSWERED: {q_answer}]" if q_answer else " [UNANSWERED]"
                        )
                        category_info = f" ({q_category})" if q_category else ""
                        parts.append(f"  Q{i}{category_info}: {q_text}{answer_status}")

            # Handle parsed_pr - show key PR details
            elif key == "parsed_pr" and isinstance(value, dict):
                parts.append("\n**CURRENT PR DETAILS:**")
                for pr_key in [
                    "title",
                    "description",
                    "platform",
                    "engine_type",
                    "program_size",
                    "tier",
                ]:
                    if pr_key in value and value[pr_key]:
                        parts.append(f"  - {pr_key}: {str(value[pr_key])[:200]}")

            # Handle pr_summary - show summary details
            elif key == "pr_summary" and isinstance(value, dict):
                parts.append("\n**PR SUMMARY:**")
                for sum_key in [
                    "title",
                    "program_size",
                    "complexity_score",
                    "key_features",
                ]:
                    if sum_key in value and value[sum_key]:
                        val = value[sum_key]
                        if isinstance(val, list):
                            parts.append(
                                f"  - {sum_key}: {', '.join(str(v) for v in val[:5])}"
                            )
                        else:
                            parts.append(f"  - {sum_key}: {str(val)[:150]}")

            # Handle breakdown - show activity breakdown
            elif key == "breakdown" and isinstance(value, list):
                if value:
                    parts.append(f"\n**COST BREAKDOWN ({len(value)} activities):**")
                    for item in value[:10]:  # Show first 10 items
                        func = item.get("pe_function", "Unknown")
                        desc = item.get("activity_description", "")[:50]
                        hours = item.get("hours", 0)
                        edited = " [EDITED]" if item.get("user_edited") else ""
                        parts.append(f"  - {func}: {desc}... ({hours}h){edited}")
                    if len(value) > 10:
                        parts.append(f"  ... and {len(value) - 10} more activities")

            # Handle similar_prs - show similar projects
            elif key == "similar_prs" and isinstance(value, list):
                if value:
                    parts.append(f"\n**SIMILAR PROJECTS ({len(value)}):**")
                    for pr in value[:5]:
                        pr_num = pr.get("pr_number", "?")
                        pr_title = pr.get("title", "Unknown")[:40]
                        similarity = pr.get("similarity", pr.get("score", 0))
                        parts.append(
                            f"  - {pr_num}: {pr_title} ({similarity:.0%} similar)"
                        )

            # Handle user_edits - show corrections
            elif key == "user_edits" and isinstance(value, list):
                if value:
                    parts.append(f"\n**USER CORRECTIONS ({len(value)}):**")
                    for edit in value[:5]:
                        orig = edit.get("original_hours", 0)
                        new = edit.get("new_hours", 0)
                        reason = edit.get("reason", "No reason")[:50]
                        parts.append(f"  - Changed {orig}h → {new}h: {reason}")

            # Handle applied_rules
            elif key == "applied_rules" and isinstance(value, list):
                if value:
                    parts.append(f"\n**APPLIED RULES ({len(value)}):**")
                    for rule in value[:5]:
                        name = rule.get("rule_name", "Unknown")
                        effect = rule.get("effect_value", 0)
                        parts.append(f"  - {name}: {effect:+.0%}")

            # Handle ml_features - brief summary
            elif key == "ml_features" and isinstance(value, dict):
                parts.append(f"\n**ML FEATURES:** {len(value)} features extracted")

            # Default handling for other types
            elif isinstance(value, list):
                if len(value) > 0:
                    parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, dict):
                parts.append(f"{key}: {list(value.keys())[:5]}")
            else:
                parts.append(f"{key}: {str(value)[:100]}")

        return "\n".join(parts)

    def _format_tools_info(self) -> str:
        """Format available tools as string."""
        tool_names = list(self.tools.keys())

        if not tool_names:
            return "No special tools available for this step."

        lines = ["Available tools for this step:"]
        for name in tool_names:
            desc = CHAT_TOOL_DESCRIPTIONS.get(name, "No description")
            lines.append(f"- {name}: {desc}")

        return "\n".join(lines)

    def _is_meta_question(self, message: str) -> bool:
        """Detect if user is asking about the system itself rather than technical content."""
        message_lower = message.lower()

        # Meta-question patterns - questions about the Q&A process, not technical queries
        meta_patterns = [
            # Questions about the questions themselves
            "should i regenerate",
            "regenerate question",
            "these questions good",
            "questions are good",
            "question good enough",
            "redo questions",
            "change the questions",
            "better questions",
            "different questions",
            "enough questions",
            "more questions needed",
            "skip questions",
            # Questions about the process
            "what should i do",
            "next step",
            "how does this work",
            "what happens if",
            "can i skip",
            "is this correct",
            "am i doing this right",
            "help me understand",
            # Questions about answers
            "answer good",
            "my answer correct",
            "answer enough",
            "should i change",
            "need more detail",
            # System capability questions
            "can you",
            "are you able",
            "do you support",
            "what can you",
            "help me with",
        ]

        for pattern in meta_patterns:
            if pattern in message_lower:
                return True

        return False

    def _detect_tool_request(self, message: str) -> str | None:
        """Detect if user is requesting a specific tool."""
        message_lower = message.lower()

        # Don't trigger tools for meta-questions - let LLM handle them naturally
        if self._is_meta_question(message):
            return None

        tool_triggers = {
            "search_similar_qa": [
                "similar question",
                "past answers",
                "qa history",
                "similar qa",
                "historical answers",
                "how did others answer",
                "previous projects",
            ],
            "generate_question": [
                "add a question about",  # More specific trigger
                "generate a question about",
                "create question",
            ],
            "explain_feature": [
                "explain the feature",
                "what does this feature mean",
                "tell me about this feature",
            ],
            "compare_prs": [
                "compare with similar",
                "show similar project",
                "find similar pr",
                "historical comparison",
            ],
            "search_knowledge": [
                "search knowledge base",
                "find in documentation",
                "look up in docs",
            ],
            "explain_estimate": [
                "explain this estimate",
                "why these hours",
                "how was this calculated",
                "explain the reasoning",
            ],
            "compare_breakdown": [
                "compare costs with historical",
                "breakdown comparison",
                "historical cost data",
            ],
            "show_rules": [
                "show applied rules",
                "what rules were used",
                "adjustment rules",
            ],
            "suggest_reason": [
                "suggest a reason for",
                "why might i change",
                "correction reason",
            ],
            "preview_learning": [
                "what will the system learn",
                "preview learning",
                "how will this help training",
            ],
            "generate_summary": [
                "generate summary",
                "create a summary",
                "give me an overview",
            ],
        }

        for tool, triggers in tool_triggers.items():
            if tool in self.tools:
                for trigger in triggers:
                    if trigger in message_lower:
                        return tool

        return None

    async def _execute_tool(self, tool_name: str, user_message: str = "") -> str | None:
        """Execute a tool and return result."""
        tool_fn = self.tools.get(tool_name)

        if not tool_fn:
            return None

        try:
            result = await tool_fn(user_message)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            return f"Tool error: {str(e)}"

    async def _generate_suggestions(
        self,
        user_message: str,
        response: str,
    ) -> list[dict[str, str]]:
        """Generate contextual suggestions for next actions."""
        try:
            recent_messages = f"User: {user_message}\nAssistant: {response[:200]}"

            state_summary = {
                "step": self.current_step.value,
                "has_breakdown": bool(self.state.get("breakdown")),
                "has_edits": bool(self.state.get("user_edits")),
            }

            prompt = SUGGESTION_GENERATION.format(
                step=self.current_step.value,
                recent_messages=recent_messages,
                state_summary=str(state_summary),
            )

            result = await self.llm.extract_json(prompt)

            return result.get("suggestions", [])[:4]

        except Exception:
            # Return default suggestions on error
            return self._get_default_suggestions()

    def _get_default_suggestions(self) -> list[dict[str, str]]:
        """Get default suggestions for current step."""
        defaults = {
            ChatStep.QA: [
                {
                    "text": "Are these questions relevant?",
                    "action": "Are these questions good for estimating this PR?",
                    "icon": "❓",
                },
                {
                    "text": "Help answering",
                    "action": "Can you help me answer these questions?",
                    "icon": "💡",
                },
                {
                    "text": "Show similar projects",
                    "action": "Show me similar historical projects",
                    "icon": "📚",
                },
                {
                    "text": "What is [term]?",
                    "action": "What does ATS mean in FPT context?",
                    "icon": "📖",
                },
            ],
            ChatStep.SUMMARY: [
                {
                    "text": "Is this summary correct?",
                    "action": "Does this summary accurately capture the PR?",
                    "icon": "✅",
                },
                {
                    "text": "Explain features",
                    "action": "Explain the extracted features",
                    "icon": "📋",
                },
                {
                    "text": "Compare with similar",
                    "action": "Compare with similar PRs",
                    "icon": "🔄",
                },
            ],
            ChatStep.ESTIMATION: [
                {
                    "text": "Why these hours?",
                    "action": "Explain this estimate - why these hours?",
                    "icon": "⏱️",
                },
                {
                    "text": "Compare to history",
                    "action": "How does this compare to historical projects?",
                    "icon": "📊",
                },
                {
                    "text": "Show applied rules",
                    "action": "What rules were applied to this estimate?",
                    "icon": "📜",
                },
            ],
            ChatStep.REVIEW: [
                {
                    "text": "Suggest correction reason",
                    "action": "Suggest a reason for my correction",
                    "icon": "✏️",
                },
                {
                    "text": "What will be learned?",
                    "action": "What will the system learn from my changes?",
                    "icon": "🧠",
                },
                {
                    "text": "Generate summary",
                    "action": "Generate a summary of the estimation",
                    "icon": "📝",
                },
            ],
        }

        return defaults.get(self.current_step, [])

    # ===== Tool Implementations with Real Vector Search =====

    async def _tool_search_similar_qa(self, user_message: str = "") -> str:
        """Search for similar Q&A from historical projects using vector search."""
        questions = self.state.get("questions", [])
        parsed_pr = self.state.get("parsed_pr", {})

        # Build search query from PR context and current questions
        search_query = ""
        if parsed_pr:
            search_query = (
                f"{parsed_pr.get('title', '')} {parsed_pr.get('description', '')}"
            )
        if questions:
            # Add first unanswered question to search
            unanswered = [q for q in questions if not q.get("answer")]
            if unanswered:
                search_query += f" {unanswered[0].get('question_text', '')}"

        if not search_query.strip():
            search_query = user_message

        try:
            # Search PR embeddings for similar projects
            results = await self._search_vectors(
                collection=PR_EMBEDDINGS,
                query=search_query,
                limit=5,
                score_threshold=0.4,
            )

            if not results:
                # Provide helpful fallback when no similar projects found
                platform = parsed_pr.get("platform", "unknown")
                engine_type = parsed_pr.get("engine_type", "unknown")
                return f"""No similar Q&A found in historical projects.

**This could mean:**
- This is a unique project type for {platform}/{engine_type}
- The knowledge base needs more historical data for this combination
- Try broadening your question or asking about specific aspects

**Suggestions:**
- Ask me to explain FPT terminology if any questions are unclear
- Provide your best estimate based on your expertise
- The system will learn from this project for future reference"""

            # Format results
            output_lines = [
                f"Found {len(results)} similar projects with Q&A history:\n"
            ]

            for i, result in enumerate(results, 1):
                payload = result.get("payload", {})
                score = result.get("score", 0)
                pr_number = payload.get("pr_number", "Unknown")
                title = payload.get("title", "No title")
                platform = payload.get("platform", "Unknown")
                total_cost = payload.get("total_cost", 0)

                output_lines.append(
                    f"{i}. **{pr_number}** - {title}\n"
                    f"   Platform: {platform} | Similarity: {score:.0%} | Cost: €{total_cost:,.0f}"
                )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Similar Q&A search failed: {e}")
            return f"Search temporarily unavailable. Error: {str(e)}"

    async def _tool_generate_question(self, user_message: str = "") -> str:
        """Generate a new clarifying question using LLM."""
        parsed_pr = self.state.get("parsed_pr", {})
        existing_questions = self.state.get("questions", [])

        if not parsed_pr:
            return "Cannot generate questions without PR data."

        try:
            # Use LLM to generate a relevant question
            existing_q_texts = [q.get("question_text", "") for q in existing_questions]

            prompt = f"""Based on this Product Request, generate ONE important clarifying question that hasn't been asked yet.

PR Title: {parsed_pr.get("title", "N/A")}
PR Description: {parsed_pr.get("description", "N/A")[:500]}
Platform: {parsed_pr.get("platform", "N/A")}
Engine Type: {parsed_pr.get("engine_type", "N/A")}

Already asked questions:
{chr(10).join(f"- {q}" for q in existing_q_texts[:5])}

User context: {user_message}

Generate a single, specific question that would help improve cost estimation accuracy.
Return ONLY the question text, nothing else."""

            question = await self.llm.fast_response(prompt)
            return f'Generated question: "{question.strip()}"'

        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return "Could not generate question. Please try again."

    async def _tool_rephrase_question(self, user_message: str = "") -> str:
        """Rephrase a question for clarity using LLM."""
        questions = self.state.get("questions", [])

        if not questions:
            return "No questions available to rephrase."

        # Find the question to rephrase (last one or based on user message)
        target_question = questions[-1].get("question_text", "")

        try:
            prompt = f"""Rephrase this technical question to be clearer and more specific:

Original: {target_question}
User guidance: {user_message}

Return ONLY the rephrased question, nothing else."""

            rephrased = await self.llm.fast_response(prompt)
            return f'Rephrased: "{rephrased.strip()}"'

        except Exception as e:
            logger.error(f"Question rephrasing failed: {e}")
            return "Could not rephrase question. Please try again."

    async def _tool_explain_feature(self, user_message: str = "") -> str:
        """Explain a specific feature using state and knowledge base."""
        pr_summary = self.state.get("pr_summary", {})
        ml_features = self.state.get("ml_features", {})

        if not pr_summary and not ml_features:
            return "No features available to explain. Complete the summary step first."

        # Search knowledge base for relevant explanations
        search_query = (
            user_message if user_message else "feature explanation automotive"
        )

        try:
            knowledge_results = await self._search_vectors(
                collection=KNOWLEDGE_CHUNKS,
                query=search_query,
                limit=3,
                score_threshold=0.3,
            )

            # Build feature summary
            features = pr_summary.get("key_features", [])
            complexity = pr_summary.get("complexity_score", "N/A")
            program_size = pr_summary.get("program_size", "N/A")

            output = [
                "**Feature Analysis:**\n",
                f"- Program Size: {program_size}",
                f"- Complexity Score: {complexity}",
                f"- Key Features: {', '.join(features[:5]) if features else 'Not extracted'}",
            ]

            if knowledge_results:
                output.append("\n**Related Knowledge:**")
                for result in knowledge_results[:2]:
                    payload = result.get("payload", {})
                    title = payload.get("title", "Document")
                    chunk_text = payload.get("chunk_text", "")[:200]
                    output.append(f"- {title}: {chunk_text}...")

            return "\n".join(output)

        except Exception as e:
            logger.error(f"Feature explanation failed: {e}")
            return f"Key features: {pr_summary.get('key_features', ['No features extracted'])}"

    async def _tool_compare_prs(self, user_message: str = "") -> str:
        """Compare current PR with similar historical PRs using vector search."""
        parsed_pr = self.state.get("parsed_pr", {})
        pr_summary = self.state.get("pr_summary", {})

        # Build comparison query
        search_query = f"{parsed_pr.get('title', '')} {parsed_pr.get('platform', '')} {parsed_pr.get('engine_type', '')}"

        if not search_query.strip():
            search_query = user_message

        try:
            # Search for similar PRs
            results = await self._search_vectors(
                collection=PR_EMBEDDINGS,
                query=search_query,
                limit=5,
                score_threshold=0.35,
            )

            if not results:
                return (
                    "No similar PRs found for comparison. This project may be unique."
                )

            # Format comparison table
            output_lines = [
                "**Similar PR Comparison:**\n",
                "| PR # | Title | Platform | Cost | Similarity |",
                "|------|-------|----------|------|------------|",
            ]

            total_costs = []
            for result in results:
                payload = result.get("payload", {})
                score = result.get("score", 0)
                pr_num = payload.get("pr_number", "?")
                title = payload.get("title", "Unknown")[:30]
                platform = payload.get("platform", "?")
                cost = payload.get("total_cost", 0)

                if cost:
                    total_costs.append(cost)

                output_lines.append(
                    f"| {pr_num} | {title} | {platform} | €{cost:,.0f} | {score:.0%} |"
                )

            if total_costs:
                avg_cost = sum(total_costs) / len(total_costs)
                output_lines.append(
                    f"\n**Average cost of similar PRs:** €{avg_cost:,.0f}"
                )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"PR comparison failed: {e}")
            return f"Comparison temporarily unavailable. Error: {str(e)}"

    async def _tool_search_knowledge(self, user_message: str = "") -> str:
        """Search the enterprise knowledge base."""
        search_query = user_message if user_message else "cost estimation guidelines"

        try:
            results = await self._search_vectors(
                collection=KNOWLEDGE_CHUNKS,
                query=search_query,
                limit=5,
                score_threshold=0.3,
            )

            if not results:
                return f"""No relevant knowledge found for: "{search_query}"

**Try searching for:**
- FPT acronyms (ATS, DOC, SCR, VCU, ECU)
- Engine families (CURSOR, NEF, F1C)
- Emissions standards (Stage V, Tier 4B, Euro VI)
- Cost estimation methodology
- Activity categories

**Or ask me directly** - I can explain FPT terminology and cost estimation concepts."""

            output_lines = ["**Knowledge Base Results:**\n"]

            for i, result in enumerate(results, 1):
                payload = result.get("payload", {})
                score = result.get("score", 0)
                title = payload.get("title", "Document")
                doc_type = payload.get("doc_type", "general")
                chunk_text = payload.get("chunk_text", "")[:300]

                output_lines.append(
                    f"{i}. **{title}** ({doc_type}) - {score:.0%} match\n"
                    f"   {chunk_text}...\n"
                )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return f"Knowledge search unavailable. Error: {str(e)}"

    async def _tool_explain_estimate(self, user_message: str = "") -> str:
        """Explain the estimation reasoning using state and similar quotations."""
        breakdown = self.state.get("breakdown", [])
        ml_prediction = self.state.get("ml_prediction", {})
        applied_rules = self.state.get("applied_rules", [])

        if not breakdown:
            return "No estimation available yet. Complete the estimation step first."

        try:
            # Calculate totals
            total_hours = sum(item.get("hours", 0) for item in breakdown)
            total_cost = sum(item.get("cost", 0) for item in breakdown)

            output_lines = [
                "**Estimation Breakdown:**\n",
                f"- Total Hours: {total_hours:,}",
                f"- Total Cost: €{total_cost:,.0f}",
                f"- Number of Activities: {len(breakdown)}",
            ]

            # Add ML prediction info
            if ml_prediction:
                confidence = ml_prediction.get("confidence", 0)
                output_lines.append(f"- ML Confidence: {confidence:.0%}")

            # Add applied rules
            if applied_rules:
                output_lines.append("\n**Applied Rules:**")
                for rule in applied_rules[:5]:
                    rule_name = rule.get("rule_name", "Unknown rule")
                    effect = rule.get("effect_value", 0)
                    output_lines.append(f"- {rule_name}: {effect:+.0%}")

            # Search for similar quotation breakdowns
            if user_message:
                similar = await self._search_vectors(
                    collection=QUOTATION_CHUNKS,
                    query=user_message,
                    limit=3,
                    score_threshold=0.4,
                )
                if similar:
                    output_lines.append("\n**Similar Historical Activities:**")
                    for s in similar:
                        payload = s.get("payload", {})
                        func = payload.get("pe_function", "Unknown")
                        hours = payload.get("hours", 0)
                        output_lines.append(f"- {func}: {hours} hours")

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Estimate explanation failed: {e}")
            return f"Explanation temporarily unavailable. Error: {str(e)}"

    async def _tool_compare_breakdown(self, user_message: str = "") -> str:
        """Compare current breakdown with historical data."""
        breakdown = self.state.get("breakdown", [])

        if not breakdown:
            return "No breakdown available for comparison."

        # Build query from breakdown activities
        activities = [item.get("activity_description", "") for item in breakdown[:3]]
        search_query = " ".join(activities) if activities else user_message

        try:
            results = await self._search_vectors(
                collection=QUOTATION_CHUNKS,
                query=search_query,
                limit=10,
                score_threshold=0.35,
            )

            if not results:
                return "No similar breakdown activities found in historical data."

            output_lines = [
                "**Historical Breakdown Comparison:**\n",
                "| Function | Description | Hist. Hours | Current |",
                "|----------|-------------|-------------|---------|",
            ]

            for result in results[:7]:
                payload = result.get("payload", {})
                func = payload.get("pe_function", "?")[:15]
                desc = payload.get("activity_description", "")[:25]
                hist_hours = payload.get("hours", 0)

                # Find matching current activity
                current_hours = "N/A"
                for item in breakdown:
                    if func.lower() in item.get("pe_function", "").lower():
                        current_hours = str(item.get("hours", 0))
                        break

                output_lines.append(
                    f"| {func} | {desc} | {hist_hours} | {current_hours} |"
                )

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Breakdown comparison failed: {e}")
            return f"Comparison unavailable. Error: {str(e)}"

    async def _tool_show_rules(self, user_message: str = "") -> str:
        """Show applied rules from the estimation."""
        applied_rules = self.state.get("applied_rules", [])

        if not applied_rules:
            # Search for relevant rules in feedback patterns
            try:
                parsed_pr = self.state.get("parsed_pr", {})
                search_query = f"{parsed_pr.get('platform', '')} {parsed_pr.get('engine_type', '')} rules"

                results = await self._search_vectors(
                    collection=FEEDBACK_PATTERNS,
                    query=search_query,
                    limit=5,
                    score_threshold=0.3,
                )

                if results:
                    output_lines = [
                        "**Potentially Applicable Rules (from history):**\n"
                    ]
                    for r in results:
                        payload = r.get("payload", {})
                        category = payload.get("reason_category", "General")
                        text = payload.get("reason_text", "")[:100]
                        change = payload.get("change_percentage", 0)
                        output_lines.append(
                            f"- **{category}**: {text} ({change:+.0f}%)"
                        )
                    return "\n".join(output_lines)

            except Exception as e:
                logger.error(f"Rules search failed: {e}")

            return "No rules were applied to this estimation. This may indicate a standard case."

        output_lines = ["**Applied Rules:**\n"]
        for rule in applied_rules:
            rule_name = rule.get("rule_name", "Unknown")
            description = rule.get("rule_description", "No description")
            effect = rule.get("effect_value", 0)
            confidence = rule.get("confidence", 0)

            output_lines.append(
                f"- **{rule_name}** ({confidence:.0%} confidence)\n"
                f"  {description}\n"
                f"  Effect: {effect:+.0%}\n"
            )

        return "\n".join(output_lines)

    async def _tool_suggest_reason(self, user_message: str = "") -> str:
        """Suggest reasons for user corrections using LLM and historical patterns."""
        user_edits = self.state.get("user_edits", [])

        # Search historical feedback patterns for similar corrections
        try:
            if user_message:
                results = await self._search_vectors(
                    collection=FEEDBACK_PATTERNS,
                    query=user_message,
                    limit=5,
                    score_threshold=0.3,
                )

                if results:
                    output_lines = [
                        "**Suggested reasons based on historical corrections:**\n"
                    ]
                    categories = set()
                    for r in results:
                        payload = r.get("payload", {})
                        category = payload.get("reason_category", "")
                        if category:
                            categories.add(category)

                    if categories:
                        output_lines.append("Common categories:")
                        for cat in list(categories)[:5]:
                            output_lines.append(f"- {cat}")

                    return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"Reason suggestion search failed: {e}")

        # Fallback to standard suggestions
        return """**Suggested correction reasons:**
- Historical precedent - similar projects took more/less time
- Team expertise - specialized skills affect estimation
- Scope clarification - requirements were adjusted
- Technical complexity - implementation harder than expected
- External dependencies - third-party constraints
- Regulatory requirements - compliance overhead"""

    async def _tool_preview_learning(self, user_message: str = "") -> str:
        """Preview what the system will learn from corrections."""
        user_edits = self.state.get("user_edits", [])

        if not user_edits:
            return "No corrections made yet. Make changes in the review step to see what the system will learn."

        output_lines = [
            f"**Learning Preview ({len(user_edits)} corrections):**\n",
            "The system will analyze these corrections to:",
        ]

        # Analyze edit patterns
        total_increase = 0
        total_decrease = 0

        for edit in user_edits:
            original = edit.get("original_value", 0)
            corrected = edit.get("corrected_value", 0)
            diff = corrected - original

            if diff > 0:
                total_increase += diff
            else:
                total_decrease += abs(diff)

        if total_increase > 0:
            output_lines.append(
                f"- Identify patterns where estimates need +{total_increase:.0f}h adjustment"
            )
        if total_decrease > 0:
            output_lines.append(
                f"- Identify patterns where estimates need -{total_decrease:.0f}h adjustment"
            )

        output_lines.append("\n**What happens next:**")
        output_lines.append("1. Immediate: Rules extracted for similar future projects")
        output_lines.append(
            "2. Batch: Model retrained when enough corrections accumulate"
        )
        output_lines.append("3. Validation: New model tested before deployment")

        return "\n".join(output_lines)

    async def _tool_generate_summary(self, user_message: str = "") -> str:
        """Generate a summary of the estimation session."""
        pr_summary = self.state.get("pr_summary", {})
        breakdown = self.state.get("breakdown", [])
        user_edits = self.state.get("user_edits", [])
        applied_rules = self.state.get("applied_rules", [])

        if not breakdown:
            return "No estimation data available for summary."

        total_hours = sum(item.get("hours", 0) for item in breakdown)
        total_cost = sum(item.get("cost", 0) for item in breakdown)
        edits_count = len(user_edits) if user_edits else 0
        rules_count = len(applied_rules) if applied_rules else 0

        # Use LLM to generate a narrative summary
        try:
            prompt = f"""Generate a concise executive summary for this cost estimation:

Project: {pr_summary.get("title", "Unknown")}
Platform: {pr_summary.get("platform", "Unknown")}
Program Size: {pr_summary.get("program_size", "Unknown")}

Key Metrics:
- Total Hours: {total_hours:,}
- Total Cost: €{total_cost:,.0f}
- Activities: {len(breakdown)}
- Rules Applied: {rules_count}
- Manual Corrections: {edits_count}

Generate a 3-4 sentence executive summary highlighting key points."""

            summary = await self.llm.fast_response(prompt)

            return f"""**Estimation Summary:**

{summary}

**Quick Stats:**
- Total: €{total_cost:,.0f} ({total_hours:,} hours)
- Activities: {len(breakdown)}
- Corrections: {edits_count}
- Rules Applied: {rules_count}"""

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"""**Estimation Summary:**

- Total Cost: €{total_cost:,.0f}
- Total Hours: {total_hours:,}
- Activities: {len(breakdown)}
- Manual Corrections: {edits_count}
- Rules Applied: {rules_count}"""

"""
FPT Cost Brain 2.0 - Legendary FPT Engineering Agent
Domain-expert AI companion for R&D cost estimation

Architecture:
- BRAIN: RAG-first knowledge retrieval + Intent classification
- HANDS: Deterministic EstimationTools for hard calculations

This agent combines:
- RAG-first knowledge (no hardcoded domain info)
- Strategic tool execution based on intent
- Unified response generation
"""

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from llm.client import get_llm_client
from vector.client import get_vector_store
from vector.collections import (
    FEEDBACK_PATTERNS,
    KNOWLEDGE_CHUNKS,
    PR_EMBEDDINGS,
    QUOTATION_CHUNKS,
)

from agents.state import EstimationState
from agents.tools import EstimationTools

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Classification of user intent for routing."""

    META = "meta"  # Questions about the system/process
    TERMINOLOGY = "terminology"  # FPT acronyms, definitions
    PR_SPECIFIC = "pr_specific"  # Questions about current PR
    COMPARISON = "comparison"  # Compare with historical data
    ESTIMATION = "estimation"  # Cost/hours questions
    GUIDANCE = "guidance"  # How-to, next steps
    GENERAL = "general"  # General conversation
    MODIFY_STATE = "modify_state"  # GOD MODE: User wants to change state


@dataclass
class RAGContext:
    """Container for RAG-retrieved context."""

    acronyms: list[dict] = field(default_factory=list)
    knowledge: list[dict] = field(default_factory=list)
    similar_prs: list[dict] = field(default_factory=list)
    feedback_patterns: list[dict] = field(default_factory=list)
    base_terminology: list[dict] = field(default_factory=list)  # Loaded at init

    def has_context(self) -> bool:
        return bool(
            self.acronyms
            or self.knowledge
            or self.similar_prs
            or self.feedback_patterns
            or self.base_terminology
        )

    def format_for_prompt(self) -> str:
        """Format RAG context for inclusion in LLM prompt."""
        parts = []

        # Base terminology (loaded from RAG at init)
        if self.base_terminology:
            parts.append("## FPT Terminology from Knowledge Base")
            for item in self.base_terminology[:10]:
                acronym = item.get("acronym", "")
                full_form = item.get("full_form", "")
                if acronym and full_form:
                    parts.append(f"- **{acronym}**: {full_form}")
            parts.append("")

        # Query-specific acronyms
        if self.acronyms:
            parts.append("## Relevant Acronyms (from your question)")
            for item in self.acronyms[:5]:
                acronym = item.get("acronym", "")
                full_form = item.get("full_form", "")
                if acronym and full_form:
                    parts.append(f"- **{acronym}**: {full_form}")
            parts.append("")

        if self.knowledge:
            parts.append("## Retrieved Knowledge")
            for item in self.knowledge[:3]:
                title = item.get("title", "Document")
                text = item.get("chunk_text", "")[:250]
                parts.append(f"### {title}")
                parts.append(f"{text}...")
                parts.append("")

        if self.similar_prs:
            parts.append("## Similar Historical Projects")
            for item in self.similar_prs[:3]:
                pr_num = item.get("pr_number", "?")
                title = item.get("title", "")[:50]
                cost = item.get("total_cost", 0)
                platform = item.get("platform", "?")
                parts.append(
                    f"- **{pr_num}**: {title} | Platform: {platform} | Cost: €{cost:,.0f}"
                )
            parts.append("")

        if self.feedback_patterns:
            parts.append("## Learned Patterns from Corrections")
            for item in self.feedback_patterns[:3]:
                category = item.get("reason_category", "")
                text = item.get("reason_text", "")[:100]
                change = item.get("change_percentage", 0)
                parts.append(f"- {category}: {text} ({change:+.0f}%)")

        return (
            "\n".join(parts)
            if parts
            else "No relevant context found in knowledge base."
        )


class FPTEngineerAgent:
    """
    Legendary FPT Engineering Agent.

    Architecture:
    - BRAIN: RAG-first knowledge retrieval + Intent classification
    - HANDS: Deterministic EstimationTools for hard calculations

    Combines semantic intelligence with deterministic capabilities:
    - Base terminology loaded from Qdrant at initialization
    - Query-specific context retrieved per message
    - Strategic tool execution based on intent + context
    - No hardcoded domain knowledge

    Modes:
    - agent_mode=False (default): Read-only assistant, keyword-based modification detection
    - agent_mode=True: Full GOD MODE with LLM-based intent understanding for modifications
    """

    def __init__(self, state: EstimationState, agent_mode: bool = False):
        self.state = state
        self.llm = get_llm_client()
        self.current_step = state.get("current_step", "qa")
        self.agent_mode = agent_mode  # GOD MODE toggle
        self._vector_store = None
        self._base_terminology: list[dict] = []  # Loaded from RAG
        self._initialized = False

        # The "Hands" - deterministic tools for hard calculations
        self.tools = EstimationTools()

        if agent_mode:
            logger.info("[GOD MODE] Agent initialized with write capabilities enabled")

    async def _ensure_initialized(self):
        """Lazy initialization - load base terminology from RAG."""
        if self._initialized:
            return

        try:
            # Load common FPT acronyms from knowledge base
            self._base_terminology = await self._load_base_terminology()
            logger.info(f"Loaded {len(self._base_terminology)} base terms from RAG")
        except Exception as e:
            logger.warning(f"Could not load base terminology: {e}")
            self._base_terminology = []

        self._initialized = True

    async def _get_vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            self._vector_store = await get_vector_store()
        return self._vector_store

    async def _load_base_terminology(self) -> list[dict]:
        """Load common FPT terminology from RAG at initialization."""
        try:
            vector_store = await self._get_vector_store()

            # Search for acronym-type documents
            # Use a general query to get common terms
            query_vector = await self.llm.embed(
                "FPT Industrial acronyms terminology ATS SCR DOC ECU engine emissions"
            )

            results = await vector_store.search(
                collection=KNOWLEDGE_CHUNKS,
                query_vector=query_vector,
                limit=20,
                filter_conditions={"doc_type": "acronym"},
                score_threshold=0.2,
            )

            return [
                {**r.get("payload", {}), "score": r.get("score", 0)} for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to load base terminology: {e}")
            return []

    # ===== Intent Classification =====

    async def classify_intent_with_llm(self, message: str) -> IntentType:
        """
        LLM-based intent classification for AGENT MODE.

        Uses natural language understanding to detect if user wants to:
        - Modify questions, answers, estimates, or other state
        - Ask questions about the process
        - Get information or guidance

        This is more accurate than keyword matching but slower.
        Only used when agent_mode=True.
        """
        # Build context about what can be modified in current step
        step_capabilities = {
            "qa": "questions (regenerate, add, remove, answer), question priorities",
            "summary": "complexity level, platform, engine type, tier/size, PR fields",
            "estimation": "activity hours, cost breakdown, confidence levels",
            "review": "activity hours, cost breakdown, add corrections with reasons",
        }

        current_capabilities = step_capabilities.get(
            self.current_step, "limited modifications"
        )

        prompt = f"""You are classifying user intent in a cost estimation system.

CURRENT STEP: {self.current_step}
MODIFIABLE IN THIS STEP: {current_capabilities}

USER MESSAGE: "{message}"

Classify the intent as ONE of these categories:

1. MODIFY_STATE - User wants to CHANGE something (add, remove, update, regenerate, improve, set, fix, correct, redo, make better, etc.)
   Examples: "regenerate the questions", "make these better", "add 50 hours to testing", "change complexity to high", "these questions aren't relevant"

2. META - Questions about the system, process, or what to do next
   Examples: "what should I do now?", "is this correct?", "help me understand"

3. TERMINOLOGY - Asking about FPT terms, acronyms, definitions
   Examples: "what does SCR mean?", "explain ATS"

4. PR_SPECIFIC - Questions about the current PR/project
   Examples: "what platform is this?", "tell me about this project"

5. COMPARISON - Comparing with historical data
   Examples: "show similar projects", "how does this compare?"

6. ESTIMATION - Questions about cost/hours/estimates
   Examples: "why so many hours?", "explain the confidence"

7. GUIDANCE - How-to questions
   Examples: "how do I proceed?", "what's the best approach?"

8. GENERAL - General conversation or unclear

IMPORTANT: In this AGENT MODE, if the user expresses ANY dissatisfaction or desire for change (even implicitly like "these questions don't make sense" or "this isn't right"), classify as MODIFY_STATE.

Respond with ONLY the intent category name (e.g., "MODIFY_STATE" or "TERMINOLOGY")."""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )

            intent_str = response.strip().upper().replace(" ", "_")

            # Map to IntentType enum
            intent_map = {
                "MODIFY_STATE": IntentType.MODIFY_STATE,
                "META": IntentType.META,
                "TERMINOLOGY": IntentType.TERMINOLOGY,
                "PR_SPECIFIC": IntentType.PR_SPECIFIC,
                "COMPARISON": IntentType.COMPARISON,
                "ESTIMATION": IntentType.ESTIMATION,
                "GUIDANCE": IntentType.GUIDANCE,
                "GENERAL": IntentType.GENERAL,
            }

            intent = intent_map.get(intent_str, IntentType.GENERAL)
            logger.info(
                f"[AGENT MODE] LLM classified intent: {intent.value} (raw: {intent_str})"
            )
            return intent

        except Exception as e:
            logger.error(f"[AGENT MODE] LLM intent classification failed: {e}")
            # Fallback to keyword-based classification
            return self._classify_intent_keywords(message)

    def _classify_intent_keywords(self, message: str) -> IntentType:
        """
        Keyword-based intent classification (fast but limited).

        Used in normal chat mode or as fallback when LLM fails.
        GOD MODE: MODIFY_STATE has highest priority - detected first.
        """
        msg_lower = message.lower()

        # GOD MODE: Detect state modification intent (highest priority)
        # These are ACTION verbs that indicate user wants to CHANGE something
        modify_patterns = [
            # Direct modification commands
            r"(change|update|set|modify|edit|adjust|fix|correct)\s+(the|this|my)?",
            r"(add|remove|delete|increase|decrease|reduce)\s+\d*\s*(hours?|h\b|activities?)?",
            r"(recalculate|regenerate|redo|rewrite|rework|refresh)\s+(the|this|my)?",
            # Specific field changes
            r"(make|mark)\s+(it|this|the)\s+(high|medium|low|complex)",
            r"(set|change).+(to|as)\s+\d+",
            r"(add|put|include)\s+\d+\s*(more\s+)?(hours?|h\b)",
            # Imperative commands
            r"^(update|change|add|remove|set|modify|regenerate|recalculate)\b",
            # Question modification (GOD MODE for Q&A step)
            r"(improve|enhance|refine|better|rewrite|regenerate|generate|create)\s+(the\s+)?(questions?)",
            r"(make|give).+(better|improved|new)\s+(questions?)",
            r"(new|different|more)\s+questions?",
        ]
        for pattern in modify_patterns:
            if re.search(pattern, msg_lower):
                return IntentType.MODIFY_STATE

        # Quick keyword check for modification
        modify_verbs = [
            "change",
            "update",
            "set",
            "modify",
            "add",
            "remove",
            "delete",
            "recalculate",
            "regenerate",
            "adjust",
            "fix",
            "increase",
            "decrease",
            "reduce",
            "correct",
            "improve",  # GOD MODE: User wants to improve something
            "enhance",
            "refine",
            "rewrite",
            "redo",
            "refresh",
            "generate",  # "generate new questions"
        ]
        if any(verb in msg_lower for verb in modify_verbs):
            # Check if it's a command (starts with verb or has "please" nearby)
            words = msg_lower.split()
            if words and words[0] in modify_verbs:
                return IntentType.MODIFY_STATE
            if "please" in msg_lower and any(
                verb in msg_lower for verb in modify_verbs
            ):
                return IntentType.MODIFY_STATE

        # Meta questions - about the system/process itself
        meta_patterns = [
            r"should (i|we) (regenerate|redo|change|skip)",
            r"(is|are) (this|these|the) (question|answer|summary|estimate)s? (good|correct|enough)",
            r"what (should|can) (i|we) do",
            r"how does (this|the system) work",
            r"next step",
            r"am i doing (this|it) right",
            r"help me (understand|with)",
            r"what happens (if|when|next)",
        ]
        for pattern in meta_patterns:
            if re.search(pattern, msg_lower):
                return IntentType.META

        # Terminology - asking about acronyms/definitions
        terminology_patterns = [
            r"what (is|does|are) [A-Z]{2,}",
            r"what does .+ (mean|stand for)",
            r"explain .+ (term|acronym|abbreviation)",
            r"define [A-Z]{2,}",
            r"(meaning|definition) of",
        ]
        for pattern in terminology_patterns:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                return IntentType.TERMINOLOGY

        # Check for explicit acronyms in message
        acronyms_found = re.findall(r"\b[A-Z]{2,6}\b", message)
        if acronyms_found and any(
            word in msg_lower for word in ["what", "explain", "mean", "?"]
        ):
            return IntentType.TERMINOLOGY

        # PR-specific questions
        pr_patterns = [
            r"(this|the|current) (pr|project|request)",
            r"(about|regarding) (the|this) (pr|project)",
            r"pr.?(code|number|title|description)",
            r"(what|which) (platform|engine|family|customer)",
        ]
        for pattern in pr_patterns:
            if re.search(pattern, msg_lower):
                return IntentType.PR_SPECIFIC

        # Comparison requests
        comparison_patterns = [
            r"(compare|similar|like) (this|to|with)",
            r"historical (project|pr|data)",
            r"(past|previous|other) (project|pr|estimate)",
            r"how does (this|it) compare",
        ]
        for pattern in comparison_patterns:
            if re.search(pattern, msg_lower):
                return IntentType.COMPARISON

        # Estimation questions
        estimation_patterns = [
            r"(how many|estimated|total) hours",
            r"(cost|budget|price|expense)",
            r"why (this|these) (hour|estimate|cost)",
            r"(confidence|accuracy|reliable)",
            r"(breakdown|activity|activities)",
        ]
        for pattern in estimation_patterns:
            if re.search(pattern, msg_lower):
                return IntentType.ESTIMATION

        # Guidance requests
        guidance_patterns = [
            r"how (do|can|should) (i|we)",
            r"(help|guide|assist) (me|us)",
            r"what.+recommend",
            r"(best|correct) (way|approach)",
        ]
        for pattern in guidance_patterns:
            if re.search(pattern, msg_lower):
                return IntentType.GUIDANCE

        return IntentType.GENERAL

    # ===== RAG Context Retrieval =====

    async def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for query."""
        return await self.llm.embed(query)

    async def _search_collection(
        self,
        collection: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.3,
        filter_conditions: dict | None = None,
    ) -> list[dict]:
        """Search a vector collection."""
        try:
            query_vector = await self._embed_query(query)
            vector_store = await self._get_vector_store()

            results = await vector_store.search(
                collection=collection,
                query_vector=query_vector,
                limit=limit,
                filter_conditions=filter_conditions,
                score_threshold=score_threshold,
            )

            return [
                {**r.get("payload", {}), "score": r.get("score", 0)} for r in results
            ]
        except Exception as e:
            logger.error(f"Search failed in {collection}: {e}")
            return []

    def _extract_acronyms(self, text: str) -> list[str]:
        """Extract potential acronyms from text."""
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
        common_words = {
            "PR",
            "OK",
            "EU",
            "US",
            "UK",
            "IT",
            "IS",
            "AS",
            "AT",
            "OR",
            "AN",
            "TO",
            "IF",
        }
        return [a for a in acronyms if a not in common_words]

    async def gather_rag_context(
        self,
        message: str,
        intent: IntentType,
    ) -> RAGContext:
        """
        Gather relevant RAG context based on message and intent.
        RAG-FIRST: All knowledge comes from vector database.
        """
        await self._ensure_initialized()

        context = RAGContext()
        context.base_terminology = self._base_terminology

        # Always look up acronyms in the message
        acronyms_in_message = self._extract_acronyms(message)
        if acronyms_in_message:
            for acronym in acronyms_in_message[:3]:
                results = await self._search_collection(
                    KNOWLEDGE_CHUNKS,
                    f"FPT acronym {acronym} meaning definition stands for",
                    limit=2,
                    score_threshold=0.35,
                    filter_conditions={"doc_type": "acronym"},
                )
                for r in results:
                    if r.get("acronym", "").upper() == acronym.upper():
                        context.acronyms.append(r)
                        break

        # Intent-specific retrieval
        if intent == IntentType.TERMINOLOGY:
            # Deep search for terminology
            knowledge = await self._search_collection(
                KNOWLEDGE_CHUNKS,
                message,
                limit=5,
                score_threshold=0.3,
            )
            context.knowledge = knowledge

        elif intent == IntentType.COMPARISON:
            # Search similar PRs
            parsed_pr = self.state.get("parsed_pr", {})
            search_query = f"{parsed_pr.get('title', '')} {parsed_pr.get('platform', '')} {message}"

            similar = await self._search_collection(
                PR_EMBEDDINGS,
                search_query,
                limit=5,
                score_threshold=0.35,
            )
            context.similar_prs = similar

        elif intent == IntentType.ESTIMATION:
            # Search quotation history and feedback patterns
            breakdown = self.state.get("breakdown", [])
            activity_names = [b.get("activity_name", "") for b in breakdown[:3]]
            search_query = f"{' '.join(activity_names)} {message}"

            quotations = await self._search_collection(
                QUOTATION_CHUNKS,
                search_query,
                limit=5,
                score_threshold=0.35,
            )
            context.similar_prs = quotations

            patterns = await self._search_collection(
                FEEDBACK_PATTERNS,
                message,
                limit=3,
                score_threshold=0.3,
            )
            context.feedback_patterns = patterns

        elif intent in [
            IntentType.PR_SPECIFIC,
            IntentType.GENERAL,
            IntentType.GUIDANCE,
        ]:
            # General knowledge search
            knowledge = await self._search_collection(
                KNOWLEDGE_CHUNKS,
                message,
                limit=3,
                score_threshold=0.3,
            )
            context.knowledge = knowledge

        return context

    # ===== Response Generation =====

    def _build_system_prompt(self, intent: IntentType, rag_context: RAGContext) -> str:
        """Build the system prompt - knowledge comes from RAG context."""

        step_context = {
            "qa": "helping the user answer clarifying questions about the PR",
            "summary": "reviewing and validating the PR summary and extracted features",
            "estimation": "explaining and refining the cost estimation breakdown",
            "review": "finalizing estimates and capturing learning from corrections",
        }

        current_task = step_context.get(
            self.current_step, "assisting with cost estimation"
        )

        prompt = f"""You are **FPT Engineering Assistant**, an expert AI companion for R&D cost estimation.

## Your Role
- A knowledgeable, friendly engineering colleague
- You communicate clearly and concisely
- You're helpful but professional

## Current Context
- **Step**: {self.current_step.upper()} - {current_task}
- **Your Goal**: Help the user successfully complete this step

## IMPORTANT: Knowledge Source
Your domain knowledge comes ONLY from the RAG context below.
- Do NOT make up acronym meanings or technical details
- If information is not in the RAG context, say "I don't have that in my knowledge base"
- Always prefer RAG context over assumptions

## RAG Context (Your Knowledge Base)
{rag_context.format_for_prompt()}

## Response Guidelines by Intent Type

**META questions** (about the system/process):
- Answer directly about the current step, questions, or process
- Guide the user on what to do next
- Be reassuring and helpful

**TERMINOLOGY questions**:
- Use ONLY definitions from RAG context above
- If term not found, say "I couldn't find that term in the knowledge base"
- Explain why the term matters for cost estimation

**PR_SPECIFIC questions**:
- Reference the current PR data from session context
- Explain what was detected and why
- Suggest corrections if something seems wrong

**COMPARISON questions**:
- Use historical data from RAG context
- Highlight relevant similarities and differences
- Provide cost benchmarks when available

**ESTIMATION questions**:
- Explain the reasoning behind estimates
- Reference applied rules and learned patterns
- Suggest adjustments if appropriate

**GUIDANCE questions**:
- Provide step-by-step help
- Be practical and actionable

## Style
- Concise but complete
- Use bullet points for clarity
- Conversational yet professional
- When mentioning acronyms, include the full form from RAG context
"""

        return prompt

    def _build_context_message(self) -> str:
        """Build context message from current state with full visibility into page content."""
        parts = ["## Current Session Data"]

        # PR Details
        parsed_pr = self.state.get("parsed_pr", {})
        if parsed_pr:
            parts.append(
                f"- **PR**: {parsed_pr.get('pr_code', 'N/A')} - {parsed_pr.get('title', 'N/A')}"
            )
            parts.append(f"- **Platform**: {parsed_pr.get('platform', 'N/A')}")
            parts.append(f"- **Engine**: {parsed_pr.get('engine_type', 'N/A')}")
            parts.append(f"- **Customer**: {parsed_pr.get('customer', 'N/A')}")
            if parsed_pr.get("description"):
                parts.append(
                    f"- **Description**: {str(parsed_pr.get('description'))[:200]}..."
                )

        # FULL QUESTION DETAILS - This is what user sees on screen
        questions = self.state.get("questions", [])
        if questions:
            answered = sum(1 for q in questions if q.get("answer"))
            parts.append(
                f"\n### QUESTIONS ON SCREEN ({answered}/{len(questions)} answered):"
            )
            for i, q in enumerate(questions, 1):
                q_text = q.get("question_text", q.get("text", str(q)))
                q_answer = q.get("answer", "")
                q_category = q.get("category", "")
                if q_answer:
                    parts.append(f"  Q{i} ({q_category}): {q_text}")
                    parts.append(f"      → Answer: {q_answer}")
                else:
                    parts.append(f"  Q{i} ({q_category}): {q_text} [UNANSWERED]")

        # Breakdown summary with key activities
        breakdown = self.state.get("breakdown", [])
        if breakdown:
            total_hours = sum(b.get("hours", 0) for b in breakdown)
            total_cost = sum(b.get("cost_eur", 0) for b in breakdown)
            parts.append(
                f"\n### COST BREAKDOWN: {total_hours:,.0f} hours (€{total_cost:,.0f})"
            )
            # Show top activities by hours
            sorted_breakdown = sorted(
                breakdown, key=lambda x: x.get("hours", 0), reverse=True
            )
            for item in sorted_breakdown[:7]:
                code = item.get("activity_code", "")
                name = item.get("activity_name", "Unknown Activity")
                hours = item.get("hours", 0)
                cost = item.get("cost_eur", 0)
                confidence = item.get("confidence_score", 0) * 100
                reasoning = item.get("reasoning", "")[:60]
                edited = " [EDITED]" if item.get("user_edited") else ""
                parts.append(
                    f"  - **{code}** {name}: {hours:,.0f}h (€{cost:,.0f}) [{confidence:.0f}% conf]{edited}"
                )
                if reasoning:
                    parts.append(f"    Reasoning: {reasoning}...")
            if len(breakdown) > 7:
                remaining_hours = sum(b.get("hours", 0) for b in sorted_breakdown[7:])
                parts.append(
                    f"  ... and {len(breakdown) - 7} more activities ({remaining_hours:,.0f}h total)"
                )

        # ML Prediction with Sizing Information
        ml_prediction = self.state.get("ml_prediction", {})
        if ml_prediction:
            parts.append("\n### ML PREDICTION (HCQE Model)")
            pred_hours = ml_prediction.get("predicted_total_hours", 0)
            pred_cost = ml_prediction.get("predicted_cost_keur", 0)
            confidence = ml_prediction.get("confidence", 0) * 100
            method = ml_prediction.get("method", "unknown")
            parts.append(
                f"  - **Predicted**: {pred_hours:,.0f} hours (€{pred_cost:,.0f}K) [{confidence:.0f}% confidence]"
            )
            parts.append(f"  - **Method**: {method}")

            # Prediction interval
            interval = ml_prediction.get("prediction_interval", {})
            if interval:
                lower_h = interval.get("lower_hours", 0)
                upper_h = interval.get("upper_hours", 0)
                parts.append(
                    f"  - **Range**: {lower_h:,.0f}h - {upper_h:,.0f}h (90% interval)"
                )

            # Program Sizing - IMPORTANT for user context
            sizing = ml_prediction.get("sizing", {})
            if sizing:
                predicted_size = sizing.get("predicted", "Unknown")
                size_conf = sizing.get("confidence", 0) * 100
                parts.append(
                    f"  - **Program Sizing**: {predicted_size} [{size_conf:.0f}% confidence]"
                )
                # Show size probabilities if available
                size_probs = sizing.get("probabilities", {})
                if size_probs:
                    prob_str = ", ".join(
                        f"{k}: {v * 100:.0f}%"
                        for k, v in sorted(size_probs.items(), key=lambda x: -x[1])[:3]
                    )
                    parts.append(f"    Probabilities: {prob_str}")

            # Reasoning from model
            reasoning = ml_prediction.get("reasoning", "")
            if reasoning:
                parts.append(f"  - **Model Reasoning**: {reasoning[:150]}...")

        # User corrections
        user_edits = self.state.get("user_edits", [])
        if user_edits:
            parts.append(f"\n### USER CORRECTIONS ({len(user_edits)}):")
            for edit in user_edits[:3]:
                orig = edit.get("original_hours", 0)
                new = edit.get("new_hours", 0)
                reason = edit.get("reason", "")[:50]
                parts.append(f"  - Changed {orig}h → {new}h: {reason}")

        # Applied rules
        applied_rules = self.state.get("applied_rules", [])
        if applied_rules:
            parts.append(f"\n### APPLIED RULES ({len(applied_rules)}):")
            for rule in applied_rules[:3]:
                name = rule.get("rule_name", "Unknown")
                effect = rule.get("effect_value", 0)
                parts.append(f"  - {name}: {effect:+.0%}")

        return "\n".join(parts)

    # ===== Strategic Tool Execution (The Bridge) =====

    def _execute_strategic_tools(
        self,
        intent: IntentType,
        message: str,
        rag_context: RAGContext,
    ) -> str | None:
        """
        The Bridge: Decides if we need hard math/logic based on Intent + Context.

        This method connects the BRAIN (RAG/Intent) with the HANDS (Tools).
        It analyzes the intent and message to determine which deterministic
        tools should be executed to provide analytical data.

        Args:
            intent: Classified user intent
            message: User's message
            rag_context: Retrieved RAG context

        Returns:
            Tool output string if tools were executed, None otherwise
        """
        msg_lower = message.lower()
        tool_outputs = []

        # ===== ESTIMATION Intent Strategies =====
        if intent == IntentType.ESTIMATION:
            breakdown = self.state.get("breakdown", [])

            # Strategy: Compare breakdown with historical
            if any(
                word in msg_lower
                for word in ["compare", "breakdown", "historical", "similar"]
            ):
                if breakdown and rag_context.similar_prs:
                    result = self.tools.compare_breakdown(
                        breakdown, rag_context.similar_prs
                    )
                    tool_outputs.append(result)

            # Strategy: Explain estimation
            if any(
                word in msg_lower for word in ["explain", "why", "how", "hours", "cost"]
            ):
                if breakdown:
                    result = self.tools.explain_estimate(
                        breakdown=breakdown,
                        ml_prediction=self.state.get("ml_prediction"),
                        applied_rules=self.state.get("applied_rules", []),
                        historical_activities=rag_context.similar_prs[:5]
                        if rag_context.similar_prs
                        else None,
                    )
                    tool_outputs.append(result)

            # Strategy: Show rules
            if any(word in msg_lower for word in ["rule", "adjustment", "applied"]):
                result = self.tools.format_rules(
                    applied_rules=self.state.get("applied_rules", []),
                    potential_rules=rag_context.feedback_patterns,
                )
                tool_outputs.append(result)

            # Strategy: Show breakdown table
            if any(
                word in msg_lower
                for word in ["table", "breakdown", "activities", "list"]
            ):
                if breakdown:
                    result = self.tools.format_breakdown_table(breakdown)
                    tool_outputs.append(result)

        # ===== COMPARISON Intent Strategies =====
        elif intent == IntentType.COMPARISON:
            parsed_pr = self.state.get("parsed_pr", {})

            # Strategy: Compare PRs
            if rag_context.similar_prs:
                result = self.tools.format_pr_comparison(
                    parsed_pr, rag_context.similar_prs
                )
                tool_outputs.append(result)

            # Strategy: Compare breakdown if available
            breakdown = self.state.get("breakdown", [])
            if breakdown and rag_context.similar_prs:
                result = self.tools.compare_breakdown(
                    breakdown, rag_context.similar_prs
                )
                tool_outputs.append(result)

        # ===== REVIEW Step Strategies =====
        if self.current_step == "review":
            user_edits = self.state.get("user_edits", [])

            # Strategy: Preview learning
            if any(
                word in msg_lower
                for word in ["learn", "training", "correction", "impact"]
            ):
                if user_edits:
                    result = self.tools.preview_learning(user_edits)
                    tool_outputs.append(result)

            # Strategy: Suggest reasons
            if any(
                word in msg_lower
                for word in ["reason", "why", "justify", "explain change"]
            ):
                result = self.tools.suggest_correction_reasons()
                tool_outputs.append(result)

            # Strategy: Generate summary stats
            if any(
                word in msg_lower for word in ["summary", "stats", "overview", "total"]
            ):
                breakdown = self.state.get("breakdown", [])
                if breakdown:
                    result = self.tools.generate_summary_stats(
                        breakdown=breakdown,
                        user_edits=user_edits,
                        applied_rules=self.state.get("applied_rules", []),
                        pr_summary=self.state.get("pr_summary", {}),
                    )
                    tool_outputs.append(result)

        # ===== Q&A Step Strategies =====
        if self.current_step == "qa":
            questions = self.state.get("questions", [])

            # Strategy: Show question status
            if any(
                word in msg_lower
                for word in ["question", "status", "progress", "answer"]
            ):
                if questions:
                    result = self.tools.format_questions_status(questions)
                    tool_outputs.append(result)

        # Return combined tool outputs or None
        if tool_outputs:
            return "\n\n---\n\n".join(tool_outputs)
        return None

    # ===== GOD MODE: State Action Handlers =====

    async def _execute_state_action(
        self,
        intent: IntentType,
        message: str,
    ) -> dict | None:
        """
        GOD MODE: The Action Dispatcher.

        Routes modification requests to the correct handler based on current step.
        Only executes when intent is MODIFY_STATE.

        Args:
            intent: Classified intent (should be MODIFY_STATE)
            message: User's natural language command

        Returns:
            Action result dict with status, action_type, details, and updated_state
            None if no action was taken
        """
        if intent != IntentType.MODIFY_STATE:
            return None

        msg_lower = message.lower()
        logger.info(f"[GOD MODE] Executing state action for step: {self.current_step}")

        # Route to appropriate handler based on current step
        if self.current_step == "qa":
            return await self._action_update_questions(msg_lower, message)
        elif self.current_step == "summary":
            return await self._action_update_summary(msg_lower, message)
        elif self.current_step in ["estimation", "review"]:
            return await self._action_update_estimation(msg_lower, message)
        else:
            logger.warning(f"[GOD MODE] Unknown step: {self.current_step}")
            return {
                "status": "error",
                "action_type": "unknown",
                "details": f"Cannot modify state in step: {self.current_step}",
                "updated_state": None,
            }

    # ===== Selective Question Regeneration Helpers =====

    def _parse_question_numbers(self, message: str) -> list[int]:
        """
        Parse question numbers from natural language.

        Supports:
        - Single: "Q1", "question 1", "#1"
        - Multiple: "Q1, Q3, Q5", "questions 1 3 5", "1, 3, and 5"
        - Range: "Q1-3", "questions 2 to 4", "1 through 5"
        - Mixed: "Q1, 3-5, and 7"

        Returns 0-indexed list of question indices.
        """
        indices = set()
        message_lower = message.lower()

        # Pattern 1: Ranges like "1-3", "2 to 4", "1 through 5"
        range_patterns = [
            r"(\d+)\s*[-–—]\s*(\d+)",  # 1-3, 1–3
            r"(\d+)\s+to\s+(\d+)",  # 1 to 3
            r"(\d+)\s+through\s+(\d+)",  # 1 through 3
        ]
        for pattern in range_patterns:
            for match in re.finditer(pattern, message_lower):
                start, end = int(match.group(1)), int(match.group(2))
                indices.update(range(start - 1, end))  # Convert to 0-indexed

        # Pattern 2: Individual numbers (Q1, question 1, #1)
        # More specific pattern to avoid matching unrelated numbers
        individual_patterns = [
            r"q(?:uestion)?s?\s*#?\s*(\d+)",  # Q1, question 1, Q#1, questions 1
            r"#(\d+)",  # #1
            r"(?:number|no\.?|nr\.?)\s*(\d+)",  # number 1, no. 1
        ]
        for pattern in individual_patterns:
            for match in re.finditer(pattern, message_lower):
                idx = int(match.group(1)) - 1  # Convert to 0-indexed
                if idx >= 0:  # Only positive indices
                    indices.add(idx)

        # Pattern 3: Bare numbers in list context (after action words)
        # Match: "redo 1, 3, 5" or "regenerate 1 and 3"
        action_words = [
            "regenerate",
            "redo",
            "rewrite",
            "improve",
            "fix",
            "update",
            "change",
        ]
        has_action = any(word in message_lower for word in action_words)
        if has_action:
            # Find standalone numbers (1-20 range for reasonable question indices)
            bare_number_pattern = r"(?:^|[\s,]|and\s+)(\d{1,2})(?=[\s,]|$|and|\s*-)"
            for match in re.finditer(bare_number_pattern, message_lower):
                idx = int(match.group(1)) - 1
                if 0 <= idx < 20:  # Reasonable question index (1-20)
                    indices.add(idx)

        return sorted(list(indices))

    def _extract_regeneration_feedback(self, message: str) -> str:
        """
        Extract user's feedback about why questions are bad.

        Examples:
        - "regenerate Q1 - it's not related to the PR" → "it's not related to the PR"
        - "redo questions 1,3 because they're too generic" → "they're too generic"
        - "improve Q2, should ask about Stage V" → "should ask about Stage V"
        """
        # Patterns to extract feedback after the question specification
        feedback_patterns = [
            r"(?:because|since|as)\s+(.+?)$",  # "because they're too generic"
            r"[-–—]\s*(.+?)$",  # "- it's not related"
            r",\s*(?:it\s+)?(?:should|needs?|must)\s+(.+?)$",  # ", should ask about X"
            r"(?:not|isn't|aren't|it's not)\s+(.+?)$",  # "not related to the PR"
            r"(?:too|very)\s+(generic|vague|broad|unclear).+$",  # "too generic"
            r"please\s+(.+?)$",  # "please make it about Stage V"
        ]

        for pattern in feedback_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                feedback = match.group(1).strip()
                # Clean up common trailing patterns
                feedback = re.sub(r"\s*please\s*$", "", feedback, flags=re.IGNORECASE)
                return feedback

        return ""

    async def _generate_replacement_questions(
        self,
        old_questions: list[dict],
        indices: list[int],
        feedback: str,
        avoid_similar_to: list[dict],
    ) -> list[dict]:
        """Generate replacement questions using LLM.

        IMPORTANT: Preserves all original question fields (priority, reason,
        suggested_answers, is_answered, etc.) and only updates the question text.
        """
        parsed_pr = self.state.get("parsed_pr", {})

        # Get question text - support both 'question' and 'question_text' field names
        def get_q_text(q: dict) -> str:
            return q.get("question") or q.get("question_text") or str(q)

        # Format questions being replaced
        old_q_text = "\n".join(
            [f"Q{idx + 1}: {get_q_text(q)}" for idx, q in zip(indices, old_questions)]
        )

        # Format questions to avoid duplicating
        keep_q_text = (
            "\n".join([f"- {get_q_text(q)}" for q in avoid_similar_to]) or "None"
        )

        prompt = f"""You are regenerating clarifying questions for R&D cost estimation.

## PR CONTEXT
PR Code: {parsed_pr.get("pr_code", "N/A")}
Title: {parsed_pr.get("title", "N/A")}
Platform: {parsed_pr.get("platform", "N/A")}
Engine: {parsed_pr.get("engine_type", "N/A")}
Description: {parsed_pr.get("description", "N/A")[:800]}

## QUESTIONS BEING REPLACED
{old_q_text}

## USER FEEDBACK
{feedback or "Make them more relevant and specific to this PR"}

## QUESTIONS TO KEEP (do NOT duplicate these)
{keep_q_text}

## INSTRUCTIONS
Generate {len(indices)} NEW clarifying question(s) to replace the ones above.
Each question should:
1. Be directly relevant to THIS specific PR
2. Help estimate cost/effort more accurately
3. Be clear, specific, and actionable
4. Be DIFFERENT from both the old questions AND the kept questions

Return as JSON array: [{{"question": "...", "reason": "why this question helps estimation"}}]
"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
            )

            # Parse LLM response - extract JSON
            import json

            json_match = re.search(r"\[[\s\S]*\]", response)
            if json_match:
                new_questions_raw = json.loads(json_match.group())

                # Build question objects - PRESERVE all original fields!
                new_questions = []
                for i, (idx, raw_q) in enumerate(zip(indices, new_questions_raw)):
                    # Start with a COPY of the original question to preserve all fields
                    updated_q = old_questions[i].copy()

                    # Only update the question text and reason
                    new_text = raw_q.get("question") or raw_q.get("question_text", "")
                    if new_text:
                        # Update the correct field based on what exists
                        if "question" in updated_q:
                            updated_q["question"] = new_text
                        else:
                            updated_q["question_text"] = new_text

                        # Update reason if provided
                        if raw_q.get("reason"):
                            updated_q["reason"] = raw_q["reason"]

                        # Reset answer since question changed
                        updated_q["answer"] = None
                        updated_q["is_answered"] = False

                    new_questions.append(updated_q)

                return new_questions

        except Exception as e:
            logger.error(f"[SELECTIVE REGEN] Failed to generate questions: {e}")

        # Fallback: return copies of original questions with modified text
        fallback_questions = []
        for i, idx in enumerate(indices):
            updated_q = old_questions[i].copy()
            fallback_text = f"What additional information about this PR would help estimate effort? (Regeneration failed)"
            if "question" in updated_q:
                updated_q["question"] = fallback_text
            else:
                updated_q["question_text"] = fallback_text
            updated_q["answer"] = None
            updated_q["is_answered"] = False
            fallback_questions.append(updated_q)

        return fallback_questions

    async def _regenerate_selected_questions(
        self,
        question_indices: list[int],
        user_message: str,
    ) -> dict:
        """
        Regenerate specific questions while keeping others unchanged.

        Args:
            question_indices: 0-indexed list of questions to regenerate
            user_message: Original user message (may contain feedback)
        """
        questions = self.state.get("questions", [])
        total_questions = len(questions)

        # Validate indices
        valid_indices = [i for i in question_indices if 0 <= i < total_questions]
        invalid_indices = [
            i + 1 for i in question_indices if i < 0 or i >= total_questions
        ]

        if not valid_indices:
            return {
                "status": "error",
                "action_type": "regenerate_selected",
                "details": f"No valid questions found. Available: 1-{total_questions}",
                "updated_state": None,
            }

        # Extract user feedback about what's wrong
        feedback = self._extract_regeneration_feedback(user_message)

        # Get existing questions to avoid duplicates
        keep_questions = [q for i, q in enumerate(questions) if i not in valid_indices]

        logger.info(
            f"[SELECTIVE REGEN] Regenerating Q#{[i + 1 for i in valid_indices]} with feedback: {feedback or 'none'}"
        )

        # Generate replacement questions
        new_questions = await self._generate_replacement_questions(
            old_questions=[questions[i] for i in valid_indices],
            indices=valid_indices,
            feedback=feedback,
            avoid_similar_to=keep_questions,
        )

        # Merge: keep unchanged questions, replace selected ones
        updated_questions = questions.copy()
        for idx, new_q in zip(valid_indices, new_questions):
            updated_questions[idx] = new_q

        self.state["questions"] = updated_questions

        # Build response message
        regenerated_nums = [i + 1 for i in valid_indices]
        msg = f"Regenerated question(s) #{', #'.join(map(str, regenerated_nums))}"
        if invalid_indices:
            msg += f" (skipped invalid: #{', #'.join(map(str, invalid_indices))})"

        return {
            "status": "success",
            "action_type": "regenerate_selected",
            "details": msg,
            "updated_state": {"questions": updated_questions},
            "regenerated": regenerated_nums,
            "unchanged": [
                i + 1 for i in range(total_questions) if i not in valid_indices
            ],
            "feedback_used": feedback or None,
        }

    # ===== End Selective Question Regeneration Helpers =====

    async def _action_update_questions(
        self,
        msg_lower: str,
        original_message: str,
    ) -> dict:
        """
        GOD MODE Handler: Q&A Step Actions.

        Supported actions:
        - Regenerate all questions
        - Regenerate specific question(s) - NEW!
        - Add custom question
        - Remove question
        - Update question answer
        """
        questions = self.state.get("questions", [])
        parsed_pr = self.state.get("parsed_pr", {})

        # ===== NEW: Check for SELECTIVE question regeneration first =====
        # This handles: "regenerate Q1", "redo questions 1, 3, 5", "improve Q1-3"
        selective_keywords = [
            "regenerate",
            "regernate",
            "regen",
            "redo",
            "rewrite",
            "improve",
            "fix",
            "update",
            "change",
            "remake",
        ]
        has_selective_action = any(kw in msg_lower for kw in selective_keywords)

        if has_selective_action:
            # Parse which questions user wants to regenerate
            question_indices = self._parse_question_numbers(original_message)

            if question_indices:
                # User specified specific questions (e.g., "Q1", "questions 1,3,5")
                logger.info(
                    f"[GOD MODE] Selective regeneration detected: Q#{[i + 1 for i in question_indices]}"
                )
                # Debug: Log state info
                parsed_pr = self.state.get("parsed_pr", {})
                logger.info(
                    f"[GOD MODE] parsed_pr keys: {list(parsed_pr.keys()) if parsed_pr else 'EMPTY'}"
                )
                logger.info(
                    f"[GOD MODE] PR title: {parsed_pr.get('title', 'NO TITLE')[:50] if parsed_pr else 'N/A'}"
                )
                return await self._regenerate_selected_questions(
                    question_indices, original_message
                )
            # If no specific numbers found, fall through to "regenerate all" logic

        # ===== Action: Regenerate ALL questions =====
        # Flexible matching - check for key words separately to handle typos
        has_action_word = any(
            word in msg_lower
            for word in [
                "regenerate",
                "regernate",
                "regen",  # common typos
                "redo",
                "refresh",
                "improve",
                "enhance",
                "refine",
                "rewrite",
                "update",
                "change",
                "better",
                "new",
                "generate",
                "create",
                "fix",
                "remake",
            ]
        )
        has_target_word = any(
            word in msg_lower for word in ["question", "qa", "q&a", "q.a", "queries"]
        )

        if has_action_word and has_target_word:
            logger.info("[GOD MODE] Regenerating all questions via LLM")

            # Use LLM to generate new questions with FULL schema
            prompt = f"""Generate 5-7 clarifying questions for this Product Request:

PR Code: {parsed_pr.get("pr_code", "N/A")}
Title: {parsed_pr.get("title", "N/A")}
Platform: {parsed_pr.get("platform", "N/A")}
Engine: {parsed_pr.get("engine_type", "N/A")}
Description: {parsed_pr.get("description", "N/A")[:500]}

Generate questions that help clarify:
1. Technical scope and complexity
2. Integration requirements
3. Testing needs
4. Timeline constraints
5. Special requirements

IMPORTANT: Return questions with the following EXACT JSON schema:
[{{
  "question": "the question text",
  "reason": "why this question helps with cost estimation",
  "category": "scope|complexity|technical|timeline|testing|integration",
  "priority": "high|medium|low",
  "suggested_answers": ["option 1", "option 2", "option 3"]
}}]

Include 3-5 suggested_answers for each question as clickable options."""

            try:
                response = await self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048,  # Increased for suggested_answers
                )

                # Parse LLM response - extract JSON
                import json

                json_match = re.search(r"\[[\s\S]*\]", response)
                if json_match:
                    raw_questions = json.loads(json_match.group())
                    # Normalize to proper Question schema with all required fields
                    new_questions = []
                    for i, q in enumerate(raw_questions):
                        normalized_q = {
                            "id": f"q_{i + 1}",
                            # Support both 'question' and 'question_text' from LLM
                            "question": q.get("question") or q.get("question_text", ""),
                            "question_text": q.get("question")
                            or q.get("question_text", ""),
                            "reason": q.get("reason", "Helps clarify project scope"),
                            "category": q.get("category", "general"),
                            # Normalize priority: 'importance' -> 'priority'
                            "priority": q.get("priority")
                            or q.get("importance", "medium"),
                            # Ensure suggested_answers is always a list
                            "suggested_answers": q.get("suggested_answers", []),
                            "answer": None,
                            "is_answered": False,
                        }
                        new_questions.append(normalized_q)

                    self.state["questions"] = new_questions

                    return {
                        "status": "success",
                        "action_type": "regenerate_questions",
                        "details": f"Generated {len(new_questions)} new questions with suggested answers",
                        "updated_state": {"questions": new_questions},
                    }
            except Exception as e:
                logger.error(f"[GOD MODE] Failed to regenerate questions: {e}")
                return {
                    "status": "error",
                    "action_type": "regenerate_questions",
                    "details": f"Failed to regenerate: {str(e)}",
                    "updated_state": None,
                }

        # Action: Add custom question
        add_match = re.search(
            r'add (?:a )?question[:\s]+["\']?(.+?)["\']?$', msg_lower, re.IGNORECASE
        )
        if add_match or "add question" in msg_lower:
            # Extract question text from original message
            question_text = (
                add_match.group(1)
                if add_match
                else original_message.split("add question")[-1].strip()
            )
            question_text = question_text.strip("\" '.:")

            if question_text and len(question_text) > 10:
                new_q = {
                    "id": f"q_custom_{len(questions) + 1}",
                    "category": "Custom",
                    "question_text": question_text,
                    "importance": "medium",
                    "answer": "",
                }
                questions.append(new_q)
                self.state["questions"] = questions

                return {
                    "status": "success",
                    "action_type": "add_question",
                    "details": f"Added custom question: {question_text[:50]}...",
                    "updated_state": {"questions": questions},
                }

        # Action: Remove question by number
        remove_match = re.search(r"(?:remove|delete) question\s*#?(\d+)", msg_lower)
        if remove_match:
            q_num = int(remove_match.group(1)) - 1  # Convert to 0-indexed
            if 0 <= q_num < len(questions):
                removed = questions.pop(q_num)
                self.state["questions"] = questions

                return {
                    "status": "success",
                    "action_type": "remove_question",
                    "details": f"Removed question #{q_num + 1}: {removed.get('question_text', '')[:40]}...",
                    "updated_state": {"questions": questions},
                }

        # Action: Answer a question
        answer_match = re.search(
            r"(?:answer|set answer for) question\s*#?(\d+)[:\s]+(.+)",
            msg_lower,
            re.IGNORECASE,
        )
        if not answer_match:
            answer_match = re.search(
                r"question\s*#?(\d+)[:\s]+answer[:\s]+(.+)", msg_lower, re.IGNORECASE
            )

        if answer_match:
            q_num = int(answer_match.group(1)) - 1
            answer_text = answer_match.group(2).strip("\" '")

            if 0 <= q_num < len(questions):
                questions[q_num]["answer"] = answer_text
                self.state["questions"] = questions

                return {
                    "status": "success",
                    "action_type": "answer_question",
                    "details": f"Set answer for question #{q_num + 1}",
                    "updated_state": {"questions": questions},
                }

        # No recognized action
        return {
            "status": "no_action",
            "action_type": "unknown",
            "details": "Could not understand the modification request for Q&A step",
            "updated_state": None,
        }

    async def _action_update_summary(
        self,
        msg_lower: str,
        original_message: str,
    ) -> dict:
        """
        GOD MODE Handler: Summary Step Actions.

        Supported actions:
        - Update complexity level
        - Update platform/engine
        - Recalculate features
        - Modify PR fields
        - Regenerate narrative summary
        - Add/edit key features
        - Add/edit risk factors
        - Add/edit dependencies
        - Add/edit special requirements
        """
        parsed_pr = self.state.get("parsed_pr", {})
        pr_summary = self.state.get("pr_summary", {})

        # Action: Regenerate the narrative summary
        has_regen_word = any(
            word in msg_lower
            for word in [
                "regenerate",
                "rewrite",
                "improve",
                "enhance",
                "redo",
                "better",
                "update",
                "regenrate",
                "regen",
            ]
        )
        has_summary_word = any(
            word in msg_lower for word in ["summary", "narrative", "overview", "text"]
        )
        if has_regen_word and has_summary_word:
            try:
                # Build context for summary regeneration
                activities = parsed_pr.get("raw_activities", [])
                answers = self.state.get("answers", {})
                similar_prs = self.state.get("similar_prs", [])

                # Build similar PRs reference with hours data
                similar_prs_text = (
                    "\n".join(
                        [
                            f"  - {sp.get('pr_code')}: {sp.get('total_hours', 0)} hours estimated"
                            for sp in similar_prs[:3]
                            if sp.get("total_hours", 0) > 0
                        ]
                    )
                    or "  No historical hours data available"
                )

                context = f"""**FPT Cost Brain - R&D Cost Estimation Tool**

IMPORTANT: This tool PREDICTS engineering effort (hours, cost, breakdown) based on the PR description.
The breakdown of activities is GENERATED by our ML system from historical patterns - NOT expected in the PR.

**Product Request Details:**
- PR Code: {parsed_pr.get("pr_code", "Unknown")}
- Title: {parsed_pr.get("title", "Unknown")}
- Customer: {parsed_pr.get("customer", "Unknown")}
- Program Family: {parsed_pr.get("program_family", "Unknown")}
- Description: {parsed_pr.get("description", "N/A")}

**Q&A Clarifications Provided:**
{answers if answers else "None provided"}

**Similar Historical PRs (used for prediction calibration):**
{similar_prs_text}

**User's guidance for regeneration:** {original_message}"""

                prompt = f"""{context}

Generate an EXECUTIVE SUMMARY for a customer manager reviewing this cost estimation request.

The summary should:
1. **Project Overview**: Explain what this PR is requesting (scope, objectives, deliverables)
2. **Technical Scope**: Describe the likely engineering work involved (based on the title and context)
3. **Estimation Basis**: Reference how similar historical projects inform our cost prediction
4. **Key Considerations**: Highlight factors that may impact the estimate

CRITICAL: Do NOT say activities are "missing" or "not defined" - our ML system PREDICTS the breakdown automatically.
Keep it professional, concise (2-3 paragraphs), suitable for management review."""

                response = await self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1500,
                )

                # Parse the response - extract sections
                summary_text = response.strip()

                # Try to extract structured sections from the response
                key_features = pr_summary.get("key_features", [])
                dependencies = pr_summary.get("dependencies", [])
                risk_factors = pr_summary.get("risk_factors", [])
                special_requirements = pr_summary.get("special_requirements", [])

                # Update pr_summary with new narrative
                pr_summary["summary_text"] = summary_text
                self.state["pr_summary"] = pr_summary

                return {
                    "status": "success",
                    "action_type": "regenerate_summary",
                    "details": "Generated new executive summary",
                    "updated_state": {"pr_summary": pr_summary},
                }

            except Exception as e:
                logger.error(f"[GOD MODE] Failed to regenerate summary: {e}")
                return {
                    "status": "error",
                    "action_type": "regenerate_summary",
                    "details": f"Failed to regenerate summary: {str(e)}",
                    "updated_state": None,
                }

        # Action: Add a key feature
        add_feature_match = re.search(
            r"add (?:key )?feature[:\s]+(.+)", original_message, re.IGNORECASE
        )
        if add_feature_match or "add feature" in msg_lower:
            feature_text = (
                add_feature_match.group(1).strip()
                if add_feature_match
                else original_message.split("add feature")[-1].strip()
            )
            feature_text = feature_text.strip("\"'.:,")

            if feature_text and len(feature_text) > 3:
                key_features = pr_summary.get("key_features", [])
                key_features.append(feature_text)
                pr_summary["key_features"] = key_features
                self.state["pr_summary"] = pr_summary

                return {
                    "status": "success",
                    "action_type": "add_feature",
                    "details": f"Added key feature: {feature_text[:50]}...",
                    "updated_state": {"pr_summary": pr_summary},
                }

        # Action: Add a risk factor
        add_risk_match = re.search(
            r"add (?:risk|risk factor)[:\s]+(.+)", original_message, re.IGNORECASE
        )
        if add_risk_match or "add risk" in msg_lower:
            risk_text = (
                add_risk_match.group(1).strip()
                if add_risk_match
                else original_message.split("add risk")[-1].strip()
            )
            risk_text = risk_text.strip("\"'.:,")

            if risk_text and len(risk_text) > 3:
                risk_factors = pr_summary.get("risk_factors", [])
                risk_factors.append(risk_text)
                pr_summary["risk_factors"] = risk_factors
                self.state["pr_summary"] = pr_summary

                return {
                    "status": "success",
                    "action_type": "add_risk",
                    "details": f"Added risk factor: {risk_text[:50]}...",
                    "updated_state": {"pr_summary": pr_summary},
                }

        # Action: Add a dependency
        add_dep_match = re.search(
            r"add (?:dependency|dep)[:\s]+(.+)", original_message, re.IGNORECASE
        )
        if add_dep_match or "add dependency" in msg_lower or "add dep" in msg_lower:
            dep_text = (
                add_dep_match.group(1).strip()
                if add_dep_match
                else original_message.split("add dep")[-1].strip()
            )
            dep_text = dep_text.strip("\"'.:,")

            if dep_text and len(dep_text) > 3:
                dependencies = pr_summary.get("dependencies", [])
                dependencies.append(dep_text)
                pr_summary["dependencies"] = dependencies
                self.state["pr_summary"] = pr_summary

                return {
                    "status": "success",
                    "action_type": "add_dependency",
                    "details": f"Added dependency: {dep_text[:50]}...",
                    "updated_state": {"pr_summary": pr_summary},
                }

        # Action: Add a special requirement
        add_req_match = re.search(
            r"add (?:special )?requirement[:\s]+(.+)", original_message, re.IGNORECASE
        )
        if add_req_match or "add requirement" in msg_lower:
            req_text = (
                add_req_match.group(1).strip()
                if add_req_match
                else original_message.split("add requirement")[-1].strip()
            )
            req_text = req_text.strip("\"'.:,")

            if req_text and len(req_text) > 3:
                special_requirements = pr_summary.get("special_requirements", [])
                special_requirements.append(req_text)
                pr_summary["special_requirements"] = special_requirements
                self.state["pr_summary"] = pr_summary

                return {
                    "status": "success",
                    "action_type": "add_requirement",
                    "details": f"Added requirement: {req_text[:50]}...",
                    "updated_state": {"pr_summary": pr_summary},
                }

        # Action: Update complexity
        complexity_match = re.search(
            r"(?:set|change|update|make)\s+(?:the\s+)?complexity\s+(?:to\s+)?(high|medium|low)",
            msg_lower,
        )
        if complexity_match:
            new_complexity = complexity_match.group(1).upper()

            # Update in both parsed_pr and pr_summary
            parsed_pr["complexity"] = new_complexity
            pr_summary["complexity"] = new_complexity

            self.state["parsed_pr"] = parsed_pr
            self.state["pr_summary"] = pr_summary

            return {
                "status": "success",
                "action_type": "update_complexity",
                "details": f"Updated complexity to: {new_complexity}",
                "updated_state": {
                    "parsed_pr": parsed_pr,
                    "pr_summary": pr_summary,
                },
            }

        # Action: Update platform
        platform_match = re.search(
            r'(?:set|change|update)\s+(?:the\s+)?platform\s+(?:to\s+)?["\']?([A-Za-z0-9_-]+)["\']?',
            msg_lower,
        )
        if platform_match:
            new_platform = platform_match.group(1).upper()

            parsed_pr["platform"] = new_platform
            self.state["parsed_pr"] = parsed_pr

            return {
                "status": "success",
                "action_type": "update_platform",
                "details": f"Updated platform to: {new_platform}",
                "updated_state": {"parsed_pr": parsed_pr},
            }

        # Action: Update engine type
        engine_match = re.search(
            r'(?:set|change|update)\s+(?:the\s+)?engine\s*(?:type)?\s+(?:to\s+)?["\']?([A-Za-z0-9_-]+)["\']?',
            msg_lower,
        )
        if engine_match:
            new_engine = engine_match.group(1).upper()

            parsed_pr["engine_type"] = new_engine
            self.state["parsed_pr"] = parsed_pr

            return {
                "status": "success",
                "action_type": "update_engine",
                "details": f"Updated engine type to: {new_engine}",
                "updated_state": {"parsed_pr": parsed_pr},
            }

        # Action: Update tier/size
        tier_match = re.search(
            r'(?:set|change|update)\s+(?:the\s+)?(?:tier|size|program.?size)\s+(?:to\s+)?["\']?(\w+)["\']?',
            msg_lower,
        )
        if tier_match:
            new_tier = tier_match.group(1).upper()

            parsed_pr["tier"] = new_tier
            pr_summary["program_size"] = new_tier

            self.state["parsed_pr"] = parsed_pr
            self.state["pr_summary"] = pr_summary

            return {
                "status": "success",
                "action_type": "update_tier",
                "details": f"Updated tier/size to: {new_tier}",
                "updated_state": {
                    "parsed_pr": parsed_pr,
                    "pr_summary": pr_summary,
                },
            }

        # Action: Recalculate/regenerate summary
        if any(
            keyword in msg_lower
            for keyword in [
                "recalculate",
                "regenerate summary",
                "refresh summary",
                "redo summary",
            ]
        ):
            # This would trigger re-extraction - mark for node re-execution
            return {
                "status": "pending_reprocess",
                "action_type": "regenerate_summary",
                "details": "Summary regeneration requested - will re-extract features from PR",
                "updated_state": None,
                "requires_reprocess": True,
            }

        # No recognized action
        return {
            "status": "no_action",
            "action_type": "unknown",
            "details": "Could not understand the modification request for Summary step",
            "updated_state": None,
        }

    async def _action_update_estimation(
        self,
        msg_lower: str,
        original_message: str,
    ) -> dict:
        """
        GOD MODE Handler: Estimation/Review Step Actions.

        Supported actions:
        - Modify activity hours
        - Add hours to activity
        - Reduce hours from activity
        - Recalculate estimation
        - Set confidence level
        """
        breakdown = self.state.get("breakdown", [])
        user_edits = self.state.get("user_edits", [])

        # Action: Set hours for specific activity by name/function
        hours_match = re.search(
            r'(?:set|change|update)\s+(?:the\s+)?(?:hours?\s+(?:for\s+)?)?["\']?([^"\']+)["\']?\s+(?:to\s+)?(\d+)\s*(?:hours?|h)?',
            msg_lower,
        )
        if not hours_match:
            # Alternative pattern: "change X hours for activity"
            hours_match = re.search(
                r'(?:set|change)\s+(\d+)\s*(?:hours?|h)\s+(?:for|to)\s+["\']?([^"\']+)["\']?',
                msg_lower,
            )
            if hours_match:
                # Swap groups since pattern is reversed
                activity_name = hours_match.group(2)
                new_hours = int(hours_match.group(1))
            else:
                activity_name = None
                new_hours = None
        else:
            activity_name = hours_match.group(1).strip()
            new_hours = int(hours_match.group(2))

        if activity_name and new_hours is not None:
            # Find matching activity
            for item in breakdown:
                item_name = f"{item.get('pe_function', '')} {item.get('activity_description', '')}".lower()
                if (
                    activity_name.lower() in item_name
                    or item_name in activity_name.lower()
                ):
                    original_hours = item.get("hours", 0)
                    item["hours"] = new_hours
                    item["cost"] = new_hours * item.get(
                        "hourly_rate", 50
                    )  # Default rate
                    item["user_edited"] = True

                    # Record edit
                    user_edits.append(
                        {
                            "activity_id": item.get("id"),
                            "activity_name": item.get("activity_description", ""),
                            "original_hours": original_hours,
                            "new_hours": new_hours,
                            "reason": f"User command: {original_message[:100]}",
                            "timestamp": "now",
                        }
                    )

                    self.state["breakdown"] = breakdown
                    self.state["user_edits"] = user_edits

                    return {
                        "status": "success",
                        "action_type": "update_hours",
                        "details": f"Updated '{item.get('activity_description', '')[:30]}' from {original_hours}h to {new_hours}h",
                        "updated_state": {
                            "breakdown": breakdown,
                            "user_edits": user_edits,
                        },
                    }

        # Action: Add/increase hours
        add_hours_match = re.search(
            r'(?:add|increase)\s+(\d+)\s*(?:hours?|h)\s+(?:to|for)\s+["\']?([^"\']+)["\']?',
            msg_lower,
        )
        if add_hours_match:
            add_amount = int(add_hours_match.group(1))
            activity_name = add_hours_match.group(2).strip()

            for item in breakdown:
                item_name = f"{item.get('pe_function', '')} {item.get('activity_description', '')}".lower()
                if activity_name.lower() in item_name:
                    original_hours = item.get("hours", 0)
                    new_hours = original_hours + add_amount
                    item["hours"] = new_hours
                    item["cost"] = new_hours * item.get("hourly_rate", 50)
                    item["user_edited"] = True

                    user_edits.append(
                        {
                            "activity_id": item.get("id"),
                            "original_hours": original_hours,
                            "new_hours": new_hours,
                            "reason": f"Added {add_amount}h via command",
                        }
                    )

                    self.state["breakdown"] = breakdown
                    self.state["user_edits"] = user_edits

                    return {
                        "status": "success",
                        "action_type": "add_hours",
                        "details": f"Added {add_amount}h to '{item.get('activity_description', '')[:30]}' ({original_hours}h → {new_hours}h)",
                        "updated_state": {
                            "breakdown": breakdown,
                            "user_edits": user_edits,
                        },
                    }

        # Action: Reduce/decrease hours
        reduce_match = re.search(
            r'(?:reduce|decrease|remove)\s+(\d+)\s*(?:hours?|h)\s+(?:from|for)\s+["\']?([^"\']+)["\']?',
            msg_lower,
        )
        if reduce_match:
            reduce_amount = int(reduce_match.group(1))
            activity_name = reduce_match.group(2).strip()

            for item in breakdown:
                item_name = f"{item.get('pe_function', '')} {item.get('activity_description', '')}".lower()
                if activity_name.lower() in item_name:
                    original_hours = item.get("hours", 0)
                    new_hours = max(0, original_hours - reduce_amount)
                    item["hours"] = new_hours
                    item["cost"] = new_hours * item.get("hourly_rate", 50)
                    item["user_edited"] = True

                    user_edits.append(
                        {
                            "activity_id": item.get("id"),
                            "original_hours": original_hours,
                            "new_hours": new_hours,
                            "reason": f"Reduced {reduce_amount}h via command",
                        }
                    )

                    self.state["breakdown"] = breakdown
                    self.state["user_edits"] = user_edits

                    return {
                        "status": "success",
                        "action_type": "reduce_hours",
                        "details": f"Reduced {reduce_amount}h from '{item.get('activity_description', '')[:30]}' ({original_hours}h → {new_hours}h)",
                        "updated_state": {
                            "breakdown": breakdown,
                            "user_edits": user_edits,
                        },
                    }

        # Action: Recalculate estimation
        if any(
            keyword in msg_lower
            for keyword in [
                "recalculate",
                "recalculate estimation",
                "redo estimation",
                "refresh estimation",
                "re-estimate",
            ]
        ):
            return {
                "status": "pending_reprocess",
                "action_type": "recalculate_estimation",
                "details": "Estimation recalculation requested - will re-run ML prediction",
                "updated_state": None,
                "requires_reprocess": True,
            }

        # Action: Update confidence level
        confidence_match = re.search(
            r"(?:set|change)\s+(?:the\s+)?confidence\s+(?:to\s+)?(\d+)%?", msg_lower
        )
        if confidence_match:
            new_confidence = int(confidence_match.group(1)) / 100

            ml_prediction = self.state.get("ml_prediction", {})
            ml_prediction["confidence"] = new_confidence
            ml_prediction["user_adjusted_confidence"] = True
            self.state["ml_prediction"] = ml_prediction

            return {
                "status": "success",
                "action_type": "update_confidence",
                "details": f"Updated confidence level to {new_confidence:.0%}",
                "updated_state": {"ml_prediction": ml_prediction},
            }

        # Action: Apply percentage increase/decrease to all activities
        pct_all_match = re.search(
            r"(?:increase|decrease|reduce|add)\s+(?:all|everything|total)\s+(?:by\s+)?(\d+)%",
            msg_lower,
        )
        if pct_all_match:
            pct = int(pct_all_match.group(1)) / 100
            is_increase = "increase" in msg_lower or "add" in msg_lower

            total_original = 0
            total_new = 0

            for item in breakdown:
                original_hours = item.get("hours", 0)
                total_original += original_hours
                if is_increase:
                    new_hours = int(original_hours * (1 + pct))
                else:
                    new_hours = int(original_hours * (1 - pct))
                item["hours"] = max(0, new_hours)
                item["cost"] = new_hours * item.get("hourly_rate", 50)
                item["user_edited"] = True
                total_new += new_hours

            self.state["breakdown"] = breakdown
            action_type = "increase" if is_increase else "decrease"

            return {
                "status": "success",
                "action_type": f"{action_type}_all",
                "details": f"{'Increased' if is_increase else 'Decreased'} all activities by {pct:.0%} ({total_original}h → {total_new}h)",
                "updated_state": {"breakdown": breakdown},
            }

        # Action: Scale specific category
        category_scale_match = re.search(
            r"(?:increase|decrease|scale)\s+(?:all\s+)?(\w+)\s+(?:activities?\s+)?(?:by\s+)?(\d+)%",
            msg_lower,
        )
        if category_scale_match:
            category = category_scale_match.group(1).lower()
            pct = int(category_scale_match.group(2)) / 100
            is_increase = "increase" in msg_lower or "scale" in msg_lower

            modified_count = 0
            for item in breakdown:
                item_category = item.get("category", "").lower()
                item_function = item.get("pe_function", "").lower()
                if category in item_category or category in item_function:
                    original_hours = item.get("hours", 0)
                    if is_increase:
                        new_hours = int(original_hours * (1 + pct))
                    else:
                        new_hours = int(original_hours * (1 - pct))
                    item["hours"] = max(0, new_hours)
                    item["cost"] = new_hours * item.get("hourly_rate", 50)
                    item["user_edited"] = True
                    modified_count += 1

            if modified_count > 0:
                self.state["breakdown"] = breakdown
                return {
                    "status": "success",
                    "action_type": "scale_category",
                    "details": f"Scaled {modified_count} {category} activities by {pct:.0%}",
                    "updated_state": {"breakdown": breakdown},
                }

        # Action: Set edit reason/justification for recent edits
        reason_match = re.search(
            r"(?:reason|because|justification)[:\s]+(.+)",
            original_message,
            re.IGNORECASE,
        )
        if reason_match and user_edits:
            reason_text = reason_match.group(1).strip()
            # Apply reason to most recent edits without a reason
            for edit in reversed(user_edits):
                if not edit.get("reason") or "User command" in edit.get("reason", ""):
                    edit["reason"] = reason_text
                    break

            self.state["user_edits"] = user_edits
            return {
                "status": "success",
                "action_type": "set_reason",
                "details": f"Added justification: {reason_text[:50]}...",
                "updated_state": {"user_edits": user_edits},
            }

        # Action: Show/explain estimation for an activity (read-only but useful)
        if "explain" in msg_lower or "why" in msg_lower:
            # This is handled by chat response, not state modification
            return {
                "status": "no_action",
                "action_type": "explain",
                "details": "Explanation will be provided in chat response",
                "updated_state": None,
            }

        # No recognized action
        return {
            "status": "no_action",
            "action_type": "unknown",
            "details": "Could not understand the modification request for Estimation step. Try: 'set hours for [activity] to [number]', 'add [N] hours to [activity]', or 'increase all by [N]%'",
            "updated_state": None,
        }

    async def _persist_state(self) -> None:
        """Persist current state to Redis for session continuity."""
        # Import here to avoid circular imports
        from services.estimation_service import save_state_to_redis

        try:
            session_id = self.state.get("session_id")
            if session_id:
                await save_state_to_redis(session_id, self.state)
                logger.info(
                    f"[GOD MODE] State persisted to Redis for session {session_id}"
                )
            else:
                logger.warning("[GOD MODE] No session_id in state, cannot persist")
        except Exception as e:
            logger.error(f"[GOD MODE] Failed to persist state: {e}")

    async def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Process a chat message with unified BRAIN + HANDS architecture.

        Pipeline:
        1. Load base terminology from RAG
        2. Classify intent (BRAIN)
        3. [GOD MODE] Execute state action if MODIFY_STATE
        4. Gather RAG context (BRAIN)
        5. Execute strategic tools (HANDS)
        6. Build unified prompt with tool data
        7. Generate LLM response
        """
        history = history or []
        action_result = None  # GOD MODE action result

        # Step 1: Ensure base terminology is loaded (BRAIN init)
        await self._ensure_initialized()

        # Step 2: Classify intent (BRAIN)
        # In AGENT MODE: Use LLM for natural language understanding
        # In CHAT MODE: Use fast keyword-based classification
        if self.agent_mode:
            intent = await self.classify_intent_with_llm(user_message)
            logger.info(f"[AGENT MODE] LLM classified intent: {intent.value}")
        else:
            intent = self._classify_intent_keywords(user_message)
            logger.info(f"[CHAT MODE] Keyword classified intent: {intent.value}")

        # Step 3: [GOD MODE] Execute state action if MODIFY_STATE
        if intent == IntentType.MODIFY_STATE:
            action_result = await self._execute_state_action(intent, user_message)
            logger.info(f"[GOD MODE] Action result: {action_result}")

            # Persist state if action was successful
            if action_result and action_result.get("status") == "success":
                await self._persist_state()

        # Step 4: Gather RAG context (BRAIN)
        rag_context = await self.gather_rag_context(user_message, intent)
        logger.info(
            f"[BRAIN] RAG context: {len(rag_context.base_terminology)} base terms, "
            f"{len(rag_context.acronyms)} acronyms, {len(rag_context.knowledge)} docs, "
            f"{len(rag_context.similar_prs)} similar PRs"
        )

        # Step 5: Execute strategic tools (HANDS)
        tool_output = self._execute_strategic_tools(intent, user_message, rag_context)
        if tool_output:
            logger.info(f"[HANDS] Tool output generated ({len(tool_output)} chars)")
        else:
            logger.info("[HANDS] No tools triggered for this query")

        # Step 6: Build unified prompt
        system_prompt = self._build_system_prompt(intent, rag_context)

        # Inject tool output into prompt if available
        if tool_output:
            system_prompt += f"""

## ANALYTICAL DATA (from Tools)
The following data was calculated by deterministic tools. Use this to answer the user's question:

{tool_output}

**IMPORTANT**: Reference this analytical data in your response. Don't recalculate - use these numbers.
"""

        # Inject GOD MODE action result into prompt
        if action_result:
            action_status = action_result.get("status", "unknown")
            action_type = action_result.get("action_type", "unknown")
            action_details = action_result.get("details", "")

            if action_status == "success":
                system_prompt += f"""

## STATE MODIFICATION EXECUTED (GOD MODE)
**Action**: {action_type}
**Status**: ✅ SUCCESS
**Details**: {action_details}

You MUST confirm this action to the user in a friendly way. Explain what was changed and what they can do next.
"""
            elif action_status == "pending_reprocess":
                system_prompt += f"""

## STATE MODIFICATION PENDING
**Action**: {action_type}
**Status**: ⏳ PENDING REPROCESS
**Details**: {action_details}

Tell the user their request requires reprocessing and will be handled when they proceed to the next step.
"""
            elif action_status == "no_action":
                system_prompt += f"""

## STATE MODIFICATION REQUEST
**Status**: ❓ NOT UNDERSTOOD
**Details**: {action_details}

The user tried to modify something but I couldn't understand the request.
Help them rephrase their command. Give examples of valid commands for the current step ({self.current_step}).
"""
            else:
                system_prompt += f"""

## STATE MODIFICATION FAILED
**Action**: {action_type}
**Status**: ❌ ERROR
**Details**: {action_details}

Apologize for the error and suggest what the user can do instead.
"""

        context_message = self._build_context_message()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
        ]

        # Add history (last 10 messages)
        for msg in history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        # Step 7: Generate response (BRAIN)
        response = await self.llm.chat(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )

        # Step 8: Generate suggestions
        suggestions = await self._generate_suggestions(user_message, response, intent)

        # Build result with GOD MODE data
        result = {
            "response": response,
            "intent": intent.value,
            "rag_context_used": rag_context.has_context(),
            "tools_used": tool_output is not None,
            "rag_stats": {
                "base_terms": len(rag_context.base_terminology),
                "acronyms_found": len(rag_context.acronyms),
                "knowledge_docs": len(rag_context.knowledge),
                "similar_prs": len(rag_context.similar_prs),
            },
            "suggestions": suggestions,
            "step": self.current_step,
        }

        # Include GOD MODE action result and updated state
        if action_result:
            result["action_executed"] = True
            result["action_result"] = {
                "status": action_result.get("status"),
                "action_type": action_result.get("action_type"),
                "details": action_result.get("details"),
            }
            # Include updated state for frontend sync
            if action_result.get("updated_state"):
                result["updated_state"] = action_result.get("updated_state")
        else:
            result["action_executed"] = False

        return result

    async def prepare_stream_context(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Prepare context for streaming - does all RAG work upfront.
        Call this before stream_response() to allow status updates between steps.
        """
        history = history or []

        # BRAIN: Initialize and gather context
        await self._ensure_initialized()
        # Use LLM classification in agent mode, keywords otherwise
        if self.agent_mode:
            intent = await self.classify_intent_with_llm(user_message)
        else:
            intent = self._classify_intent_keywords(user_message)
        rag_context = await self.gather_rag_context(user_message, intent)

        # HANDS: Execute strategic tools
        tool_output = self._execute_strategic_tools(intent, user_message, rag_context)

        # Build unified prompt
        system_prompt = self._build_system_prompt(intent, rag_context)

        # Inject tool output if available
        if tool_output:
            system_prompt += f"""

## ANALYTICAL DATA (from Tools)
{tool_output}

**IMPORTANT**: Reference this analytical data in your response.
"""

        context_message = self._build_context_message()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_message},
        ]

        for msg in history[-10:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_message})

        return {
            "messages": messages,
            "intent": intent,
            "rag_context": rag_context,
        }

    async def stream_response(
        self,
        prepared_context: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Stream LLM response from prepared context. Call prepare_stream_context() first."""
        messages = prepared_context["messages"]
        async for chunk in self.llm.chat_stream(messages):
            yield chunk

    async def chat_stream(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat response with unified BRAIN + HANDS architecture."""
        # Prepare context (this does RAG work)
        prepared = await self.prepare_stream_context(user_message, history)
        # Stream response
        async for chunk in self.stream_response(prepared):
            yield chunk

    async def _generate_suggestions(
        self,
        user_message: str,
        response: str,
        intent: IntentType,
    ) -> list[dict[str, str]]:
        """Generate contextual follow-up suggestions."""

        # Intent-based suggestions
        intent_suggestions = {
            IntentType.META: [
                {
                    "text": "Continue to next step",
                    "action": "I'm ready to proceed",
                    "icon": "➡️",
                },
                {
                    "text": "More guidance",
                    "action": "What should I do here?",
                    "icon": "❓",
                },
            ],
            IntentType.TERMINOLOGY: [
                {
                    "text": "Related terms",
                    "action": "What related FPT terms should I know?",
                    "icon": "📖",
                },
                {
                    "text": "Back to task",
                    "action": "Let's continue with the estimation",
                    "icon": "➡️",
                },
            ],
            IntentType.PR_SPECIFIC: [
                {
                    "text": "Compare similar",
                    "action": "Show me similar historical projects",
                    "icon": "📊",
                },
                {
                    "text": "Check accuracy",
                    "action": "Is this information correct?",
                    "icon": "✅",
                },
            ],
            IntentType.COMPARISON: [
                {
                    "text": "More details",
                    "action": "Tell me more about the most similar project",
                    "icon": "🔍",
                },
                {
                    "text": "Cost breakdown",
                    "action": "How did their costs break down?",
                    "icon": "💰",
                },
            ],
            IntentType.ESTIMATION: [
                {
                    "text": "Explain confidence",
                    "action": "Why this confidence level?",
                    "icon": "📈",
                },
                {
                    "text": "Applied rules",
                    "action": "What rules affect this estimate?",
                    "icon": "📜",
                },
            ],
            IntentType.GUIDANCE: [
                {
                    "text": "Show example",
                    "action": "Can you show me an example?",
                    "icon": "💡",
                },
                {
                    "text": "Step by step",
                    "action": "Walk me through this",
                    "icon": "👣",
                },
            ],
            IntentType.GENERAL: [
                {
                    "text": "Help with PR",
                    "action": "Help me understand this PR",
                    "icon": "📋",
                },
                {
                    "text": "FPT terminology",
                    "action": "Explain common FPT terms",
                    "icon": "📖",
                },
            ],
            IntentType.MODIFY_STATE: [
                {
                    "text": "Undo change",
                    "action": "Can you undo the last change?",
                    "icon": "↩️",
                },
                {
                    "text": "See current state",
                    "action": "Show me the current values",
                    "icon": "👁️",
                },
                {
                    "text": "More changes",
                    "action": "What else can I modify?",
                    "icon": "✏️",
                },
                {
                    "text": "Continue",
                    "action": "I'm happy with these changes, let's proceed",
                    "icon": "➡️",
                },
            ],
        }

        return intent_suggestions.get(intent, [])[:4]

    # ===== Utility Methods =====

    async def explain_acronym(self, acronym: str) -> str:
        """Quick lookup for a specific acronym from RAG."""
        results = await self._search_collection(
            KNOWLEDGE_CHUNKS,
            f"FPT acronym {acronym} meaning definition stands for",
            limit=3,
            score_threshold=0.3,
            filter_conditions={"doc_type": "acronym"},
        )

        for r in results:
            if r.get("acronym", "").upper() == acronym.upper():
                return f"**{r['acronym']}**: {r['full_form']}"

        # Not found in RAG
        return f"'{acronym}' not found in FPT knowledge base. Please add it to the acronyms database."

    async def get_pr_summary(self) -> str:
        """Get a quick summary of the current PR."""
        parsed_pr = self.state.get("parsed_pr", {})

        if not parsed_pr:
            return "No PR loaded in current session."

        return f"""**Current PR Summary:**
- **Code**: {parsed_pr.get("pr_code", "N/A")}
- **Title**: {parsed_pr.get("title", "N/A")}
- **Platform**: {parsed_pr.get("platform", "N/A")}
- **Engine Type**: {parsed_pr.get("engine_type", "N/A")}
- **Customer**: {parsed_pr.get("customer", "N/A")}
- **Activities**: {len(parsed_pr.get("raw_activities", []))} detected"""

"""
FPT Cost Brain 2.0 - LangGraph Estimation Graph
Main workflow orchestration with conditional routing
"""

from typing import Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from agents.state import EstimationState, StepStatus


class EstimationGraph:
    """
    LangGraph-based estimation workflow.

    The workflow follows these steps:
    1. intake -> Parse and validate PR file
    2. qa -> Generate questions, wait for answers
    3. summary -> Analyze PR, extract features, find similar PRs
    4. estimation -> Run prediction, apply rules
    5. review -> Allow edits, capture feedback
    6. export -> Generate PE02 documents
    7. learning -> Extract rules from corrections

    Conditional edges handle:
    - Invalid PR files -> error state
    - Skipped Q&A -> jump to summary
    - User requests re-estimation -> back to estimation
    """

    def __init__(self, checkpointer: AsyncPostgresSaver | None = None):
        self.checkpointer = checkpointer
        self._db_session = None
        self.graph = self._build_graph()

    def set_db_session(self, db_session):
        """Set the database session for nodes that need it (e.g., learning)."""
        self._db_session = db_session

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the graph with our state type
        workflow = StateGraph(EstimationState)

        # Add all nodes
        workflow.add_node("intake", self._intake_node)
        workflow.add_node("qa_generate", self._qa_generate_node)
        workflow.add_node("qa_wait", self._qa_wait_node)
        workflow.add_node("summary", self._summary_node)
        workflow.add_node("estimation", self._estimation_node)
        workflow.add_node("review", self._review_node)
        workflow.add_node("export", self._export_node)
        workflow.add_node("learning", self._learning_node)
        workflow.add_node("error", self._error_node)

        # Set entry point
        workflow.set_entry_point("intake")

        # Add edges with conditional routing
        # LAZY LOADING: After intake, go directly to qa_wait (not qa_generate)
        # Questions will be generated on-demand when user enters Q&A step
        workflow.add_conditional_edges(
            "intake",
            self._route_after_intake,
            {
                "qa_wait": "qa_wait",  # Skip qa_generate during upload
                "qa_generate": "qa_generate",  # Only used for explicit generation
                "error": "error",
            },
        )

        workflow.add_edge("qa_generate", "qa_wait")

        workflow.add_conditional_edges(
            "qa_wait",
            self._route_after_qa,
            {
                "summary": "summary",
                "end": END,  # Pause at END when waiting for input
            },
        )

        # After summary, pause for user to review before estimation
        workflow.add_conditional_edges(
            "summary",
            self._route_after_summary,
            {
                "estimation": "estimation",
                "end": END,  # Pause at END for user to review summary
            },
        )

        workflow.add_conditional_edges(
            "estimation",
            self._route_after_estimation,
            {
                "review": "review",
                "error": "error",
            },
        )

        workflow.add_conditional_edges(
            "review",
            self._route_after_review,
            {
                "export": "export",
                "estimation": "estimation",  # Re-estimate
                "end": END,  # Pause at END when waiting for user review
            },
        )

        workflow.add_edge("export", "learning")
        workflow.add_edge("learning", END)
        workflow.add_edge("error", END)

        return workflow

    # ===== Node Implementations =====

    async def _intake_node(self, state: EstimationState) -> EstimationState:
        """Parse and validate the PR file."""
        from agents.nodes.intake_node import process_intake

        return await process_intake(state)

    async def _qa_generate_node(self, state: EstimationState) -> EstimationState:
        """Generate clarifying questions."""
        from agents.nodes.qa_node import generate_questions

        return await generate_questions(state)

    async def _qa_wait_node(self, state: EstimationState) -> EstimationState:
        """Wait for user to answer questions (LAZY LOADING entry point)."""
        # Set current_step to qa_wait so frontend knows to trigger question generation
        state["current_step"] = "qa_wait"
        # This is an interrupt point - the graph pauses here
        # until the user provides answers
        state["step_status"]["qa"] = StepStatus.WAITING_INPUT
        return state

    async def _summary_node(self, state: EstimationState) -> EstimationState:
        """Analyze PR and generate summary."""
        from agents.nodes.summary_node import process_summary

        return await process_summary(state)

    async def _estimation_node(self, state: EstimationState) -> EstimationState:
        """Run estimation and generate breakdown."""
        from agents.nodes.estimation_node import process_estimation, set_db_session

        # Set the database session for rule retrieval
        if self._db_session:
            set_db_session(self._db_session)

        return await process_estimation(state)

    async def _review_node(self, state: EstimationState) -> EstimationState:
        """Handle review step."""
        # This is an interrupt point - the graph pauses here
        # until the user finalizes or requests changes
        state["step_status"]["review"] = StepStatus.WAITING_INPUT
        state["current_step"] = "review"
        return state

    async def _export_node(self, state: EstimationState) -> EstimationState:
        """Generate export documents."""
        from agents.nodes.export_node import process_export

        return await process_export(state)

    async def _learning_node(self, state: EstimationState) -> EstimationState:
        """Extract learning from user corrections."""
        from agents.nodes.learning_node import process_learning, set_db_session

        # Set the database session for learning operations
        if self._db_session:
            set_db_session(self._db_session)

        return await process_learning(state)

    async def _error_node(self, state: EstimationState) -> EstimationState:
        """Handle error state."""
        state["current_step"] = "error"
        return state

    # ===== Routing Functions =====

    def _route_after_intake(
        self, state: EstimationState
    ) -> Literal["qa_wait", "qa_generate", "error"]:
        """Route after intake based on validation result.

        LAZY LOADING: By default, skip qa_generate and go to qa_wait.
        Questions will be generated on-demand when user enters Q&A step.
        Only use qa_generate if explicitly requested via _generate_questions flag.
        """
        if not state.get("is_valid", False):
            return "error"

        # Check if explicit question generation was requested
        if state.get("_generate_questions", False):
            return "qa_generate"

        # Default: skip to qa_wait (lazy loading)
        return "qa_wait"

    def _route_after_qa(self, state: EstimationState) -> Literal["summary", "end"]:
        """Route after Q&A based on completion status."""
        if state.get("qa_complete", False) or state.get("qa_skipped", False):
            return "summary"
        # Return "end" to pause and wait for user input
        return "end"

    def _route_after_summary(
        self, state: EstimationState
    ) -> Literal["estimation", "end"]:
        """Route after summary - pause for user to review summary before estimation."""
        # Continue to estimation if explicitly requested
        if state.get("summary_reviewed", False):
            return "estimation"
        # Pause at END to show summary to user
        return "end"

    def _route_after_estimation(
        self, state: EstimationState
    ) -> Literal["review", "error"]:
        """Route after estimation based on success."""
        if state.get("breakdown") and len(state["breakdown"]) > 0:
            return "review"
        return "error"

    def _route_after_review(
        self, state: EstimationState
    ) -> Literal["export", "estimation", "end"]:
        """Route after review based on user action."""
        if state.get("is_finalized", False):
            return "export"
        # Check if user requested re-estimation
        if state.get("_reestimate_requested", False):
            return "estimation"
        # Return "end" to pause and wait for user review
        return "end"

    # ===== Public Methods =====

    def compile(self):
        """Compile the graph for execution."""
        if self.checkpointer:
            return self.graph.compile(checkpointer=self.checkpointer)
        return self.graph.compile()

    async def run(
        self,
        initial_state: EstimationState,
        config: dict | None = None,
    ) -> EstimationState:
        """Run the graph from initial state."""
        compiled = self.compile()
        result = await compiled.ainvoke(initial_state, config=config)
        return result

    async def resume(
        self,
        thread_id: str,
        updates: dict,
        config: dict | None = None,
    ) -> EstimationState:
        """Resume the graph from a checkpoint with updates."""
        compiled = self.compile()
        config = config or {}
        config["configurable"] = {"thread_id": thread_id}

        # Get current state
        state = await compiled.aget_state(config)

        # Apply updates
        current_state = state.values
        current_state.update(updates)

        # Resume execution
        result = await compiled.ainvoke(current_state, config=config)
        return result

    async def get_state(self, thread_id: str) -> EstimationState | None:
        """Get the current state for a thread."""
        if not self.checkpointer:
            return None

        compiled = self.compile()
        config = {"configurable": {"thread_id": thread_id}}
        state = await compiled.aget_state(config)
        return state.values if state else None


# Global checkpointer singleton - managed for app lifecycle
_checkpointer_instance: AsyncPostgresSaver | None = None
_checkpointer_context = None


async def get_checkpointer(database_url: str) -> AsyncPostgresSaver:
    """Get or create the singleton checkpointer instance."""
    global _checkpointer_instance, _checkpointer_context

    if _checkpointer_instance is None:
        # Create the context manager
        _checkpointer_context = AsyncPostgresSaver.from_conn_string(database_url)
        # Enter the context manager to get the checkpointer
        _checkpointer_instance = await _checkpointer_context.__aenter__()
        # Setup the tables
        await _checkpointer_instance.setup()

    return _checkpointer_instance


async def cleanup_checkpointer():
    """Cleanup the checkpointer on app shutdown."""
    global _checkpointer_instance, _checkpointer_context

    if _checkpointer_context is not None:
        await _checkpointer_context.__aexit__(None, None, None)
        _checkpointer_instance = None
        _checkpointer_context = None


async def create_estimation_graph(
    database_url: str | None = None,
) -> EstimationGraph:
    """
    Create an estimation graph with PostgreSQL checkpointing.

    Args:
        database_url: PostgreSQL connection URL for checkpointing.
                     If None, runs without persistence.

    Returns:
        Configured EstimationGraph instance.
    """
    checkpointer = None

    if database_url:
        # Get or create the singleton checkpointer
        checkpointer = await get_checkpointer(database_url)

    return EstimationGraph(checkpointer=checkpointer)


# Convenience function for creating a simple graph without persistence
def create_simple_graph() -> EstimationGraph:
    """Create a simple estimation graph without persistence."""
    return EstimationGraph()

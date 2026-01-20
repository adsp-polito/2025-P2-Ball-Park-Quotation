"""
FPT Cost Brain 2.0 - Export Node
Generate PE02 PPTX and Excel exports using PE02Generator
"""

from datetime import datetime, timezone

from agents.state import EstimationState, ExportResult, StepStatus
from export.pe02_generator import ExportOptions, PE02Data, PE02Generator


async def process_export(state: EstimationState) -> EstimationState:
    """
    Process the export step: generate PE02 documents.

    This step:
    1. Generates PE02 PowerPoint presentation
    2. Generates detailed Excel breakdown
    3. Optionally bundles both into ZIP
    """
    state["step_status"]["export"] = StepStatus.IN_PROGRESS
    state["current_step"] = "export"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    breakdown = state.get("breakdown", [])
    export_format = state.get("export_format", "pptx")
    export_language = state.get("export_language", "en")

    if not breakdown:
        state["error_message"] = "No breakdown data available for export"
        state["error_step"] = "export"
        state["step_status"]["export"] = StepStatus.ERROR
        return state

    try:
        # Create PE02 data from state
        pe02_data = PE02Data.from_state(state)

        # Configure export options
        options = ExportOptions(
            language=export_language,
            include_confidence=True,
            include_reasoning=False,
            include_similar_prs=True,
        )

        # Initialize generator
        generator = PE02Generator(options=options)

        export_result: ExportResult = {
            "pptx_path": None,
            "xlsx_path": None,
            "pptx_bytes": None,
            "xlsx_bytes": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Generate based on format
        if export_format == "bundle":
            bundle_bytes = generator.generate_bundle(pe02_data)
            export_result["bundle_bytes"] = bundle_bytes
        else:
            if export_format in ["pptx", "bundle"]:
                export_result["pptx_bytes"] = generator.generate_pptx(pe02_data)

            if export_format in ["xlsx", "bundle"]:
                export_result["xlsx_bytes"] = generator.generate_xlsx(pe02_data)

        state["export_result"] = export_result
        state["step_status"]["export"] = StepStatus.COMPLETED

    except Exception as e:
        state["error_message"] = f"Export failed: {str(e)}"
        state["error_step"] = "export"
        state["step_status"]["export"] = StepStatus.ERROR

    return state

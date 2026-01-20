"""
FPT Cost Brain 2.0 - Intake Node
Parse and validate PR Excel files

Ported comprehensive feature detection from v1 parser:
- Product family detection (E0C0, NEF, CURSOR, F1, E5F0)
- Emissions detection (Stage V, Tier 4B, China NRIV, Euro VI)
- Sector detection (AG, CE, PT)
- ATS technology detection
- Boolean flags for change types
- Raw text extraction for LLM processing
"""

import io
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from agents.state import EstimationState, ParsedPR, StepStatus
from app.debug_logging import log_parsed_pr, log_error_details

logger = logging.getLogger(__name__)

# ============================================================================
# DICTATOR LOGIC v2: Strict Sector Lookup (Module 1.5)
# Priority: Platform/Machine > Engine Hint > Keywords
# Key Insight: Product Line identifies ENGINE TYPE, not SECTOR!
#              Same engine can be used in both AG and CE applications.
# ============================================================================

# Load sector lookup from reference data (lazy loading)
_SECTOR_LOOKUP: dict | None = None


def _load_sector_lookup() -> dict[str, Any]:
    """Load sector lookup from ref_sector_lookup.json (lazy loading)."""
    global _SECTOR_LOOKUP
    if _SECTOR_LOOKUP is not None:
        return _SECTOR_LOOKUP

    lookup_path = (
        Path(__file__).parent.parent.parent / "data" / "ref_sector_lookup.json"
    )
    try:
        with open(lookup_path) as f:
            loaded = json.load(f)
            # Normalize codes to uppercase for consistent comparison
            loaded["_engine_codes_upper"] = {
                c.upper() for c in loaded.get("valid_engine_codes", [])
            }
            loaded["_ats_codes_upper"] = {
                c.upper() for c in loaded.get("valid_ats_codes", [])
            }
            _SECTOR_LOOKUP = loaded
            logger.info(
                f"Loaded sector lookup v2: "
                f"{len(_SECTOR_LOOKUP.get('by_platform', {}))} platforms, "
                f"{len(_SECTOR_LOOKUP.get('by_machine', {}))} machines, "
                f"{len(_SECTOR_LOOKUP.get('engine_family_hint', {}))} engine hints"
            )
    except FileNotFoundError:
        logger.warning(f"Sector lookup not found at {lookup_path}, using empty lookup")
        _SECTOR_LOOKUP = {
            "valid_engine_codes": [],
            "valid_ats_codes": [],
            "_engine_codes_upper": set(),
            "_ats_codes_upper": set(),
            "by_platform": {},
            "by_machine": {},
            "engine_family_hint": {},
        }

    return _SECTOR_LOOKUP


def is_valid_engine_code(code: str) -> bool:
    """Check if a product line code is a valid ENGINE (not ATS)."""
    lookup = _load_sector_lookup()
    return code.upper() in lookup.get("_engine_codes_upper", set())


def is_ats_code(code: str) -> bool:
    """Check if a code is an ATS (After Treatment System) component."""
    lookup = _load_sector_lookup()
    return code.upper() in lookup.get("_ats_codes_upper", set())


def resolve_sector_strictly(
    parsed_pr_data: dict, keyword_guess: str = "AG"
) -> tuple[str, str]:
    """
    Determine Sector with STRICT priority (Dictator Logic v2):

    KEY INSIGHT: Product Line identifies ENGINE TYPE, not SECTOR!
    The same E6N0 engine can power both tractors (AG) and loaders (CE).
    Sector is determined by APPLICATION (Platform/Machine), not engine.

    Priority:
    1. Explicit 'Sector' field in PR (Highest Priority)
    2. Platform match ("CE Heavy" → CE, "Large Tractor" → AG)
    3. Machine type match ("excavator" in text → CE)
    4. Engine family HINT (E0N* → likely AG, but NOT definitive!)
    5. Keyword guess (Lowest Priority / Fallback)

    Returns:
        tuple: (sector, source) - sector value and how it was determined
    """
    lookup = _load_sector_lookup()

    # 1. Check Explicit Sector Field (Highest Priority)
    explicit_sector = parsed_pr_data.get("sector_explicit")
    if explicit_sector in ["AG", "CE"]:
        return explicit_sector, "explicit_field"

    # 2. Platform Match (High Confidence - Platform DEFINES sector)
    platform = parsed_pr_data.get("platform") or ""
    platform = str(platform).strip()
    if platform:
        # Exact match
        if platform in lookup.get("by_platform", {}):
            return lookup["by_platform"][platform], f"platform:{platform}"
        # Partial match (e.g., "CE Heavy Equipment" contains "CE Heavy")
        platform_lower = platform.lower()
        for plat_key, sector in lookup.get("by_platform", {}).items():
            if plat_key.lower() in platform_lower or platform_lower in plat_key.lower():
                return sector, f"platform_partial:{plat_key}"

    # 3. Machine Type Match (Medium Confidence - Machine implies sector)
    raw_text = str(parsed_pr_data.get("raw_text", "")).lower()
    machine_field = str(parsed_pr_data.get("machine", "")).lower()
    search_text = f"{machine_field} {raw_text}"

    for machine, sector in lookup.get("by_machine", {}).items():
        if machine in search_text:
            return sector, f"machine:{machine}"

    # 4. Engine Family HINT (Low Confidence - just a hint, not definitive!)
    # E0N family is PRIMARILY used in AG, but can be in CE too
    product_line = (
        parsed_pr_data.get("product_family") or parsed_pr_data.get("engine") or ""
    )
    product_line = str(product_line).strip().upper()

    if product_line:
        # Check engine_family_hint (prefix matching)
        for family_prefix, sector in lookup.get("engine_family_hint", {}).items():
            family_prefix_upper = family_prefix.upper()
            if (
                product_line.startswith(family_prefix_upper)
                or family_prefix_upper in product_line
            ):
                return sector, f"engine_hint:{family_prefix}(~{product_line})"

    # 5. Fallback to keyword guess (Last Resort)
    return keyword_guess, "keyword_fallback"


# ============================================================================
# Feature Detection Patterns (Ported from v1 pr_excel_parser.py)
# ============================================================================

# Product family detection patterns
PRODUCT_FAMILIES: dict[str, list[str]] = {
    "E0C0": ["E0C0", "E9C0"],
    "NEF": ["NEF", "N45", "N67", "E0N6", "E6N0"],
    "CURSOR": ["CURSOR", "C87", "C9", "C11", "C13"],
    "F1": ["F1", "F1C", "F1A"],
    "E5F0": ["E5F0", "E3F6", "E5FC"],
}

# Emission standard detection patterns
EMISSIONS: dict[str, list[str]] = {
    "Stage V": ["Stage V", "Stage 5", "StageV"],
    "Tier 4B": ["Tier 4B", "Tier 4b", "Tier4B", "T4B"],
    "China NRIV": ["China IV", "China NRIV", "NRIV", "China 4"],
    "Euro VI": ["Euro VI", "Euro 6", "EuroVI"],
}

# Sector detection patterns
SECTORS: dict[str, list[str]] = {
    "AG": ["AG", "Agriculture", "Tractor", "Harvester", "AGST", "AGSV"],
    "CE": ["CE", "Construction", "Excavator", "Loader", "CEWI", "CESSL"],
    "PT": ["PT", "Powertrain", "Truck"],
}

# ATS (Aftertreatment System) technology detection patterns
ATS_TECH: dict[str, list[str]] = {
    "DOC_SCRoF": ["DOC_SCRoF", "SCRoF", "DOC+SCRoF"],
    "DOC_SCR-T": ["DOC_SCR-T", "SCR-T", "DOC+SCR"],
    "SCR_only": ["SCR_only", "SCR only"],
    "DOC_only": ["DOC_only", "DOC only"],
}

# Keywords for boolean flag detection
HARDWARE_KEYWORDS = ["HARDWARE", "NEW COMPONENT", "NEW PART", "INJECTOR", "TURBO"]
CALIBRATION_KEYWORDS = ["CALIBRATION", "CALIBRAT", "CAL RELEASE", "TUNING"]
ATS_KEYWORDS = ["ATS", "SCR", "DOC", "DPF", "AFTERTREATMENT"]
SOFTWARE_KEYWORDS = ["SOFTWARE", "SW", "VCU", "ECU", "EMS", "FIRMWARE"]

# Component-specific keywords (for sizing classification)
TURBO_KEYWORDS = ["TURBO", "TURBOCHARGER", "TURBOCHARGING", "WASTEGATE", "VGT"]
INJECTOR_KEYWORDS = ["INJECTOR", "INJECTION", "FUEL INJECTOR", "COMMON RAIL"]
FUEL_RAIL_KEYWORDS = ["FUEL RAIL", "RAIL PRESSURE", "HIGH PRESSURE PUMP", "HPP"]
EGR_KEYWORDS = ["EGR", "EXHAUST GAS RECIRCULATION", "EGR VALVE", "EGR COOLER"]
COOLING_KEYWORDS = ["COOLING", "COOLANT", "RADIATOR", "WATER PUMP", "THERMOSTAT"]

# Power and torque patterns
POWER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kW|KW|kw)", re.IGNORECASE)
TORQUE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:Nm|NM|nm|N\.m)", re.IGNORECASE)
POWER_INCREASE_PATTERN = re.compile(
    r"(?:power\s*(?:increase|upgrade|boost)|increase.*?power|upgrade.*?power).*?(\d+(?:\.\d+)?)\s*(?:kW|KW|kw)",
    re.IGNORECASE,
)
TORQUE_INCREASE_PATTERN = re.compile(
    r"(?:torque\s*(?:increase|upgrade|boost)|increase.*?torque|upgrade.*?torque).*?(\d+(?:\.\d+)?)\s*(?:Nm|NM|nm)",
    re.IGNORECASE,
)

# Form noise to filter out (common Excel form fields with no content)
FORM_NOISE_PATTERNS = [
    r"^Sign-off:?$",
    r"^Name$",
    r"^Date$",
    r"^Writer$",
    r"^PCM$",
    r"^EICC Responsible$",
    r"^PM Manager$",
    r"^Func\.\s*\d+$",
    r"^Dev\.\s*\d+$",
    r"^PB\s*\d+$",
    r"^\d+P$",
    r"^Job\d*$",
    r"^Qty$",
    r"^Ex-Works$",
    r"^Vehicle Date$",
    r"^Requested Time$",
    r"^Product$",
    r"^Engine$",
]
FORM_NOISE_RE = re.compile("|".join(FORM_NOISE_PATTERNS), re.IGNORECASE)


async def process_intake(state: EstimationState) -> EstimationState:
    """
    Process the intake step: parse and validate PR file.

    Reads the uploaded Excel file, extracts relevant fields,
    and validates against the expected schema.
    """
    intake_start = time.time()
    logger.info("=" * 70)
    logger.info("📥 INTAKE NODE STARTED")
    logger.info("=" * 70)

    # Skip if intake was already completed (avoid re-parsing)
    # Check both enum and string value (after JSON deserialization)
    intake_status = state.get("step_status", {}).get("intake")
    if intake_status in (StepStatus.COMPLETED, "completed"):
        logger.info("  ⏭️ Intake already completed, skipping")
        return state

    # Skip if we already have parsed_pr (state was loaded from Redis)
    if state.get("parsed_pr") and state.get("is_valid"):
        logger.info("  ⏭️ Parsed PR already exists, skipping")
        state["step_status"]["intake"] = StepStatus.COMPLETED
        return state

    state["step_status"]["intake"] = StepStatus.IN_PROGRESS
    state["current_step"] = "intake"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    file_bytes = state.get("pr_file_bytes")
    filename = state.get("pr_filename", "unknown.xlsx")
    logger.info(f"  📄 File: {filename} ({len(file_bytes) if file_bytes else 0} bytes)")

    if not file_bytes:
        logger.info("  🎭 No file uploaded, using DEMO mode")
        # Create demo PR data for testing without file upload
        demo_pr: ParsedPR = {
            "pr_code": f"DEMO-{state['session_id'][:8].upper()}",
            "title": "Demo Product Request - Stress Analysis Package",
            "description": "Demonstration PR for testing the estimation workflow",
            "program_family": "A320 Family",
            "customer": "Airbus",
            "project_phase": "Preliminary Design",
            "raw_activities": [
                {"code": "STR-001", "name": "Static Stress Analysis", "hours": 120},
                {"code": "STR-002", "name": "Fatigue Analysis", "hours": 80},
                {"code": "STR-003", "name": "Damage Tolerance", "hours": 60},
                {"code": "DOC-001", "name": "Technical Documentation", "hours": 40},
            ],
            "raw_data": {
                "filename": "demo_pr.xlsx",
                "demo_mode": True,
                "shape": (10, 5),
                "columns": ["Activity", "Description", "Hours", "Rate", "Cost"],
            },
            "validation_errors": [],
        }

        state["parsed_pr"] = demo_pr
        state["validation_result"] = {
            "valid": True,
            "errors": [],
            "warnings": ["Using demo data - no file was uploaded"],
            "extracted_fields": {
                "pr_code": True,
                "title": True,
                "program_family": True,
                "customer": True,
                "activities_count": 4,
            },
        }
        state["is_valid"] = True
        state["step_status"]["intake"] = StepStatus.COMPLETED
        logger.info(f"  ✅ Demo PR created in {time.time() - intake_start:.2f}s")
        return state

    try:
        # Parse the Excel file
        logger.info("  📊 Parsing Excel file...")
        parse_start = time.time()
        parsed_pr = await parse_pr_file(file_bytes, filename)
        logger.info(f"  ✅ Parsed in {time.time() - parse_start:.2f}s")
        log_parsed_pr(logger, parsed_pr)

        # Validate the parsed data
        logger.info("  🔍 Validating parsed data...")
        validation_result = await validate_pr_data(parsed_pr)
        logger.info(
            f"  Validation: {'✅ VALID' if validation_result.get('valid') else '❌ INVALID'}"
        )
        if validation_result.get("errors"):
            logger.warning(f"  Errors: {validation_result['errors']}")
        if validation_result.get("warnings"):
            logger.info(f"  Warnings: {validation_result['warnings']}")

        state["parsed_pr"] = parsed_pr
        state["validation_result"] = validation_result
        state["is_valid"] = validation_result.get("valid", False)

        if state["is_valid"]:
            state["step_status"]["intake"] = StepStatus.COMPLETED

            # Run early feature extraction (rule-based only, no LLM)
            # This populates feature_extraction_result for Q&A to generate
            # targeted questions based on missing/low-confidence features
            logger.info("  🔧 Running early feature extraction...")
            feature_start = time.time()
            feature_extraction_result = await _run_early_feature_extraction(parsed_pr)
            state["feature_extraction_result"] = feature_extraction_result
            logger.info(
                f"  ✅ Feature extraction in {time.time() - feature_start:.2f}s: "
                f"confidence={feature_extraction_result['confidence']:.0%}, "
                f"missing={len(feature_extraction_result['missing_features'])} features"
            )
        else:
            state["error_message"] = "; ".join(validation_result.get("errors", []))
            state["error_step"] = "intake"
            logger.error(f"  ❌ Validation failed: {state['error_message']}")
            state["step_status"]["intake"] = StepStatus.ERROR

    except Exception as e:
        state["validation_result"] = {"valid": False, "errors": [str(e)]}
        state["is_valid"] = False
        state["error_message"] = f"Failed to parse PR file: {str(e)}"
        state["error_step"] = "intake"
        state["step_status"]["intake"] = StepStatus.ERROR

    return state


def extract_structured_pr_text(df: pd.DataFrame, filename: str) -> str:
    """
    Extract structured, LLM-optimized text from FPT PR Excel format.

    Produces clean, well-structured output that:
    1. Preserves label-value relationships
    2. Groups related information into sections
    3. Filters out form noise and empty fields
    4. Provides context for LLM analysis

    Args:
        df: DataFrame read from Excel without header
        filename: Original filename for PR code extraction

    Returns:
        Well-structured text optimized for LLM processing
    """
    sections: dict[str, list[str]] = {
        "header": [],
        "project_info": [],
        "technical": [],
        "description": [],
        "engine_dressing": [],
        "revision_history": [],
        "other": [],
    }

    current_section = "header"
    pr_code = ""
    title = ""
    description_lines: list[str] = []
    in_description = False

    pr_match = re.search(r"PR_(\d+)(?:_rev_([A-Z]))?", filename, re.IGNORECASE)
    if pr_match:
        pr_code = f"PR_{pr_match.group(1)}"
        if pr_match.group(2):
            pr_code += f" (Revision {pr_match.group(2)})"

    for i, row in df.iterrows():
        row_texts = []
        for j, cell in enumerate(row):
            if pd.notna(cell):
                cell_str = str(cell).strip()
                if cell_str and not FORM_NOISE_RE.match(cell_str):
                    row_texts.append((j, cell_str))

        if not row_texts:
            continue

        combined_row = " | ".join(t[1] for t in row_texts)
        first_cell = row_texts[0][1] if row_texts else ""

        if "PRODUCT REQUEST" in combined_row.upper():
            for _, text in row_texts:
                if re.match(r"PR_\d+", text):
                    pr_code = text
            continue

        if first_cell.startswith("Title:"):
            title = first_cell.replace("Title:", "").strip()
            for _, text in row_texts[1:]:
                if text and text != title:
                    title += " " + text
            sections["header"].append(f"Title: {title}")
            continue

        if first_cell.startswith("Platform:") or first_cell.startswith("Plant:"):
            info_parts = []
            for _, text in row_texts:
                if ":" in text:
                    info_parts.append(text)
                elif info_parts:
                    info_parts[-1] += " " + text
            for part in info_parts:
                sections["project_info"].append(part)
            continue

        if first_cell.startswith("Engine:") or first_cell.startswith("Tier:"):
            info_parts = []
            for _, text in row_texts:
                if ":" in text:
                    info_parts.append(text)
                elif info_parts:
                    info_parts[-1] += " " + text
            for part in info_parts:
                sections["technical"].append(part)
            continue

        if first_cell == "Description:":
            in_description = True
            continue

        if first_cell == "Engine Dressing:":
            in_description = False
            current_section = "engine_dressing"
            continue

        if first_cell == "Shipping Conditions:":
            current_section = "other"
            continue

        if first_cell == "Revision History:":
            current_section = "revision_history"
            continue

        if in_description:
            desc_text = " ".join(t[1] for t in row_texts)
            if desc_text and len(desc_text) > 3:
                description_lines.append(desc_text)
            continue

        if first_cell.startswith("Vehicle Models:"):
            models = first_cell.replace("Vehicle Models:", "").strip()
            for _, text in row_texts[1:]:
                models += " " + text
            if models.strip():
                sections["project_info"].append(f"Vehicle Models: {models.strip()}")
            continue

        if current_section == "revision_history" and len(row_texts) >= 2:
            rev_text = " | ".join(t[1] for t in row_texts)
            if any(c.isalnum() for c in rev_text):
                sections["revision_history"].append(rev_text)
            continue

    if description_lines:
        sections["description"] = description_lines

    output_parts = []
    output_parts.append("=" * 60)
    output_parts.append(f"FPT PRODUCT REQUEST: {pr_code}")
    output_parts.append("=" * 60)

    if sections["header"]:
        output_parts.append("")
        for line in sections["header"]:
            output_parts.append(line)

    if sections["project_info"]:
        output_parts.append("")
        output_parts.append("--- PROJECT INFORMATION ---")
        for line in sections["project_info"]:
            output_parts.append(f"  {line}")

    if sections["technical"]:
        output_parts.append("")
        output_parts.append("--- TECHNICAL SPECIFICATIONS ---")
        for line in sections["technical"]:
            output_parts.append(f"  {line}")

    if sections["description"]:
        output_parts.append("")
        output_parts.append("--- PROJECT DESCRIPTION ---")
        for line in sections["description"]:
            output_parts.append(f"  {line}")

    if sections["revision_history"]:
        valid_revisions = [r for r in sections["revision_history"] if len(r) > 10]
        if valid_revisions:
            output_parts.append("")
            output_parts.append("--- REVISION HISTORY ---")
            for line in valid_revisions[:5]:
                output_parts.append(f"  {line}")

    output_parts.append("")
    output_parts.append("=" * 60)

    return "\n".join(output_parts)


def generate_llm_context(parsed: dict, raw_text: str) -> str:
    """
    Generate comprehensive LLM-optimized context from parsed PR data.

    This function creates a ChatGPT-quality structured text that includes
    ALL relevant information for accurate cost estimation.

    The output format is designed for LLM consumption with:
    - Clear section headers
    - Complete information (no truncation)
    - Context clues for understanding
    - Consistent formatting
    """
    lines: list[str] = []

    # Header
    pr_code = parsed.get("pr_code", "UNKNOWN")
    revision = parsed.get("revision", "")
    revision_str = f" (Revision {revision})" if revision else ""

    lines.append("=" * 70)
    lines.append(f"FPT PRODUCT REQUEST: {pr_code}{revision_str}")
    lines.append("=" * 70)

    # Title
    title = parsed.get("title", "")
    if title:
        lines.append("")
        lines.append(f"TITLE: {title}")

    # PR Types (critical for estimation!)
    pr_types = parsed.get("pr_types", [])
    if pr_types:
        lines.append("")
        lines.append(f"PR TYPE: {', '.join(pr_types)}")
    else:
        lines.append("")
        lines.append("PR TYPE: Not specified (likely standard engineering change)")

    # Project Information Section
    lines.append("")
    lines.append("─" * 40)
    lines.append("PROJECT INFORMATION")
    lines.append("─" * 40)

    info_fields = [
        ("Platform", parsed.get("platform")),
        ("Plant", parsed.get("plant")),
        ("Engine", parsed.get("engine")),
        ("Tier/Emission Standard", parsed.get("tier")),
        ("Product Family", parsed.get("product_family")),
        (
            "Sector",
            f"{parsed.get('sector', 'N/A')} (source: {parsed.get('sector_source', 'N/A')})",
        ),
        ("Priority", parsed.get("priority")),
        ("Target Date", parsed.get("target_date")),
        ("EPP No", parsed.get("epp_no")),
        ("Worktask No", parsed.get("worktask_no")),
    ]

    for label, value in info_fields:
        if value and str(value).strip():
            lines.append(f"  {label}: {value}")

    # Vehicle Models
    vehicle_models = parsed.get("vehicle_models", [])
    if vehicle_models:
        lines.append("")
        lines.append("  Vehicle Models:")
        for model in vehicle_models:
            lines.append(f"    - {model}")

    # Countries/Markets
    countries = parsed.get("countries", [])
    if countries:
        lines.append("")
        lines.append(f"  Target Markets: {', '.join(countries)}")

    # Description Section (CRITICAL for estimation)
    lines.append("")
    lines.append("─" * 40)
    lines.append("PROJECT DESCRIPTION")
    lines.append("─" * 40)

    description = parsed.get("description", "")
    if description:
        for desc_line in description.split("\n"):
            if desc_line.strip():
                lines.append(f"  {desc_line.strip()}")
    else:
        lines.append("  No description provided.")

    # Motivation Section
    motivation = parsed.get("motivation", "")
    if motivation:
        lines.append("")
        lines.append("─" * 40)
        lines.append("MOTIVATION / BUSINESS CASE")
        lines.append("─" * 40)
        for mot_line in motivation.split("\n"):
            if mot_line.strip():
                lines.append(f"  {mot_line.strip()}")

    # Technical Changes Summary (for ML feature alignment)
    lines.append("")
    lines.append("─" * 40)
    lines.append("TECHNICAL SCOPE ANALYSIS")
    lines.append("─" * 40)

    change_flags = [
        ("Hardware Changes", parsed.get("hardware_change", False)),
        ("Calibration Changes", parsed.get("calibration_change", False)),
        ("ATS/Aftertreatment Changes", parsed.get("ats_change", False)),
        ("Software/VCU Changes", parsed.get("software_vcu_change", False)),
        ("Turbo-Related", parsed.get("turbo_related", False)),
        ("Injector-Related", parsed.get("injectors_related", False)),
        ("Fuel Rail-Related", parsed.get("fuel_rail_related", False)),
        ("EGR-Related", parsed.get("EGR_related", False)),
        ("Cooling System-Related", parsed.get("cooling_system_related", False)),
    ]

    detected_changes = [label for label, value in change_flags if value]
    if detected_changes:
        lines.append("  Detected Change Types:")
        for change in detected_changes:
            lines.append(f"    ✓ {change}")
    else:
        lines.append("  No specific change types detected from keywords.")

    # Power/Torque if specified
    power_kw = parsed.get("power_kw")
    torque_nm = parsed.get("torque_nm")
    power_increase = parsed.get("power_increase_kw")
    torque_increase = parsed.get("torque_increase_nm")

    if any([power_kw, torque_nm, power_increase, torque_increase]):
        lines.append("")
        lines.append("  Power/Torque Specifications:")
        if power_kw:
            lines.append(f"    Power: {power_kw} kW")
        if torque_nm:
            lines.append(f"    Torque: {torque_nm} Nm")
        if power_increase:
            lines.append(f"    Power Increase: +{power_increase} kW")
        if torque_increase:
            lines.append(f"    Torque Increase: +{torque_increase} Nm")

    # Engine Dressing
    engine_dressing = parsed.get("engine_dressing", "")
    if engine_dressing:
        lines.append("")
        lines.append("─" * 40)
        lines.append("ENGINE DRESSING / CONFIGURATION")
        lines.append("─" * 40)
        for ed_line in engine_dressing.split("\n"):
            if ed_line.strip():
                lines.append(f"  {ed_line.strip()}")

    # Shipping Conditions
    shipping = parsed.get("shipping_conditions", "")
    if shipping:
        lines.append("")
        lines.append("─" * 40)
        lines.append("SHIPPING CONDITIONS")
        lines.append("─" * 40)
        for ship_line in shipping.split("\n"):
            if ship_line.strip():
                lines.append(f"  {ship_line.strip()}")

    # Revision History (if available)
    revision_history = parsed.get("revision_history", [])
    if revision_history:
        lines.append("")
        lines.append("─" * 40)
        lines.append("REVISION HISTORY")
        lines.append("─" * 40)
        for rev in revision_history[:5]:  # Limit to 5 entries
            lines.append(f"  {rev}")

    # Emissions Standard
    emissions = parsed.get("emissions", "")
    ats_tech = parsed.get("ats_tech", "")
    if emissions or ats_tech:
        lines.append("")
        lines.append("─" * 40)
        lines.append("EMISSIONS & AFTERTREATMENT")
        lines.append("─" * 40)
        if emissions:
            lines.append(f"  Emission Standard: {emissions}")
        if ats_tech:
            lines.append(f"  ATS Technology: {ats_tech}")

    # Footer
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF PRODUCT REQUEST DOCUMENT")
    lines.append("=" * 70)

    return "\n".join(lines)


async def parse_pr_file(file_bytes: bytes, filename: str) -> ParsedPR:
    """Parse PR Excel file and extract relevant data."""
    # Read Excel file without header to handle FPT format
    df_raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=None)

    # Also read with default header for generic parsing
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)

    # Initialize parsed data
    parsed: ParsedPR = {
        "pr_code": "",
        "title": "",
        "description": "",
        "program_family": "",
        "customer": "",
        "project_phase": "",
        "raw_activities": [],
        "raw_data": {},
        "validation_errors": [],
    }

    # Check if this is FPT PR format (has "PRODUCT REQUEST" in first row)
    is_fpt_format = False
    if len(df_raw) > 0:
        first_row_str = " ".join(str(x) for x in df_raw.iloc[0].values if pd.notna(x))
        if "PRODUCT REQUEST" in first_row_str.upper():
            is_fpt_format = True

    if is_fpt_format:
        # Parse FPT PR format
        parsed = parse_fpt_pr_format(df_raw, filename)
    else:
        # Use generic parsing
        parsed = parse_generic_format(df, filename)

    # Store raw data for reference
    parsed["raw_data"] = {
        "filename": filename,
        "shape": df_raw.shape,
        "columns": list(df.columns) if not is_fpt_format else ["FPT PR Format"],
        "sample_rows": df.head(5).to_dict("records") if not is_fpt_format else [],
    }

    return parsed


def parse_fpt_pr_format(df: pd.DataFrame, filename: str) -> ParsedPR:
    """
    Parse FPT-specific PR Excel format.

    Includes comprehensive feature detection ported from v1:
    - Product family, emissions, sector, ATS tech
    - Boolean flags for change types
    - Raw text extraction for LLM processing
    """
    parsed: ParsedPR = {
        "pr_code": "",
        "title": "",
        "description": "",
        "program_family": "",
        "customer": "",
        "project_phase": "",
        "raw_activities": [],
        "raw_data": {},
        "validation_errors": [],
    }

    # Build raw text from all cells (for feature detection and LLM)
    raw_texts: list[str] = []
    for _, row in df.iterrows():
        for cell in row:
            if pd.notna(cell) and str(cell).strip():
                raw_texts.append(str(cell).strip())
    raw_text = "\n".join(raw_texts)

    # Extract PR code and revision from filename first (most reliable)
    # Support multiple patterns:
    #   PR_18094_rev_D_2HMK_AG.xls (underscores, lowercase rev)
    #   PR_19111_Rev_C_2I1V.xls (underscores, uppercase Rev)
    pr_match = re.search(
        r"PR[_\s]?(\d+)[_\s]?[Rr]ev[_\s]?([A-Z])", filename, re.IGNORECASE
    )
    if pr_match:
        parsed["pr_code"] = f"PR_{pr_match.group(1)}"
        parsed["revision"] = pr_match.group(2).upper()
    elif "PR" in filename.upper():
        # Fallback: extract PR number from filename
        pr_num_match = re.search(r"PR[_\s]?(\d+)", filename, re.IGNORECASE)
        if pr_num_match:
            parsed["pr_code"] = f"PR_{pr_num_match.group(1)}"
        else:
            base_name = filename.split(".")[0]
            parsed["pr_code"] = base_name

    # Also try Row 0 cells (FPT location varies: Col 7 or Col 8)
    # Format 1: "PR_18094_rev_D" at Col 8
    # Format 2: "PR 19111 Rev C" at Col 7
    if not parsed["pr_code"] and len(df) > 0:
        for col_idx in [8, 7, 6, 9]:  # Check multiple columns
            if col_idx < len(df.columns):
                pr_code_cell = df.iloc[0, col_idx]
                if pd.notna(pr_code_cell):
                    cell_str = str(pr_code_cell).strip()
                    # Match "PR 19111 Rev C" or "PR_18094_rev_D" patterns
                    cell_match = re.search(r"PR[_\s]?(\d+)", cell_str, re.IGNORECASE)
                    if cell_match:
                        parsed["pr_code"] = f"PR_{cell_match.group(1)}"
                        # Extract revision if present
                        rev_match = re.search(
                            r"[Rr]ev[_\s]?([A-Z])", cell_str, re.IGNORECASE
                        )
                        if rev_match and not parsed.get("revision"):
                            parsed["revision"] = rev_match.group(1).upper()
                        break

    # =========================================================================
    # COMPREHENSIVE SECTION EXTRACTION (ChatGPT-quality parsing)
    # =========================================================================

    # Initialize all extraction variables
    engine_value = ""
    tier_value = ""
    platform_value = ""
    plant_value = ""
    priority_value = ""
    target_date = ""
    epp_no = ""
    worktask_no = ""
    product_profile_no = ""

    # Multi-line section tracking
    sections_content: dict[str, list[str]] = {
        "description": [],
        "motivation": [],
        "vehicle_models": [],
        "engine_dressing": [],
        "shipping_conditions": [],
        "revision_history": [],
        "countries": [],
    }

    # PR Type detection (checkboxes)
    pr_types_detected: list[str] = []
    PR_TYPE_KEYWORDS = ["New Engine Rating", "BOM", "Homologation", "Special Project"]

    # Section markers for multi-line extraction
    SECTION_MARKERS = {
        "Description:": "description",
        "Motivation:": "motivation",
        "Vehicle Models:": "vehicle_models",
        "Engine Dressing:": "engine_dressing",
        "Shipping Conditions:": "shipping_conditions",
        "Revision History:": "revision_history",
        "COUNTRIES SOLD TO:": "countries",
        "Countries:": "countries",
    }

    # End markers that signal section boundary
    END_MARKERS = [
        "Engine Dressing:",
        "Shipping Conditions:",
        "Revision History:",
        "Target Cost",
        "Motivation:",
        "PR Type",
        "Forecast Cycle",
        "Sign-off:",
        "Writer",
        "PCM",
        "Vehicle Date",
        "Product Profile",
    ]

    current_section: str | None = None

    for i in range(min(100, len(df))):
        row = df.iloc[i]
        row_cells: list[str] = []

        for j, cell in enumerate(row):
            if pd.isna(cell):
                continue
            cell_str = str(cell).strip()
            if not cell_str or cell_str in ["nan", "NaN"]:
                continue
            row_cells.append(cell_str)

            # === SINGLE-VALUE FIELD EXTRACTION ===

            # Title
            if cell_str.startswith("Title:"):
                parsed["title"] = cell_str.replace("Title:", "").strip()
                # Also get continuation from next cells
                for k in range(j + 1, len(row)):
                    next_cell = row.iloc[k] if k < len(row) else None
                    if pd.notna(next_cell) and str(next_cell).strip():
                        next_str = str(next_cell).strip()
                        if not any(
                            next_str.startswith(m)
                            for m in ["Platform:", "Plant:", "Date:"]
                        ):
                            parsed["title"] += " " + next_str

            # Platform
            if cell_str.startswith("Platform:"):
                platform_value = cell_str.replace("Platform:", "").strip()

            # Engine (handle both "Engine: E0N6" and "Engine_E0N6")
            if cell_str.startswith("Engine:"):
                engine_value = cell_str.replace("Engine:", "").strip()
            elif cell_str.startswith("Engine_"):
                engine_value = cell_str.replace("Engine_", "").strip()

            # Tier / Emission standard
            if cell_str.startswith("Tier:"):
                tier_value = cell_str.replace("Tier:", "").strip()

            # Plant
            if cell_str.startswith("Plant:"):
                plant_value = cell_str.replace("Plant:", "").strip()

            # Priority
            if cell_str.startswith("Priority:"):
                priority_value = cell_str.replace("Priority:", "").strip()

            # Target Date
            if cell_str.startswith("Target Date"):
                target_date = cell_str.split(":")[-1].strip() if ":" in cell_str else ""

            # EPP No
            if cell_str.startswith("EPP No"):
                epp_no = cell_str.split(":")[-1].strip() if ":" in cell_str else ""

            # Worktask No
            if cell_str.startswith("Worktask No"):
                worktask_no = cell_str.split(":")[-1].strip() if ":" in cell_str else ""

            # Product Profile No
            if cell_str.startswith("Product Profile No"):
                product_profile_no = (
                    cell_str.split(":")[-1].strip() if ":" in cell_str else ""
                )

            # === PR TYPE DETECTION (checkbox row) ===
            if cell_str == "PR Type":
                # Check subsequent cells in this row for "X" marks
                for type_idx, pr_type in enumerate(PR_TYPE_KEYWORDS):
                    check_col = j + 2 + (type_idx * 2)  # Typical spacing
                    if check_col < len(row):
                        check_cell = (
                            row.iloc[check_col] if check_col < len(row) else None
                        )
                        if (
                            pd.notna(check_cell)
                            and str(check_cell).strip().upper() == "X"
                        ):
                            pr_types_detected.append(pr_type)

            # Also detect X marks in rows following PR Type
            if cell_str.upper() == "X" and i > 0:
                # Check if this row has PR type markers
                row_text = " ".join(str(c) for c in row if pd.notna(c))
                for pr_type in PR_TYPE_KEYWORDS:
                    if pr_type.upper() in row_text.upper():
                        if pr_type not in pr_types_detected:
                            pr_types_detected.append(pr_type)

            # === SECTION MARKER DETECTION ===
            for marker, section_name in SECTION_MARKERS.items():
                if cell_str.startswith(marker) or cell_str == marker.rstrip(":"):
                    current_section = section_name
                    # Add any content after the marker
                    content_after = cell_str.replace(marker, "").strip()
                    if content_after and len(content_after) > 2:
                        sections_content[section_name].append(content_after)
                    break

            # === END MARKER DETECTION ===
            for end_marker in END_MARKERS:
                if cell_str.startswith(end_marker) and current_section:
                    if not any(cell_str.startswith(m) for m in SECTION_MARKERS.keys()):
                        current_section = None
                    break

        # === MULTI-LINE SECTION CONTENT ===
        if current_section and row_cells:
            combined_row = " ".join(row_cells)
            # Skip if this row is just a section header
            is_marker = any(
                combined_row.startswith(m.rstrip(":")) for m in SECTION_MARKERS.keys()
            )
            is_end = any(combined_row.startswith(e) for e in END_MARKERS)

            if not is_marker and not is_end and len(combined_row) > 3:
                # Filter out form noise
                if not FORM_NOISE_RE.match(combined_row):
                    sections_content[current_section].append(combined_row)

    # Store extracted basic fields
    parsed["platform"] = platform_value
    parsed["engine"] = engine_value
    parsed["tier"] = tier_value
    parsed["plant"] = plant_value
    parsed["customer"] = plant_value  # Alias for backward compatibility
    parsed["priority"] = priority_value
    parsed["target_date"] = target_date
    parsed["epp_no"] = epp_no
    parsed["worktask_no"] = worktask_no
    parsed["product_profile_no"] = product_profile_no
    parsed["pr_types"] = pr_types_detected

    # Store multi-line sections
    parsed["description"] = (
        "\n".join(sections_content["description"])
        if sections_content["description"]
        else ""
    )
    parsed["motivation"] = (
        "\n".join(sections_content["motivation"])
        if sections_content["motivation"]
        else ""
    )
    parsed["vehicle_models"] = sections_content["vehicle_models"]
    parsed["engine_dressing"] = (
        "\n".join(sections_content["engine_dressing"])
        if sections_content["engine_dressing"]
        else ""
    )
    parsed["shipping_conditions"] = (
        "\n".join(sections_content["shipping_conditions"])
        if sections_content["shipping_conditions"]
        else ""
    )
    parsed["revision_history"] = sections_content["revision_history"]
    parsed["countries"] = sections_content["countries"]

    # Set program_family from platform or engine
    if platform_value:
        parsed["program_family"] = platform_value
    elif engine_value:
        parsed["program_family"] = engine_value

    # Set project_phase from tier
    if tier_value:
        parsed["project_phase"] = tier_value

    # =========================================================================
    # Feature Detection (Ported from v1 pr_excel_parser.py)
    # =========================================================================
    text_upper = raw_text.upper()

    # Detect product family
    product_family = ""
    for family, patterns in PRODUCT_FAMILIES.items():
        for pattern in patterns:
            if pattern.upper() in text_upper:
                product_family = family
                break
        if product_family:
            break
    parsed["product_family"] = product_family

    # Detect emissions standard
    emissions = ""
    for emission, patterns in EMISSIONS.items():
        for pattern in patterns:
            if pattern.upper() in text_upper:
                emissions = emission
                break
        if emissions:
            break
    # Also check tier field specifically
    if not emissions and tier_value:
        for emission, patterns in EMISSIONS.items():
            for pattern in patterns:
                if pattern.upper() in tier_value.upper():
                    emissions = emission
                    break
    parsed["emissions"] = emissions

    # =========================================================================
    # DICTATOR LOGIC: Strict Sector Resolution (Module 1.5)
    # Priority: ref_Product lookup > keyword guessing
    # =========================================================================

    # First, get keyword-based guess as fallback
    keyword_sector = ""
    for sec, patterns in SECTORS.items():
        for pattern in patterns:
            if pattern.upper() in text_upper:
                keyword_sector = sec
                break
        if keyword_sector:
            break

    # Also check filename (e.g., PR_18094_rev_D_2HMK_AG.xls)
    if not keyword_sector:
        filename_upper = filename.upper()
        if "_AG" in filename_upper or "AG_" in filename_upper:
            keyword_sector = "AG"
        elif "_CE" in filename_upper or "CE_" in filename_upper:
            keyword_sector = "CE"
        elif "_PT" in filename_upper or "PT_" in filename_upper:
            keyword_sector = "PT"

    # Apply STRICT sector resolution (Dictator Logic)
    # This OVERRIDES keyword guess if reference data matches
    sector, sector_source = resolve_sector_strictly(
        parsed_pr_data={
            "product_family": product_family,
            "engine": engine_value,
            "platform": platform_value,
            "raw_text": raw_text,
        },
        keyword_guess=keyword_sector or "AG",
    )
    parsed["sector"] = sector
    parsed["sector_source"] = sector_source  # Track how sector was determined
    logger.info(f"[INTAKE] Sector resolved: {sector} (source: {sector_source})")

    # Detect ATS technology
    ats_tech = ""
    for tech, patterns in ATS_TECH.items():
        for pattern in patterns:
            if pattern.upper() in text_upper:
                ats_tech = tech
                break
        if ats_tech:
            break
    parsed["ats_tech"] = ats_tech

    # Detect boolean flags for change types
    parsed["hardware_change"] = any(kw in text_upper for kw in HARDWARE_KEYWORDS)
    parsed["calibration_change"] = any(kw in text_upper for kw in CALIBRATION_KEYWORDS)
    parsed["ats_change"] = any(kw in text_upper for kw in ATS_KEYWORDS)
    parsed["software_vcu_change"] = any(kw in text_upper for kw in SOFTWARE_KEYWORDS)

    # Detect component-specific changes (for sizing classification)
    parsed["turbo_related"] = any(kw in text_upper for kw in TURBO_KEYWORDS)
    parsed["injectors_related"] = any(kw in text_upper for kw in INJECTOR_KEYWORDS)
    parsed["fuel_rail_related"] = any(kw in text_upper for kw in FUEL_RAIL_KEYWORDS)
    parsed["EGR_related"] = any(kw in text_upper for kw in EGR_KEYWORDS)
    parsed["cooling_system_related"] = any(kw in text_upper for kw in COOLING_KEYWORDS)

    # Detect power and torque values
    power_values = POWER_PATTERN.findall(raw_text)
    torque_values = TORQUE_PATTERN.findall(raw_text)
    power_increase_match = POWER_INCREASE_PATTERN.search(raw_text)
    torque_increase_match = TORQUE_INCREASE_PATTERN.search(raw_text)

    parsed["power_kw"] = float(power_values[0]) if power_values else None
    parsed["torque_nm"] = float(torque_values[0]) if torque_values else None
    parsed["power_increase_kw"] = (
        float(power_increase_match.group(1)) if power_increase_match else None
    )
    parsed["torque_increase_nm"] = (
        float(torque_increase_match.group(1)) if torque_increase_match else None
    )

    # Store BOTH raw text (for feature detection) and structured text (for LLM)
    # The structured text is optimized for LLM processing with clear sections
    # Use new comprehensive generate_llm_context for ChatGPT-quality output
    parsed["raw_text"] = raw_text  # Keep for keyword detection
    llm_context = generate_llm_context(parsed, raw_text)
    parsed["structured_text"] = llm_context  # LLM-optimized version
    parsed["llm_context"] = llm_context  # Alias for clarity in LLM prompts

    # =========================================================================
    # Consolidated extracted_features dict (for sizing and estimation)
    # =========================================================================
    parsed["extracted_features"] = {
        # Basic identifiers
        "pr_code": parsed.get("pr_code"),
        "product_family": product_family,
        "emissions": emissions,
        "sector": sector,
        "ats_tech": ats_tech,
        # Boolean change flags
        "hardware_change": parsed.get("hardware_change", False),
        "calibration_change": parsed.get("calibration_change", False),
        "ats_change": parsed.get("ats_change", False),
        "software_vcu_change": parsed.get("software_vcu_change", False),
        # Component-specific flags (for sizing classification)
        "turbo_related": parsed.get("turbo_related", False),
        "injectors_related": parsed.get("injectors_related", False),
        "fuel_rail_related": parsed.get("fuel_rail_related", False),
        "EGR_related": parsed.get("EGR_related", False),
        "cooling_system_related": parsed.get("cooling_system_related", False),
        # Power and torque
        "power_kw": parsed.get("power_kw"),
        "torque_nm": parsed.get("torque_nm"),
        "power_increase_kw": parsed.get("power_increase_kw"),
        "torque_increase_nm": parsed.get("torque_increase_nm"),
        # Additional metadata
        "platform": platform_value,
        "engine": engine_value,
        "tier": tier_value,
    }

    # Extract activities (returns empty for FPT - estimation node generates them)
    activities = extract_fpt_activities(df)
    parsed["raw_activities"] = activities

    return parsed


def extract_fpt_activities(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Extract activities from FPT PR format.

    NOTE: FPT PR files are REQUEST documents - they describe requirements,
    not activities with hours. The actual engineering activities are
    PREDICTED by the estimation node based on the PR content.

    Returns empty list so estimation node generates proper activities.
    """
    # FPT PR files don't contain structured activity data with hours
    # Return empty list to trigger default activity generation in estimation node
    return []


def parse_generic_format(df: pd.DataFrame, filename: str) -> ParsedPR:
    """Parse generic Excel format (non-FPT)."""
    parsed: ParsedPR = {
        "pr_code": "",
        "title": "",
        "description": "",
        "program_family": "",
        "customer": "",
        "project_phase": "",
        "raw_activities": [],
        "raw_data": {},
        "validation_errors": [],
    }

    # Try to extract PR code from various possible locations
    pr_code = extract_field(df, ["PR", "PR Code", "PR Number", "PR_Code", "pr_id"])
    if pr_code and is_valid_pr_code(str(pr_code)):
        parsed["pr_code"] = str(pr_code)
    else:
        # Try to extract from filename
        if "PR" in filename.upper():
            parsed["pr_code"] = filename.split(".")[0]

    # Extract title - avoid column headers
    title = extract_field(df, ["Title", "Project Title", "Description"])
    if title and is_valid_field_value(str(title)):
        parsed["title"] = str(title)

    # Extract program family
    family = extract_field(
        df,
        ["Program Family", "Family", "Program", "Famiglia Programma", "product_family"],
    )
    if family and is_valid_field_value(str(family)):
        parsed["program_family"] = str(family)

    # Extract customer
    customer = extract_field(df, ["Customer", "Cliente", "Client"])
    if customer and is_valid_field_value(str(customer)):
        parsed["customer"] = str(customer)

    # Extract project phase
    phase = extract_field(df, ["Phase", "Project Phase", "Fase"])
    if phase and is_valid_field_value(str(phase)):
        parsed["project_phase"] = str(phase)

    # Extract activities (look for activity-like columns)
    activities = extract_activities(df)
    parsed["raw_activities"] = activities

    return parsed


def is_valid_pr_code(value: str) -> bool:
    """Check if value looks like a valid PR code."""
    if not value:
        return False
    value_lower = value.lower().strip()
    # Reject common column headers
    invalid_values = {"name", "title", "pr", "code", "number", "id", "nan", "none", ""}
    if value_lower in invalid_values:
        return False
    # Must contain at least one digit to be a real PR code
    return any(c.isdigit() for c in value)


def is_valid_field_value(value: str) -> bool:
    """Check if value is a real field value, not a column header."""
    if not value:
        return False
    value_lower = value.lower().strip()
    # Reject common column headers
    invalid_values = {
        "name",
        "title",
        "pr",
        "code",
        "number",
        "id",
        "nan",
        "none",
        "",
        "description",
        "customer",
        "client",
        "phase",
        "family",
        "program",
    }
    return value_lower not in invalid_values


def extract_field(df: pd.DataFrame, possible_names: list[str]) -> Any:
    """Extract a field value trying multiple possible column names."""
    # Check column names
    for name in possible_names:
        for col in df.columns:
            if name.lower() in str(col).lower():
                # Return first non-null value in this column
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    return non_null.iloc[0]

    # Check first row for key-value pairs
    if len(df) > 0:
        first_row = df.iloc[0]
        for name in possible_names:
            for i, val in enumerate(first_row):
                if name.lower() in str(val).lower():
                    # Return next cell value
                    if i + 1 < len(first_row):
                        return first_row.iloc[i + 1]

    return None


def extract_activities(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract activity rows from the DataFrame."""
    activities = []

    # Look for activity code column
    activity_col = None
    hours_col = None

    for col in df.columns:
        col_str = str(col).lower()
        if any(
            x in col_str for x in ["activity", "attività", "task", "code", "codice"]
        ):
            activity_col = col
        if any(x in col_str for x in ["hour", "ore", "effort", "h"]):
            hours_col = col

    if activity_col:
        for _, row in df.iterrows():
            activity_val = row.get(activity_col)
            if pd.notna(activity_val) and str(activity_val).strip():
                activity = {
                    "code": str(activity_val),
                    "name": str(activity_val),
                    "hours": 0,
                }

                if hours_col and pd.notna(row.get(hours_col)):
                    try:
                        activity["hours"] = float(row[hours_col])
                    except (ValueError, TypeError):
                        pass

                activities.append(activity)

    return activities


async def validate_pr_data(parsed_pr: ParsedPR) -> dict[str, Any]:
    """Validate the parsed PR data."""
    errors = []
    warnings = []

    # PR code is optional - system will generate fallback if missing
    if not parsed_pr.get("pr_code"):
        warnings.append("PR code not found - will be auto-generated")

    # Title is recommended but not required
    if not parsed_pr.get("title"):
        warnings.append("Title not found - will use filename")

    # Warnings for optional fields
    if not parsed_pr.get("program_family"):
        warnings.append("Program family not found - will need to be specified")

    if not parsed_pr.get("raw_activities"):
        warnings.append("No activities could be extracted from the file")

    # Check for minimum data
    raw_data = parsed_pr.get("raw_data", {})
    if raw_data.get("shape", (0, 0))[0] < 2:
        errors.append("File appears to be empty or has insufficient data")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "extracted_fields": {
            "pr_code": bool(parsed_pr.get("pr_code")),
            "title": bool(parsed_pr.get("title")),
            "program_family": bool(parsed_pr.get("program_family")),
            "customer": bool(parsed_pr.get("customer")),
            "activities_count": len(parsed_pr.get("raw_activities", [])),
        },
    }


# ============================================================================
# EARLY FEATURE EXTRACTION (Rule-based only, for Q&A question generation)
# ============================================================================

# High-impact features that Q&A should prioritize asking about
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


async def _run_early_feature_extraction(parsed_pr: dict) -> dict:
    """
    Run early feature extraction to identify missing features for Q&A.

    This is a lightweight rule-based extraction (no LLM calls) that runs
    during intake to populate feature_extraction_result. The Q&A node uses
    this to generate targeted questions for missing/uncertain features.

    Args:
        parsed_pr: Parsed PR data from parse_pr_file()

    Returns:
        dict with keys:
            - confidence: float (0-1) overall extraction confidence
            - missing_features: list[str] features that couldn't be extracted
            - extraction_method: str always "rule_based_early"
            - extracted: dict[str, any] extracted feature values
    """
    missing_features = []
    extracted = {}
    confidence_score = 1.0

    # Check sector
    sector = parsed_pr.get("sector")
    if sector and sector != "UNKNOWN":
        extracted["sector"] = sector
    else:
        missing_features.append("sector")
        confidence_score -= 0.1

    # Check product family
    product_family = parsed_pr.get("product_family")
    if product_family:
        extracted["product_family"] = product_family
    else:
        missing_features.append("product_family")
        confidence_score -= 0.1

    # Check emissions
    emissions = parsed_pr.get("emissions")
    if emissions:
        extracted["emissions"] = emissions
    else:
        missing_features.append("emissions")
        confidence_score -= 0.05

    # Check change flags from parsed_pr
    flags = parsed_pr.get("flags", {})

    # Hardware change
    if (
        flags.get("hardware_change")
        or flags.get("turbo_related")
        or flags.get("injectors_related")
    ):
        extracted["hardware_change"] = True
    elif "turbo" in str(parsed_pr).lower() or "injector" in str(parsed_pr).lower():
        extracted["hardware_change"] = True
    else:
        # Can't determine - mark as missing
        missing_features.append("hardware_change")
        confidence_score -= 0.1

    # Calibration change
    if flags.get("calibration_change") or flags.get("calibration_related"):
        extracted["calibration_change"] = True
    elif "calibrat" in str(parsed_pr).lower():
        extracted["calibration_change"] = True
    else:
        missing_features.append("calibration_change")
        confidence_score -= 0.1

    # ATS change
    if flags.get("ATS_change") or flags.get("aftertreatment"):
        extracted["ATS_change"] = True
    elif "ats" in str(parsed_pr).lower() or "aftertreatment" in str(parsed_pr).lower():
        extracted["ATS_change"] = True
    else:
        missing_features.append("ATS_change")
        confidence_score -= 0.05

    # Power increase
    power = parsed_pr.get("power_increase_kw")
    if power and power > 0:
        extracted["power_increase_kw"] = power
    else:
        missing_features.append("power_increase_kw")
        confidence_score -= 0.05

    # Sizing (usually not known at intake)
    sizing = parsed_pr.get("sizing_program")
    if sizing:
        extracted["sizing_program"] = sizing
    else:
        missing_features.append("sizing_program")
        confidence_score -= 0.05

    # Clamp confidence to [0.3, 1.0]
    confidence_score = max(0.3, min(1.0, confidence_score))

    logger.info(
        f"[EARLY_EXTRACTION] confidence={confidence_score:.0%}, "
        f"missing={missing_features}, extracted={list(extracted.keys())}"
    )

    return {
        "confidence": confidence_score,
        "missing_features": missing_features,
        "extraction_method": "rule_based_early",
        "extracted": extracted,
    }

"""
HCQE Feature Extractor v4

Extracts INPUT features from PR content using LLM "Virtual Manager" approach.
Uses ref_Sizing lookup table for sizing classification.

Architecture:
    PR Raw Text → LLM Agent (CoT with ref_Sizing) → Structured Features → HCQE Model

INPUT Features (18+ total for v4):
    Binary (5): ATS_change, application_tractor, calibration_change, hardware_change, software_VCU_change
    Numeric (3): power_increase_kw, torque_increase_nm, num_functions
    Sizing (4): sizing_PE_* features (now NUMERIC 0-4, not text)
    Categorical (5): Product_Family, ATS_tech, Emissions, R&D_type, PR_Type
    Derived (2): sector, is_ce

Key v4 changes:
- ATS_change is CORRECTED (was inverted: 12% should have ATS, not 97%)
- Sizing is NUMERIC 0-4 (X-Small=0, Small=1, Mid=2, Large=3, Full=4)
- NEW features: Product_Family, ATS_tech, Emissions, R&D_type, PR_Type
- sector derived from Global_Product_Platform (AG/CE has ×3 cost impact)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

# Import sizing prompts
try:
    from llm.prompts_sizing import get_sizing_prompt, REF_SIZING_CONTEXT
except ImportError:
    get_sizing_prompt = None
    REF_SIZING_CONTEXT = ""

logger = logging.getLogger(__name__)

# ============================================================================
# FEATURE SCHEMA (v4 - 18+ INPUT FEATURES)
# ============================================================================

# v4 NUMERIC SIZING ENCODING (replaces text)
# X-Small=0, Small=1, Mid=2, Large=3, Full=4
SIZING_NUMERIC_ENCODING = {
    "X-small": 0,
    "X-Small": 0,
    "Small": 1,
    "Mid": 2,
    "Medium": 2,
    "Large": 3,
    "X-Large": 4,
    "Full": 4,
}

# Platform to sector mapping (for derivation)
PLATFORM_TO_SECTOR = {
    "Small/Medium Tractors": "AG",
    "Crop Production, Hay & Forage": "AG",
    "Crop Harvesting": "AG",
    "Large Tractors": "AG",
    "CE Light": "CE",
    "CE Heavy": "CE",
    "Compact Wheel Loader": "CE",
    "Excavator": "CE",
}

# These features have ≥50% coverage in training data
HCQE_INPUT_FEATURES = {
    # === BINARY FEATURES (100% coverage) ===
    "ATS_change": {
        "type": "binary",
        "description": "After-Treatment System modifications required (DPF, SCR, DOC)?",
        "default": 0,  # v4 CORRECTED: default is 0 (most PRs don't have ATS changes)
        "keywords": [
            "ats",
            "aftertreatment",
            "after-treatment",
            "dpf",
            "scr",
            "doc",
        ],
    },
    "application_tractor": {
        "type": "binary",
        "description": "Is application a tractor?",
        "default": 0,
        "keywords": ["tractor", "agricultural"],
    },
    "calibration_change": {
        "type": "binary",
        "description": "Calibration work required (tuning, mapping, parameters)?",
        "default": 0,
        "keywords": ["calibration", "calibrat", "tuning", "mapping", "parameter"],
    },
    "hardware_change": {
        "type": "binary",
        "description": "Hardware changes required (turbo, injectors, etc.)?",
        "default": 0,
        "keywords": [
            "turbo",
            "injector",
            "hardware",
            "component",
        ],
    },
    "software_VCU_change": {
        "type": "binary",
        "description": "VCU/ECU software changes required?",
        "default": 0,
        "keywords": ["software", "sw", "vcu", "ecu", "code", "firmware"],
    },
    # === NUMERIC FEATURES (50-73% coverage) ===
    "power_increase_kw": {
        "type": "numeric",
        "description": "Power increase target in kW (0 if no increase)",
        "default": 0,
        "range": [0, 1000],
        "keywords": ["kw", "kilowatt", "power", "hp", "horsepower"],
    },
    "torque_increase_nm": {
        "type": "numeric",
        "description": "Torque increase target in Nm (0 if no increase)",
        "default": 0,
        "range": [0, 5000],
        "keywords": ["nm", "torque", "newton"],
    },
    "num_functions": {
        "type": "numeric",
        "description": "Number of R&D functions involved in the project",
        "default": 20,
        "range": [5, 80],
        "correlation_with_cost": 0.44,
    },
    # === SIZING FEATURES (v4: NUMERIC 0-4, not text) ===
    "sizing_PE_base_powertrain": {
        "type": "numeric",  # Changed from ordinal to numeric
        "description": "Base powertrain engineering scope (0=X-Small to 4=Full)",
        "default": 2,  # Mid
        "range": [0, 4],
    },
    "sizing_PE_system_assembly": {
        "type": "numeric",
        "description": "System assembly engineering scope (0-4)",
        "default": 2,
        "range": [0, 4],
    },
    "sizing_PE_installation_application_homologation": {
        "type": "numeric",
        "description": "Installation/application/homologation scope (0-4)",
        "default": 2,
        "range": [0, 4],
    },
    "sizing_program": {
        "type": "numeric",
        "description": "Overall program sizing (0-4)",
        "default": 2,
        "range": [0, 4],
    },
    # === NEW v4 CATEGORICAL FEATURES ===
    "product_family": {
        "type": "categorical",
        "description": "Engine product family code",
        "default": "E0N0",
        "allowed_values": ["E0N0", "E5F0", "E0C0", "E8S0", "E0V0"],
    },
    "ats_tech": {
        "type": "categorical",
        "description": "After-treatment system technology",
        "default": "DOC_SCRoF",
        "allowed_values": [
            "DOC_SCRoF",
            "DOC_DPF",
            "DOC_only",
            "DOC_SCR-T",
            "SCR_only",
        ],
    },
    "emissions": {
        "type": "categorical",
        "description": "Target emission standard",
        "default": "Stage V",
        "allowed_values": [
            "Stage V",
            "Tier 4B",
            "Tier 4F",
            "Tier 3",
            "Tier 2",
            "China NRIV",
            "Trem V",
        ],
    },
    "rd_type": {
        "type": "ordinal",
        "description": "R&D complexity type (1=Full, 2=Partial, 3=Minimal)",
        "default": 1,
        "allowed_values": [1, 2, 3],
    },
    "pr_type": {
        "type": "categorical",
        "description": "Product Request type",
        "default": "New engine",
        "allowed_values": [
            "New engine",
            "BOM (Update engine)",
            "BOM",
            "Homologation",
        ],
    },
    # === DERIVED FEATURES ===
    "sector": {
        "type": "categorical",
        "description": "Business sector: AG (Agricultural) or CE (Construction Equipment)",
        "default": "AG",
        "allowed_values": ["AG", "CE"],
        "keywords": {
            "AG": [
                "tractor",
                "harvester",
                "combine",
                "sprayer",
                "agricultural",
                "farm",
            ],
            "CE": [
                "excavator",
                "loader",
                "grader",
                "telehandler",
                "construction",
                "crawler",
            ],
        },
        "cost_multiplier": {"AG": 1.0, "CE": 0.3},
    },
}

# ============================================================================
# MISSING FEATURE → QUESTION MAPPING
# Maps each feature to a targeted clarifying question when extraction fails
# ============================================================================

MISSING_FEATURE_QUESTIONS: dict[str, dict] = {
    "power_increase_kw": {
        "question": "What is the target power output or power increase for this project (in kW)?",
        "reason": "Power specifications significantly impact engineering effort and testing scope",
        "category": "technical",
        "priority": "high",
        "suggested_answers": [
            "No power increase (same as baseline)",
            "Small increase (<20 kW)",
            "Medium increase (20-50 kW)",
            "Large increase (>50 kW)",
            "New power rating: [specify kW]",
        ],
    },
    "torque_increase_nm": {
        "question": "What is the target torque output or torque increase (in Nm)?",
        "reason": "Torque changes affect powertrain design and validation requirements",
        "category": "technical",
        "priority": "medium",
        "suggested_answers": [
            "No torque increase",
            "Minor increase (<100 Nm)",
            "Moderate increase (100-300 Nm)",
            "Significant increase (>300 Nm)",
        ],
    },
    "hardware_change": {
        "question": "Does this project require any hardware modifications to the engine or powertrain?",
        "reason": "Hardware changes significantly increase design, prototype, and testing costs",
        "category": "hardware",
        "priority": "high",
        "suggested_answers": [
            "No hardware changes - calibration/software only",
            "Minor component changes (sensors, actuators)",
            "Moderate changes (injectors, turbo optimization)",
            "Major changes (new turbo, significant component redesign)",
        ],
    },
    "calibration_change": {
        "question": "What level of engine calibration work is required?",
        "reason": "Calibration scope directly affects development and testing effort",
        "category": "calibration",
        "priority": "high",
        "suggested_answers": [
            "No calibration changes",
            "Parameter adjustments only",
            "Emissions recalibration",
            "Full engine calibration",
            "Multi-application calibration",
        ],
    },
    "ATS_change": {
        "question": "Are any aftertreatment system (ATS) modifications required?",
        "reason": "ATS changes have major cost implications due to emissions compliance testing",
        "category": "ats",
        "priority": "high",
        "suggested_answers": [
            "No ATS changes",
            "ATS calibration only",
            "Component optimization (DOC/DPF/SCR)",
            "Integration of existing ATS",
            "New or redesigned ATS system",
        ],
    },
    "software_VCU_change": {
        "question": "What software or ECU/VCU changes are needed?",
        "reason": "Software complexity affects development time and validation requirements",
        "category": "software",
        "priority": "high",
        "suggested_answers": [
            "No software changes",
            "Parameter updates only",
            "New diagnostic features",
            "Control strategy modifications",
            "Multi-ECU coordination changes",
        ],
    },
    "emissions": {
        "question": "What is the target emissions certification standard?",
        "reason": "Emissions standard determines testing and certification requirements",
        "category": "emissions",
        "priority": "high",
        "suggested_answers": [
            "Stage V (EU non-road)",
            "Tier 4B/Final (US)",
            "China NRIV",
            "Euro VI (on-road)",
            "Multiple standards",
        ],
    },
    "sector": {
        "question": "Which market sector is this project targeting?",
        "reason": "Sector significantly affects application requirements and testing scope",
        "category": "application",
        "priority": "high",
        "suggested_answers": [
            "AG - Agriculture (tractors, harvesters, sprayers)",
            "CE - Construction Equipment (excavators, loaders)",
            "PT - Powertrain/Trucks",
            "Marine applications",
            "Power generation",
        ],
    },
    "product_family": {
        "question": "Which engine product family does this project target?",
        "reason": "Product family determines baseline complexity and historical reference data",
        "category": "technical",
        "priority": "high",
        "suggested_answers": [
            "NEF (N45/N67/E0N6)",
            "CURSOR (C87/C9/C11/C13)",
            "E0C0/E9C0 (Light duty)",
            "F1 series (F1C/F1A)",
            "E5F0 series (Industrial)",
        ],
    },
    "sizing_program": {
        "question": "What is the overall scope/sizing of this program?",
        "reason": "Program sizing is a key predictor of total R&D effort",
        "category": "complexity",
        "priority": "high",
        "suggested_answers": [
            "X-Small (minimal changes, <500 hours)",
            "Small (minor project, 500-1500 hours)",
            "Mid (standard project, 1500-4000 hours)",
            "Large (significant development, 4000-8000 hours)",
            "Full (major new development, >8000 hours)",
        ],
    },
    "num_functions": {
        "question": "Approximately how many R&D functions/departments will be involved?",
        "reason": "Number of involved functions correlates strongly with project cost",
        "category": "complexity",
        "priority": "medium",
        "suggested_answers": [
            "Few (5-10 functions)",
            "Moderate (10-20 functions)",
            "Many (20-40 functions)",
            "Extensive (40+ functions)",
        ],
    },
    "ats_tech": {
        "question": "What aftertreatment technology configuration will be used?",
        "reason": "ATS technology affects emissions compliance approach and testing",
        "category": "ats",
        "priority": "medium",
        "suggested_answers": [
            "DOC + SCRoF (SCR on Filter)",
            "DOC + DPF (Diesel Particulate Filter)",
            "DOC only (Diesel Oxidation Catalyst)",
            "DOC + SCR-T (Twin SCR)",
            "SCR only",
        ],
    },
    "rd_type": {
        "question": "What type of R&D effort does this project represent?",
        "reason": "R&D type affects resource allocation and validation depth",
        "category": "complexity",
        "priority": "medium",
        "suggested_answers": [
            "Full R&D (new development)",
            "Partial R&D (derivative work)",
            "Minimal R&D (minor updates)",
        ],
    },
    "pr_type": {
        "question": "What type of Product Request is this?",
        "reason": "PR type determines the scope of deliverables and testing",
        "category": "technical",
        "priority": "medium",
        "suggested_answers": [
            "New engine development",
            "BOM update (existing engine)",
            "Homologation only",
            "Re-certification",
        ],
    },
}

# Output targets (what we predict, NOT used as inputs)
HCQE_OUTPUT_TARGETS = {
    "total_cost_eur": {
        "type": "numeric",
        "description": "Total R&D cost in EUR (PRIMARY TARGET)",
        "primary": True,
    },
    "design_hours": {
        "type": "numeric",
        "description": "Design engineering hours",
    },
    "bench_development_hours": {
        "type": "numeric",
        "description": "Bench development test hours",
    },
    "bench_durability_hours": {
        "type": "numeric",
        "description": "Bench durability test hours",
    },
    "calibration_hours": {
        "type": "numeric",
        "description": "Calibration engineering hours",
    },
    "vehicle_hours": {
        "type": "numeric",
        "description": "Vehicle test hours",
    },
}

# ============================================================================
# LLM PROMPT FOR FEATURE EXTRACTION
# ============================================================================

FEATURE_EXTRACTION_PROMPT = """You are an FPT R&D cost estimation expert. Extract technical features from this Product Request.

## PRODUCT REQUEST
**Title**: {title}
**Description**: {description}
**Program Family**: {program_family}
**Scope**: {scope}

## ACTIVITIES MENTIONED
{activities}

## USER CLARIFICATIONS (Q&A)
{qa_answers}

## SIMILAR HISTORICAL PROJECTS
{similar_prs}

## FEATURE EXTRACTION TASK

Extract exactly these features. Use the context to determine values:

### BINARY FEATURES (0 or 1)
1. hardware_change: Any engine hardware changes (turbo, injectors, fuel system, components)? 1=yes, 0=no
2. calibration_change: Calibration/tuning work required? 1=yes, 0=no
3. ATS_change: After-treatment system changes (DPF, SCR, DOC)? 1=yes, 0=no
4. software_VCU_change: VCU/ECU software changes? 1=yes, 0=no
5. application_tractor: Is this for a tractor application? 1=yes (tractor/agricultural), 0=no (CE/other)

### NUMERIC FEATURES
6. power_increase_kw: Target power increase in kW (0 if no change, extract number if mentioned)
7. torque_increase_nm: Target torque increase in Nm (0 if no change, extract number if mentioned)
8. num_functions: Estimated number of R&D functions involved (5-80, based on scope complexity)

### CATEGORICAL FEATURES
9. emissions: Target emission standard. Values: "Stage V", "Tier 4B", "Tier 4F", "Tier 3", "Tier 2", "China NRIV", "Trem V"
10. product_family: Engine product family code. Values: "E0N0", "E5F0", "E0C0", "E8S0", "E0V0"
11. sector: Business sector. Values: "AG" (agricultural/tractors), "CE" (construction equipment)

### SIZING FEATURES (estimate based on scope complexity)
Use: "X-small", "Small", "Mid", "Large", or "Full"
12. sizing_PE_base_powertrain: Base powertrain engineering scope
13. sizing_PE_system_assembly: System assembly (engine + ATS) scope
14. sizing_PE_installation_application_homologation: Installation/application/homologation scope
15. sizing_program: Overall program size

### SIZING GUIDELINES:
- X-small: Minor changes, <50h total, single component
- Small: Simple changes, 50-200h, 2-3 components
- Mid: Moderate scope, 200-500h, several systems affected
- Large: Complex project, 500-1500h, major redesign
- Full: Complete new development, >1500h, all systems

Respond ONLY with a valid JSON object:
```json
{{
  "hardware_change": 0,
  "calibration_change": 1,
  "ATS_change": 0,
  "software_VCU_change": 0,
  "application_tractor": 1,
  "power_increase_kw": 85,
  "torque_increase_nm": 450,
  "num_functions": 25,
  "emissions": "Stage V",
  "product_family": "E0N0",
  "sector": "AG",
  "sizing_PE_base_powertrain": "Mid",
  "sizing_PE_system_assembly": "Mid",
  "sizing_PE_installation_application_homologation": "Mid",
  "sizing_program": "Mid"
}}
```"""


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class HCQEFeatures:
    """Extracted features for HCQE model (12 INPUT features)."""

    features: dict[str, Any]
    confidence: float
    extraction_method: str  # "llm", "rule_based", "qa_enhanced"
    raw_response: str | None = None
    missing_features: list[str] = field(default_factory=list)

    def to_model_input(self) -> dict[str, float]:
        """Convert to numeric format expected by HCQE model."""
        result = {}
        for name, value in self.features.items():
            schema = HCQE_INPUT_FEATURES.get(name)
            if not schema:
                continue

            if schema["type"] == "binary":
                result[name] = 1.0 if value else 0.0
            elif schema["type"] == "numeric":
                result[name] = float(value) if value is not None else 0.0
            elif schema["type"] == "ordinal":
                # Convert sizing string to numeric
                encoding = schema.get("encoding", {})
                if isinstance(value, str):
                    result[f"{name}_encoded"] = float(
                        encoding.get(value, 3)
                    )  # Default to Mid
                else:
                    result[f"{name}_encoded"] = float(value) if value else 3.0
            # Skip categorical for now (handled separately)

        return result


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================


def get_default_features() -> dict[str, Any]:
    """Return default feature values."""
    return {name: schema["default"] for name, schema in HCQE_INPUT_FEATURES.items()}


async def extract_hcqe_features(
    parsed_pr: dict,
    qa_answers: dict | None = None,
    similar_prs: list[dict] | None = None,
    llm=None,
) -> HCQEFeatures:
    """
    Extract HCQE features from PR content using LLM.

    Args:
        parsed_pr: Parsed Product Request data
        qa_answers: User's Q&A answers
        similar_prs: Similar historical PRs for context
        llm: LLM client instance

    Returns:
        HCQEFeatures with extracted feature dict
    """
    # Start with defaults
    features = get_default_features()
    missing_features = []

    # Build context strings
    title = parsed_pr.get("title", "Unknown Project")
    description = parsed_pr.get("description", "") or ""
    program_family = parsed_pr.get("program_family", "Unknown")
    scope = parsed_pr.get("scope", "") or parsed_pr.get("technical_scope", "") or ""

    # Format activities
    activities_list = parsed_pr.get("raw_activities", [])
    activities_str = (
        "\n".join(
            [
                f"- {a.get('name', 'Unknown')}: {a.get('description', '')}"
                for a in activities_list[:20]
            ]
        )
        or "No activities specified"
    )

    # Format Q&A answers
    qa_str = "No Q&A answers provided"
    if qa_answers:
        qa_str = "\n".join([f"- {k}: {v}" for k, v in qa_answers.items()])

    # Format similar PRs
    similar_str = "No similar projects found"
    if similar_prs:
        similar_str = "\n".join(
            [
                f"- {sp.get('pr_code', 'Unknown')}: {sp.get('total_cost_keur', 0):.0f} K€, sizing={sp.get('sizing_program', 'Unknown')}"
                for sp in similar_prs[:5]
            ]
        )

    # Try LLM extraction
    if llm is not None:
        try:
            prompt = FEATURE_EXTRACTION_PROMPT.format(
                title=title,
                description=description[:3000],
                program_family=program_family,
                scope=scope[:2000],
                activities=activities_str,
                qa_answers=qa_str,
                similar_prs=similar_str,
            )

            response = await llm.reason(prompt=prompt)
            raw_response = response if isinstance(response, str) else str(response)

            # Parse JSON from response
            extracted = _parse_json_from_response(raw_response)

            if extracted:
                # Merge extracted features with defaults
                for key, value in extracted.items():
                    if key in features:
                        normalized = _normalize_feature_value(key, value)
                        if normalized is not None:
                            features[key] = normalized
                        else:
                            missing_features.append(key)

                # Enhance with Q&A answers
                if qa_answers:
                    features = _enhance_from_qa(features, qa_answers)

                logger.info(f"LLM extracted {len(extracted)} features for HCQE v3")

                return HCQEFeatures(
                    features=features,
                    confidence=0.85,
                    extraction_method="llm",
                    raw_response=raw_response,
                    missing_features=missing_features,
                )
        except Exception as e:
            logger.warning(f"LLM feature extraction failed: {e}")

    # Fallback: Rule-based extraction from parsed PR
    features, missing_features = _extract_features_rule_based(
        parsed_pr, qa_answers, similar_prs, features
    )

    return HCQEFeatures(
        features=features,
        confidence=0.6,
        extraction_method="rule_based",
        missing_features=missing_features,
    )


def _parse_json_from_response(response: str) -> dict | None:
    """Extract JSON object from LLM response."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _normalize_feature_value(feature_name: str, value: Any) -> Any:
    """Normalize feature value to expected type."""
    schema = HCQE_INPUT_FEATURES.get(feature_name)
    if not schema:
        return None

    feature_type = schema["type"]

    if feature_type == "binary":
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, str):
            return 1 if value.lower() in ("yes", "true", "1", "y") else 0
        return 1 if value else 0

    elif feature_type == "numeric":
        try:
            val = float(value)
            # Check range if defined
            if "range" in schema:
                min_val, max_val = schema["range"]
                val = max(min_val, min(max_val, val))
            return val
        except (ValueError, TypeError):
            return schema["default"]

    elif feature_type == "ordinal":
        # Accept both string and numeric
        if isinstance(value, str):
            encoding = schema.get("encoding", {})
            if value in encoding:
                return value  # Keep as string, convert later
        try:
            # If numeric, keep it
            return float(value)
        except (ValueError, TypeError):
            return schema["default"]

    elif feature_type == "categorical":
        if value is None or value == "":
            return schema.get("default", "")
        return str(value)

    return value


def _enhance_from_qa(features: dict[str, Any], qa_answers: dict) -> dict[str, Any]:
    """
    Enhance features from Q&A answers.

    This function maps user answers to specific ML features, enabling the Q&A
    system to act as a "feature completion" mechanism that reduces uncertainty
    in the ML pipeline input.
    """
    for question, answer in qa_answers.items():
        q_lower = question.lower()
        a_lower = str(answer).lower()

        # === POWER/TORQUE EXTRACTION ===
        if "power" in q_lower or "kw" in q_lower or "horsepower" in q_lower:
            match = re.search(r"(\d+)", str(answer))
            if match:
                try:
                    value = float(match.group(1))
                    if "hp" in a_lower or "horsepower" in q_lower:
                        value = value * 0.746  # Convert HP to kW
                    features["power_increase_kw"] = value
                except ValueError:
                    pass

        if "torque" in q_lower or "nm" in q_lower:
            match = re.search(r"(\d+)", str(answer))
            if match:
                try:
                    features["torque_increase_nm"] = float(match.group(1))
                except ValueError:
                    pass

        # === HARDWARE SCOPE ANSWERS ===
        if "hardware" in q_lower or "component" in q_lower:
            hardware_keywords = {
                "injector": ["injector", "nozzle", "injection"],
                "turbo": ["turbo", "turbocharger", "compressor", "boost"],
                "sensor": ["sensor", "actuator", "transducer"],
                "engine block": ["block", "piston", "cylinder", "crankshaft"],
                "multiple": ["multiple", "several", "many", "various"],
            }
            # Check ALL hardware types (don't break early)
            for hw_type, keywords in hardware_keywords.items():
                if any(kw in a_lower for kw in keywords):
                    features["hardware_change"] = 1
                    if hw_type == "turbo":
                        features["turbo_related"] = 1
                    elif hw_type == "injector":
                        features["injectors_related"] = 1

            # Estimate complexity from scope description
            if any(
                word in a_lower
                for word in ["major", "significant", "extensive", "complete", "full"]
            ):
                features["hardware_complexity"] = 3  # High
            elif any(word in a_lower for word in ["minor", "small", "simple", "basic"]):
                features["hardware_complexity"] = 1  # Low
            else:
                features["hardware_complexity"] = 2  # Medium

        # === ATS MODIFICATION DETAILS ===
        if "aftertreatment" in q_lower or "ats" in q_lower:
            features["ATS_change"] = 1

            ats_tech_map = {
                "DOC_SCRoF": ["scrof", "scr-on-filter", "scr on filter"],
                "DOC_DPF": ["dpf", "particulate filter"],
                "DOC_only": ["doc only", "oxidation catalyst only"],
                "DOC_SCR-T": ["scr-t", "scr twin", "dual scr"],
                "SCR_only": ["scr only", "scr system"],
            }
            for tech, keywords in ats_tech_map.items():
                if any(kw in a_lower for kw in keywords):
                    features["ats_tech"] = tech
                    break

            # ATS redesign scope
            if any(
                word in a_lower for word in ["complete", "redesign", "new", "full"]
            ):
                features["ats_complexity"] = 3  # Major redesign
            elif any(word in a_lower for word in ["integration", "adapt", "optimize"]):
                features["ats_complexity"] = 2  # Moderate
            else:
                features["ats_complexity"] = 1  # Minor

        # === CALIBRATION SCOPE ===
        if "calibration" in q_lower or "tuning" in q_lower:
            features["calibration_change"] = 1

            if any(
                word in a_lower
                for word in ["full", "complete", "multi-application", "performance"]
            ):
                features["calibration_complexity"] = 3  # Full calibration
            elif any(
                word in a_lower for word in ["emissions", "emission", "compliance"]
            ):
                features["calibration_complexity"] = 2  # Emissions focus
            elif any(word in a_lower for word in ["minor", "parameter", "adjustment"]):
                features["calibration_complexity"] = 1  # Minor adjustments
            else:
                features["calibration_complexity"] = 2  # Default medium

        # === SOFTWARE/ECU SCOPE ===
        if "software" in q_lower or "ecu" in q_lower or "vcu" in q_lower:
            features["software_VCU_change"] = 1

            if any(
                word in a_lower
                for word in ["new control", "strategy", "multi-ecu", "coordination"]
            ):
                features["software_complexity"] = 3  # Major SW development
            elif any(word in a_lower for word in ["diagnostic", "obd", "fault"]):
                features["software_complexity"] = 2  # Diagnostics focus
            elif any(word in a_lower for word in ["parameter", "update", "flash"]):
                features["software_complexity"] = 1  # Parameter updates only
            else:
                features["software_complexity"] = 2  # Default medium

        # === TESTING/VALIDATION SCOPE ===
        if "test" in q_lower or "validation" in q_lower:
            test_scope_map = {
                3: ["full", "certification", "customer-specific", "field test"],
                2: ["vehicle", "bench + vehicle", "validation program"],
                1: ["bench only", "limited", "basic"],
            }
            for scope_level, keywords in test_scope_map.items():
                if any(kw in a_lower for kw in keywords):
                    features["testing_scope"] = scope_level
                    break

            if "certification" in a_lower:
                features["certification_required"] = 1

        # === EMISSIONS STANDARD ===
        if "emission" in q_lower or "stage" in q_lower or "tier" in q_lower:
            emission_patterns = [
                ("Stage V", ["stage v", "stage 5", "sv"]),
                ("Tier 4B", ["tier 4b", "tier 4 b", "t4b"]),
                ("Tier 4F", ["tier 4f", "tier 4 final", "t4f"]),
                ("Tier 3", ["tier 3", "t3"]),
                ("Tier 2", ["tier 2", "t2"]),
                ("China NRIV", ["china", "nriv", "nr4"]),
                ("Trem V", ["trem v", "trem 5"]),
            ]
            for level, patterns in emission_patterns:
                if any(p in a_lower for p in patterns):
                    features["emissions"] = level
                    break

        # === SIZING/COMPLEXITY ===
        if "sizing" in q_lower or "scope" in q_lower or "complexity" in q_lower:
            sizing_map = {
                0: ["x-small", "xsmall", "minimal", "trivial"],
                1: ["small", "minor", "low"],
                2: ["mid", "medium", "moderate", "standard"],
                3: ["large", "significant", "high"],
                4: ["full", "x-large", "xlarge", "very high", "first-of-kind"],
            }
            for size_val, keywords in sizing_map.items():
                if any(kw in a_lower for kw in keywords):
                    features["sizing_program"] = size_val
                    break

        # === TIMELINE/URGENCY ===
        if "timeline" in q_lower or "deadline" in q_lower or "delivery" in q_lower:
            if any(word in a_lower for word in ["urgent", "asap", "< 3 month"]):
                features["timeline_urgency"] = 3  # Rush
            elif any(word in a_lower for word in ["accelerated", "3-6 month"]):
                features["timeline_urgency"] = 2  # Accelerated
            else:
                features["timeline_urgency"] = 1  # Standard

        # === PRODUCT FAMILY ===
        if "engine" in q_lower or "product family" in q_lower or "family" in q_lower:
            family_patterns = [
                ("E0N0", ["nef", "n45", "n67", "e0n"]),
                ("E5F0", ["cursor", "c87", "c9", "c11", "c13", "e5f"]),
                ("E0C0", ["e0c", "e9c", "light duty"]),
                ("E8S0", ["f1", "f1c", "f1a", "e8s"]),
                ("E0V0", ["e0v", "e5fc", "industrial"]),
            ]
            for family, patterns in family_patterns:
                if any(p in a_lower for p in patterns):
                    features["product_family"] = family
                    break

        # === SECTOR/APPLICATION ===
        if "sector" in q_lower or "application" in q_lower:
            if any(
                word in a_lower
                for word in ["ag", "agriculture", "tractor", "harvester", "farm"]
            ):
                features["sector"] = "AG"
                features["application_tractor"] = 1 if "tractor" in a_lower else 0
            elif any(
                word in a_lower
                for word in [
                    "ce",
                    "construction",
                    "excavator",
                    "loader",
                    "grader",
                ]
            ):
                features["sector"] = "CE"
                features["application_tractor"] = 0

    return features


def _extract_features_rule_based(
    parsed_pr: dict,
    qa_answers: dict | None,
    similar_prs: list[dict] | None,
    features: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Rule-based feature extraction as fallback."""
    missing = []

    # Combine all text for keyword matching
    text = " ".join(
        [
            str(parsed_pr.get("title", "")),
            str(parsed_pr.get("description", "")),
            str(parsed_pr.get("scope", "")),
            str(parsed_pr.get("technical_scope", "")),
        ]
    ).lower()

    # === SECTOR DETECTION (HIGH IMPACT: ×5 cost difference!) ===
    ag_keywords = [
        "tractor",
        "harvester",
        "combine",
        "sprayer",
        "agricultural",
        "farm",
        "crop",
    ]
    ce_keywords = [
        "excavator",
        "loader",
        "grader",
        "telehandler",
        "construction",
        "crawler",
        "skid",
    ]

    ag_score = sum(1 for kw in ag_keywords if kw in text)
    ce_score = sum(1 for kw in ce_keywords if kw in text)

    if ag_score > ce_score:
        features["sector"] = "AG"
    elif ce_score > ag_score:
        features["sector"] = "CE"
    else:
        # Check parsed_pr for sector info
        sector = parsed_pr.get("sector", "")
        if sector in ["AG", "CE"]:
            features["sector"] = sector
        else:
            features["sector"] = "AG"  # Default to AG (more common in dataset)
            missing.append("sector")

    # === NUM_FUNCTIONS (r = +0.44 correlation with cost) ===
    num_func = parsed_pr.get("num_functions")
    if num_func is not None:
        features["num_functions"] = float(num_func)
    else:
        # Estimate from activities count
        activities = parsed_pr.get("raw_activities", [])
        features["num_functions"] = max(10, min(70, len(activities) * 2 + 10))

    # Hardware change detection (turbo, injectors, fuel system, components)
    if any(
        kw in text
        for kw in [
            "turbo",
            "injector",
            "hardware",
            "component",
            "performance",
            "power upgrade",
            "hp increase",
        ]
    ):
        features["hardware_change"] = 1

    # Calibration detection
    if any(kw in text for kw in ["calibrat", "tuning", "mapping", "parameter"]):
        features["calibration_change"] = 1

    # ATS detection
    if any(
        kw in text
        for kw in ["ats", "aftertreatment", "after-treatment", "dpf", "scr", "doc"]
    ):
        features["ATS_change"] = 1

    # Software detection
    if any(kw in text for kw in ["software", "sw", "vcu", "ecu", "code", "firmware"]):
        features["software_VCU_change"] = 1

    # Power/torque extraction from text
    power_match = re.search(r"(\d+)\s*(?:kw|kilowatt)", text, re.IGNORECASE)
    if power_match:
        features["power_increase_kw"] = float(power_match.group(1))
    else:
        missing.append("power_increase_kw")

    torque_match = re.search(r"(\d+)\s*nm", text, re.IGNORECASE)
    if torque_match:
        features["torque_increase_nm"] = float(torque_match.group(1))
    else:
        missing.append("torque_increase_nm")

    # Emissions detection (schema key is "emissions", not "emission_level")
    emission_patterns = [
        ("Stage V", r"stage\s*v"),
        ("Tier 4B", r"tier\s*4\s*b"),
        ("Tier 4F", r"tier\s*4\s*f"),
        ("Tier 3", r"tier\s*3"),
        ("Tier 2", r"tier\s*2"),
        ("China NRIV", r"china\s*nr|nriv"),
        ("Trem V", r"trem\s*v"),
    ]
    emission_found = False
    for level, pattern in emission_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            features["emissions"] = level
            emission_found = True
            break
    if not emission_found:
        missing.append("emissions")

    # Application tractor detection (binary: is this a tractor application?)
    # Schema uses "application_tractor" as binary 0/1, not text "application"
    tractor_keywords = [
        "tractor",
        "agricultural",
        "farm",
        "harvester",
        "combine",
        "sprayer",
    ]
    if any(kw in text for kw in tractor_keywords):
        features["application_tractor"] = 1
    else:
        features["application_tractor"] = 0

    # Get sizing from similar PRs (use their ground truth sizing labels, NOT cost!)
    # This is the correct approach - sizing comes from PR content analysis, not from cost
    if similar_prs:
        # Count sizing labels from similar PRs (use their actual ground truth labels)
        sizing_counts = {}
        for sp in similar_prs:
            # Get sizing from similar PR's ground truth label (numeric 0-4 or string)
            sp_sizing = sp.get("sizing_program")
            if sp_sizing is not None:
                # Convert numeric to string if needed
                if isinstance(sp_sizing, (int, float)):
                    sizing_map_reverse = {0: "X-small", 1: "Small", 2: "Mid", 3: "Large", 4: "Full"}
                    sp_sizing = sizing_map_reverse.get(int(sp_sizing), "Mid")
                sizing_counts[sp_sizing] = sizing_counts.get(sp_sizing, 0) + 1

        if sizing_counts:
            # Use MODE (most common sizing) from similar PRs
            sizing = max(sizing_counts.keys(), key=lambda x: sizing_counts[x])
            logger.debug(f"Sizing from similar PRs (MODE): {sizing} (counts: {sizing_counts})")
        else:
            # Fallback: use parsed_pr sizing if available
            pr_sizing = parsed_pr.get("sizing_program") or parsed_pr.get("extracted_features", {}).get("sizing_program")
            if pr_sizing:
                sizing = pr_sizing if isinstance(pr_sizing, str) else "Mid"
            else:
                sizing = "Mid"  # Default fallback
                missing.append("sizing_program")
                logger.debug("No sizing found in similar PRs or parsed_pr, defaulting to Mid")

        features["sizing_PE_base_powertrain"] = sizing
        features["sizing_PE_system_assembly"] = sizing
        features["sizing_PE_installation_application_homologation"] = sizing
        features["sizing_program"] = sizing
    else:
        # Mark sizing as missing if no similar PRs
        missing.extend(
            [
                "sizing_PE_base_powertrain",
                "sizing_PE_system_assembly",
                "sizing_PE_installation_application_homologation",
                "sizing_program",
            ]
        )

    # Enhance with Q&A if available
    if qa_answers:
        features = _enhance_from_qa(features, qa_answers)

    return features, missing


def features_to_list(features: dict[str, Any]) -> list[dict]:
    """Convert features dict to list format for state storage."""
    result = []
    for name, value in features.items():
        schema = HCQE_INPUT_FEATURES.get(name, {})
        result.append(
            {
                "name": name,
                "value": value,
                "type": schema.get("type", "unknown"),
                "description": schema.get("description", ""),
                "source": "hcqe_extractor_v3",
            }
        )
    return result


def validate_features(features: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate features against schema."""
    errors = []

    for name, schema in HCQE_INPUT_FEATURES.items():
        value = features.get(name)

        if value is None:
            errors.append(f"Missing feature: {name}")
            continue

        if schema["type"] == "binary":
            if value not in (0, 1):
                errors.append(f"{name} must be 0 or 1, got {value}")

        elif schema["type"] == "numeric":
            try:
                float(value)
            except (ValueError, TypeError):
                errors.append(f"{name} must be numeric, got {value}")

        elif schema["type"] == "ordinal":
            encoding = schema.get("encoding", {})
            if isinstance(value, str) and value not in encoding:
                errors.append(
                    f"{name} must be one of {list(encoding.keys())}, got {value}"
                )

    return len(errors) == 0, errors


# Export feature names for other modules
INPUT_FEATURE_NAMES = list(HCQE_INPUT_FEATURES.keys())
OUTPUT_TARGET_NAMES = list(HCQE_OUTPUT_TARGETS.keys())

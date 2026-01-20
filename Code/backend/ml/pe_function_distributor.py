"""
PE Function Hours Distributor
=============================

Distributes total hours across PE functions based on:
1. Affected functions (from change flags)
2. Historical weights (from similar PRs)
3. Base weights (domain knowledge)
4. Hour-type applicability rules (from rule_hours.json)

Key improvement: Only affected functions get hours, with proper re-normalization.
Hour-type constraints: Each PE function has specific hour types enabled/disabled.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# HOUR TYPE APPLICABILITY RULES (from rule_hours.json)
# ============================================================================
# Each PE function has specific hour types that can be assigned.
# Disabled types should be null/0 in the output.

HOUR_COLUMNS = [
    "manpower_hrs",
    "bench_durability_hrs",
    "bench_development_hrs",
    "bench_special_hrs",
    "vehicle_hrs",
]

# Lazy-loaded from rule_hours.json
_HOUR_TYPE_RULES: dict[str, dict[str, bool]] | None = None


def _load_hour_type_rules() -> dict[str, dict[str, bool]]:
    """Load hour type applicability rules from rule_hours.json."""
    global _HOUR_TYPE_RULES
    if _HOUR_TYPE_RULES is not None:
        return _HOUR_TYPE_RULES

    # Try multiple paths (v2/backend context vs root context)
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "Dataset" / "csv_exports" / "rule_hours.json",
        Path(__file__).parent.parent / "data" / "rule_hours.json",
        Path("Dataset/csv_exports/rule_hours.json"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)

                # Build lookup: function_description → enabled dict
                rules = {}
                for item in data.get("function_hour_applicability", []):
                    func_name = item["function_description"]
                    rules[func_name] = item["enabled"]

                # Store default for unknown functions
                rules["_default"] = data.get("recommended_handling", {}).get(
                    "unknown_function_policy", {}
                ).get("enabled", {
                    "manpower_hrs": True,
                    "bench_durability_hrs": False,
                    "bench_development_hrs": False,
                    "bench_special_hrs": False,
                    "vehicle_hrs": False,
                })

                _HOUR_TYPE_RULES = rules
                logger.info(f"Loaded hour type rules for {len(rules) - 1} PE functions")
                return _HOUR_TYPE_RULES
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

    # Fallback defaults if file not found
    logger.warning("rule_hours.json not found, using defaults (manpower only)")
    _HOUR_TYPE_RULES = {
        "_default": {
            "manpower_hrs": True,
            "bench_durability_hrs": False,
            "bench_development_hrs": False,
            "bench_special_hrs": False,
            "vehicle_hrs": False,
        }
    }
    return _HOUR_TYPE_RULES


def get_enabled_hour_types(function_name: str) -> dict[str, bool]:
    """Get enabled hour types for a PE function."""
    rules = _load_hour_type_rules()

    # Normalize function name for matching (handle slight variations)
    normalized = function_name.strip()

    # Try exact match first
    if normalized in rules:
        return rules[normalized]

    # Try partial match for common variations
    for rule_name in rules:
        if rule_name == "_default":
            continue
        # Handle "Aftertreatment(ATS)" vs "ATS, Mat & Fluids" variations
        if normalized.lower() in rule_name.lower() or rule_name.lower() in normalized.lower():
            return rules[rule_name]

    # Return default
    return rules.get("_default", {
        "manpower_hrs": True,
        "bench_durability_hrs": False,
        "bench_development_hrs": False,
        "bench_special_hrs": False,
        "vehicle_hrs": False,
    })

# PE Function → Change type mapping (derived from ref_features_by_function.json)
# Maps which change flags trigger which PE functions
#
# IMPORTANT: Calibration-only projects should NOT trigger hardware-heavy functions
# Design/Prototype/Materials primarily need hardware_change or ATS_change
PE_FUNCTION_TRIGGERS: dict[str, list[str]] = {
    # Always included (overhead) - minimal weight
    "Project Management": ["sizing"],
    "Others (Travels, Dataloggers, contingencies)": [],

    # Hardware-intensive (require hardware_change or ATS_change)
    "Design": ["hardware_change", "ATS_change"],  # NO calibration_change!
    "Cost Engineering": ["hardware_change", "ATS_change"],
    "Basic technologies, Simulation, Virtual Validation": ["hardware_change", "ATS_change"],
    "ATS, Mat & Fluids": ["hardware_change", "ATS_change", "fluids_related"],
    "Testing / Endurance - Engine & ATS": ["hardware_change", "ATS_change"],
    "Prototype": ["hardware_change", "ATS_change"],  # Physical prototypes
    "Materials & Travels": ["hardware_change", "ATS_change"],
    "Laboratories": ["hardware_change", "ATS_change", "fluids_related"],
    "Contracts / Fees - Other Suppliers": ["hardware_change", "ATS_change", "fluids_related"],

    # Software/Calibration-related (triggered by software or calibration changes)
    "Control System & Software (CS&SW; EMS)": ["software_VCU_change", "calibration_change"],
    "OBD & Diagnostics": ["software_VCU_change", "calibration_change"],
    "CP&E; Dev&Rel": ["calibration_change", "emission_related"],  # Core calibration
    "Application Engineering": ["calibration_change", "emission_related"],
    "Technical Certification": ["calibration_change", "emission_related"],
    "Contracts / Fees - Supplier_B": ["software_VCU_change", "calibration_change"],

    # Vehicle testing (emission/PEMS)
    "Vehicle": ["emission_related", "requires_vehicle_test"],

    # Documentation
    "Advanced Troubleshooting & Tech. Docu.": ["technical_documentation", "calibration_change"],
}

# Base weights for each PE function (from historical analysis)
# These represent typical % of total hours when function is affected
PE_FUNCTION_BASE_WEIGHTS: dict[str, float] = {
    "Project Management": 0.08,
    "Cost Engineering": 0.02,
    "Design": 0.15,
    "Basic technologies, Simulation, Virtual Validation": 0.05,
    "ATS, Mat & Fluids": 0.06,
    "Control System & Software (CS&SW; EMS)": 0.10,
    "OBD & Diagnostics": 0.04,
    "CP&E; Dev&Rel": 0.12,
    "Testing / Endurance - Engine & ATS": 0.10,
    "Application Engineering": 0.08,
    "Vehicle": 0.05,
    "Advanced Troubleshooting & Tech. Docu.": 0.03,
    "Technical Certification": 0.04,
    "Prototype": 0.03,
    "Materials & Travels": 0.02,
    "Laboratories": 0.02,
    "Contracts / Fees - Supplier_B": 0.01,
    "Contracts / Fees - Other Suppliers": 0.01,
    "Others (Travels, Dataloggers, contingencies)": 0.02,
}

# PE Function → PE02 Activity Code mapping
PE_FUNCTION_TO_PE02: dict[str, str] = {
    "Project Management": "A1",
    "Cost Engineering": "A1",
    "Design": "A2",
    "Basic technologies, Simulation, Virtual Validation": "A4",
    "ATS, Mat & Fluids": "A3",
    "Control System & Software (CS&SW; EMS)": "A4",
    "OBD & Diagnostics": "B1-C",
    "CP&E; Dev&Rel": "B1",
    "Testing / Endurance - Engine & ATS": "B2",
    "Application Engineering": "C",
    "Vehicle": "D3",
    "Advanced Troubleshooting & Tech. Docu.": "F",
    "Technical Certification": "D1+D2",
    "Prototype": "B3",
    "Materials & Travels": "E",
    "Laboratories": "B2",
    "Contracts / Fees - Supplier_B": "E",
    "Contracts / Fees - Other Suppliers": "E",
    "Others (Travels, Dataloggers, contingencies)": "G",
}

# Reverse mapping: PE02 Code → Primary PE Function Name
# Used when estimation_node needs to get hour-type distribution by activity code
PE02_TO_PE_FUNCTION: dict[str, str] = {
    "A1": "Project Management",
    "A2": "Design",
    "A3": "ATS, Mat & Fluids",
    "A4": "Control System & Software (CS&SW; EMS)",
    "B1": "CP&E; Dev&Rel",
    "B1-C": "OBD & Diagnostics",
    "B2": "Testing / Endurance - Engine & ATS",
    "B3": "Prototype",
    "C": "Application Engineering",
    "D1": "Technical Certification",
    "D2": "Technical Certification",
    "D1+D2": "Technical Certification",
    "D3": "Vehicle",
    "E": "Materials & Travels",
    "F": "Advanced Troubleshooting & Tech. Docu.",
    "F1": "Advanced Troubleshooting & Tech. Docu.",
    "F2": "Advanced Troubleshooting & Tech. Docu.",
    "G": "Others (Travels, Dataloggers, contingencies)",
}

# Hourly rates by PE function (from price_rate_db.json)
PE_FUNCTION_RATES: dict[str, float] = {
    "Project Management": 89.0,
    "Cost Engineering": 59.0,
    "Design": 59.0,
    "Basic technologies, Simulation, Virtual Validation": 59.0,
    "ATS, Mat & Fluids": 59.0,
    "Control System & Software (CS&SW; EMS)": 59.0,
    "OBD & Diagnostics": 59.0,
    "CP&E; Dev&Rel": 59.0,
    "Testing / Endurance - Engine & ATS": 44.0,
    "Application Engineering": 59.0,
    "Vehicle": 107.0,  # Vehicle/PEMS rate
    "Advanced Troubleshooting & Tech. Docu.": 89.0,
    "Technical Certification": 59.0,
    "Prototype": 106.0,
    "Materials & Travels": 59.0,
    "Laboratories": 59.0,
    "Contracts / Fees - Supplier_B": 59.0,
    "Contracts / Fees - Other Suppliers": 59.0,
    "Others (Travels, Dataloggers, contingencies)": 59.0,
}


def get_affected_pe_functions(features: dict[str, Any]) -> list[str]:
    """
    Determine which PE functions are affected based on change flags.

    Returns list of affected PE function names.
    """
    affected = []

    for func_name, triggers in PE_FUNCTION_TRIGGERS.items():
        # Special case: PM and Others always included (overhead)
        if func_name in ["Project Management", "Others (Travels, Dataloggers, contingencies)"]:
            affected.append(func_name)
            continue

        # Check if any trigger is active
        for trigger in triggers:
            value = features.get(trigger, 0)
            if value and value not in [0, False, "0", "false", ""]:
                affected.append(func_name)
                break

    return affected


def count_affected_pe_functions(features: dict[str, Any]) -> int:
    """Count how many PE functions are affected."""
    return len(get_affected_pe_functions(features))


def distribute_hours_to_pe_functions(
    total_hours: float,
    features: dict[str, Any],
    historical_weights: dict[str, float] | None = None,
    blend_ratio: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Distribute total hours across PE functions based on affected functions.

    Key improvements over fixed distribution:
    1. Only affected functions get hours (non-affected = 0)
    2. Weights are re-normalized to sum to 100%
    3. Blends historical + base weights for accuracy

    Args:
        total_hours: Total hours from HCQE model
        features: ML features dict with change flags
        historical_weights: Optional weights from similar PRs (0-1 per function)
        blend_ratio: Ratio for historical vs base weights (0.7 = 70% historical)

    Returns:
        List of dicts with: function, pe02_code, hours, rate, cost_eur
    """
    affected_functions = get_affected_pe_functions(features)

    if not affected_functions:
        logger.warning("No affected functions found, using minimal distribution")
        affected_functions = ["Project Management", "Design", "Others (Travels, Dataloggers, contingencies)"]

    # Calculate raw weights for affected functions only
    raw_weights: dict[str, float] = {}

    for func in affected_functions:
        base_weight = PE_FUNCTION_BASE_WEIGHTS.get(func, 0.05)

        if historical_weights and func in historical_weights:
            hist_weight = historical_weights[func]
            # Blend: 70% historical + 30% base (configurable)
            blended = blend_ratio * hist_weight + (1 - blend_ratio) * base_weight
            raw_weights[func] = blended
        else:
            raw_weights[func] = base_weight

    # CRITICAL: Re-normalize to sum to 1.0
    total_raw = sum(raw_weights.values())
    if total_raw <= 0:
        total_raw = 1.0

    normalized_weights = {k: v / total_raw for k, v in raw_weights.items()}

    # Build breakdown with hour-type distribution
    breakdown: list[dict[str, Any]] = []

    for func, weight in normalized_weights.items():
        hours = round(total_hours * weight, 1)
        if hours <= 0:
            continue

        rate = PE_FUNCTION_RATES.get(func, 59.0)
        cost_eur = hours * rate
        pe02_code = PE_FUNCTION_TO_PE02.get(func, "G")

        # Get hour-type breakdown for this function
        hour_types = distribute_hours_by_type(func, hours, features)

        breakdown.append({
            "function": func,
            "pe02_code": pe02_code,
            "hours": hours,
            "weight": round(weight, 4),
            "hourly_rate": rate,
            "cost_eur": round(cost_eur, 2),
            "cost_keur": round(cost_eur / 1000, 2),
            # Hour-type breakdown (respecting rule_hours.json)
            "manpower_hrs": hour_types.get("manpower_hrs"),
            "bench_durability_hrs": hour_types.get("bench_durability_hrs"),
            "bench_development_hrs": hour_types.get("bench_development_hrs"),
            "bench_special_hrs": hour_types.get("bench_special_hrs"),
            "vehicle_hrs": hour_types.get("vehicle_hrs"),
        })

    # Sort by hours descending
    breakdown.sort(key=lambda x: x["hours"], reverse=True)

    logger.info(
        f"Distributed {total_hours:.0f}h across {len(breakdown)} PE functions "
        f"(of {len(PE_FUNCTION_TRIGGERS)} total, {len(affected_functions)} affected)"
    )

    return breakdown


# ============================================================================
# HOUR-TYPE DISTRIBUTION (Based on rule_hours.json constraints)
# ============================================================================

# Default hour-type weights when multiple types are enabled
# Based on domain knowledge: bench development is typically the largest component
HOUR_TYPE_DEFAULT_WEIGHTS = {
    "manpower_hrs": 0.30,           # Engineering manpower
    "bench_development_hrs": 0.35,  # Bench development testing
    "bench_durability_hrs": 0.20,   # Durability testing
    "bench_special_hrs": 0.05,      # Special tests (emissions, etc.)
    "vehicle_hrs": 0.10,            # Vehicle/PEMS testing
}

# Function-specific hour-type weight adjustments
# Some functions have specific distributions based on domain knowledge
FUNCTION_HOUR_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "Project Management": {
        "manpower_hrs": 1.0,  # 100% manpower
    },
    "Cost Engineering": {
        "manpower_hrs": 1.0,
    },
    "Design": {
        "manpower_hrs": 1.0,
    },
    "CP&E; Dev&Rel": {
        "manpower_hrs": 0.25,
        "bench_development_hrs": 0.45,
        "bench_durability_hrs": 0.20,
        "bench_special_hrs": 0.10,
    },
    "Testing / Endurance - Engine & ATS": {
        "manpower_hrs": 0.20,
        "bench_development_hrs": 0.40,
        "bench_durability_hrs": 0.40,
    },
    "Technical Certification": {
        "manpower_hrs": 0.30,
        "bench_development_hrs": 0.35,
        "bench_durability_hrs": 0.35,
    },
    "OBD & Diagnostics": {
        "manpower_hrs": 0.40,
        "bench_development_hrs": 0.35,
        "vehicle_hrs": 0.25,
    },
    "Vehicle": {
        "manpower_hrs": 0.30,
        "vehicle_hrs": 0.70,
    },
}


def distribute_hours_by_type(
    function_name: str,
    total_hours: float,
    features: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """
    Distribute hours across 5 hour types for a PE function.

    Respects enabled/disabled constraints from rule_hours.json.
    Disabled hour types return None (not 0).

    Args:
        function_name: PE function name
        total_hours: Total hours for this function
        features: ML features (for context-based adjustments)

    Returns:
        Dict with 5 hour types, disabled ones as None
    """
    enabled = get_enabled_hour_types(function_name)

    # Count enabled types
    enabled_types = [ht for ht in HOUR_COLUMNS if enabled.get(ht, False)]

    if not enabled_types:
        # No hour types enabled (e.g., Contracts/Others) - return all None
        return {ht: None for ht in HOUR_COLUMNS}

    # Get function-specific weights or use defaults
    custom_weights = FUNCTION_HOUR_TYPE_WEIGHTS.get(function_name, {})

    # Calculate weights for enabled types only
    raw_weights: dict[str, float] = {}
    for ht in enabled_types:
        if ht in custom_weights:
            raw_weights[ht] = custom_weights[ht]
        else:
            raw_weights[ht] = HOUR_TYPE_DEFAULT_WEIGHTS.get(ht, 0.2)

    # Normalize weights to sum to 1.0
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        total_weight = 1.0

    # Build result
    result: dict[str, float | None] = {}
    for ht in HOUR_COLUMNS:
        if ht in enabled_types and ht in raw_weights:
            weight = raw_weights[ht] / total_weight
            result[ht] = round(total_hours * weight, 1)
        else:
            result[ht] = None

    return result


def aggregate_to_clusters(pe_breakdown: list[dict[str, Any]]) -> dict[str, float]:
    """
    Aggregate PE function hours into 8 clusters for backwards compatibility.

    Cluster mapping:
    - hardware: Design, ATS Mat & Fluids
    - calibration: CP&E, OBD
    - testing: Testing, Laboratories, Vehicle
    - software: Control System, Basic Tech
    - documentation: Tech Docu, Tech Certification
    - ats: ATS, Mat & Fluids (already in hardware, but separated)
    - installation: Application Engineering
    - dependent: PM, Cost Engineering, Prototype, Materials
    """
    # Map PE functions to 8 clusters (using "dataset" instead of "dependent")
    PE_TO_CLUSTER = {
        "Project Management": "dataset",
        "Cost Engineering": "dataset",
        "Design": "hardware",
        "Basic technologies, Simulation, Virtual Validation": "software",
        "ATS, Mat & Fluids": "ats",
        "Control System & Software (CS&SW; EMS)": "software",
        "OBD & Diagnostics": "calibration",
        "CP&E; Dev&Rel": "calibration",
        "Testing / Endurance - Engine & ATS": "testing",
        "Application Engineering": "installation",
        "Vehicle": "testing",
        "Advanced Troubleshooting & Tech. Docu.": "documentation",
        "Technical Certification": "documentation",
        "Prototype": "dataset",
        "Materials & Travels": "dataset",
        "Laboratories": "testing",
        "Contracts / Fees - Supplier_B": "dataset",
        "Contracts / Fees - Other Suppliers": "dataset",
        "Others (Travels, Dataloggers, contingencies)": "dataset",
    }

    # 8 clusters matching HCQE predictor: hardware, calibration, testing, ats,
    # software, documentation, installation, dataset (renamed from "dependent")
    cluster_hours: dict[str, float] = {
        "hardware": 0.0,
        "calibration": 0.0,
        "testing": 0.0,
        "software": 0.0,
        "documentation": 0.0,
        "ats": 0.0,
        "installation": 0.0,
        "dataset": 0.0,  # Renamed from "dependent" to match HCQE predictor
    }

    for item in pe_breakdown:
        func = item["function"]
        hours = item["hours"]
        cluster = PE_TO_CLUSTER.get(func, "dataset")  # Default to dataset
        cluster_hours[cluster] += hours

    # Round all values
    return {k: round(v, 1) for k, v in cluster_hours.items()}


def calculate_effective_rate(pe_breakdown: list[dict[str, Any]]) -> float:
    """Calculate weighted average hourly rate from PE breakdown."""
    total_hours = sum(item["hours"] for item in pe_breakdown)
    total_cost = sum(item["cost_eur"] for item in pe_breakdown)

    if total_hours <= 0:
        return 59.0  # Default rate

    return round(total_cost / total_hours, 2)


def aggregate_hour_types(pe_breakdown: list[dict[str, Any]]) -> dict[str, float]:
    """
    Aggregate hour types across all PE functions.

    Returns totals for each hour type column.
    """
    totals = {ht: 0.0 for ht in HOUR_COLUMNS}

    for item in pe_breakdown:
        for ht in HOUR_COLUMNS:
            value = item.get(ht)
            if value is not None:
                totals[ht] += value

    return {k: round(v, 1) for k, v in totals.items()}


def validate_hour_distribution(pe_breakdown: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Validate that hour distribution respects rule_hours.json constraints.

    Returns validation results with any violations.
    """
    violations = []
    valid_count = 0

    for item in pe_breakdown:
        func = item["function"]
        enabled = get_enabled_hour_types(func)

        for ht in HOUR_COLUMNS:
            value = item.get(ht)
            is_enabled = enabled.get(ht, False)

            if not is_enabled and value is not None and value > 0:
                violations.append({
                    "function": func,
                    "hour_type": ht,
                    "value": value,
                    "error": "Hours assigned to disabled hour type",
                })
            elif is_enabled and value is not None and value >= 0:
                valid_count += 1

    return {
        "valid": len(violations) == 0,
        "valid_assignments": valid_count,
        "violations": violations,
    }



def distribute_hours_by_activity_code(
    activity_code: str,
    total_hours: float,
    features: dict[str, Any] | None = None,
) -> dict[str, float | None]:
    """
    Distribute hours across 5 hour types for an activity code (PE02 code).

    Maps activity code (A1, B1, etc.) to PE function name, then uses
    distribute_hours_by_type() with rule_hours.json constraints.

    Args:
        activity_code: PE02 activity code (e.g., "A1", "B1", "D1+D2")
        total_hours: Total hours for this activity
        features: ML features (for context-based adjustments)

    Returns:
        Dict with keys: manpower, bench_dev, bench_special, bench_dur, vehicle
        (uses legacy column names for backward compatibility with estimation_node)
    """
    # Map activity code to PE function name
    function_name = PE02_TO_PE_FUNCTION.get(activity_code, "Others (Travels, Dataloggers, contingencies)")

    # Get hour-type distribution using rule_hours.json
    hour_types = distribute_hours_by_type(function_name, total_hours, features)

    # Convert to legacy column names (for backward compatibility)
    return {
        "manpower": hour_types.get("manpower_hrs") or 0.0,
        "bench_dev": hour_types.get("bench_development_hrs") or 0.0,
        "bench_special": hour_types.get("bench_special_hrs") or 0.0,
        "bench_dur": hour_types.get("bench_durability_hrs") or 0.0,
        "vehicle": hour_types.get("vehicle_hrs") or 0.0,
    }

"""
FPT Cost Brain 2.0 - Unified PR Embedding Text Builder
======================================================
Single source of truth for PR text representation used in both
indexing (populate script) and querying (summary_node).

This module solves the QUERY-DOCUMENT MISMATCH problem by ensuring
identical text construction for both index-time and query-time embeddings.

Author: FPT Cost Brain Research Team
"""

from typing import Any

# Sizing proximity groups for hybrid filtering
SIZING_PROXIMITY = {
    "X-small": ["X-small", "Small"],
    "Small": ["X-small", "Small", "Mid"],
    "Mid": ["Small", "Mid", "Large"],
    "Large": ["Mid", "Large", "Full"],
    "Full": ["Large", "Full"],
    "nan": ["X-small", "Small", "Mid", "Large", "Full"],  # Unknown matches all
    "Unknown": ["X-small", "Small", "Mid", "Large", "Full"],
}

# Feature weights for deterministic reranking (hardcoded per spec)
FEATURE_WEIGHTS = {
    "sector": 0.25,  # Critical for cost structure (AG vs CE)
    "sizing": 0.20,  # Critical for scale
    "product_family": 0.15,  # Engine family correlation
    "customer_platform": 0.15,  # Application type
    "emissions": 0.10,  # Regulatory scope
    "hardware_change": 0.05,  # Cost driver flag
    "calibration_change": 0.05,  # Cost driver flag
    "ats_change": 0.03,  # Cost driver flag
    "software_vcu_change": 0.02,  # Cost driver flag
}

# Ensemble weights (feature vs vector similarity)
ENSEMBLE_FEATURE_WEIGHT = 0.6
ENSEMBLE_VECTOR_WEIGHT = 0.4


def build_pr_embedding_text(pr: dict[str, Any]) -> str:
    """
    Build deterministic text representation of a PR for embedding.

    This function MUST be used identically for:
    - Indexing (populate_pr_embeddings_v3.py)
    - Querying (summary_node.generate_pr_text)

    Args:
        pr: Dictionary containing PR data (from CSV or parsed PR)

    Returns:
        Deterministic text string for embedding generation

    Field priority (most important first):
        1. Sector (AG/CE) - determines cost structure
        2. Platform - application type
        3. Product Family - engine family
        4. Emissions - regulatory scope
        5. Sizing - program scale
        6. Type - project type
        7. Change flags - cost drivers
        8. Description - additional context
    """
    # Normalize sector with multiple fallbacks
    sector = _normalize_field(pr.get("sector") or pr.get("Sector"), default="Unknown")

    # Normalize platform with multiple fallbacks
    platform = _normalize_field(
        pr.get("customer_platform")
        or pr.get("Customer_Platform")
        or pr.get("platform"),
        default="Unknown",
    )

    # Normalize product family
    product_family = _normalize_field(
        pr.get("product_family")
        or pr.get("Product_Family")
        or pr.get("program_family"),
        default="Unknown",
    )

    # Normalize emissions
    emissions = _normalize_field(
        pr.get("emissions") or pr.get("Emissions"), default="Unknown"
    )

    # Normalize sizing with multiple fallbacks
    sizing = _normalize_field(
        pr.get("sizing")
        or pr.get("Sizing")
        or pr.get("sizing_program")
        or pr.get("program_size"),
        default="Unknown",
    )

    # Normalize PR type
    pr_type = _normalize_field(
        pr.get("pr_type") or pr.get("PR_Type") or pr.get("project_type"),
        default="Unknown",
    )

    # Boolean flags with consistent yes/no format
    hw_change = _bool_to_yesno(pr.get("hardware_change") or pr.get("Hardware_Change"))
    cal_change = _bool_to_yesno(
        pr.get("calibration_change") or pr.get("Calibration_Change")
    )
    ats_change = _bool_to_yesno(
        pr.get("ats_change") or pr.get("ATS_change") or pr.get("ATS_Change")
    )
    sw_change = _bool_to_yesno(
        pr.get("software_vcu_change")
        or pr.get("software_VCU_change")
        or pr.get("Software_VCU_Change")
    )

    # Build description from title + raw_text (per user decision)
    description = _build_description(pr)

    # Construct the final text in fixed order
    text = (
        f"Product Request. "
        f"Sector: {sector}. "
        f"Platform: {platform}. "
        f"Engine family: {product_family}. "
        f"Emissions: {emissions}. "
        f"Sizing: {sizing}. "
        f"Type: {pr_type}. "
        f"Hardware change: {hw_change}. "
        f"Calibration change: {cal_change}. "
        f"ATS change: {ats_change}. "
        f"Software VCU change: {sw_change}. "
        f"Description: {description}"
    )

    return text


def _normalize_field(value: Any, default: str = "Unknown") -> str:
    """Normalize a field value to a clean string."""
    if value is None:
        return default
    str_value = str(value).strip()
    if str_value.lower() in ("", "nan", "none", "null", "n/a"):
        return default
    return str_value


def _bool_to_yesno(value: Any) -> str:
    """Convert a value to consistent yes/no string."""
    if value is None:
        return "no"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if value > 0 else "no"
    str_value = str(value).strip().lower()
    return "yes" if str_value in ("1", "true", "yes", "y") else "no"


def _build_description(pr: dict[str, Any], max_length: int = 200) -> str:
    """
    Build description from available fields.

    Priority:
    1. pr_name + customer_platform (as title)
    2. raw_text if available
    3. title field
    4. description field
    """
    parts = []

    # PR name/code
    pr_name = pr.get("pr_name") or pr.get("pr_code") or pr.get("pr_id")
    if pr_name:
        parts.append(str(pr_name))

    # Customer platform as part of title
    platform = pr.get("customer_platform") or pr.get("Customer_Platform")
    if platform and str(platform).lower() not in ("nan", "none", "unknown"):
        parts.append(str(platform))

    # Title if different from above
    title = pr.get("title")
    if title and str(title) not in parts:
        parts.append(str(title))

    # Raw text for additional context
    raw_text = pr.get("raw_text")
    if raw_text and len(str(raw_text)) > 10:
        parts.append(str(raw_text)[:100])

    # Join and truncate
    description = " - ".join(parts)
    if len(description) > max_length:
        description = description[: max_length - 3] + "..."

    return description if description else "No description"


def calculate_feature_similarity(pr1: dict[str, Any], pr2: dict[str, Any]) -> float:
    """
    Calculate feature-based similarity between two PRs.

    Uses weighted matching of structured fields for deterministic,
    explainable similarity scoring.

    Args:
        pr1: First PR dictionary
        pr2: Second PR dictionary

    Returns:
        Similarity score between 0.0 and 1.0
    """
    score = 0.0

    # Sector match (exact)
    sector1 = _normalize_field(pr1.get("sector") or pr1.get("Sector"))
    sector2 = _normalize_field(pr2.get("sector") or pr2.get("Sector"))
    if sector1 == sector2 and sector1 != "Unknown":
        score += FEATURE_WEIGHTS["sector"]

    # Sizing match (proximity-based)
    sizing1 = _normalize_field(
        pr1.get("sizing") or pr1.get("Sizing") or pr1.get("sizing_program")
    )
    sizing2 = _normalize_field(
        pr2.get("sizing") or pr2.get("Sizing") or pr2.get("sizing_program")
    )
    sizing_score = _calculate_sizing_proximity(sizing1, sizing2)
    score += FEATURE_WEIGHTS["sizing"] * sizing_score

    # Product family match (exact)
    family1 = _normalize_field(pr1.get("product_family") or pr1.get("Product_Family"))
    family2 = _normalize_field(pr2.get("product_family") or pr2.get("Product_Family"))
    if family1 == family2 and family1 != "Unknown":
        score += FEATURE_WEIGHTS["product_family"]

    # Platform match (partial - contains check)
    platform1 = _normalize_field(
        pr1.get("customer_platform") or pr1.get("Customer_Platform")
    )
    platform2 = _normalize_field(
        pr2.get("customer_platform") or pr2.get("Customer_Platform")
    )
    platform_score = _calculate_platform_similarity(platform1, platform2)
    score += FEATURE_WEIGHTS["customer_platform"] * platform_score

    # Emissions match (exact)
    emissions1 = _normalize_field(pr1.get("emissions") or pr1.get("Emissions"))
    emissions2 = _normalize_field(pr2.get("emissions") or pr2.get("Emissions"))
    if emissions1 == emissions2 and emissions1 != "Unknown":
        score += FEATURE_WEIGHTS["emissions"]

    # Change flags match
    for flag in [
        "hardware_change",
        "calibration_change",
        "ats_change",
        "software_vcu_change",
    ]:
        val1 = _bool_to_yesno(pr1.get(flag) or pr1.get(flag.replace("_", "_")))
        val2 = _bool_to_yesno(pr2.get(flag) or pr2.get(flag.replace("_", "_")))
        if val1 == val2:
            score += FEATURE_WEIGHTS[flag]

    return min(1.0, score)


def _calculate_sizing_proximity(sizing1: str, sizing2: str) -> float:
    """
    Calculate sizing proximity score.

    Returns:
        1.0 for exact match
        0.5 for adjacent size
        0.0 for distant or unknown
    """
    if sizing1 == sizing2:
        return 1.0

    # Get proximity groups
    group1 = SIZING_PROXIMITY.get(sizing1, [])
    group2 = SIZING_PROXIMITY.get(sizing2, [])

    # Check if in each other's proximity group
    if sizing2 in group1 or sizing1 in group2:
        return 0.5

    return 0.0


def _calculate_platform_similarity(platform1: str, platform2: str) -> float:
    """
    Calculate platform similarity using keyword matching.

    Returns:
        1.0 for exact match
        0.5-0.8 for partial match (shared keywords)
        0.0 for no match
    """
    if platform1 == platform2:
        return 1.0

    if platform1 == "Unknown" or platform2 == "Unknown":
        return 0.0

    # Extract keywords
    keywords1 = set(platform1.lower().split())
    keywords2 = set(platform2.lower().split())

    # Remove common words
    stopwords = {"the", "a", "an", "for", "and", "or", "of", "with"}
    keywords1 -= stopwords
    keywords2 -= stopwords

    if not keywords1 or not keywords2:
        return 0.0

    # Jaccard similarity
    intersection = keywords1 & keywords2
    union = keywords1 | keywords2

    if not union:
        return 0.0

    return len(intersection) / len(union)


def calculate_ensemble_score(
    vector_score: float,
    feature_score: float,
) -> float:
    """
    Calculate final ensemble score combining vector and feature similarity.

    Formula: 0.6 * feature_score + 0.4 * vector_score

    Args:
        vector_score: Cosine similarity from vector search (0-1)
        feature_score: Feature-based similarity (0-1)

    Returns:
        Combined score (0-1)
    """
    return (
        ENSEMBLE_FEATURE_WEIGHT * feature_score + ENSEMBLE_VECTOR_WEIGHT * vector_score
    )


def get_sizing_filter_values(sizing: str) -> list[str]:
    """
    Get sizing values for Qdrant filter based on proximity groups.

    Args:
        sizing: The sizing value of the query PR

    Returns:
        List of sizing values to include in filter
    """
    normalized = _normalize_field(sizing)
    return SIZING_PROXIMITY.get(normalized, list(SIZING_PROXIMITY.keys()))

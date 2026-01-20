"""
Lookup-Based Cost Estimator

Primary estimation approach using ref_Sizing lookup tables.
ML model is used only for calibration (±20% adjustment).

Architecture:
    1. LLM Agent classifies PR → sector + sizing
    2. Lookup table provides base estimate
    3. ML model provides optional calibration
    4. Agent provides reasoning and explanation

This approach is more reliable than pure ML with only 37 training samples.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# REFERENCE DATA (from ref_Sizing Excel sheet)
# ============================================================================


class Sector(str, Enum):
    """Business sector classification."""

    AG = "AG"  # Agricultural (tractors, harvesters, combines)
    CE = "CE"  # Construction Equipment (excavators, loaders)


class SizingLevel(str, Enum):
    """Project sizing classification (from ref_Sizing)."""

    X_SMALL = "X-small"
    SMALL = "Small"
    MID = "Mid"
    LARGE = "Large"
    FULL = "Full"


# Cost lookup table (K EUR) - from ref_Sizing historical data
# AG projects are typically 2-5× more expensive than CE
REF_SIZING_COSTS = {
    Sector.AG: {
        SizingLevel.X_SMALL: 500,
        SizingLevel.SMALL: 1500,
        SizingLevel.MID: 5000,
        SizingLevel.LARGE: 6500,
        SizingLevel.FULL: 7500,
    },
    Sector.CE: {
        SizingLevel.X_SMALL: 130,
        SizingLevel.SMALL: 1000,
        SizingLevel.MID: 1000,
        SizingLevel.LARGE: 4600,
        SizingLevel.FULL: 5000,
    },
}

# Sizing descriptions for LLM context (from ref_Sizing)
SIZING_DESCRIPTIONS = {
    SizingLevel.X_SMALL: {
        "PE_base": "Minimum modification (only adaptation); Minimum NC; No validation effort",
        "PE_system": "Minimum modification (only adaptation); Minimum NC; No validation",
        "installation": "Minimum installation; Minimum Cals effort; No homologation",
        "build_stages": "Only Pilot required",
    },
    SizingLevel.SMALL: {
        "PE_base": "Light modification of existing product; Low NC; Low validation effort",
        "PE_system": "Light modification; Low NC; Low validation",
        "installation": "Low installation effort; Limited Cals Review; Homologation",
        "build_stages": "PP, Pilot required",
    },
    SizingLevel.MID: {
        "PE_base": "Medium modification of existing concepts (no impact on manufacturing); Medium NC; Medium validation",
        "PE_system": "Medium modification (no manufacturing impact); Medium NC; Medium validation",
        "installation": "Medium installation effort; Medium Cals Review; Homologation; RGT",
        "build_stages": "Gamma, PP, Pilot required",
    },
    SizingLevel.LARGE: {
        "PE_base": "Heavy modification of existing concepts with impact on manufacturing; High/Medium NC; High/Medium validation",
        "PE_system": "Heavy modification with manufacturing impact; High/Medium NC; High/Medium validation",
        "installation": "Medium Installation effort; Medium Cals review; Homologation; RGT",
        "build_stages": "Beta, Gamma, PP, Pilot required",
    },
    SizingLevel.FULL: {
        "PE_base": "New concept required; High level of New Content (NC); New serviceability requirements; High validation effort",
        "PE_system": "New concept required; High NC; New serviceability requirements; High validation",
        "installation": "First installation; New SW & Cals; New Emission Stage; RGT",
        "build_stages": "All build stages required (Alpha, Beta, Gamma, PP, Pilot)",
    },
}

# Sector keywords for classification
SECTOR_KEYWORDS = {
    Sector.AG: [
        "tractor",
        "harvester",
        "combine",
        "sprayer",
        "agricultural",
        "farm",
        "crop",
        "forage",
        "baler",
        "planter",
        "seeder",
    ],
    Sector.CE: [
        "excavator",
        "loader",
        "grader",
        "telehandler",
        "construction",
        "crawler",
        "skid",
        "backhoe",
        "dozer",
        "compactor",
        "paver",
    ],
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SizingClassification:
    """Result of LLM sizing classification."""

    sector: Sector
    sizing_PE_base: SizingLevel
    sizing_PE_system: SizingLevel
    sizing_installation: SizingLevel
    sizing_program: SizingLevel
    confidence: float  # 0.0 - 1.0
    reasoning: str


@dataclass
class LookupEstimate:
    """Estimate from lookup table."""

    point_estimate_keur: float
    low_estimate_keur: float
    high_estimate_keur: float
    sector: Sector
    sizing_level: SizingLevel  # Aggregated sizing
    confidence: float
    source: str = "ref_Sizing lookup"


@dataclass
class CalibratedEstimate:
    """Final estimate with ML calibration."""

    lookup_estimate_keur: float
    ml_adjustment_keur: float
    final_estimate_keur: float
    low_estimate_keur: float
    high_estimate_keur: float
    confidence: float
    reasoning: str


# ============================================================================
# LOOKUP ESTIMATOR
# ============================================================================


class LookupEstimator:
    """
    Primary cost estimator using ref_Sizing lookup tables.

    This is more reliable than ML alone with limited training data.
    """

    def __init__(self):
        self.costs = REF_SIZING_COSTS
        self.descriptions = SIZING_DESCRIPTIONS

    def detect_sector(self, text: str) -> tuple[Sector, float]:
        """
        Detect sector (AG/CE) from PR text using keyword matching.

        Returns:
            (sector, confidence)
        """
        text_lower = text.lower()

        ag_score = sum(1 for kw in SECTOR_KEYWORDS[Sector.AG] if kw in text_lower)
        ce_score = sum(1 for kw in SECTOR_KEYWORDS[Sector.CE] if kw in text_lower)

        total = ag_score + ce_score
        if total == 0:
            # Default to AG (more expensive = safer estimate)
            return Sector.AG, 0.5

        if ag_score > ce_score:
            return Sector.AG, ag_score / total
        elif ce_score > ag_score:
            return Sector.CE, ce_score / total
        else:
            # Tie - default to AG (conservative)
            return Sector.AG, 0.5

    def aggregate_sizing(self, classification: SizingClassification) -> SizingLevel:
        """
        Aggregate multiple sizing levels into single overall sizing.

        Uses the maximum sizing (most conservative/expensive estimate).
        """
        sizing_order = [
            SizingLevel.X_SMALL,
            SizingLevel.SMALL,
            SizingLevel.MID,
            SizingLevel.LARGE,
            SizingLevel.FULL,
        ]

        sizes = [
            classification.sizing_PE_base,
            classification.sizing_PE_system,
            classification.sizing_installation,
            classification.sizing_program,
        ]

        # Get maximum sizing level
        max_idx = max(sizing_order.index(s) for s in sizes)
        return sizing_order[max_idx]

    def estimate_from_sizing(
        self,
        sector: Sector,
        sizing: SizingLevel,
        confidence: float = 0.6,
    ) -> LookupEstimate:
        """
        Get cost estimate from lookup table.

        Args:
            sector: AG or CE
            sizing: Sizing level
            confidence: Classification confidence

        Returns:
            LookupEstimate with point estimate and range
        """
        base_cost = self.costs[sector][sizing]

        # Confidence affects the range width
        # Lower confidence = wider range
        range_factor = 0.4 + (1 - confidence) * 0.2  # 0.4 to 0.6

        return LookupEstimate(
            point_estimate_keur=base_cost,
            low_estimate_keur=base_cost * (1 - range_factor),
            high_estimate_keur=base_cost * (1 + range_factor * 1.5),
            sector=sector,
            sizing_level=sizing,
            confidence=confidence,
        )

    def estimate_from_classification(
        self,
        classification: SizingClassification,
    ) -> LookupEstimate:
        """
        Get estimate from full sizing classification.
        """
        sizing = self.aggregate_sizing(classification)
        return self.estimate_from_sizing(
            sector=classification.sector,
            sizing=sizing,
            confidence=classification.confidence,
        )

    def quick_estimate(
        self,
        pr_text: str,
        sizing_hint: Optional[str] = None,
    ) -> LookupEstimate:
        """
        Quick estimate from PR text without full LLM classification.

        Uses keyword matching for sector and sizing hint if provided.
        """
        sector, confidence = self.detect_sector(pr_text)

        # Parse sizing hint or default to Mid
        sizing = SizingLevel.MID
        if sizing_hint:
            sizing_hint_lower = sizing_hint.lower()
            for level in SizingLevel:
                if level.value.lower() in sizing_hint_lower:
                    sizing = level
                    break

        return self.estimate_from_sizing(sector, sizing, confidence)

    def get_sizing_context(self) -> str:
        """
        Get ref_Sizing context for LLM prompts.
        """
        lines = ["## SIZING REFERENCE TABLE (ref_Sizing)\n"]

        for level in SizingLevel:
            desc = self.descriptions[level]
            lines.append(f"### {level.value}")
            lines.append(f"- PE Base: {desc['PE_base']}")
            lines.append(f"- PE System: {desc['PE_system']}")
            lines.append(f"- Installation: {desc['installation']}")
            lines.append(f"- Build Stages: {desc['build_stages']}")
            lines.append("")

        lines.append("### COST BY SECTOR AND SIZING (K EUR)")
        lines.append("| Sizing | AG | CE |")
        lines.append("|--------|----|----|")
        for level in SizingLevel:
            ag = self.costs[Sector.AG][level]
            ce = self.costs[Sector.CE][level]
            lines.append(f"| {level.value} | {ag:,} | {ce:,} |")

        return "\n".join(lines)


# ============================================================================
# ML CALIBRATOR (Secondary role)
# ============================================================================


class MLCalibrator:
    """
    ML-based calibration for lookup estimates.

    Only adjusts by ±20% based on specific features.
    NEVER replaces the lookup estimate entirely.
    """

    MAX_ADJUSTMENT = 0.20  # ±20% max

    def __init__(self, model=None):
        self.model = model
        self._is_fitted = model is not None

    def calibrate(
        self,
        lookup_estimate: LookupEstimate,
        features: dict,
    ) -> CalibratedEstimate:
        """
        Apply ML calibration to lookup estimate.

        Args:
            lookup_estimate: Base estimate from lookup table
            features: Extracted PR features for ML model

        Returns:
            CalibratedEstimate with adjusted values
        """
        base = lookup_estimate.point_estimate_keur
        adjustment = 0.0
        reasoning = "Lookup-based estimate"

        if self._is_fitted and self.model is not None:
            try:
                # Get ML prediction
                ml_pred = self._get_ml_prediction(features)

                # Calculate adjustment (capped at ±20%)
                diff = ml_pred - base
                max_adj = base * self.MAX_ADJUSTMENT
                adjustment = max(-max_adj, min(max_adj, diff))

                if abs(adjustment) > 0.01 * base:
                    direction = "up" if adjustment > 0 else "down"
                    pct = abs(adjustment / base) * 100
                    reasoning = f"Lookup estimate adjusted {direction} by {pct:.0f}% based on ML calibration"

            except Exception as e:
                logger.warning(f"ML calibration failed: {e}")
                adjustment = 0.0

        final = base + adjustment

        # Adjust confidence based on how much we relied on ML
        conf_penalty = abs(adjustment / base) * 0.1 if base > 0 else 0
        confidence = max(0.4, lookup_estimate.confidence - conf_penalty)

        return CalibratedEstimate(
            lookup_estimate_keur=base,
            ml_adjustment_keur=adjustment,
            final_estimate_keur=final,
            low_estimate_keur=lookup_estimate.low_estimate_keur + adjustment * 0.8,
            high_estimate_keur=lookup_estimate.high_estimate_keur + adjustment * 1.2,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _get_ml_prediction(self, features: dict) -> float:
        """Get prediction from ML model."""
        if self.model is None:
            raise ValueError("No ML model loaded")

        # TODO: Integrate with HCQEPredictor
        # For now, return 0 (no adjustment)
        return 0.0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_lookup_estimator() -> LookupEstimator:
    """Factory function for LookupEstimator."""
    return LookupEstimator()


def quick_estimate(pr_text: str, sizing_hint: Optional[str] = None) -> dict:
    """
    Quick estimate from PR text.

    Returns dict for easy JSON serialization.
    """
    estimator = LookupEstimator()
    estimate = estimator.quick_estimate(pr_text, sizing_hint)

    return {
        "point_estimate_keur": estimate.point_estimate_keur,
        "low_estimate_keur": estimate.low_estimate_keur,
        "high_estimate_keur": estimate.high_estimate_keur,
        "sector": estimate.sector.value,
        "sizing_level": estimate.sizing_level.value,
        "confidence": estimate.confidence,
        "source": estimate.source,
    }

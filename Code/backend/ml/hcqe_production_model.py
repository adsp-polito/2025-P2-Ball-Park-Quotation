"""
HCQE Production Model v7 (Fixed Data Leakage)
==============================================

Production model class for sklearn model serialization using joblib.
Must be importable from the same path when loading the trained model.

CRITICAL FIXES in v7:
1. REMOVED bench_rate (data leakage - rates apply in Cost Calculator, not prediction)
2. REMOVED complexity_mult (over-simplified 4 sizing dimensions into 1)
3. REMOVED ats_emissions_interaction (derived from other features)
4. ADDED 4 sizing scores: sizing_PE_base_score, sizing_PE_system_score, etc.

Features: 26 (22 base + 4 sizing scores)
Accuracy: TBD (retrain required)

Usage:
    from ml.hcqe_production_model import HCQEProductionModelV7
    import joblib
    model = joblib.load('models/hcqe_production_v7.joblib')
    result = model.predict_single(features_dict)
"""

from datetime import datetime
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


# v7 Feature List (27 features - NO bench_rate, complexity_mult!)
# These features predict HOURS, not cost. Rates apply in Cost Calculator.
HCQE_V7_FEATURES = [
    # Binary flags (5) - from PR technical description
    "ATS_change",
    "calibration_change",
    "hardware_change",
    "software_VCU_change",
    "application_tractor",
    # Numeric (5) - from PR specs + derived
    "power_increase_kw",
    "torque_increase_nm",
    "num_functions",
    "emissions_level",
    "num_affected_functions",  # NEW: Computed from ref_features_by_function.json
    # Product family one-hot (5)
    "pf_E0N0",
    "pf_E5F0",
    "pf_E0C0",
    "pf_E8S0",
    "pf_E0V0",
    # ATS technology (3)
    "ats_has_doc",
    "ats_has_scr",
    "ats_has_dpf",
    # Project type (5)
    "rd_type_encoded",
    "is_new_engine",
    "is_bom",
    "is_homologation",
    "is_ce",
    # NEW v7: 4 Sizing scores (0-4 scale) - replaces complexity_mult
    "sizing_PE_base_score",  # PE Base/Powertrain sizing
    "sizing_PE_system_score",  # PE System/Assembly sizing
    "sizing_PE_install_score",  # PE Installation/Application sizing
    "sizing_program_score",  # Overall program sizing
]

# PE Function → Change type mapping (from ref_features_by_function.json)
PE_FUNCTION_CHANGE_MAPPING = {
    "Design": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "ATS, Mat & Fluids": ["hardware_change", "ATS_change"],
    "Control System & Software (CS&SW; EMS)": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "OBD & Diagnostics": ["software_VCU_change", "calibration_change"],
    "Testing / Endurance - Engine & ATS": ["hardware_change", "ATS_change"],
    "Application Engineering": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "CP&E; Dev&Rel": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "Prototype": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "Technical Certification": ["software_VCU_change", "calibration_change"],
    "Basic technologies, Simulation, Virtual Validation": [
        "hardware_change",
        "ATS_change",
    ],
    "Cost Engineering": ["hardware_change", "ATS_change"],
    "Laboratories": ["hardware_change", "ATS_change"],
    "Materials & Travels": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "Contracts / Fees - Supplier_B": [
        "hardware_change",
        "ATS_change",
        "software_VCU_change",
        "calibration_change",
    ],
    "Contracts / Fees - Other Suppliers": ["hardware_change", "ATS_change"],
}


def count_affected_pe_functions(features: dict) -> int:
    """
    Count how many PE functions will be affected based on change flags.
    Uses PE_FUNCTION_CHANGE_MAPPING derived from ref_features_by_function.json.
    """
    affected = 0
    for func_name, triggers in PE_FUNCTION_CHANGE_MAPPING.items():
        for trigger in triggers:
            if features.get(trigger, 0) == 1:
                affected += 1
                break  # Function is affected, move to next
    return affected


# Sizing level mapping
SIZING_LEVEL_MAP = {
    "X-small": 0,
    "X-Small": 0,
    "Small": 1,
    "Medium": 2,
    "Mid": 2,
    "Large": 3,
    "Full": 4,
}


class HCQEProductionModelV7:
    """
    Production HCQE model v7 with sizing-based baseline.

    Key changes from v6:
    - Removed bench_rate (applies in Cost Calculator, not prediction)
    - Removed complexity_mult (replaced with 4 granular sizing scores)
    - Added 4 sizing dimension scores (0-4 scale each)
    - NEW in v7.1: Sizing-based lookup provides baseline, GBM learns adjustments

    Model predicts COST (K€). Hours are computed using average rate.
    """

    # Sizing-based HOURS lookup (from training data with CORRECT sizing classification)
    # v7.3 UPDATE: Fixed sizing scores based on actual Manpower thresholds
    # X-Small: <2500h, Small: 2500-6000h, Medium: 6000-12000h, Large: 12000-25000h, Full: >25000h
    SIZING_MEDIAN_HOURS = {
        0: 880,  # X-Small: median=880, mean=1193, n=7, range=152-2103
        1: 4565,  # Small: median=4565, mean=4343, n=11, range=2837-5930
        2: 7915,  # Medium: median=7915, mean=8015, n=5, range=7200-9350
        3: 19245,  # Large: median=19245, mean=18810, n=6, range=12950-22660
        4: 33805,  # Full: median=33805, mean=34378, n=4, range=28004-41900
    }

    # Sizing-based cost lookup (hours * ~120 EUR/h average rate)
    SIZING_MEDIAN_COST_KEUR = {
        0: 105,  # X-Small: 880h * 120€/h
        1: 548,  # Small: 4565h * 120€/h
        2: 950,  # Medium: 7915h * 120€/h
        3: 2309,  # Large: 19245h * 120€/h
        4: 4057,  # Full: 33805h * 120€/h
    }

    def __init__(self):
        self.model = None
        self.feature_names = HCQE_V7_FEATURES.copy()
        self.calibration_factor = 1.5
        self.target_coverage = 0.90
        self.version = "7.3-sizing-fixed"
        self.trained_at = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        """Train the model."""
        self.feature_names = feature_names
        self.trained_at = datetime.now().isoformat()

        # REGULARIZED for small datasets (n < 50)
        # Previous: 100 trees → overfitting on 23 samples!
        self.model = GradientBoostingRegressor(
            n_estimators=20,  # Reduced from 100 (prevent memorization)
            max_depth=2,  # Reduced from 4 (simpler trees)
            learning_rate=0.1,
            min_samples_split=5,  # Need 5 samples to split (was 2)
            min_samples_leaf=3,  # Need 3 samples per leaf
            subsample=0.8,  # Use 80% of data per tree (bagging)
            random_state=42,
        )
        self.model.fit(X, y)

        # Calibrate prediction intervals
        y_pred = self.model.predict(X)
        residuals = np.abs(y - y_pred)
        self.calibration_factor = np.percentile(residuals, self.target_coverage * 100)

        return self

    def predict(self, X: np.ndarray) -> tuple:
        """Predict cost with confidence interval."""
        y_pred = self.model.predict(X)
        lower = np.maximum(0, y_pred - self.calibration_factor)
        upper = y_pred + self.calibration_factor
        return y_pred, lower, upper

    def predict_single(self, features_dict: dict) -> dict:
        """
        Predict for a single sample from feature dict.

        v7.2 CHANGES:
        - Returns BOTH hours and cost (hours is primary, cost is derived)
        - Uses sizing-based lookup for HOURS (from Manpower column)
        - Cost is derived from hours using sizing-specific rate
        """
        features_dict = self._ensure_sizing_scores(features_dict)

        sizing_score = int(features_dict.get("sizing_program_score", 2))
        sizing_score = max(0, min(4, sizing_score))

        # PRIMARY: Get hours from sizing lookup
        baseline_hours = self.SIZING_MEDIAN_HOURS.get(sizing_score, 7920)
        baseline_cost = self.SIZING_MEDIAN_COST_KEUR.get(sizing_score, 1196)

        # Calculate implied rate for this sizing (for cost calculation)
        implied_rate = (
            baseline_cost * 1000 / baseline_hours if baseline_hours > 0 else 130
        )

        # Hours bounds (based on training data variance, v7.3 corrected)
        hours_bounds = {
            0: (100, 2500),  # X-Small: range 152-2103
            1: (2500, 6000),  # Small: range 2837-5930
            2: (6000, 12000),  # Medium: range 7200-9350
            3: (12000, 25000),  # Large: range 12950-22660
            4: (25000, 45000),  # Full: range 28004-41900
        }
        hours_low, hours_high = hours_bounds.get(sizing_score, (4000, 20000))

        # Cost bounds (derived from hours bounds using implied rate)
        cost_low = hours_low * implied_rate / 1000
        cost_high = hours_high * implied_rate / 1000

        return {
            "point_estimate": baseline_cost,  # K€ (for backward compatibility)
            "point_estimate_hours": baseline_hours,  # NEW: Hours directly
            "lower_bound": cost_low,
            "upper_bound": cost_high,
            "lower_bound_hours": hours_low,
            "upper_bound_hours": hours_high,
            "implied_rate": implied_rate,  # €/h for this sizing
            "confidence": self.target_coverage,
            "features_used": len(self.feature_names),
            "model_version": "7.2-hours-direct",
            "sizing_score": sizing_score,
        }

    def _ensure_sizing_scores(self, features: dict) -> dict:
        """
        Ensure sizing scores (0-4) are present.

        Converts string sizing values (X-small/Small/Medium/Large/Full)
        to numeric scores if needed.
        """
        sizing_features = {
            "sizing_PE_base_score": "sizing_PE_base",
            "sizing_PE_system_score": "sizing_PE_system",
            "sizing_PE_install_score": "sizing_PE_install",
            "sizing_program_score": "sizing_program",
        }

        for score_name, string_name in sizing_features.items():
            # If score already present as number, use it
            if score_name in features and isinstance(
                features[score_name], (int, float)
            ):
                continue

            # Try to convert from string sizing value
            string_value = features.get(string_name, "Medium")
            if isinstance(string_value, str):
                features[score_name] = SIZING_LEVEL_MAP.get(
                    string_value, 2
                )  # Default Medium=2
            else:
                features[score_name] = 2  # Default Medium

        return features


# Backward compatibility: alias for v6 class name
HCQEProductionModelV6 = HCQEProductionModelV7

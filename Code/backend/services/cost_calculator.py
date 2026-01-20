"""
FPT Cost Brain 2.0 - Cost Calculator Service
=============================================

Applies hourly rates to convert HCQE hours predictions into EUR cost.

CRITICAL: This is the ONLY place where rates are applied!
- HCQE model predicts HOURS only (no bench_rate in features)
- CostCalculator applies rates from ref_hourly_rates.json
- This separation prevents data leakage and allows independent updates

Usage:
    from services.cost_calculator import CostCalculator
    calculator = CostCalculator()
    cost_breakdown = calculator.calculate_cost(hours_breakdown, product_family)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdownItem:
    """Single cost breakdown item with hours and cost."""

    cluster: str
    activity: str
    hours: float
    rate_eur_h: float
    cost_eur: float
    rate_source: str  # e.g., "manpower:standard", "bench:development"


@dataclass
class CostCalculationResult:
    """Complete cost calculation result."""

    total_hours: float
    total_cost_eur: float
    total_cost_keur: float
    breakdown: list[CostBreakdownItem]
    product_family: str
    rates_version: str


# Cluster to rate category mapping
CLUSTER_RATE_MAPPING = {
    # Hardware-related clusters → standard manpower + bench
    "hardware": {"manpower": "standard", "bench": "development"},
    "turbo": {"manpower": "standard", "bench": "development"},
    "injector": {"manpower": "standard", "bench": "development"},
    "cooling": {"manpower": "standard", "bench": "development"},
    "egr": {"manpower": "standard", "bench": "development"},
    # Calibration → ATS function rate + bench
    "calibration": {"manpower": "ats", "bench": "development"},
    "ats": {"manpower": "ats", "bench": "development"},
    # Testing → testing rate + bench
    "testing": {"manpower": "testing", "bench": "durability"},
    "bench": {"manpower": "testing", "bench": "durability"},
    "vehicle": {"manpower": "testing", "bench": "vehicle_pems"},
    "field": {"manpower": "testing", "bench": "vehicle_pems"},
    "dataset": {"manpower": "testing", "bench": None},
    # Software → standard rate
    "software": {"manpower": "software", "bench": None},
    # Documentation → premium rate
    "documentation": {"manpower": "premium", "bench": None},
    # Installation → standard rate
    "installation": {"manpower": "standard", "bench": None},
    # Dependent (combined)
    "dependent": {"manpower": "standard", "bench": None},
}

# Function name to rate category mapping
FUNCTION_RATE_CATEGORIES = {
    "testing": 44,  # Testing / Endurance
    "standard": 59,  # Design, Engineering, etc.
    "ats": 59,  # Aftertreatment
    "software": 59,  # Control System & Software
    "premium": 89,  # Project Management, Tech Doc
    "prototype": 106,  # Prototype development
}


class CostCalculator:
    """
    Applies hourly rates to convert hours into EUR cost.

    CRITICAL: Rates are loaded from ref_hourly_rates.json and applied
    ONLY in this service - not in the ML model.
    """

    def __init__(self, rates_path: Path | str | None = None):
        """
        Initialize CostCalculator with rates from JSON file.

        Args:
            rates_path: Path to price_rate_db.json. If None, uses default location.
        """
        if rates_path is None:
            # Primary: Dataset/csv_exports/price_rate_db.json
            rates_path = (
                Path(__file__).parent.parent.parent.parent
                / "Dataset"
                / "csv_exports"
                / "price_rate_db.json"
            )
            # Fallback to v2 data location
            if not rates_path.exists():
                rates_path = (
                    Path(__file__).parent.parent
                    / "data"
                    / "knowledge"
                    / "price_rate_db.json"
                )

        self.rates_path = Path(rates_path)
        self._rates_data: dict[str, Any] = {}
        self._load_rates()

    def _load_rates(self) -> None:
        """Load rates from price_rate_db.json file."""
        if not self.rates_path.exists():
            logger.warning(f"Rates file not found at {self.rates_path}, using defaults")
            self._rates_data = self._get_default_rates()
            return

        try:
            with open(self.rates_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            # Convert price_rate_db.json format to internal format
            self._rates_data = self._convert_price_rate_db(raw_data)
            logger.info(f"Loaded rates from {self.rates_path}")
        except Exception as e:
            logger.error(f"Failed to load rates: {e}, using defaults")
            self._rates_data = self._get_default_rates()

    def _convert_price_rate_db(self, raw_data: dict) -> dict:
        """Convert price_rate_db.json format to internal format."""
        # Build manpower rates lookup
        manpower_rates = {}
        for item in raw_data.get("manpower_hourly_rates_by_function", []):
            func = item.get("function", "")
            rate = item.get("rate")
            if rate is not None:
                manpower_rates[func] = {"function": func, "rate_eur_h": rate}

        # Build bench rates by product family
        bench_rates = {}
        for item in raw_data.get("bench_rates_place_1", []):
            pf = item.get("product_family", "")
            bench_rates[pf] = {
                "durability": item.get("durability"),
                "development": item.get("development"),
                "vehicle_pems": item.get("vehicle_pems"),
                "special_rigs": item.get("special_rigs"),
            }

        return {
            "version": "price_rate_db",
            "manpower_rates": manpower_rates,
            "bench_rates_by_product_family": bench_rates,
        }

    def _get_default_rates(self) -> dict[str, Any]:
        """Return default rates if JSON file not available."""
        return {
            "version": "default",
            "manpower_rates": {
                "1": {"function": "Project Management", "rate_eur_h": 89},
                "5": {
                    "function": "Aftertreatment(ATS), Mat & Fluids",
                    "rate_eur_h": 59,
                },
                "6": {"function": "Control System & Software", "rate_eur_h": 59},
                "9": {"function": "Testing / Endurance", "rate_eur_h": 44},
            },
            "bench_rates_by_product_family": {
                "E5F0": {"durability": 146, "development": 394, "vehicle_pems": 107},
                "E8S0": {"durability": 146, "development": 394, "vehicle_pems": 107},
                "E0N0": {"durability": 163, "development": 404, "vehicle_pems": 107},
                "E0V0": {"durability": 302, "development": 498, "vehicle_pems": 107},
                "E0C0": {"durability": 217, "development": 441, "vehicle_pems": 107},
            },
        }

    def get_manpower_rate(self, category: str) -> float:
        """
        Get manpower hourly rate by category.

        Args:
            category: One of 'testing', 'standard', 'ats', 'software', 'premium', 'prototype'

        Returns:
            Hourly rate in EUR
        """
        # Use predefined category rates
        if category in FUNCTION_RATE_CATEGORIES:
            return FUNCTION_RATE_CATEGORIES[category]

        # Search in manpower_rates by function name
        manpower_rates = self._rates_data.get("manpower_rates", {})
        for key, data in manpower_rates.items():
            if isinstance(data, dict):
                func = data.get("function", "").lower()
                if category.lower() in func:
                    rate = data.get("rate_eur_h")
                    if rate is not None:
                        return rate

        # Default standard rate
        return 59.0

    def get_bench_rate(self, bench_type: str, product_family: str) -> float:
        """
        Get bench hourly rate by type and product family.

        Args:
            bench_type: One of 'durability', 'development', 'vehicle_pems', 'special_rigs'
            product_family: Product family code (E5F0, E8S0, E0N0, E0V0, E0C0)

        Returns:
            Hourly rate in EUR
        """
        bench_rates = self._rates_data.get("bench_rates_by_product_family", {})

        # Normalize product family
        pf = product_family.upper()
        if pf not in bench_rates:
            # Try to find matching family
            for known_pf in bench_rates.keys():
                if known_pf in pf or pf in known_pf:
                    pf = known_pf
                    break
            else:
                pf = "E5F0"  # Default

        family_rates = bench_rates.get(pf, {})
        rate = family_rates.get(bench_type)

        if rate is None:
            # Default development rate
            return bench_rates.get("E5F0", {}).get("development", 394)

        return rate

    def calculate_cost(
        self,
        breakdown: list[dict[str, Any]],
        product_family: str = "E5F0",
        bench_hours_ratio: float = 0.3,
    ) -> CostCalculationResult:
        """
        Calculate total cost from hours breakdown.

        CRITICAL: This is where rates are applied - ONLY HERE, not in ML model!

        Args:
            breakdown: List of dicts with 'cluster', 'activity', 'hours'
            product_family: Product family for bench rates
            bench_hours_ratio: Ratio of hours that are bench time (default 30%)

        Returns:
            CostCalculationResult with breakdown and totals
        """
        cost_items: list[CostBreakdownItem] = []
        total_hours = 0.0
        total_cost = 0.0

        for item in breakdown:
            cluster = item.get("cluster", "dependent").lower()
            activity = item.get("activity", "Unknown")
            hours = float(item.get("hours", 0))

            if hours <= 0:
                continue

            # Get rate mapping for this cluster
            rate_mapping = CLUSTER_RATE_MAPPING.get(
                cluster, {"manpower": "standard", "bench": None}
            )

            # Calculate manpower cost
            manpower_category = rate_mapping.get("manpower", "standard")
            manpower_rate = self.get_manpower_rate(manpower_category)

            # Determine bench usage
            bench_type = rate_mapping.get("bench")
            if bench_type:
                # Split hours between manpower and bench
                manpower_hours = hours * (1 - bench_hours_ratio)
                bench_hours = hours * bench_hours_ratio
                bench_rate = self.get_bench_rate(bench_type, product_family)

                # Calculate costs
                manpower_cost = manpower_hours * manpower_rate
                bench_cost = bench_hours * bench_rate
                item_cost = manpower_cost + bench_cost
                rate_source = f"manpower:{manpower_category}+bench:{bench_type}"
                effective_rate = item_cost / hours if hours > 0 else 0
            else:
                # Manpower only
                item_cost = hours * manpower_rate
                rate_source = f"manpower:{manpower_category}"
                effective_rate = manpower_rate

            cost_items.append(
                CostBreakdownItem(
                    cluster=cluster,
                    activity=activity,
                    hours=hours,
                    rate_eur_h=effective_rate,
                    cost_eur=item_cost,
                    rate_source=rate_source,
                )
            )

            total_hours += hours
            total_cost += item_cost

        return CostCalculationResult(
            total_hours=total_hours,
            total_cost_eur=total_cost,
            total_cost_keur=total_cost / 1000,
            breakdown=cost_items,
            product_family=product_family,
            rates_version=self._rates_data.get("version", "unknown"),
        )

    def get_average_rate(self, product_family: str = "E5F0") -> float:
        """
        Get weighted average hourly rate for quick estimates.

        Based on typical cost distribution:
        - 70% manpower at standard rate
        - 30% bench at development rate

        Args:
            product_family: Product family for bench rate

        Returns:
            Weighted average EUR/h
        """
        manpower_rate = self.get_manpower_rate("standard")
        bench_rate = self.get_bench_rate("development", product_family)

        # Typical weight: 70% manpower, 30% bench
        return 0.7 * manpower_rate + 0.3 * bench_rate


# Singleton instance
_cost_calculator: CostCalculator | None = None


def get_cost_calculator() -> CostCalculator:
    """Get singleton CostCalculator instance."""
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator()
    return _cost_calculator

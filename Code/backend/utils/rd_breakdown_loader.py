"""
FPT Cost Brain 2.0 - R&D Breakdown Data Loader
==============================================
Loads and parses PE02 function-level hour breakdown from db_RandD_output.csv.
Used to enrich Similar PR retrieval with detailed CBR context.

PE02 Functions (20):
1. Project Management
2. Cost Engineering
3. Design
4. Basic technologies, Simulation, Virtual Validation
5. Aftertreatment(ATS), Mat & Fluids
6. Control System & Software (CS&SW; EMS)
7. OBD & Diagnostics
8. CP&E; Dev&Rel
9. Testing / Endurance - Engine & ATS
10. Application Engineering
11. Vehicle
13. Technical Certification
14. Prototype
15. Materials & Travels
16. Laboratories
17. Contracts / Fees - Supplier_B
18. Contracts / Fees - Other Suppliers
19. Others (Travels, Dataloggers, contingencies)
20. TOTAL

Author: FPT Cost Brain Research Team
"""

import csv
import json
from pathlib import Path
from typing import Any

# Standard PE02 function IDs and their canonical names
PE02_FUNCTIONS = {
    1: "Project Management",
    2: "Cost Engineering",
    3: "Design",
    4: "Basic Technologies & Simulation",
    5: "Aftertreatment (ATS) & Materials",
    6: "Control System & Software",
    7: "OBD & Diagnostics",
    8: "CP&E Development & Release",
    9: "Testing / Endurance",
    10: "Application Engineering",
    11: "Vehicle",
    13: "Technical Certification",
    14: "Prototype",
    15: "Materials & Travels",
    16: "Laboratories",
    17: "Contracts - Supplier B",
    18: "Contracts - Other Suppliers",
    19: "Others",
    20: "TOTAL",
}


class FunctionBreakdown:
    """Breakdown of hours/cost for a single PE02 function."""

    def __init__(
        self,
        function_id: int,
        function_name: str,
        activities: str = "",
        manpower_hrs: float = 0.0,
        bench_durability_hrs: float = 0.0,
        bench_development_hrs: float = 0.0,
        bench_special_hrs: float = 0.0,
        vehicle_hrs: float = 0.0,
        cost_keur: float = 0.0,
    ):
        self.function_id = function_id
        self.function_name = function_name
        self.activities = activities
        self.manpower_hrs = manpower_hrs
        self.bench_durability_hrs = bench_durability_hrs
        self.bench_development_hrs = bench_development_hrs
        self.bench_special_hrs = bench_special_hrs
        self.vehicle_hrs = vehicle_hrs
        self.cost_keur = cost_keur

    @property
    def total_hours(self) -> float:
        """Total hours across all categories."""
        return (
            self.manpower_hrs
            + self.bench_durability_hrs
            + self.bench_development_hrs
            + self.bench_special_hrs
            + self.vehicle_hrs
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "function_id": self.function_id,
            "function_name": self.function_name,
            "activities": self.activities,
            "manpower_hrs": self.manpower_hrs,
            "bench_durability_hrs": self.bench_durability_hrs,
            "bench_development_hrs": self.bench_development_hrs,
            "bench_special_hrs": self.bench_special_hrs,
            "vehicle_hrs": self.vehicle_hrs,
            "total_hrs": self.total_hours,
            "cost_keur": self.cost_keur,
        }

    def __repr__(self) -> str:
        return (
            f"FunctionBreakdown({self.function_id}: {self.function_name}, "
            f"{self.total_hours:.0f}h, {self.cost_keur:.1f}k€)"
        )


class PRBreakdown:
    """Complete PE02 breakdown for a single PR."""

    def __init__(self, pr_id: str):
        self.pr_id = pr_id
        self.functions: dict[int, FunctionBreakdown] = {}

    def add_function(self, func: FunctionBreakdown) -> None:
        """Add or merge a function breakdown."""
        if func.function_id in self.functions:
            # Merge: sum hours and costs, concatenate activities
            existing = self.functions[func.function_id]
            existing.manpower_hrs += func.manpower_hrs
            existing.bench_durability_hrs += func.bench_durability_hrs
            existing.bench_development_hrs += func.bench_development_hrs
            existing.bench_special_hrs += func.bench_special_hrs
            existing.vehicle_hrs += func.vehicle_hrs
            existing.cost_keur += func.cost_keur
            if func.activities and func.activities not in existing.activities:
                sep = "; " if existing.activities else ""
                existing.activities += f"{sep}{func.activities}"
        else:
            self.functions[func.function_id] = func

    @property
    def total_manpower_hrs(self) -> float:
        """Total manpower hours excluding TOTAL function."""
        return sum(f.manpower_hrs for fid, f in self.functions.items() if fid != 20)

    @property
    def total_cost_keur(self) -> float:
        """Total cost from TOTAL function or sum of all."""
        if 20 in self.functions:
            return self.functions[20].cost_keur
        return sum(f.cost_keur for f in self.functions.values())

    def get_top_functions(
        self, n: int = 5, by: str = "cost"
    ) -> list[FunctionBreakdown]:
        """Get top N functions by cost or hours."""
        funcs = [f for fid, f in self.functions.items() if fid != 20]  # Exclude TOTAL
        if by == "cost":
            funcs.sort(key=lambda x: x.cost_keur, reverse=True)
        else:
            funcs.sort(key=lambda x: x.total_hours, reverse=True)
        return funcs[:n]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pr_id": self.pr_id,
            "total_manpower_hrs": self.total_manpower_hrs,
            "total_cost_keur": self.total_cost_keur,
            "functions": {
                fid: f.to_dict() for fid, f in sorted(self.functions.items())
            },
        }

    def to_context_string(self, top_n: int = 5) -> str:
        """
        Generate concise context string for LLM.

        Format:
        PR 18094_D (731k€, 3226h):
        - OBD & Diagnostics: 1350h, 71.6k€ (Specific ATS diagnosis development)
        - Application Engineering: 976h, 55.7k€ (QG development and releases)
        ...
        """
        lines = [
            f"PR {self.pr_id} ({self.total_cost_keur:.0f}k€, {self.total_manpower_hrs:.0f}h):"
        ]

        top_funcs = self.get_top_functions(n=top_n, by="cost")
        for func in top_funcs:
            if func.cost_keur > 0 or func.total_hours > 0:
                activity_hint = ""
                if func.activities:
                    # Truncate long activities
                    activity_text = func.activities[:80]
                    if len(func.activities) > 80:
                        activity_text += "..."
                    activity_hint = f" ({activity_text})"
                lines.append(
                    f"  - {func.function_name}: {func.total_hours:.0f}h, "
                    f"{func.cost_keur:.1f}k€{activity_hint}"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PRBreakdown({self.pr_id}: {len(self.functions)} functions, {self.total_cost_keur:.0f}k€)"


def _parse_float(value: str) -> float:
    """Parse float handling European comma format and empty values."""
    if not value or value.strip() == "":
        return 0.0
    # Handle European comma decimal separator
    value = value.replace(",", ".").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_rd_breakdown(csv_path: str | Path | None = None) -> dict[str, PRBreakdown]:
    """
    Load R&D breakdown data from CSV file.

    Args:
        csv_path: Path to db_RandD_output.csv. If None, uses default location.

    Returns:
        Dictionary mapping PR ID to PRBreakdown object.
    """
    if csv_path is None:
        # Default path relative to backend
        csv_path = (
            Path(__file__).parent.parent.parent.parent
            / "Dataset"
            / "csv_exports"
            / "db_RandD_output.csv"
        )

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    breakdowns: dict[str, PRBreakdown] = {}

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pr_id = row.get("PR id", "").strip()
            if not pr_id:
                continue

            # Get or create PR breakdown
            if pr_id not in breakdowns:
                breakdowns[pr_id] = PRBreakdown(pr_id)

            # Parse function data
            try:
                function_id = int(row.get("Function ID", "0") or "0")
            except ValueError:
                continue

            if function_id == 0:
                continue

            function_name = row.get("Function Description", "").strip()
            if not function_name:
                function_name = PE02_FUNCTIONS.get(
                    function_id, f"Function {function_id}"
                )

            func = FunctionBreakdown(
                function_id=function_id,
                function_name=function_name,
                activities=row.get("Main Activities Description", "").strip(),
                manpower_hrs=_parse_float(row.get("Manpower hrs", "")),
                bench_durability_hrs=_parse_float(row.get("Bench Durability hrs", "")),
                bench_development_hrs=_parse_float(
                    row.get("Bench Development hrs", "")
                ),
                bench_special_hrs=_parse_float(
                    row.get("Bench Special (Baro-climatic chamber, NVH) hrs", "")
                ),
                vehicle_hrs=_parse_float(row.get("Vehicle hrs", "")),
                cost_keur=_parse_float(row.get("Cost k€", "")),
            )

            breakdowns[pr_id].add_function(func)

    return breakdowns


def get_breakdown_for_pr(
    pr_id: str, breakdowns: dict[str, PRBreakdown] | None = None
) -> PRBreakdown | None:
    """
    Get breakdown for a specific PR.

    Args:
        pr_id: The PR identifier (e.g., "18094_D")
        breakdowns: Pre-loaded breakdowns dict, or None to load fresh.

    Returns:
        PRBreakdown object or None if not found.
    """
    if breakdowns is None:
        breakdowns = load_rd_breakdown()

    # Try exact match first
    if pr_id in breakdowns:
        return breakdowns[pr_id]

    # Try with common suffixes (_A, _B, _C, etc.)
    base_id = pr_id.rsplit("_", 1)[0] if "_" in pr_id else pr_id
    for suffix in ["_A", "_B", "_C", "_D", "_E", "_F"]:
        key = f"{base_id}{suffix}"
        if key in breakdowns:
            return breakdowns[key]

    return None


def export_breakdowns_to_json(output_path: str | Path | None = None) -> Path:
    """
    Export all breakdowns to JSON for embedding payload.

    Args:
        output_path: Output JSON path. If None, uses default location.

    Returns:
        Path to the exported JSON file.
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent.parent
            / "data_prepared"
            / "rag_history"
            / "pr_rd_breakdown.json"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    breakdowns = load_rd_breakdown()

    export_data = {pr_id: bd.to_dict() for pr_id, bd in breakdowns.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    return output_path


# Pre-loaded breakdowns cache (lazy initialization)
_BREAKDOWN_CACHE: dict[str, PRBreakdown] | None = None


def get_cached_breakdowns() -> dict[str, PRBreakdown]:
    """Get cached breakdowns (loads once on first call)."""
    global _BREAKDOWN_CACHE
    if _BREAKDOWN_CACHE is None:
        _BREAKDOWN_CACHE = load_rd_breakdown()
    return _BREAKDOWN_CACHE


def generate_cbr_context(
    similar_prs: list[dict[str, Any]], top_n_functions: int = 5
) -> str:
    """
    Generate Case-Based Reasoning (CBR) context string for LLM estimation prompts.

    Extracts R&D breakdown from similar PRs and formats it as a reference for estimation.

    Args:
        similar_prs: List of SimilarPR dictionaries with rd_breakdown field
        top_n_functions: Number of top functions to include per PR

    Returns:
        Formatted context string for LLM prompt injection
    """
    if not similar_prs:
        return "No similar historical PRs found for reference."

    lines = ["## Historical Similar PRs - R&D Breakdown Reference\n"]

    prs_with_breakdown = [sp for sp in similar_prs if sp.get("rd_breakdown")]

    if not prs_with_breakdown:
        # Fallback: show basic cost info even without breakdown
        lines.append("Note: Detailed function breakdown not available for these PRs.\n")
        for sp in similar_prs[:3]:
            pr_code = sp.get("pr_code", "Unknown")
            total_cost = sp.get("total_cost_eur", 0)
            total_hours = sp.get("total_hours", 0)
            similarity = sp.get("similarity_score", 0)
            sector = sp.get("sector", "Unknown")
            sizing = sp.get("sizing", "Unknown")
            lines.append(
                f"- {pr_code}: {total_cost:.0f}k€, {total_hours:.0f}h "
                f"(Sector: {sector}, Sizing: {sizing}, Similarity: {similarity:.0%})"
            )
        return "\n".join(lines)

    # Generate detailed breakdown context
    for i, sp in enumerate(prs_with_breakdown[:3], 1):
        pr_code = sp.get("pr_code", "Unknown")
        breakdown = sp.get("rd_breakdown", {})
        total_cost = breakdown.get("total_cost_keur", sp.get("total_cost_eur", 0))
        total_hours = breakdown.get("total_manpower_hrs", sp.get("total_hours", 0))
        similarity = sp.get("similarity_score", 0)
        sector = sp.get("sector", "Unknown")
        sizing = sp.get("sizing", "Unknown")

        lines.append(f"### {i}. {pr_code} (Similarity: {similarity:.0%})")
        lines.append(f"- Sector: {sector}, Sizing: {sizing}")
        lines.append(f"- Total: {total_cost:.0f}k€, {total_hours:.0f}h")
        lines.append("- Function breakdown:")

        functions = breakdown.get("functions", [])
        for func in functions[:top_n_functions]:
            func_name = func.get("name", "Unknown")
            func_hours = func.get("hours", 0)
            func_cost = func.get("cost_keur", 0)
            activities = func.get("activities", "")
            activity_hint = (
                f" - {activities[:60]}..."
                if len(activities) > 60
                else (f" - {activities}" if activities else "")
            )
            lines.append(
                f"    • {func_name}: {func_hours:.0f}h, {func_cost:.1f}k€{activity_hint}"
            )

        lines.append("")

    # Add guidance note
    lines.append("**Estimation Guidance:**")
    lines.append(
        "Use these similar PRs as reference baselines. Adjust hours/costs based on:"
    )
    lines.append("- Scope differences (more/fewer components)")
    lines.append("- Complexity variations (new tech vs. incremental)")
    lines.append("- Certification requirements (China NRIV, Stage V, etc.)")

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    breakdowns = load_rd_breakdown()
    print(f"Loaded {len(breakdowns)} PR breakdowns")

    # Show sample
    for pr_id, bd in list(breakdowns.items())[:3]:
        print(f"\n{bd}")
        print(bd.to_context_string(top_n=3))

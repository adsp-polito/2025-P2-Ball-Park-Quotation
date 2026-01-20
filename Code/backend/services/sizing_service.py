"""
FPT Cost Brain - Rule-Based Sizing Service

Strict rule-based classifier that:
1. Finds matching rule from 45 rules in ref_sizing.json
2. Returns sizing + reasoning with specific rule_id
3. Uses LLM for rule selection, keyword matching as fallback

Architecture:
    PR Text → SizingService.classify_sizing() → Match rules → Return SizingResult
                                                      ↓
                                                3 strategies:
                                                1. LLM selects rule_id (primary)
                                                2. Keyword matching (fallback)
                                                3. Default = Medium (ultimate fallback)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================


class SizingLevel(str, Enum):
    """Project sizing classification (from ref_Sizing)."""

    FULL = "Full"
    LARGE = "Large"
    MEDIUM = "Medium"
    SMALL = "Small"
    X_SMALL = "X-small"


class SizingMethod(str, Enum):
    """Method used for sizing classification."""

    LLM = "llm"
    KEYWORD = "keyword"
    DEFAULT = "default"


# Domain mapping from ref_sizing.json to state field names
DOMAIN_MAPPING = {
    ("Product Engineering", "Base Engine"): "pe_base_powertrain",
    ("Product Engineering", "System (engine+ATS)"): "pe_system_assembly",
    (
        "Product Engineering",
        "Installation/ Application / Homologation",
    ): "pe_installation_application",
    ("Manufacturing", "Plant - base engine"): "manufacturing_base_engine",
    ("Manufacturing", "Plant - ATS"): "manufacturing_ats",
    ("Purchasing", "Sourcing"): "purchasing_sourcing",
    ("Purchasing", "Supplier Quality"): "purchasing_supplier_quality",
    ("Customer Manager", "Build stages"): "customer_build_stages",
    ("Program Manager", "Overall"): "program_manager_overall",
}

# Reverse mapping for rule ID generation
DOMAIN_PREFIXES = {
    "pe_base_powertrain": "PE_BASE",
    "pe_system_assembly": "PE_SYS",
    "pe_installation_application": "PE_INST",
    "manufacturing_base_engine": "MFG_BASE",
    "manufacturing_ats": "MFG_ATS",
    "purchasing_sourcing": "PUR_SRC",
    "purchasing_supplier_quality": "PUR_SQ",
    "customer_build_stages": "CM_BUILD",
    "program_manager_overall": "PM_OVR",
}

# Sizing level abbreviations for rule IDs
SIZING_ABBREV = {
    SizingLevel.FULL: "F",
    SizingLevel.LARGE: "L",
    SizingLevel.MEDIUM: "M",
    SizingLevel.SMALL: "S",
    SizingLevel.X_SMALL: "XS",
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SizingRule:
    """A single sizing rule from ref_sizing.json."""

    rule_id: str  # e.g., PE_BASE_F_001
    function: str  # e.g., Product Engineering
    sub_function: str  # e.g., Base Engine
    development_effort: str  # Full description
    sizing: SizingLevel
    domain_key: str  # e.g., pe_base_powertrain
    keywords: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Extract keywords from development_effort if not provided."""
        if not self.keywords:
            self.keywords = self._extract_keywords()

    def _extract_keywords(self) -> list[str]:
        """Extract searchable keywords from development_effort description."""
        text = self.development_effort.lower()

        # Common keyword patterns for sizing classification
        keyword_patterns = {
            # Full-level indicators
            "new concept": ["new concept", "first installation", "new serviceability"],
            "high level": ["high level", "high validation", "high nc"],
            "all build": ["all build stages", "alpha"],
            # Large-level indicators
            "heavy modification": [
                "heavy modification",
                "manufacturing impact",
                "high/medium",
            ],
            "beta gamma": ["beta", "gamma, pp"],
            # Medium-level indicators
            "medium modification": [
                "medium modification",
                "medium level",
                "no manufacturing impact",
            ],
            "medium cals": ["medium cals", "medium installation"],
            # Small-level indicators
            "light modification": [
                "light modification",
                "low level",
                "low nc",
                "limited",
            ],
            "pp pilot": ["pp, pilot", "pp pilot"],
            # X-Small-level indicators
            "minimum": ["minimum", "only adaptation", "only pilot", "no validation"],
        }

        extracted = []
        for category, patterns in keyword_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    extracted.append(pattern)

        # Also extract standalone important words
        important_words = [
            "new",
            "high",
            "medium",
            "low",
            "minimum",
            "heavy",
            "light",
            "all",
            "only",
            "beta",
            "gamma",
            "pilot",
            "pp",
            "alpha",
            "rgt",
            "homologation",
            "calibration",
            "cals",
            "tooling",
            "apqp",
            "sourcing",
        ]

        words = re.findall(r"\b\w+\b", text)
        for word in words:
            if word in important_words and word not in extracted:
                extracted.append(word)

        return extracted


@dataclass
class SizingResult:
    """Result of sizing classification for a single domain."""

    sizing: str  # Full, Large, Medium, Small, X-small
    reasoning: str  # Human-readable explanation
    rule_id: str  # Reference to applied rule
    confidence: float  # 0.0-1.0
    method: str  # llm, keyword, default

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sizing": self.sizing,
            "reasoning": self.reasoning,
            "rule_id": self.rule_id,
            "confidence": self.confidence,
            "method": self.method,
        }


@dataclass
class ProgramSizingResult:
    """Complete program sizing across all 9 domains + aggregated."""

    # Product Engineering (3)
    pe_base_powertrain: SizingResult
    pe_system_assembly: SizingResult
    pe_installation_application: SizingResult

    # Manufacturing (2)
    manufacturing_base_engine: SizingResult
    manufacturing_ats: SizingResult

    # Purchasing (2)
    purchasing_sourcing: SizingResult
    purchasing_supplier_quality: SizingResult

    # Customer Manager (1)
    customer_build_stages: SizingResult

    # Program Manager (1)
    program_manager_overall: SizingResult

    # Aggregated (max of all)
    program_overall: SizingResult

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching ProgramSizingExtended TypedDict."""
        return {
            "pe_base_powertrain": self.pe_base_powertrain.to_dict(),
            "pe_system_assembly": self.pe_system_assembly.to_dict(),
            "pe_installation_application": self.pe_installation_application.to_dict(),
            "manufacturing_base_engine": self.manufacturing_base_engine.to_dict(),
            "manufacturing_ats": self.manufacturing_ats.to_dict(),
            "purchasing_sourcing": self.purchasing_sourcing.to_dict(),
            "purchasing_supplier_quality": self.purchasing_supplier_quality.to_dict(),
            "customer_build_stages": self.customer_build_stages.to_dict(),
            "program_manager_overall": self.program_manager_overall.to_dict(),
            "program_overall": self.program_overall.to_dict(),
        }


# ============================================================================
# SIZING SERVICE
# ============================================================================


class SizingService:
    """
    Rule-based sizing classifier using ref_sizing.json.

    Primary strategy: LLM selects rule_id from list of rules
    Fallback: Keyword matching
    Ultimate fallback: Default to Medium
    """

    def __init__(self, rules_path: Optional[str] = None):
        """
        Initialize SizingService.

        Args:
            rules_path: Path to ref_sizing.json. If None, uses default location.
        """
        self.rules_path = rules_path or self._get_default_rules_path()
        self.rules: list[SizingRule] = []
        self.rules_by_domain: dict[str, list[SizingRule]] = {}
        self.rules_by_id: dict[str, SizingRule] = {}
        self._load_rules()

    def _get_default_rules_path(self) -> str:
        """Get default path to ref_sizing.json."""
        # Try multiple paths
        possible_paths = [
            Path(__file__).parent.parent.parent.parent
            / "data_prepared"
            / "rag_knowledge"
            / "ref_sizing.json",
            Path(__file__).parent.parent / "data" / "knowledge" / "ref_sizing.json",
            Path("/Users/tenxengineer/2_Projects/4. University/02_FINAL/2.ADSP/PROJECT")
            / "data_prepared"
            / "rag_knowledge"
            / "ref_sizing.json",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Fallback to first path
        return str(possible_paths[0])

    def _load_rules(self) -> None:
        """Load and index sizing rules from ref_sizing.json."""
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sizing_rules = data.get("sizing_rules", [])
            rule_counters: dict[str, int] = {}

            for rule_data in sizing_rules:
                function = rule_data.get("function", "")
                sub_function = rule_data.get("sub_function", "")
                sizing_str = rule_data.get("sizing", "Medium")
                development_effort = rule_data.get("development_effort", "")

                # Get domain key
                domain_key = DOMAIN_MAPPING.get(
                    (function, sub_function), "program_manager_overall"
                )

                # Parse sizing level
                sizing = self._parse_sizing_level(sizing_str)

                # Generate unique rule ID
                rule_id = self._generate_rule_id(domain_key, sizing, rule_counters)

                rule = SizingRule(
                    rule_id=rule_id,
                    function=function,
                    sub_function=sub_function,
                    development_effort=development_effort,
                    sizing=sizing,
                    domain_key=domain_key,
                )

                self.rules.append(rule)
                self.rules_by_id[rule_id] = rule

                if domain_key not in self.rules_by_domain:
                    self.rules_by_domain[domain_key] = []
                self.rules_by_domain[domain_key].append(rule)

            logger.info(f"Loaded {len(self.rules)} sizing rules from {self.rules_path}")

        except FileNotFoundError:
            logger.warning(
                f"Rules file not found: {self.rules_path}. Using empty rules."
            )
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rules JSON: {e}")

    def _parse_sizing_level(self, sizing_str: str) -> SizingLevel:
        """Parse sizing string to SizingLevel enum."""
        sizing_map = {
            "full": SizingLevel.FULL,
            "large": SizingLevel.LARGE,
            "medium": SizingLevel.MEDIUM,
            "mid": SizingLevel.MEDIUM,
            "small": SizingLevel.SMALL,
            "x-small": SizingLevel.X_SMALL,
            "xsmall": SizingLevel.X_SMALL,
        }
        return sizing_map.get(sizing_str.lower().strip(), SizingLevel.MEDIUM)

    def _generate_rule_id(
        self, domain_key: str, sizing: SizingLevel, counters: dict[str, int]
    ) -> str:
        """Generate unique rule ID."""
        prefix = DOMAIN_PREFIXES.get(domain_key, "UNK")
        sizing_abbrev = SIZING_ABBREV.get(sizing, "M")

        # Create counter key
        counter_key = f"{prefix}_{sizing_abbrev}"
        counters[counter_key] = counters.get(counter_key, 0) + 1

        return f"{prefix}_{sizing_abbrev}_{counters[counter_key]:03d}"

    async def classify_sizing(
        self,
        pr_text: str,
        parsed_pr: dict[str, Any],
        llm=None,
    ) -> ProgramSizingResult:
        """
        Classify program sizing across all 9 domains.

        Args:
            pr_text: Full PR text for analysis
            parsed_pr: Parsed PR data dictionary
            llm: Optional LLM client for rule selection

        Returns:
            ProgramSizingResult with sizing for each domain
        """
        domain_results: dict[str, SizingResult] = {}

        # Classify each domain
        for domain_key in DOMAIN_PREFIXES.keys():
            domain_rules = self.rules_by_domain.get(domain_key, [])

            if not domain_rules:
                # No rules for this domain, use default
                domain_results[domain_key] = self._create_default_result(domain_key)
                continue

            # Try LLM-based classification first
            if llm is not None:
                result = await self._classify_domain_with_llm(
                    domain_key, domain_rules, pr_text, parsed_pr, llm
                )
                if result is not None:
                    domain_results[domain_key] = result
                    continue

            # TRY FEATURE-BASED HEURISTIC FIRST (uses parsed_pr data)
            # This is more reliable than keywords when PR data is available
            result = self._classify_with_pr_features(
                domain_key, domain_rules, parsed_pr
            )
            if result is not None:
                domain_results[domain_key] = result
                continue

            # Fallback to keyword matching
            result = self._classify_domain_with_keywords(
                domain_key, domain_rules, pr_text, parsed_pr
            )
            domain_results[domain_key] = result

        # Calculate aggregated sizing (max of all)
        program_overall = self._aggregate_sizing(domain_results)

        return ProgramSizingResult(
            pe_base_powertrain=domain_results.get(
                "pe_base_powertrain", self._create_default_result("pe_base_powertrain")
            ),
            pe_system_assembly=domain_results.get(
                "pe_system_assembly", self._create_default_result("pe_system_assembly")
            ),
            pe_installation_application=domain_results.get(
                "pe_installation_application",
                self._create_default_result("pe_installation_application"),
            ),
            manufacturing_base_engine=domain_results.get(
                "manufacturing_base_engine",
                self._create_default_result("manufacturing_base_engine"),
            ),
            manufacturing_ats=domain_results.get(
                "manufacturing_ats", self._create_default_result("manufacturing_ats")
            ),
            purchasing_sourcing=domain_results.get(
                "purchasing_sourcing",
                self._create_default_result("purchasing_sourcing"),
            ),
            purchasing_supplier_quality=domain_results.get(
                "purchasing_supplier_quality",
                self._create_default_result("purchasing_supplier_quality"),
            ),
            customer_build_stages=domain_results.get(
                "customer_build_stages",
                self._create_default_result("customer_build_stages"),
            ),
            program_manager_overall=domain_results.get(
                "program_manager_overall",
                self._create_default_result("program_manager_overall"),
            ),
            program_overall=program_overall,
        )

    async def _classify_domain_with_llm(
        self,
        domain_key: str,
        domain_rules: list[SizingRule],
        pr_text: str,
        parsed_pr: dict[str, Any],
        llm,
    ) -> Optional[SizingResult]:
        """
        Use LLM to select the best matching rule for a domain.

        Returns None if LLM classification fails.
        """
        from llm.prompts_sizing import SIZING_RULE_SELECTION_PROMPT

        try:
            # Format rules for prompt
            rules_list = self._format_rules_for_prompt(domain_rules)

            # Build prompt
            prompt = SIZING_RULE_SELECTION_PROMPT.format(
                domain_name=domain_key.replace("_", " ").title(),
                rules_list=rules_list,
                pr_text=pr_text[:3000],  # Limit text length
            )

            # Call LLM - use fast_response for rule selection (quick classification task)
            # Our LLMClient returns string directly, not LangChain AIMessage
            response_text = await llm.fast_response(
                prompt=prompt,
                system_prompt="You are an R&D sizing classification expert. Select the best matching rule and respond with valid JSON.",
            )

            # Parse JSON response
            result = self._parse_llm_response(response_text, domain_rules)
            if result:
                return result

        except Exception as e:
            logger.warning(f"LLM classification failed for {domain_key}: {e}")

        return None

    def _format_rules_for_prompt(self, rules: list[SizingRule]) -> str:
        """Format rules list for LLM prompt."""
        lines = []
        for rule in rules:
            lines.append(
                f"- **{rule.rule_id}** ({rule.sizing.value}): {rule.development_effort[:200]}"
            )
        return "\n".join(lines)

    def _parse_llm_response(
        self, response_text: str, domain_rules: list[SizingRule]
    ) -> Optional[SizingResult]:
        """Parse LLM JSON response to SizingResult."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group())

            selected_rule_id = data.get("selected_rule_id", "")
            confidence = float(data.get("confidence", 0.7))
            reasoning = data.get("reasoning", "")

            # Find the rule
            rule = self.rules_by_id.get(selected_rule_id)
            if rule is None:
                # Try fuzzy match
                for r in domain_rules:
                    if r.rule_id == selected_rule_id or r.sizing.value in reasoning:
                        rule = r
                        break

            if rule is None:
                return None

            return SizingResult(
                sizing=rule.sizing.value,
                reasoning=f"Rule {rule.rule_id}: {reasoning}",
                rule_id=rule.rule_id,
                confidence=confidence,
                method=SizingMethod.LLM.value,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse LLM response: {e}")
            return None

    def _classify_domain_with_keywords(
        self,
        domain_key: str,
        domain_rules: list[SizingRule],
        pr_text: str,
        parsed_pr: dict[str, Any],
    ) -> SizingResult:
        """
        Use keyword matching to select the best rule.

        Fallback method when LLM is unavailable or fails.

        IMPORTANT: When multiple rules have the same score, prefer SMALLER sizing
        (more conservative estimate). Also check for exclusive keywords like "only"
        which indicate minimal scope.
        """
        pr_text_lower = pr_text.lower()

        # Sizing priority order (smaller = higher priority when scores equal)
        sizing_priority = {
            SizingLevel.X_SMALL: 0,
            SizingLevel.SMALL: 1,
            SizingLevel.MEDIUM: 2,
            SizingLevel.LARGE: 3,
            SizingLevel.FULL: 4,
        }

        # Exclusive keywords that override to smaller sizing
        exclusive_small_keywords = ["only", "minimum", "minor", "light", "limited"]
        has_exclusive_small = any(
            kw in pr_text_lower for kw in exclusive_small_keywords
        )

        # Negative keywords that indicate LARGER scope
        large_scope_keywords = [
            "all build stages",
            "alpha",
            "new concept",
            "high level",
            "heavy modification",
        ]
        has_large_scope = any(kw in pr_text_lower for kw in large_scope_keywords)

        # Score each rule by keyword matches
        best_rule: Optional[SizingRule] = None
        best_score = 0
        best_priority = 999  # Lower = smaller sizing

        for rule in domain_rules:
            score = 0
            matched_keywords = []
            for keyword in rule.keywords:
                if keyword in pr_text_lower:
                    score += 1
                    matched_keywords.append(keyword)

            rule_priority = sizing_priority.get(rule.sizing, 2)

            # Determine if this rule is better
            is_better = False
            if score > best_score:
                is_better = True
            elif score == best_score and score > 0:
                # Same score - prefer smaller sizing (lower priority number)
                # UNLESS text has large scope keywords
                if has_large_scope and rule_priority > best_priority:
                    is_better = True  # Large scope → prefer larger sizing
                elif has_exclusive_small and rule_priority < best_priority:
                    is_better = True  # Exclusive small → prefer smaller sizing
                elif not has_large_scope and rule_priority < best_priority:
                    is_better = True  # Default: prefer smaller sizing

            if is_better:
                best_score = score
                best_rule = rule
                best_priority = rule_priority

        if best_rule and best_score > 0:
            # Boost confidence if exclusive keywords match the sizing direction
            confidence = min(0.8, 0.4 + best_score * 0.1)
            if has_exclusive_small and best_priority <= 1:  # X-Small or Small
                confidence = min(0.85, confidence + 0.1)

            return SizingResult(
                sizing=best_rule.sizing.value,
                reasoning=f"Rule {best_rule.rule_id}: Matched keywords: {', '.join(best_rule.keywords[:5])}",
                rule_id=best_rule.rule_id,
                confidence=confidence,
                method=SizingMethod.KEYWORD.value,
            )

        # No keyword matches, return default (Medium)
        return self._create_default_result(domain_key)

    def _classify_with_pr_features(
        self,
        domain_key: str,
        domain_rules: list[SizingRule],
        parsed_pr: dict[str, Any],
    ) -> Optional[SizingResult]:
        """
        Use parsed PR features for accurate sizing classification.

        This heuristic uses structured data from parsed_pr to determine sizing
        more reliably than keyword matching, especially for Homologation projects.

        Heuristic logic:
        1. BOM updates (no other changes) → X-Small
        2. Homologation + NO hardware + NO new engine → Small
        3. Homologation + hardware changes → Medium
        4. New engine → Large or Full
        5. Otherwise → None (fall back to keyword matching)

        Based on ref_sizing.json rule patterns and training data analysis.
        """
        is_homologation = parsed_pr.get("is_homologation", False)
        is_bom = parsed_pr.get("is_bom", False)
        is_new_engine = parsed_pr.get("is_new_engine", False)
        hardware_change = parsed_pr.get("hardware_change", False)
        ats_change = parsed_pr.get("ATS_change", False)
        calibration_change = parsed_pr.get("calibration_change", True)

        # Calculate change complexity score
        change_count = sum(
            [
                bool(hardware_change),
                bool(ats_change),
                bool(parsed_pr.get("software_VCU_change", False)),
                bool(calibration_change),
            ]
        )

        # Determine target sizing based on features
        target_sizing: Optional[SizingLevel] = None
        reasoning = ""
        confidence = 0.7  # Base confidence for feature-based classification

        # Rule 1: BOM-only updates are minimal
        if is_bom and not is_new_engine and not hardware_change:
            target_sizing = SizingLevel.X_SMALL
            reasoning = "BOM update with no hardware changes → X-Small"
            confidence = 0.85

        # Rule 2: Homologation without hardware = Small (certification only)
        elif is_homologation and not hardware_change and not is_new_engine:
            # Homologation without hardware changes = certification updates only
            # This matches "Low installation effort; Limited Cals Review; Homologation"
            target_sizing = SizingLevel.SMALL
            reasoning = (
                "Homologation without hardware changes → Small "
                "(certification/calibration update only)"
            )
            confidence = 0.80

        # Rule 3: Homologation WITH hardware = Medium or larger
        elif is_homologation and hardware_change:
            target_sizing = SizingLevel.MEDIUM
            reasoning = "Homologation with hardware changes → Medium"
            confidence = 0.75

        # Rule 4: New engine = Large or Full
        elif is_new_engine:
            if hardware_change and ats_change:
                target_sizing = SizingLevel.FULL
                reasoning = "New engine with full changes → Full"
                confidence = 0.80
            else:
                target_sizing = SizingLevel.LARGE
                reasoning = "New engine development → Large"
                confidence = 0.75

        # Rule 5: Minimal changes (calibration only) without homologation
        elif change_count <= 1 and calibration_change and not hardware_change:
            target_sizing = SizingLevel.X_SMALL
            reasoning = "Calibration-only change → X-Small"
            confidence = 0.70

        if target_sizing is None:
            return None  # Fall back to keyword matching

        # Find matching rule for this sizing in the domain
        matching_rule = None
        for rule in domain_rules:
            if rule.sizing == target_sizing:
                matching_rule = rule
                break

        if matching_rule:
            return SizingResult(
                sizing=target_sizing.value,
                reasoning=f"[Feature-based] {reasoning}. Rule: {matching_rule.rule_id}",
                rule_id=matching_rule.rule_id,
                confidence=confidence,
                method=SizingMethod.KEYWORD.value,  # Using KEYWORD to indicate non-LLM
            )

        # No matching rule found, but we have a sizing decision
        # Create a synthetic rule ID
        prefix = DOMAIN_PREFIXES.get(domain_key, "UNK")
        abbrev = SIZING_ABBREV.get(target_sizing, "M")
        synthetic_rule_id = f"{prefix}_{abbrev}_FEATURE"

        return SizingResult(
            sizing=target_sizing.value,
            reasoning=f"[Feature-based] {reasoning}",
            rule_id=synthetic_rule_id,
            confidence=confidence,
            method=SizingMethod.KEYWORD.value,
        )

    def _create_default_result(self, domain_key: str) -> SizingResult:
        """Create default Medium sizing result."""
        # Find Medium rule for this domain if exists
        domain_rules = self.rules_by_domain.get(domain_key, [])
        medium_rule = None
        for rule in domain_rules:
            if rule.sizing == SizingLevel.MEDIUM:
                medium_rule = rule
                break

        rule_id = (
            medium_rule.rule_id
            if medium_rule
            else f"{DOMAIN_PREFIXES.get(domain_key, 'UNK')}_M_DEFAULT"
        )

        return SizingResult(
            sizing=SizingLevel.MEDIUM.value,
            reasoning=f"Default sizing (Medium) applied for {domain_key.replace('_', ' ')}",
            rule_id=rule_id,
            confidence=0.5,
            method=SizingMethod.DEFAULT.value,
        )

    def _aggregate_sizing(
        self, domain_results: dict[str, SizingResult]
    ) -> SizingResult:
        """
        Aggregate all domain sizing into overall program sizing.

        STRATEGY: Use MODE (most frequent) sizing among explicit matches.
        This is more representative than MAX for R&D projects where
        one outlier domain shouldn't dominate the overall classification.

        Fallback order:
        1. MODE of explicit matches (LLM/keyword)
        2. MODE of all matches
        3. Default Medium
        """
        from collections import Counter

        sizing_order = [
            SizingLevel.X_SMALL.value,
            SizingLevel.SMALL.value,
            SizingLevel.MEDIUM.value,
            SizingLevel.LARGE.value,
            SizingLevel.FULL.value,
        ]

        # Separate explicit matches from defaults
        explicit_results = {
            k: v
            for k, v in domain_results.items()
            if v.method in [SizingMethod.LLM.value, SizingMethod.KEYWORD.value]
        }

        # Use explicit results if available, otherwise use all results
        results_to_aggregate = explicit_results if explicit_results else domain_results

        if not results_to_aggregate:
            return SizingResult(
                sizing=SizingLevel.MEDIUM.value,
                reasoning="No domain results available, using default Medium",
                rule_id="PROGRAM_OVERALL_DEFAULT",
                confidence=0.5,
                method="aggregation",
            )

        # Count sizing frequencies
        sizing_counts = Counter(r.sizing for r in results_to_aggregate.values())

        # Find MODE (most common sizing)
        most_common = sizing_counts.most_common()
        mode_sizing = most_common[0][0]
        mode_count = most_common[0][1]

        # If there's a tie, prefer smaller sizing (more conservative)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Multiple sizes with same frequency - pick smallest
            tied_sizings = [s for s, c in most_common if c == mode_count]
            try:
                min_idx = min(sizing_order.index(s) for s in tied_sizings)
                mode_sizing = sizing_order[min_idx]
            except ValueError:
                pass

        # Find all domains that contributed to the mode
        contributing_domains = [
            k for k, v in results_to_aggregate.items() if v.sizing == mode_sizing
        ]

        # Calculate average confidence from explicit results only
        if explicit_results:
            confidences = [r.confidence for r in explicit_results.values()]
        else:
            confidences = [r.confidence for r in domain_results.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Build reasoning with frequency info
        freq_info = ", ".join(f"{s}:{c}" for s, c in most_common[:3])

        return SizingResult(
            sizing=mode_sizing,
            reasoning=f"Mode of {len(results_to_aggregate)} explicit domains ({freq_info}): {', '.join(contributing_domains[:3])}",
            rule_id="PROGRAM_OVERALL",
            confidence=avg_confidence,
            method="aggregation",
        )

    def get_rules_for_domain(self, domain_key: str) -> list[SizingRule]:
        """Get all rules for a specific domain."""
        return self.rules_by_domain.get(domain_key, [])

    def get_rule_by_id(self, rule_id: str) -> Optional[SizingRule]:
        """Get a specific rule by ID."""
        return self.rules_by_id.get(rule_id)

    def get_all_rules_context(self) -> str:
        """Get formatted context of all rules for LLM prompts."""
        lines = ["## SIZING RULES REFERENCE\n"]

        for domain_key, rules in self.rules_by_domain.items():
            domain_name = domain_key.replace("_", " ").title()
            lines.append(f"### {domain_name}")

            for rule in rules:
                lines.append(f"- **{rule.rule_id}** [{rule.sizing.value}]:")
                lines.append(f"  {rule.development_effort[:150]}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================


def create_sizing_service(rules_path: Optional[str] = None) -> SizingService:
    """Factory function to create SizingService."""
    return SizingService(rules_path=rules_path)

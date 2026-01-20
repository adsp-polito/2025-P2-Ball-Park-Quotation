"""
FPT Cost Brain - Sizing Classification Prompts

LLM prompts for "Virtual Manager" sizing classification using ref_Sizing lookup table.
"""

# Reference Sizing descriptions for LLM context
REF_SIZING_CONTEXT = """
## SIZING REFERENCE TABLE (ref_Sizing)

Use this table to classify the project sizing based on PR description.

### Product Engineering - Base Engine
| Sizing | Development Effort Description |
|--------|-------------------------------|
| Full | New concept required; High level of New Content (NC); New serviceability requirements; High validation effort |
| Large | Heavy modification of existing concepts with impact on manufacturing; High/Medium NC; High/Medium validation |
| Medium | Medium modification of existing concepts (no impact on manufacturing); Medium NC; Medium validation |
| Small | Light modification of existing product; Low NC; Low validation effort |
| X-small | Minimum modification (only adaptation); Minimum NC; No validation effort |

### Product Engineering - System (Engine + ATS)
| Sizing | Development Effort Description |
|--------|-------------------------------|
| Full | New concept required; High NC; New serviceability requirements; High validation |
| Large | Heavy modification with manufacturing impact; High/Medium NC; High/Medium validation |
| Medium | Medium modification (no manufacturing impact); Medium NC; Medium validation |
| Small | Light modification; Low NC; Low validation |
| X-small | Minimum modification (only adaptation); Minimum NC; No validation |

### Product Engineering - Installation/Application/Homologation
| Sizing | Development Effort Description |
|--------|-------------------------------|
| Full | First installation; New SW & Cals; New Emission Stage; RGT |
| Large | Medium Installation effort; Medium Cals review; Homologation; RGT |
| Medium | Medium installation effort; Medium Cals Review; Homologation; RGT |
| Small | Low installation effort; Limited Cals Review; Homologation |
| X-small | Minimum installation; Minimum Cals effort; No homologation |

### Customer Manager - Build Stages
| Sizing | Build Stages Required |
|--------|----------------------|
| Full | All build stages required (Alpha, Beta, Gamma, PP, Pilot) |
| Large | Beta, Gamma, PP, Pilot required |
| Medium | Gamma, PP, Pilot required |
| Small | PP, Pilot required |
| X-small | Only Pilot required |

### COST IMPACT BY SIZING (historical data)
| Sizing | AG Sector (K EUR) | CE Sector (K EUR) |
|--------|-------------------|-------------------|
| X-small | ~500 | ~130 |
| Small | ~1,500 | ~1,000 |
| Mid | ~5,000 | ~1,000 |
| Large | ~6,500 | ~4,600 |
| Full | ~2,200 | ~1,100 |

⚠️ NOTE: AG (Agricultural) projects are typically 2-5× more expensive than CE (Construction Equipment)!
"""

SIZING_CLASSIFICATION_PROMPT = """You are an experienced FPT R&D Program Manager with 20+ years of experience in powertrain development.

Your task is to analyze this Product Request (PR) and classify the project sizing using your expertise.

{ref_sizing_context}

---

## PRODUCT REQUEST TO ANALYZE

**PR ID**: {pr_id}
**Title**: {title}
**Description**:
{description}

**Technical Scope**:
{scope}

**Activities Mentioned**:
{activities}

---

## YOUR TASK

Think step-by-step like an experienced Program Manager:

### Step 1: Identify the Sector
- Is this Agricultural (AG) or Construction Equipment (CE)?
- Look for keywords: tractor, harvester, combine → AG; excavator, loader, grader → CE

### Step 2: Analyze Base Engine Changes
- What modifications are needed to the base engine?
- New components? Turbo upgrade? Injector changes?
- Match to ref_Sizing "Base Engine" descriptions

### Step 3: Analyze System (Engine+ATS) Changes
- What ATS modifications? DPF, SCR, DOC changes?
- New emission standard required?
- Match to ref_Sizing "System" descriptions

### Step 4: Analyze Installation/Application
- First installation or modification?
- New calibration required?
- Homologation needed?
- Match to ref_Sizing "Installation" descriptions

### Step 5: Determine Build Stages
- How many prototypes needed?
- Which validation stages?
- Match to ref_Sizing "Build Stages" descriptions

### Step 6: Extract Numeric Features
- Power increase (kW)?
- Torque increase (Nm)?
- Number of applications/machines affected?

---

## OUTPUT FORMAT

Respond with a JSON object:
```json
{{
  "reasoning": "Brief explanation of your sizing decisions...",
  "sector": "AG or CE",
  "sizing_PE_base_powertrain": "X-small|Small|Mid|Large|Full",
  "sizing_PE_system_assembly": "X-small|Small|Mid|Large|Full",
  "sizing_PE_installation_application_homologation": "X-small|Small|Mid|Large|Full",
  "sizing_program": "X-small|Small|Mid|Large|Full",
  "power_increase_kw": 0,
  "torque_increase_nm": 0,
  "emission_level": "Stage V|Tier 4B|Tier 4F|Tier 3|...",
  "calibration_change": 0 or 1,
  "ATS_change": 0 or 1,
  "software_VCU_change": 0 or 1,
  "engine_performance_component_change": 0 or 1,
  "num_applications": 1,
  "confidence": 0.0 to 1.0
}}
```
"""

# Shorter prompt for Q&A clarification
# ============================================================================
# RULE-BASED SIZING SELECTION PROMPT
# ============================================================================

SIZING_RULE_SELECTION_PROMPT = """You are an expert FPT R&D Program Manager.

Your task is to select the BEST matching sizing rule for the domain: **{domain_name}**

### AVAILABLE RULES:
{rules_list}

### PR TEXT TO ANALYZE:
{pr_text}

### INSTRUCTIONS:
1. Read the PR text carefully
2. Identify keywords that match the "development_effort" descriptions in the rules
3. Select the ONE rule that best matches the PR scope
4. If uncertain, prefer HIGHER sizing (more conservative estimate)

### RESPONSE FORMAT (JSON only, no explanation outside JSON):
```json
{{
  "selected_rule_id": "PE_BASE_L_001",
  "confidence": 0.85,
  "reasoning": "PR mentions 'heavy modification' and 'manufacturing impact' which matches Large sizing rule"
}}
```

IMPORTANT:
- Return ONLY valid JSON
- selected_rule_id MUST be one of the rule IDs from the list above
- confidence should be 0.0-1.0 based on how well the PR matches the rule
- reasoning should explain WHY this rule was selected (2-3 sentences max)
"""


SIZING_CLARIFICATION_QUESTIONS = """
Based on the PR analysis, I need clarification on the following:

1. **Sector**: Is this project for Agricultural (AG) or Construction Equipment (CE)?
   - AG includes: tractors, harvesters, combines, sprayers
   - CE includes: excavators, wheel loaders, graders, telehandlers

2. **Base Engine Scope**: What level of engine modification is required?
   - Full: New engine concept from scratch
   - Large: Major redesign with new components
   - Medium: Moderate changes to existing design
   - Small: Minor modifications
   - X-small: Only adaptation/parameter changes

3. **ATS/Emission Scope**: What after-treatment changes are needed?
   - New emission certification required?
   - New DPF/SCR/DOC components?
   - Only calibration updates?

4. **Build Stages**: What validation is required?
   - All stages (Alpha through Pilot)?
   - Only final stages (PP, Pilot)?
   - Pilot only?
"""


def get_sizing_prompt(pr_data: dict) -> str:
    """Generate sizing classification prompt for a PR."""
    return SIZING_CLASSIFICATION_PROMPT.format(
        ref_sizing_context=REF_SIZING_CONTEXT,
        pr_id=pr_data.get("pr_id", "Unknown"),
        title=pr_data.get("title", "Unknown"),
        description=pr_data.get("description", "")[:2000],
        scope=pr_data.get("scope", pr_data.get("technical_scope", ""))[:1500],
        activities="\n".join(
            [
                f"- {a.get('name', '')}: {a.get('description', '')}"
                for a in pr_data.get("raw_activities", [])[:15]
            ]
        )
        or "No activities specified",
    )

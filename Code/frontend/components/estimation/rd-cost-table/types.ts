/**
 * R&D Cost Table - Type Definitions
 *
 * Types for the enterprise-grade Excel-like R&D cost estimation table
 * matching FPT's PE02 template format.
 */

// ============================================================================
// Flat Table Row Structure (matching PE02 template)
// ============================================================================

/**
 * R&D Cost Table Row - FLAT structure matching PE02 template.
 *
 * Columns: Function ID | PE Function | Main Activities Description |
 *          Manpower | Bench(Dev) | Bench(Special) | Bench(Dur) | Vehicle tests | Investment [k€]
 *
 * Function IDs use PE02 codes: A1, A2, B1, B2, C, D1, D2, D3, E, F, G
 */
export interface RDTableRow {
  id: string;

  // PE02 Identifier columns
  functionId: string; // A1, A2, B1, B2, C, D1, D2, D3, E, F, G
  peFunction: string; // PE Function name (e.g., "Project Management")

  // Activities description
  mainActivitiesDescription: string;

  // PE02 Effort Hours columns
  manpower: number; // Manpower hours
  benchDev: number; // Bench Development hours
  benchSpecial: number; // Bench Special hours (NVH, climatic, etc.) - NEW
  benchDur: number; // Bench Durability hours
  vehicleTests: number; // Vehicle tests hours (roller, PEMS)

  // Cost in k€ (PE02 standard)
  investmentKEur: number; // Investment [k€] - PRIMARY cost field

  // AI prediction metadata
  confidence: number;
  predictedValue?: number;

  // UI state
  isEdited?: boolean;

  // Formula support
  formulas?: Record<string, string>; // column -> formula string
}

// Legacy type for backwards compatibility (deprecated)
export type RowType = "flat";

// Legacy resource types (deprecated - now using summary columns)
export type ResourceType =
  | "engineer"
  | "seniorEngineer"
  | "technician"
  | "specialist"
  | "manager"
  | "external";

export const RESOURCE_TYPES: ResourceType[] = [
  "engineer",
  "seniorEngineer",
  "technician",
  "specialist",
  "manager",
  "external",
];

export const RESOURCE_LABELS: Record<ResourceType, string> = {
  engineer: "Engineer",
  seniorEngineer: "Sr. Engineer",
  technician: "Technician",
  specialist: "Specialist",
  manager: "Manager",
  external: "External",
};

// ============================================================================
// DPO Reasoning (for learning system)
// ============================================================================

export type ChangeTag =
  | "complexity"
  | "riskBuffer"
  | "scopeChange"
  | "resourceConstraint"
  | "historicalKnowledge"
  | "customerRequirement"
  | "regulatory"
  | "integration";

export const CHANGE_TAGS: {
  value: ChangeTag;
  label: string;
  description: string;
}[] = [
  {
    value: "complexity",
    label: "Complexity",
    description: "Task more complex than predicted",
  },
  {
    value: "riskBuffer",
    label: "Risk Buffer",
    description: "Added safety margin",
  },
  {
    value: "scopeChange",
    label: "Scope Change",
    description: "Requirements changed from original PR",
  },
  {
    value: "resourceConstraint",
    label: "Resource Constraint",
    description: "Limited availability of specific resources",
  },
  {
    value: "historicalKnowledge",
    label: "Historical Knowledge",
    description: "Based on past similar projects",
  },
  {
    value: "customerRequirement",
    label: "Customer Requirement",
    description: "Specific customer constraint",
  },
  {
    value: "regulatory",
    label: "Regulatory",
    description: "Compliance/certification requirements",
  },
  {
    value: "integration",
    label: "Integration",
    description: "Dependencies on other systems",
  },
];

export interface RowReasoning {
  id: string;
  rowId: string;
  columnName: string;
  reasoningText: string;
  changeTags: ChangeTag[];
  originalValue: number;
  newValue: number;
  confidenceAtChange: number;
  createdAt: Date;
  createdBy?: string;
}

// ============================================================================
// Version History
// ============================================================================

export interface TableVersion {
  id: string;
  sessionId: string;
  versionNumber: number;
  data: RDTableRow[];
  createdBy: string;
  createdByName?: string;
  createdAt: Date;
  isFinalized: boolean;
  changeCount: number;
  changesFromPrevious?: VersionDiff[];
}

export interface VersionDiff {
  rowId: string;
  columnName: string;
  oldValue: number | string;
  newValue: number | string;
  changeType: "added" | "modified" | "deleted";
}

// ============================================================================
// Cell & Selection
// ============================================================================

export interface CellPosition {
  rowId: string;
  columnId: string;
}

export interface CellRange {
  start: CellPosition;
  end: CellPosition;
}

export interface CellState {
  value: number | string;
  displayValue: string;
  isFormula: boolean;
  hasError: boolean;
  errorMessage?: string;
  isEditing: boolean;
  isSelected: boolean;
  isInRange: boolean;
}

export type CellType = "text" | "number" | "currency" | "formula" | "locked";

export interface ColumnDefinition {
  id: string;
  header: string;
  type: CellType;
  width: number;
  minWidth?: number;
  maxWidth?: number;
  frozen?: boolean;
  editable: boolean;
  resourceType?: ResourceType;
}

// ============================================================================
// Confidence Levels (4-tier system)
// ============================================================================

export type ConfidenceLevel = "veryHigh" | "high" | "medium" | "low";

export const CONFIDENCE_THRESHOLDS: Record<
  ConfidenceLevel,
  { min: number; max: number }
> = {
  veryHigh: { min: 0.9, max: 1.0 },
  high: { min: 0.7, max: 0.9 },
  medium: { min: 0.5, max: 0.7 },
  low: { min: 0, max: 0.5 },
};

export const CONFIDENCE_STYLES: Record<
  ConfidenceLevel,
  { bg: string; icon: string | null }
> = {
  veryHigh: { bg: "", icon: null },
  high: { bg: "", icon: null },
  medium: { bg: "bg-amber-50", icon: "warning-yellow" },
  low: { bg: "bg-red-50", icon: "warning-red" },
};

export function getConfidenceLevel(confidence: number): ConfidenceLevel {
  if (confidence >= 0.9) return "veryHigh";
  if (confidence >= 0.7) return "high";
  if (confidence >= 0.5) return "medium";
  return "low";
}

// ============================================================================
// Undo/Redo Actions
// ============================================================================

export type ActionType =
  | "cellEdit"
  | "rowAdd"
  | "rowDelete"
  | "rowMove"
  | "bulkEdit"
  | "formatChange"
  | "reset";

export interface TableAction {
  id: string;
  type: ActionType;
  timestamp: Date;
  description: string;
  beforeState: Partial<RDTableRow>[];
  afterState: Partial<RDTableRow>[];
  affectedRows: string[];
}

// ============================================================================
// Table Props & State
// ============================================================================

export interface RDCostTableProps {
  sessionId: string;
  initialData: RDTableRow[];
  onDataChange: (data: RDTableRow[]) => void;
  onFinalize: () => void;
  onExport: (format: "pptx" | "xlsx" | "pdf" | "csv") => void;
  isLoading?: boolean;
  isReadOnly?: boolean;
}

export interface RDCostTableState {
  data: RDTableRow[];
  flattenedRows: RDTableRow[]; // Flattened for virtualization
  selectedCell: CellPosition | null;
  selectedRange: CellRange | null;
  editingCell: CellPosition | null;
  editValue: string;
  expandedRows: Set<string>;
  expandedReasoningRows: Set<string>;
  isDirty: boolean;
  lastSaved: Date | null;
}

// ============================================================================
// API Request/Response Types
// ============================================================================

export interface RDTableUpdateRequest {
  rowId: string;
  columnName: string;
  newValue: number | string;
  reasoning?: {
    text: string;
    tags: ChangeTag[];
  };
}

export interface RDTableVersionResponse {
  id: string;
  sessionId: string;
  versionNumber: number;
  data: RDTableRow[];
  createdBy: string;
  createdAt: string;
  isFinalized: boolean;
}

export interface RDTableFinalizeResponse {
  success: boolean;
  versionId: string;
  versionNumber: number;
  exportPrompt: boolean;
}

// ============================================================================
// FPT Styling Constants - Excel-like PE02 appearance
// ============================================================================

export const FPT_COLORS = {
  // Header styling - dark maroon matching PE02 Excel template
  headerBg: "#8B0000", // Dark red/maroon from target design
  headerBgDark: "#5c0000", // Darker shade for borders
  headerText: "#FFFFFF",

  // Row styling
  rowAlternate: "#F9FAFB", // Very subtle alternating
  rowHover: "#F3F4F6",

  // Border styling - softer for Excel look
  borderColor: "#D1D5DB", // gray-300 equivalent
  borderColorStrong: "#9CA3AF", // gray-400 for emphasis

  // Cell states
  editingBg: "#FFFFFF", // White when editing
  editingBorder: "#3B82F6", // Blue ring
  selectedBg: "#EFF6FF", // blue-50
  editedBg: "#FEF9C3", // yellow-100 for edited cells
  errorBg: "#FEE2E2", // red-100

  // Total row
  totalRowBg: "#F3F4F6", // gray-100

  // Disabled cell (non-applicable column)
  disabledBg: "#F3F4F6", // gray-100 - subtle disabled look
  disabledText: "#9CA3AF", // gray-400
} as const;

// ============================================================================
// PE02 Effort Column Rules (FPT Standard)
// ============================================================================
// Each activity category can ONLY estimate specific effort columns.
// Non-applicable columns should show as disabled/grayed out.

export type EffortColumn =
  | "manpower"
  | "benchDev"
  | "benchSpecial"
  | "benchDur"
  | "vehicleTests";

/**
 * Allowed effort columns per activity code (FPT PE02 standard).
 * Columns not in this list should be disabled for the row.
 */
export const PE02_ALLOWED_COLUMNS: Record<string, EffortColumn[]> = {
  // A-series (CP&E): Design work can use manpower, benchDev, benchSpecial
  A: ["manpower", "benchDev", "benchSpecial"],
  A1: ["manpower", "benchDev", "benchSpecial"],
  A2: ["manpower", "benchDev", "benchSpecial"],
  A3: ["manpower", "benchDev", "benchSpecial"],
  A4: ["manpower", "benchDev", "benchSpecial"],

  // B-series: Each has specific rules
  B1: ["manpower"], // Calibration is manpower only
  "B1-C": ["manpower"],
  B2: ["benchDur"], // Reliability is ONLY bench durability
  B3: ["manpower"], // Material handling is manpower

  // C: Application uses manpower and vehicle testing
  C: ["manpower", "vehicleTests"],
  "C.Vehicle": ["manpower", "vehicleTests"],

  // D-series: Certification
  D1: ["manpower", "benchDev"], // Certification cost
  D2: ["manpower", "benchDev", "benchDur"], // DF test - all bench types
  "D1+D2": ["manpower", "benchDev", "benchDur"], // Combined tech cert
  D3: ["manpower", "vehicleTests"], // PEMS is vehicle testing

  // E, F, G: Administrative - manpower only
  E: ["manpower"],
  F: ["manpower"],
  F1: ["manpower"],
  F2: ["manpower"],
  G: ["manpower"],
};

/**
 * Check if an effort column is allowed for a given function ID.
 */
export function isColumnAllowed(
  functionId: string,
  column: EffortColumn,
): boolean {
  // Try exact match first
  const allowed = PE02_ALLOWED_COLUMNS[functionId];
  if (allowed) {
    return allowed.includes(column);
  }

  // Try category prefix (e.g., "A" for "A1")
  const prefix = functionId.charAt(0);
  const prefixAllowed = PE02_ALLOWED_COLUMNS[prefix];
  if (prefixAllowed) {
    return prefixAllowed.includes(column);
  }

  // Default: only manpower allowed
  return column === "manpower";
}

/**
 * Map column IDs to EffortColumn type for the check.
 */
export const COLUMN_TO_EFFORT: Record<string, EffortColumn | null> = {
  manpower: "manpower",
  benchDev: "benchDev",
  benchSpecial: "benchSpecial",
  benchDur: "benchDur",
  vehicleTests: "vehicleTests",
  // Non-effort columns return null (always allowed)
  functionId: null,
  peFunction: null,
  mainActivitiesDescription: null,
  investmentKEur: null,
};

/**
 * R&D Cost Table - Barrel Export
 *
 * Enterprise-grade Excel-like R&D cost estimation table.
 */

// Main component
export { RDCostTable } from "./RDCostTable";

// Types
export * from "./types";

// Core utilities
export {
  HyperFormulaEngine,
  getFormulaEngine,
  destroyFormulaEngine,
} from "./HyperFormulaEngine";

export { SelectionManager, useSelectionManager } from "./SelectionManager";

export {
  UndoRedoManager,
  useUndoRedoManager,
  useUndoRedoKeyboard,
  createCellEditAction,
  createRowAddAction,
  createRowDeleteAction,
  createRowMoveAction,
  createBulkEditAction,
  createResetAction,
} from "./UndoRedoManager";

// Sub-components (for custom implementations)
export { RDCostTableCell, RDCostTableHeaderCell } from "./RDCostTableCell";
export { RDCostTableRow } from "./RDCostTableRow";
export { RDCostTableHeader, createDefaultColumns } from "./RDCostTableHeader";
export { RDCostTableToolbar } from "./RDCostTableToolbar";
export { ReasoningPanel, ReasoningToast } from "./ReasoningPanel";
export {
  ChangeTagSelector,
  ChangeTagDisplay,
  TagBadge,
} from "./ChangeTagSelector";
export { VersionHistory } from "./VersionHistory";

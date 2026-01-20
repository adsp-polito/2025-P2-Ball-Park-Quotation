"use client";

/**
 * R&D Cost Table Row Component
 *
 * FLAT row structure matching PE02 Excel template.
 * Clean spreadsheet aesthetic without extra UI chrome.
 */

import React, { memo, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  type RDTableRow,
  type ColumnDefinition,
  type EffortColumn,
  isColumnAllowed,
  COLUMN_TO_EFFORT,
} from "./types";
import { RDCostTableCell } from "./RDCostTableCell";
import { HyperFormulaEngine } from "./HyperFormulaEngine";

// ============================================================================
// Types
// ============================================================================

export interface RDCostTableRowProps {
  row: RDTableRow;
  columns: ColumnDefinition[];
  rowIndex: number;
  hasReasoning: boolean;
  isReasoningExpanded: boolean;
  formulaEngine: HyperFormulaEngine | null;
  selectionState: {
    isCellSelected: (rowId: string, columnId: string) => boolean;
    isCellInRange: (rowId: string, columnId: string) => boolean;
    editingCell: { rowId: string; columnId: string } | null;
  };
  onToggleReasoning: (rowId: string) => void;
  onCellSelect: (rowId: string, columnId: string) => void;
  onCellRangeStart: (rowId: string, columnId: string) => void;
  onCellRangeExtend: (rowId: string, columnId: string) => void;
  onCellDoubleClick: (rowId: string, columnId: string) => void;
  onCellValueChange: (
    rowId: string,
    columnId: string,
    value: string | number,
  ) => void;
  onEditComplete: () => void;
  onEditCancel: () => void;
}

// ============================================================================
// Row Component - Clean Excel-style row
// ============================================================================

export const RDCostTableRow = memo(function RDCostTableRow({
  row,
  columns,
  rowIndex,
  formulaEngine,
  selectionState,
  onCellSelect,
  onCellRangeStart,
  onCellRangeExtend,
  onCellDoubleClick,
  onCellValueChange,
  onEditComplete,
  onEditCancel,
}: RDCostTableRowProps) {
  // ==========================================================================
  // Cell Value Getters - FLAT structure
  // ==========================================================================

  const getCellValue = useCallback(
    (columnId: string): number | string => {
      switch (columnId) {
        // PE02 identifier columns
        case "functionId":
          return row.functionId || "";
        case "peFunction":
          return row.peFunction || "";
        case "mainActivitiesDescription":
          return row.mainActivitiesDescription || "";
        // PE02 effort columns
        case "manpower":
          return row.manpower || 0;
        case "benchDev":
          return row.benchDev || 0;
        case "benchSpecial":
          return row.benchSpecial || 0;
        case "benchDur":
          return row.benchDur || 0;
        case "vehicleTests":
          return row.vehicleTests || 0;
        // PE02 cost column (k€)
        case "investmentKEur":
          return row.investmentKEur || 0;
        default:
          return "";
      }
    },
    [row],
  );

  const getCellDisplayValue = useCallback(
    (columnId: string): string => {
      // Check if formula engine has computed value
      if (formulaEngine) {
        const value = formulaEngine.getCellValue(row.id, columnId as any);
        if (value !== null && value !== undefined) {
          return String(value);
        }
      }
      return String(getCellValue(columnId));
    },
    [row.id, formulaEngine, getCellValue],
  );

  const isFormula = useCallback(
    (columnId: string): boolean => {
      return formulaEngine?.isFormula(row.id, columnId as any) ?? false;
    },
    [row.id, formulaEngine],
  );

  const hasError = useCallback(
    (columnId: string): boolean => {
      return formulaEngine?.hasError(row.id, columnId as any) ?? false;
    },
    [row.id, formulaEngine],
  );

  const getErrorMessage = useCallback(
    (columnId: string): string | undefined => {
      return (
        formulaEngine?.getErrorMessage(row.id, columnId as any) ?? undefined
      );
    },
    [row.id, formulaEngine],
  );

  // ==========================================================================
  // Render - Clean Excel-style row
  // ==========================================================================

  return (
    <div
      className={cn(
        "flex items-stretch",
        "border-b border-gray-300", // Softer border for Excel look
        "bg-white",
      )}
      role="row"
      aria-rowindex={rowIndex + 3} // +3 for two header rows
      data-row-id={row.id}
    >
      {/* Row number column - matching Excel row header style */}
      <div
        className={cn(
          "flex items-center justify-center shrink-0",
          "border-r border-gray-300 bg-gray-100",
          "sticky left-0 z-10",
          "text-xs text-gray-600 font-medium",
        )}
        style={{ width: 40 }}
      >
        {rowIndex + 1}
      </div>

      {/* Data Cells */}
      {columns.map((column, colIndex) => {
        const cellValue = getCellValue(column.id);
        const displayValue = getCellDisplayValue(column.id);
        const cellIsFormula = isFormula(column.id);
        const cellHasError = hasError(column.id);
        const errorMessage = getErrorMessage(column.id);
        const isEditing =
          selectionState.editingCell?.rowId === row.id &&
          selectionState.editingCell?.columnId === column.id;
        const isSelected = selectionState.isCellSelected(row.id, column.id);
        const isInRange = selectionState.isCellInRange(row.id, column.id);
        const isLastColumn = colIndex === columns.length - 1;

        // Check if this effort column is allowed for this function ID (FPT PE02 rule)
        const effortColumn = COLUMN_TO_EFFORT[column.id];
        const isColumnDisabled =
          effortColumn !== null &&
          !isColumnAllowed(row.functionId, effortColumn as EffortColumn);

        return (
          <RDCostTableCell
            key={column.id}
            rowId={row.id}
            columnId={column.id}
            value={cellValue}
            displayValue={displayValue}
            cellType={column.type}
            isFormula={cellIsFormula}
            hasError={cellHasError}
            errorMessage={errorMessage}
            isEditing={isEditing}
            isSelected={isSelected}
            isInRange={isInRange}
            isEditable={column.editable && !isColumnDisabled}
            isDisabled={isColumnDisabled}
            isFrozen={column.frozen ?? false}
            isEdited={row.isEdited}
            width={column.width}
            isLastColumn={isLastColumn}
            onSelect={onCellSelect}
            onStartRangeSelect={onCellRangeStart}
            onExtendRange={onCellRangeExtend}
            onDoubleClick={onCellDoubleClick}
            onValueChange={onCellValueChange}
            onEditComplete={onEditComplete}
            onEditCancel={onEditCancel}
          />
        );
      })}
    </div>
  );
});

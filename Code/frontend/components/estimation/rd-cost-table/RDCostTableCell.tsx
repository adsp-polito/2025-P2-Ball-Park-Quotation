"use client";

/**
 * R&D Cost Table Cell Component
 *
 * Editable cell with formula support, validation, and confidence-based styling.
 * Supports double-click to edit, Tab/Enter navigation, and error display.
 */

import React, {
  memo,
  useRef,
  useState,
  useEffect,
  useCallback,
  type KeyboardEvent,
  type ChangeEvent,
  type FocusEvent,
} from "react";
import { cn } from "@/lib/utils";
import { type CellType, FPT_COLORS } from "./types";

// ============================================================================
// Types
// ============================================================================

export interface RDCostTableCellProps {
  rowId: string;
  columnId: string;
  value: number | string;
  displayValue: string;
  cellType: CellType;
  isFormula: boolean;
  hasError: boolean;
  errorMessage?: string;
  isEditing: boolean;
  isSelected: boolean;
  isInRange: boolean;
  isEditable: boolean;
  isDisabled?: boolean; // FPT PE02 rule: column not applicable for this activity
  isFrozen: boolean;
  isEdited?: boolean; // Yellow highlight for edited cells
  isLastColumn?: boolean;
  width: number;
  onSelect: (rowId: string, columnId: string) => void;
  onStartRangeSelect: (rowId: string, columnId: string) => void;
  onExtendRange: (rowId: string, columnId: string) => void;
  onDoubleClick: (rowId: string, columnId: string) => void;
  onValueChange: (
    rowId: string,
    columnId: string,
    value: string | number,
  ) => void;
  onEditComplete: () => void;
  onEditCancel: () => void;
}

// ============================================================================
// Cell Component
// ============================================================================

export const RDCostTableCell = memo(function RDCostTableCell({
  rowId,
  columnId,
  value,
  displayValue,
  cellType,
  isFormula,
  hasError,
  errorMessage,
  isEditing,
  isSelected,
  isInRange,
  isEditable,
  isDisabled = false,
  isFrozen,
  isEdited,
  isLastColumn,
  width,
  onSelect,
  onStartRangeSelect,
  onExtendRange,
  onDoubleClick,
  onValueChange,
  onEditComplete,
  onEditCancel,
}: RDCostTableCellProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [localValue, setLocalValue] = useState(String(value));
  const [validationError, setValidationError] = useState<string | null>(null);

  // Sync local value when value prop changes (and not editing)
  useEffect(() => {
    if (!isEditing) {
      setLocalValue(String(value));
      setValidationError(null);
    }
  }, [value, isEditing]);

  // Focus input when entering edit mode
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  // ==========================================================================
  // Event Handlers
  // ==========================================================================

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onSelect(rowId, columnId);
    },
    [rowId, columnId, onSelect],
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.shiftKey) {
        onExtendRange(rowId, columnId);
      } else {
        onStartRangeSelect(rowId, columnId);
      }
    },
    [rowId, columnId, onStartRangeSelect, onExtendRange],
  );

  const handleMouseEnter = useCallback(
    (e: React.MouseEvent) => {
      if (e.buttons === 1) {
        // Left mouse button held
        onExtendRange(rowId, columnId);
      }
    },
    [rowId, columnId, onExtendRange],
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isEditable) {
        onDoubleClick(rowId, columnId);
      }
    },
    [rowId, columnId, isEditable, onDoubleClick],
  );

  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const newValue = e.target.value;
      setLocalValue(newValue);

      // Validate based on cell type
      if (cellType === "number" || cellType === "currency") {
        // Allow formula input
        if (newValue.startsWith("=")) {
          setValidationError(null);
          return;
        }

        // Validate number
        if (newValue !== "" && isNaN(Number(newValue))) {
          setValidationError("Must be a number");
        } else {
          setValidationError(null);
        }
      }
    },
    [cellType],
  );

  // Direct handlers - no useCallback to avoid stale closure issues
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      if (!validationError) {
        // Commit directly
        let finalValue: string | number = localValue;
        if (
          (cellType === "number" || cellType === "currency") &&
          !localValue.startsWith("=")
        ) {
          finalValue = localValue === "" ? 0 : Number(localValue);
        }
        console.log("[CELL ENTER] Committing:", {
          rowId,
          columnId,
          localValue,
          finalValue,
        });
        onValueChange(rowId, columnId, finalValue);
        onEditComplete();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      setLocalValue(String(value));
      setValidationError(null);
      onEditCancel();
    } else if (e.key === "Tab") {
      e.preventDefault();
      if (!validationError) {
        let finalValue: string | number = localValue;
        if (
          (cellType === "number" || cellType === "currency") &&
          !localValue.startsWith("=")
        ) {
          finalValue = localValue === "" ? 0 : Number(localValue);
        }
        onValueChange(rowId, columnId, finalValue);
        onEditComplete();
      }
    }
  };

  const handleBlur = (_e: FocusEvent<HTMLInputElement>) => {
    if (!validationError) {
      let finalValue: string | number = localValue;
      if (
        (cellType === "number" || cellType === "currency") &&
        !localValue.startsWith("=")
      ) {
        finalValue = localValue === "" ? 0 : Number(localValue);
      }
      console.log("[CELL BLUR] Committing:", { rowId, columnId, finalValue });
      onValueChange(rowId, columnId, finalValue);
      onEditComplete();
    } else {
      setLocalValue(String(value));
      setValidationError(null);
      onEditCancel();
    }
  };

  const commitValue = () => {
    let finalValue: string | number = localValue;
    if (
      (cellType === "number" || cellType === "currency") &&
      !localValue.startsWith("=")
    ) {
      finalValue = localValue === "" ? 0 : Number(localValue);
    }
    onValueChange(rowId, columnId, finalValue);
    onEditComplete();
  };

  // ==========================================================================
  // Render Helpers
  // ==========================================================================

  const formatDisplayValue = (val: string): string => {
    if (hasError) {
      return errorMessage || "#ERROR!";
    }

    if (cellType === "currency") {
      const num = parseFloat(val);
      if (!isNaN(num)) {
        return new Intl.NumberFormat("en-US", {
          minimumFractionDigits: 0,
          maximumFractionDigits: 0,
        }).format(num);
      }
    }

    if (cellType === "number") {
      const num = parseFloat(val);
      if (!isNaN(num)) {
        return new Intl.NumberFormat("en-US", {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2,
        }).format(num);
      }
    }

    return val;
  };

  // ==========================================================================
  // Styles - Clean Excel-like appearance
  // ==========================================================================

  const cellStyles = cn(
    // Base styles - clean grid borders
    "relative flex items-center h-8 px-2 text-sm",
    "transition-colors duration-75",
    // Border - right border except for last column
    !isLastColumn && "border-r border-gray-300",
    // Background based on state (priority order matters)
    {
      // Disabled cells (FPT PE02 rule: column not applicable)
      "bg-gray-100 text-gray-400": isDisabled,
      // Editing state
      "bg-white ring-2 ring-inset ring-blue-500": isEditing && !isDisabled,
      // Selected cell (when not editing)
      "bg-blue-50 ring-2 ring-inset ring-blue-400":
        isSelected && !isEditing && !isDisabled,
      // In range but not selected
      "bg-blue-50/50": isInRange && !isSelected && !isEditing && !isDisabled,
      // Error
      "bg-red-50": hasError && !isEditing && !isSelected && !isDisabled,
      // Edited cells get yellow highlight (like Excel)
      "bg-yellow-100":
        isEdited &&
        !isEditing &&
        !isSelected &&
        !isInRange &&
        !hasError &&
        !isDisabled,
      // Frozen/identifier columns
      "bg-gray-50/50":
        isFrozen &&
        !isSelected &&
        !isInRange &&
        !hasError &&
        !isEditing &&
        !isEdited &&
        !isDisabled,
    },
    // Text alignment
    {
      "justify-start": cellType === "text",
      "justify-end font-mono": cellType === "number" || cellType === "currency",
      "justify-center": cellType === "formula" || isDisabled,
    },
    // Cursor
    {
      "cursor-cell": isEditable && !isDisabled,
      "cursor-not-allowed": isDisabled,
      "cursor-default": !isEditable && !isDisabled,
    },
  );

  // ==========================================================================
  // Render
  // ==========================================================================

  return (
    <div
      className={cellStyles}
      style={{ width, minWidth: width, maxWidth: width }}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
      onMouseEnter={handleMouseEnter}
      onDoubleClick={handleDoubleClick}
      role="gridcell"
      aria-selected={isSelected}
      aria-readonly={!isEditable}
      tabIndex={isSelected ? 0 : -1}
      data-row-id={rowId}
      data-column-id={columnId}
    >
      {isEditing ? (
        // Edit Mode
        <div className="absolute inset-0 flex items-center">
          <input
            ref={inputRef}
            type="text"
            value={localValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            className={cn(
              "w-full h-full px-2 text-sm outline-none",
              "bg-white border-2 border-blue-500",
              {
                "text-right": cellType === "number" || cellType === "currency",
                "border-red-500": validationError,
              },
            )}
            aria-label={`Edit ${columnId}`}
            aria-invalid={!!validationError}
          />
          {validationError && (
            <div className="absolute -bottom-6 left-0 z-10 px-2 py-1 text-xs text-white bg-red-500 rounded shadow">
              {validationError}
            </div>
          )}
        </div>
      ) : isDisabled ? (
        // Disabled Mode - FPT PE02 rule: column not applicable for this activity
        <span
          className="text-gray-400 text-xs select-none"
          title="Column not applicable for this activity (FPT PE02 rule)"
        >
          —
        </span>
      ) : (
        // View Mode - Clean Excel display
        <>
          {/* Formula indicator - small green corner */}
          {isFormula && !hasError && (
            <span
              className="absolute top-0 left-0 w-0 h-0 border-l-[6px] border-t-[6px] border-l-green-600 border-t-transparent border-t-green-600 border-l-transparent"
              style={{
                borderTopColor: "transparent",
                borderLeftColor: "#16a34a",
              }}
            />
          )}

          {/* Value display */}
          <span
            className={cn("truncate", {
              "text-red-600 font-medium": hasError,
              "text-gray-400 italic": !displayValue && !hasError,
              "text-gray-900": displayValue && !hasError,
            })}
            title={hasError ? errorMessage : displayValue}
          >
            {formatDisplayValue(displayValue) || ""}
          </span>
        </>
      )}
    </div>
  );
});

// ============================================================================
// Cell Wrapper for Memoization Optimization
// ============================================================================

export interface CellData {
  rowId: string;
  columnId: string;
  value: number | string;
  displayValue: string;
  cellType: CellType;
  isFormula: boolean;
  hasError: boolean;
  errorMessage?: string;
  isEditable: boolean;
  isFrozen: boolean;
  confidence?: number;
  width: number;
}

/**
 * Get cell data key for comparison in memoization.
 */
export function getCellDataKey(cell: CellData): string {
  return `${cell.rowId}-${cell.columnId}-${cell.value}-${cell.isFormula}-${cell.hasError}-${cell.confidence}`;
}

// ============================================================================
// Header Cell Component
// ============================================================================

export interface RDCostTableHeaderCellProps {
  columnId: string;
  label: string;
  width: number;
  isFrozen: boolean;
  isResizable: boolean;
  onResize?: (columnId: string, newWidth: number) => void;
}

export const RDCostTableHeaderCell = memo(function RDCostTableHeaderCell({
  columnId,
  label,
  width,
  isFrozen,
  isResizable,
  onResize,
}: RDCostTableHeaderCellProps) {
  const [isResizing, setIsResizing] = useState(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(width);

  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsResizing(true);
      startXRef.current = e.clientX;
      startWidthRef.current = width;

      const handleMouseMove = (moveEvent: MouseEvent) => {
        const diff = moveEvent.clientX - startXRef.current;
        const newWidth = Math.max(50, startWidthRef.current + diff);
        onResize?.(columnId, newWidth);
      };

      const handleMouseUp = () => {
        setIsResizing(false);
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    },
    [columnId, width, onResize],
  );

  return (
    <div
      className={cn(
        "relative flex items-center justify-center h-10 px-2 text-sm font-semibold text-white border-r border-b",
        "border-red-900 select-none",
        { "sticky left-0 z-10": isFrozen },
      )}
      style={{
        width,
        minWidth: width,
        maxWidth: width,
        backgroundColor: FPT_COLORS.headerBg,
      }}
      role="columnheader"
    >
      <span className="truncate">{label}</span>

      {/* Resize handle */}
      {isResizable && (
        <div
          className={cn(
            "absolute right-0 top-0 bottom-0 w-1 cursor-col-resize",
            "hover:bg-white/30",
            { "bg-white/50": isResizing },
          )}
          onMouseDown={handleResizeStart}
        />
      )}
    </div>
  );
});

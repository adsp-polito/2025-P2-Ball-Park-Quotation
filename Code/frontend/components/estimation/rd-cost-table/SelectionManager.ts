/**
 * Selection Manager
 *
 * Handles cell selection, range selection, and keyboard navigation
 * for the R&D Cost Table. Provides Excel-like selection behavior.
 */

import type { CellPosition, CellRange, ColumnDefinition } from "./types";

// ============================================================================
// Types
// ============================================================================

export type SelectionMode = "cell" | "range" | "none";

export interface SelectionState {
  mode: SelectionMode;
  activeCell: CellPosition | null;
  range: CellRange | null;
  isEditing: boolean;
  isDragging: boolean;
}

export interface NavigationOptions {
  wrap: boolean; // Wrap to next/prev row at boundaries
  skipLocked: boolean; // Skip non-editable columns
  skipHidden: boolean; // Skip collapsed rows
}

export type NavigationDirection = "up" | "down" | "left" | "right";

export interface SelectionChangeEvent {
  previousState: SelectionState;
  currentState: SelectionState;
  trigger: "click" | "keyboard" | "drag" | "programmatic";
}

// ============================================================================
// Selection Manager Class
// ============================================================================

export class SelectionManager {
  private state: SelectionState;
  private columns: ColumnDefinition[];
  private rowIds: string[];
  private visibleRowIds: Set<string>;
  private listeners: Set<(event: SelectionChangeEvent) => void>;
  private options: NavigationOptions;

  constructor(
    columns: ColumnDefinition[],
    rowIds: string[] = [],
    options: Partial<NavigationOptions> = {},
  ) {
    this.columns = columns;
    this.rowIds = rowIds;
    this.visibleRowIds = new Set(rowIds);
    this.listeners = new Set();
    this.options = {
      wrap: true,
      skipLocked: true,
      skipHidden: true,
      ...options,
    };
    this.state = {
      mode: "none",
      activeCell: null,
      range: null,
      isEditing: false,
      isDragging: false,
    };
  }

  // ==========================================================================
  // State Accessors
  // ==========================================================================

  getState(): SelectionState {
    return { ...this.state };
  }

  getActiveCell(): CellPosition | null {
    return this.state.activeCell ? { ...this.state.activeCell } : null;
  }

  getRange(): CellRange | null {
    return this.state.range
      ? {
          start: { ...this.state.range.start },
          end: { ...this.state.range.end },
        }
      : null;
  }

  isEditing(): boolean {
    return this.state.isEditing;
  }

  isCellSelected(rowId: string, columnId: string): boolean {
    if (!this.state.activeCell) return false;
    return (
      this.state.activeCell.rowId === rowId &&
      this.state.activeCell.columnId === columnId
    );
  }

  isCellInRange(rowId: string, columnId: string): boolean {
    if (!this.state.range) return false;

    const startRowIndex = this.rowIds.indexOf(this.state.range.start.rowId);
    const endRowIndex = this.rowIds.indexOf(this.state.range.end.rowId);
    const cellRowIndex = this.rowIds.indexOf(rowId);

    const startColIndex = this.getColumnIndex(this.state.range.start.columnId);
    const endColIndex = this.getColumnIndex(this.state.range.end.columnId);
    const cellColIndex = this.getColumnIndex(columnId);

    const minRow = Math.min(startRowIndex, endRowIndex);
    const maxRow = Math.max(startRowIndex, endRowIndex);
    const minCol = Math.min(startColIndex, endColIndex);
    const maxCol = Math.max(startColIndex, endColIndex);

    return (
      cellRowIndex >= minRow &&
      cellRowIndex <= maxRow &&
      cellColIndex >= minCol &&
      cellColIndex <= maxCol
    );
  }

  // ==========================================================================
  // Selection Operations
  // ==========================================================================

  /**
   * Select a single cell.
   */
  selectCell(
    rowId: string,
    columnId: string,
    trigger: SelectionChangeEvent["trigger"] = "programmatic",
  ): void {
    const previousState = { ...this.state };

    this.state = {
      mode: "cell",
      activeCell: { rowId, columnId },
      range: null,
      isEditing: false,
      isDragging: false,
    };

    this.notifyListeners(previousState, trigger);
  }

  /**
   * Start range selection.
   */
  startRangeSelection(rowId: string, columnId: string): void {
    const previousState = { ...this.state };

    this.state = {
      mode: "range",
      activeCell: { rowId, columnId },
      range: {
        start: { rowId, columnId },
        end: { rowId, columnId },
      },
      isEditing: false,
      isDragging: true,
    };

    this.notifyListeners(previousState, "drag");
  }

  /**
   * Extend range selection (during drag).
   */
  extendRangeSelection(rowId: string, columnId: string): void {
    if (!this.state.isDragging || !this.state.range) return;

    const previousState = { ...this.state };

    this.state = {
      ...this.state,
      range: {
        start: this.state.range.start,
        end: { rowId, columnId },
      },
    };

    this.notifyListeners(previousState, "drag");
  }

  /**
   * End range selection (mouse up).
   */
  endRangeSelection(): void {
    if (!this.state.isDragging) return;

    const previousState = { ...this.state };
    this.state.isDragging = false;

    // If start and end are the same, switch to cell mode
    if (
      this.state.range &&
      this.state.range.start.rowId === this.state.range.end.rowId &&
      this.state.range.start.columnId === this.state.range.end.columnId
    ) {
      this.state.mode = "cell";
      this.state.range = null;
    }

    this.notifyListeners(previousState, "drag");
  }

  /**
   * Clear selection.
   */
  clearSelection(): void {
    const previousState = { ...this.state };

    this.state = {
      mode: "none",
      activeCell: null,
      range: null,
      isEditing: false,
      isDragging: false,
    };

    this.notifyListeners(previousState, "programmatic");
  }

  /**
   * Enter edit mode for current cell.
   */
  enterEditMode(): void {
    if (!this.state.activeCell) return;

    const column = this.columns.find(
      (c) => c.id === this.state.activeCell!.columnId,
    );
    if (column && !column.editable) return;

    const previousState = { ...this.state };
    this.state.isEditing = true;
    this.notifyListeners(previousState, "keyboard");
  }

  /**
   * Exit edit mode.
   */
  exitEditMode(): void {
    const previousState = { ...this.state };
    this.state.isEditing = false;
    this.notifyListeners(previousState, "keyboard");
  }

  // ==========================================================================
  // Keyboard Navigation
  // ==========================================================================

  /**
   * Handle keyboard navigation.
   */
  handleKeyDown(event: KeyboardEvent): boolean {
    // If editing, let the INPUT handle Enter/Tab to save value first
    // Only handle Escape here (cancel without saving)
    if (this.state.isEditing) {
      if (event.key === "Escape") {
        this.exitEditMode();
        return true;
      }
      // DON'T handle Enter/Tab here - let input save value first via onEditComplete
      return false;
    }

    // Not editing - handle navigation
    switch (event.key) {
      case "ArrowUp":
        this.navigate("up", event.shiftKey);
        return true;
      case "ArrowDown":
        this.navigate("down", event.shiftKey);
        return true;
      case "ArrowLeft":
        this.navigate("left", event.shiftKey);
        return true;
      case "ArrowRight":
        this.navigate("right", event.shiftKey);
        return true;
      case "Tab":
        event.preventDefault();
        this.navigate(event.shiftKey ? "left" : "right");
        return true;
      case "Enter":
        if (this.state.activeCell) {
          this.enterEditMode();
        }
        return true;
      case "F2":
        if (this.state.activeCell) {
          this.enterEditMode();
        }
        return true;
      case "Escape":
        this.clearSelection();
        return true;
      case "Home":
        this.navigateToStart(event.ctrlKey || event.metaKey);
        return true;
      case "End":
        this.navigateToEnd(event.ctrlKey || event.metaKey);
        return true;
      default:
        // If it's a printable character and we have a selection, enter edit mode
        if (
          this.state.activeCell &&
          event.key.length === 1 &&
          !event.ctrlKey &&
          !event.metaKey
        ) {
          this.enterEditMode();
          return false; // Let the character through to the input
        }
        return false;
    }
  }

  /**
   * Navigate in a direction.
   */
  navigate(direction: NavigationDirection, extendRange: boolean = false): void {
    if (!this.state.activeCell) return;

    const currentRowIndex = this.rowIds.indexOf(this.state.activeCell.rowId);
    const currentColIndex = this.getColumnIndex(this.state.activeCell.columnId);

    let newRowIndex = currentRowIndex;
    let newColIndex = currentColIndex;

    switch (direction) {
      case "up":
        newRowIndex = this.findNextVisibleRow(currentRowIndex, -1);
        break;
      case "down":
        newRowIndex = this.findNextVisibleRow(currentRowIndex, 1);
        break;
      case "left":
        newColIndex = this.findNextEditableColumn(currentColIndex, -1);
        if (newColIndex === currentColIndex && this.options.wrap) {
          // Wrap to previous row, last column
          const prevRow = this.findNextVisibleRow(currentRowIndex, -1);
          if (prevRow !== currentRowIndex) {
            newRowIndex = prevRow;
            newColIndex = this.findLastEditableColumn();
          }
        }
        break;
      case "right":
        newColIndex = this.findNextEditableColumn(currentColIndex, 1);
        if (newColIndex === currentColIndex && this.options.wrap) {
          // Wrap to next row, first column
          const nextRow = this.findNextVisibleRow(currentRowIndex, 1);
          if (nextRow !== currentRowIndex) {
            newRowIndex = nextRow;
            newColIndex = this.findFirstEditableColumn();
          }
        }
        break;
    }

    const newRowId = this.rowIds[newRowIndex];
    const newColumnId = this.columns[newColIndex]?.id;

    if (!newRowId || !newColumnId) return;

    if (extendRange) {
      this.extendSelectionTo(newRowId, newColumnId);
    } else {
      this.selectCell(newRowId, newColumnId, "keyboard");
    }
  }

  /**
   * Navigate to start (Home key).
   */
  navigateToStart(toFirstRow: boolean): void {
    if (!this.state.activeCell) return;

    const firstColIndex = this.findFirstEditableColumn();
    const firstRowIndex = toFirstRow
      ? this.findNextVisibleRow(-1, 1)
      : this.rowIds.indexOf(this.state.activeCell.rowId);

    this.selectCell(
      this.rowIds[firstRowIndex],
      this.columns[firstColIndex].id,
      "keyboard",
    );
  }

  /**
   * Navigate to end (End key).
   */
  navigateToEnd(toLastRow: boolean): void {
    if (!this.state.activeCell) return;

    const lastColIndex = this.findLastEditableColumn();
    const lastRowIndex = toLastRow
      ? this.findNextVisibleRow(this.rowIds.length, -1)
      : this.rowIds.indexOf(this.state.activeCell.rowId);

    this.selectCell(
      this.rowIds[lastRowIndex],
      this.columns[lastColIndex].id,
      "keyboard",
    );
  }

  /**
   * Extend selection to a cell (Shift + navigation).
   */
  private extendSelectionTo(rowId: string, columnId: string): void {
    const previousState = { ...this.state };

    if (this.state.mode !== "range" || !this.state.range) {
      // Start new range from current active cell
      this.state = {
        ...this.state,
        mode: "range",
        range: {
          start: this.state.activeCell!,
          end: { rowId, columnId },
        },
      };
    } else {
      // Extend existing range
      this.state.range = {
        ...this.state.range,
        end: { rowId, columnId },
      };
    }

    this.notifyListeners(previousState, "keyboard");
  }

  // ==========================================================================
  // Helper Methods
  // ==========================================================================

  private getColumnIndex(columnId: string): number {
    return this.columns.findIndex((c) => c.id === columnId);
  }

  private findNextVisibleRow(currentIndex: number, direction: 1 | -1): number {
    let index = currentIndex + direction;

    while (index >= 0 && index < this.rowIds.length) {
      if (
        !this.options.skipHidden ||
        this.visibleRowIds.has(this.rowIds[index])
      ) {
        return index;
      }
      index += direction;
    }

    return currentIndex; // Stay in place if no valid row found
  }

  private findNextEditableColumn(
    currentIndex: number,
    direction: 1 | -1,
  ): number {
    let index = currentIndex + direction;

    while (index >= 0 && index < this.columns.length) {
      const column = this.columns[index];
      if (!this.options.skipLocked || column.editable) {
        return index;
      }
      index += direction;
    }

    return currentIndex; // Stay in place if no valid column found
  }

  private findFirstEditableColumn(): number {
    for (let i = 0; i < this.columns.length; i++) {
      if (!this.options.skipLocked || this.columns[i].editable) {
        return i;
      }
    }
    return 0;
  }

  private findLastEditableColumn(): number {
    for (let i = this.columns.length - 1; i >= 0; i--) {
      if (!this.options.skipLocked || this.columns[i].editable) {
        return i;
      }
    }
    return this.columns.length - 1;
  }

  // ==========================================================================
  // Configuration Updates
  // ==========================================================================

  /**
   * Update the list of row IDs.
   */
  setRowIds(rowIds: string[]): void {
    this.rowIds = rowIds;
    this.visibleRowIds = new Set(rowIds);

    // Clear selection if active cell is no longer valid
    if (
      this.state.activeCell &&
      !rowIds.includes(this.state.activeCell.rowId)
    ) {
      this.clearSelection();
    }
  }

  /**
   * Update visible row IDs (for collapsed groups).
   */
  setVisibleRowIds(visibleRowIds: Set<string>): void {
    this.visibleRowIds = visibleRowIds;

    // Move selection to nearest visible row if current is hidden
    if (
      this.state.activeCell &&
      !visibleRowIds.has(this.state.activeCell.rowId)
    ) {
      const currentIndex = this.rowIds.indexOf(this.state.activeCell.rowId);
      const nextVisibleIndex = this.findNextVisibleRow(currentIndex, 1);
      if (nextVisibleIndex !== currentIndex) {
        this.selectCell(
          this.rowIds[nextVisibleIndex],
          this.state.activeCell.columnId,
          "programmatic",
        );
      } else {
        this.clearSelection();
      }
    }
  }

  /**
   * Update column definitions.
   */
  setColumns(columns: ColumnDefinition[]): void {
    this.columns = columns;
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  /**
   * Subscribe to selection changes.
   */
  onSelectionChange(
    listener: (event: SelectionChangeEvent) => void,
  ): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify listeners of selection change.
   */
  private notifyListeners(
    previousState: SelectionState,
    trigger: SelectionChangeEvent["trigger"],
  ): void {
    const event: SelectionChangeEvent = {
      previousState,
      currentState: this.getState(),
      trigger,
    };

    this.listeners.forEach((listener) => listener(event));
  }

  // ==========================================================================
  // Copy/Paste Support
  // ==========================================================================

  /**
   * Get all cells in the current selection (for copy).
   */
  getSelectedCells(): CellPosition[] {
    if (this.state.mode === "cell" && this.state.activeCell) {
      return [{ ...this.state.activeCell }];
    }

    if (this.state.mode === "range" && this.state.range) {
      const cells: CellPosition[] = [];

      const startRowIndex = this.rowIds.indexOf(this.state.range.start.rowId);
      const endRowIndex = this.rowIds.indexOf(this.state.range.end.rowId);
      const startColIndex = this.getColumnIndex(
        this.state.range.start.columnId,
      );
      const endColIndex = this.getColumnIndex(this.state.range.end.columnId);

      const minRow = Math.min(startRowIndex, endRowIndex);
      const maxRow = Math.max(startRowIndex, endRowIndex);
      const minCol = Math.min(startColIndex, endColIndex);
      const maxCol = Math.max(startColIndex, endColIndex);

      for (let r = minRow; r <= maxRow; r++) {
        for (let c = minCol; c <= maxCol; c++) {
          cells.push({
            rowId: this.rowIds[r],
            columnId: this.columns[c].id,
          });
        }
      }

      return cells;
    }

    return [];
  }

  /**
   * Get selection bounds (for paste destination).
   */
  getSelectionBounds(): {
    startRow: number;
    endRow: number;
    startCol: number;
    endCol: number;
  } | null {
    if (!this.state.activeCell) return null;

    if (this.state.mode === "cell") {
      const rowIndex = this.rowIds.indexOf(this.state.activeCell.rowId);
      const colIndex = this.getColumnIndex(this.state.activeCell.columnId);
      return {
        startRow: rowIndex,
        endRow: rowIndex,
        startCol: colIndex,
        endCol: colIndex,
      };
    }

    if (this.state.mode === "range" && this.state.range) {
      const startRowIndex = this.rowIds.indexOf(this.state.range.start.rowId);
      const endRowIndex = this.rowIds.indexOf(this.state.range.end.rowId);
      const startColIndex = this.getColumnIndex(
        this.state.range.start.columnId,
      );
      const endColIndex = this.getColumnIndex(this.state.range.end.columnId);

      return {
        startRow: Math.min(startRowIndex, endRowIndex),
        endRow: Math.max(startRowIndex, endRowIndex),
        startCol: Math.min(startColIndex, endColIndex),
        endCol: Math.max(startColIndex, endColIndex),
      };
    }

    return null;
  }
}

// ============================================================================
// React Hook
// ============================================================================

import { useState, useCallback, useEffect, useRef } from "react";

export function useSelectionManager(
  columns: ColumnDefinition[],
  rowIds: string[],
  options?: Partial<NavigationOptions>,
) {
  const managerRef = useRef<SelectionManager | null>(null);
  const [state, setState] = useState<SelectionState>({
    mode: "none",
    activeCell: null,
    range: null,
    isEditing: false,
    isDragging: false,
  });

  // Initialize manager
  if (!managerRef.current) {
    managerRef.current = new SelectionManager(columns, rowIds, options);
  }

  // Update manager when columns/rows change
  useEffect(() => {
    managerRef.current?.setColumns(columns);
  }, [columns]);

  useEffect(() => {
    managerRef.current?.setRowIds(rowIds);
  }, [rowIds]);

  // Subscribe to state changes
  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;

    const unsubscribe = manager.onSelectionChange((event) => {
      setState(event.currentState);
    });

    return unsubscribe;
  }, []);

  // Memoized handlers
  const selectCell = useCallback((rowId: string, columnId: string) => {
    managerRef.current?.selectCell(rowId, columnId, "click");
  }, []);

  const startRangeSelection = useCallback((rowId: string, columnId: string) => {
    managerRef.current?.startRangeSelection(rowId, columnId);
  }, []);

  const extendRangeSelection = useCallback(
    (rowId: string, columnId: string) => {
      managerRef.current?.extendRangeSelection(rowId, columnId);
    },
    [],
  );

  const endRangeSelection = useCallback(() => {
    managerRef.current?.endRangeSelection();
  }, []);

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    return managerRef.current?.handleKeyDown(event) ?? false;
  }, []);

  const enterEditMode = useCallback(() => {
    managerRef.current?.enterEditMode();
  }, []);

  const exitEditMode = useCallback(() => {
    managerRef.current?.exitEditMode();
  }, []);

  const clearSelection = useCallback(() => {
    managerRef.current?.clearSelection();
  }, []);

  const setVisibleRowIds = useCallback((visibleRowIds: Set<string>) => {
    managerRef.current?.setVisibleRowIds(visibleRowIds);
  }, []);

  const isCellSelected = useCallback(
    (rowId: string, columnId: string) => {
      return managerRef.current?.isCellSelected(rowId, columnId) ?? false;
    },
    [state], // Depend on state to re-evaluate
  );

  const isCellInRange = useCallback(
    (rowId: string, columnId: string) => {
      return managerRef.current?.isCellInRange(rowId, columnId) ?? false;
    },
    [state],
  );

  const getSelectedCells = useCallback(() => {
    return managerRef.current?.getSelectedCells() ?? [];
  }, [state]);

  return {
    state,
    selectCell,
    startRangeSelection,
    extendRangeSelection,
    endRangeSelection,
    handleKeyDown,
    enterEditMode,
    exitEditMode,
    clearSelection,
    setVisibleRowIds,
    isCellSelected,
    isCellInRange,
    getSelectedCells,
    manager: managerRef.current,
  };
}

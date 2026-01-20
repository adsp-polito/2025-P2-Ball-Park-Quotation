/**
 * Undo/Redo Manager
 *
 * Manages action-level undo/redo history for the R&D Cost Table.
 * Groups related changes into single actions (e.g., drag-drop = single action).
 */

import type { RDTableRow, ActionType, TableAction } from "./types";

// ============================================================================
// Configuration
// ============================================================================

const MAX_HISTORY_SIZE = 50;

// ============================================================================
// Types
// ============================================================================

export interface UndoRedoState {
  canUndo: boolean;
  canRedo: boolean;
  undoDescription: string | null;
  redoDescription: string | null;
  historyLength: number;
  currentPosition: number;
}

export interface ActionData {
  type: ActionType;
  description: string;
  affectedRows: string[];
  beforeState: Partial<RDTableRow>[];
  afterState: Partial<RDTableRow>[];
}

export type UndoRedoCallback = (
  action: TableAction,
  direction: "undo" | "redo",
) => void;

// ============================================================================
// Undo/Redo Manager Class
// ============================================================================

export class UndoRedoManager {
  private history: TableAction[];
  private currentPosition: number; // Points to next action to undo
  private listeners: Set<() => void>;
  private applyCallback: UndoRedoCallback | null;
  private isApplying: boolean;

  constructor() {
    this.history = [];
    this.currentPosition = -1;
    this.listeners = new Set();
    this.applyCallback = null;
    this.isApplying = false;
  }

  // ==========================================================================
  // State Accessors
  // ==========================================================================

  getState(): UndoRedoState {
    return {
      canUndo: this.canUndo(),
      canRedo: this.canRedo(),
      undoDescription: this.getUndoDescription(),
      redoDescription: this.getRedoDescription(),
      historyLength: this.history.length,
      currentPosition: this.currentPosition,
    };
  }

  canUndo(): boolean {
    return this.currentPosition >= 0;
  }

  canRedo(): boolean {
    return this.currentPosition < this.history.length - 1;
  }

  getUndoDescription(): string | null {
    if (!this.canUndo()) return null;
    return this.history[this.currentPosition].description;
  }

  getRedoDescription(): string | null {
    if (!this.canRedo()) return null;
    return this.history[this.currentPosition + 1].description;
  }

  // ==========================================================================
  // Action Recording
  // ==========================================================================

  /**
   * Record a new action in history.
   * Clears any redo history when new action is recorded.
   */
  recordAction(action: ActionData): TableAction {
    // Don't record if we're currently applying an undo/redo
    if (this.isApplying) {
      return this.createTableAction(action);
    }

    // Clear redo history (everything after current position)
    this.history = this.history.slice(0, this.currentPosition + 1);

    // Create the action entry
    const tableAction = this.createTableAction(action);

    // Add to history
    this.history.push(tableAction);
    this.currentPosition = this.history.length - 1;

    // Trim if over max size
    if (this.history.length > MAX_HISTORY_SIZE) {
      const trimCount = this.history.length - MAX_HISTORY_SIZE;
      this.history = this.history.slice(trimCount);
      this.currentPosition -= trimCount;
    }

    this.notifyListeners();
    return tableAction;
  }

  /**
   * Create a table action object.
   */
  private createTableAction(action: ActionData): TableAction {
    return {
      id: crypto.randomUUID(),
      type: action.type,
      timestamp: new Date(),
      description: action.description,
      beforeState: action.beforeState,
      afterState: action.afterState,
      affectedRows: action.affectedRows,
    };
  }

  // ==========================================================================
  // Undo/Redo Operations
  // ==========================================================================

  /**
   * Undo the last action.
   * Returns the action that was undone, or null if nothing to undo.
   */
  undo(): TableAction | null {
    if (!this.canUndo()) return null;

    const action = this.history[this.currentPosition];
    this.currentPosition--;

    this.isApplying = true;
    try {
      this.applyCallback?.(action, "undo");
    } finally {
      this.isApplying = false;
    }

    this.notifyListeners();
    return action;
  }

  /**
   * Redo the previously undone action.
   * Returns the action that was redone, or null if nothing to redo.
   */
  redo(): TableAction | null {
    if (!this.canRedo()) return null;

    this.currentPosition++;
    const action = this.history[this.currentPosition];

    this.isApplying = true;
    try {
      this.applyCallback?.(action, "redo");
    } finally {
      this.isApplying = false;
    }

    this.notifyListeners();
    return action;
  }

  /**
   * Set the callback to apply undo/redo changes.
   */
  setApplyCallback(callback: UndoRedoCallback): void {
    this.applyCallback = callback;
  }

  // ==========================================================================
  // History Management
  // ==========================================================================

  /**
   * Get full history (for debugging/display).
   */
  getHistory(): TableAction[] {
    return [...this.history];
  }

  /**
   * Get recent history (most recent N actions).
   */
  getRecentHistory(count: number = 10): TableAction[] {
    const start = Math.max(0, this.history.length - count);
    return this.history.slice(start);
  }

  /**
   * Clear all history.
   */
  clear(): void {
    this.history = [];
    this.currentPosition = -1;
    this.notifyListeners();
  }

  // ==========================================================================
  // Batch Operations
  // ==========================================================================

  private batchAction: ActionData | null = null;
  private batchChanges: {
    beforeState: Partial<RDTableRow>[];
    afterState: Partial<RDTableRow>[];
    affectedRows: Set<string>;
  } | null = null;

  /**
   * Begin a batch operation (groups multiple changes).
   */
  beginBatch(type: ActionType, description: string): void {
    this.batchAction = {
      type,
      description,
      affectedRows: [],
      beforeState: [],
      afterState: [],
    };
    this.batchChanges = {
      beforeState: [],
      afterState: [],
      affectedRows: new Set(),
    };
  }

  /**
   * Add a change to the current batch.
   */
  addToBatch(
    rowId: string,
    beforeState: Partial<RDTableRow>,
    afterState: Partial<RDTableRow>,
  ): void {
    if (!this.batchChanges || !this.batchAction) {
      // Not in batch mode, record as single action
      this.recordAction({
        type: "cellEdit",
        description: "Edit cell",
        affectedRows: [rowId],
        beforeState: [beforeState],
        afterState: [afterState],
      });
      return;
    }

    this.batchChanges.beforeState.push(beforeState);
    this.batchChanges.afterState.push(afterState);
    this.batchChanges.affectedRows.add(rowId);
  }

  /**
   * End the batch operation and record as single action.
   */
  endBatch(): TableAction | null {
    if (!this.batchAction || !this.batchChanges) return null;

    const action: ActionData = {
      ...this.batchAction,
      beforeState: this.batchChanges.beforeState,
      afterState: this.batchChanges.afterState,
      affectedRows: Array.from(this.batchChanges.affectedRows),
    };

    this.batchAction = null;
    this.batchChanges = null;

    // Only record if there were actual changes
    if (action.beforeState.length > 0) {
      return this.recordAction(action);
    }

    return null;
  }

  /**
   * Cancel the current batch without recording.
   */
  cancelBatch(): void {
    this.batchAction = null;
    this.batchChanges = null;
  }

  /**
   * Check if currently in batch mode.
   */
  isInBatch(): boolean {
    return this.batchAction !== null;
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  /**
   * Subscribe to state changes.
   */
  onStateChange(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify listeners of state change.
   */
  private notifyListeners(): void {
    this.listeners.forEach((listener) => listener());
  }
}

// ============================================================================
// Action Helpers
// ============================================================================

/**
 * Create a cell edit action.
 */
export function createCellEditAction(
  rowId: string,
  columnName: string,
  oldValue: number | string,
  newValue: number | string,
  beforeRow: Partial<RDTableRow>,
  afterRow: Partial<RDTableRow>,
): ActionData {
  return {
    type: "cellEdit",
    description: `Edit ${columnName}`,
    affectedRows: [rowId],
    beforeState: [beforeRow],
    afterState: [afterRow],
  };
}

/**
 * Create a row add action.
 */
export function createRowAddAction(newRow: RDTableRow): ActionData {
  return {
    type: "rowAdd",
    description: `Add row`,
    affectedRows: [newRow.id],
    beforeState: [], // Empty - row didn't exist
    afterState: [newRow],
  };
}

/**
 * Create a row delete action.
 */
export function createRowDeleteAction(deletedRow: RDTableRow): ActionData {
  return {
    type: "rowDelete",
    description: `Delete row`,
    affectedRows: [deletedRow.id],
    beforeState: [deletedRow],
    afterState: [], // Empty - row no longer exists
  };
}

/**
 * Create a row move action.
 */
export function createRowMoveAction(
  rowId: string,
  fromIndex: number,
  toIndex: number,
  beforeState: RDTableRow[],
  afterState: RDTableRow[],
): ActionData {
  return {
    type: "rowMove",
    description: `Move row`,
    affectedRows: [rowId],
    beforeState,
    afterState,
  };
}

/**
 * Create a bulk edit action (e.g., paste).
 */
export function createBulkEditAction(
  description: string,
  affectedRows: string[],
  beforeState: Partial<RDTableRow>[],
  afterState: Partial<RDTableRow>[],
): ActionData {
  return {
    type: "bulkEdit",
    description,
    affectedRows,
    beforeState,
    afterState,
  };
}

/**
 * Create a reset action.
 */
export function createResetAction(
  beforeState: RDTableRow[],
  afterState: RDTableRow[],
): ActionData {
  return {
    type: "reset",
    description: "Reset to predicted values",
    affectedRows: beforeState.map((r) => r.id!),
    beforeState,
    afterState,
  };
}

// ============================================================================
// React Hook
// ============================================================================

import { useState, useCallback, useEffect, useRef } from "react";

export function useUndoRedoManager() {
  const managerRef = useRef<UndoRedoManager | null>(null);
  const [state, setState] = useState<UndoRedoState>({
    canUndo: false,
    canRedo: false,
    undoDescription: null,
    redoDescription: null,
    historyLength: 0,
    currentPosition: -1,
  });

  // Initialize manager
  if (!managerRef.current) {
    managerRef.current = new UndoRedoManager();
  }

  // Subscribe to state changes
  useEffect(() => {
    const manager = managerRef.current;
    if (!manager) return;

    const unsubscribe = manager.onStateChange(() => {
      setState(manager.getState());
    });

    return unsubscribe;
  }, []);

  // Memoized handlers
  const recordAction = useCallback((action: ActionData) => {
    return managerRef.current?.recordAction(action);
  }, []);

  const undo = useCallback(() => {
    return managerRef.current?.undo();
  }, []);

  const redo = useCallback(() => {
    return managerRef.current?.redo();
  }, []);

  const clear = useCallback(() => {
    managerRef.current?.clear();
  }, []);

  const beginBatch = useCallback((type: ActionType, description: string) => {
    managerRef.current?.beginBatch(type, description);
  }, []);

  const addToBatch = useCallback(
    (
      rowId: string,
      beforeState: Partial<RDTableRow>,
      afterState: Partial<RDTableRow>,
    ) => {
      managerRef.current?.addToBatch(rowId, beforeState, afterState);
    },
    [],
  );

  const endBatch = useCallback(() => {
    return managerRef.current?.endBatch();
  }, []);

  const cancelBatch = useCallback(() => {
    managerRef.current?.cancelBatch();
  }, []);

  const setApplyCallback = useCallback((callback: UndoRedoCallback) => {
    managerRef.current?.setApplyCallback(callback);
  }, []);

  const getHistory = useCallback(() => {
    return managerRef.current?.getHistory() ?? [];
  }, []);

  return {
    state,
    recordAction,
    undo,
    redo,
    clear,
    beginBatch,
    addToBatch,
    endBatch,
    cancelBatch,
    setApplyCallback,
    getHistory,
    manager: managerRef.current,
  };
}

// ============================================================================
// Keyboard Shortcut Hook
// ============================================================================

export function useUndoRedoKeyboard(
  undo: () => void,
  redo: () => void,
  canUndo: boolean,
  canRedo: boolean,
) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const isCtrlOrCmd = event.ctrlKey || event.metaKey;

      if (isCtrlOrCmd && event.key === "z") {
        if (event.shiftKey) {
          // Ctrl+Shift+Z = Redo
          if (canRedo) {
            event.preventDefault();
            redo();
          }
        } else {
          // Ctrl+Z = Undo
          if (canUndo) {
            event.preventDefault();
            undo();
          }
        }
      }

      // Ctrl+Y = Redo (Windows convention)
      if (isCtrlOrCmd && event.key === "y" && canRedo) {
        event.preventDefault();
        redo();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo, canUndo, canRedo]);
}

"use client";

/**
 * R&D Cost Table - Main Component
 *
 * Enterprise-grade Excel-like table for R&D cost estimation.
 * FLAT structure matching PE02 template - no hierarchy.
 */

import React, {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { cn } from "@/lib/utils";
import {
  type RDTableRow,
  type RDCostTableProps,
  type ColumnDefinition,
  type RowReasoning,
  type TableVersion,
} from "./types";
import {
  HyperFormulaEngine,
  getFormulaEngine,
  destroyFormulaEngine,
} from "./HyperFormulaEngine";
import { useSelectionManager } from "./SelectionManager";
import {
  useUndoRedoManager,
  useUndoRedoKeyboard,
  createCellEditAction,
} from "./UndoRedoManager";
import { RDCostTableHeader, createDefaultColumns } from "./RDCostTableHeader";
import { RDCostTableRow } from "./RDCostTableRow";
import { RDCostTableToolbar } from "./RDCostTableToolbar";
import { ReasoningToast } from "./ReasoningPanel";
import { VersionHistory } from "./VersionHistory";

// ============================================================================
// Constants
// ============================================================================

const ROW_HEIGHT = 32;
const REASONING_THRESHOLD = 0.15; // 15% change triggers reasoning prompt
const DEBOUNCE_MS = 100; // Quick sync for demo (was 2000ms)

// ============================================================================
// Main Component - FLAT table (no hierarchy)
// ============================================================================

export const RDCostTable = memo(function RDCostTable({
  sessionId: _sessionId,
  initialData,
  onDataChange,
  onFinalize,
  onExport,
  isLoading = false,
  isReadOnly = false,
}: RDCostTableProps) {
  // =========================================================================
  // Refs
  // =========================================================================
  const containerRef = useRef<HTMLDivElement>(null);
  const formulaEngineRef = useRef<HyperFormulaEngine | null>(null);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // =========================================================================
  // State
  // =========================================================================
  const [data, setData] = useState<RDTableRow[]>(initialData);
  const [columns, setColumns] = useState<ColumnDefinition[]>(
    createDefaultColumns(),
  );
  const [expandedReasoningRows, setExpandedReasoningRows] = useState<
    Set<string>
  >(new Set());
  const [reasonings, setReasonings] = useState<Map<string, RowReasoning>>(
    new Map(),
  );
  const [versions, setVersions] = useState<TableVersion[]>([]);
  const [currentVersionId, setCurrentVersionId] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showVersionHistory, setShowVersionHistory] = useState(false);

  // Reasoning prompt state
  const [pendingReasoningPrompt, setPendingReasoningPrompt] = useState<{
    rowId: string;
    columnName: string;
    originalValue: number;
    newValue: number;
    changePercent: number;
    confidence: number;
  } | null>(null);

  // =========================================================================
  // Derived State - FLAT (no tree flattening needed)
  // =========================================================================

  const rowIds = useMemo(() => data.map((r) => r.id), [data]);

  // =========================================================================
  // Initialize Managers
  // =========================================================================

  // Selection manager
  const selection = useSelectionManager(columns, rowIds, {
    wrap: true,
    skipLocked: true,
    skipHidden: true,
  });

  // Undo/redo manager
  const undoRedo = useUndoRedoManager();

  // Keyboard shortcuts for undo/redo
  useUndoRedoKeyboard(
    undoRedo.undo,
    undoRedo.redo,
    undoRedo.state.canUndo,
    undoRedo.state.canRedo,
  );

  // =========================================================================
  // Initialize Formula Engine (client-side only)
  // =========================================================================

  useEffect(() => {
    // Only initialize on client side
    const engine = getFormulaEngine();
    if (engine) {
      formulaEngineRef.current = engine;
      engine.initializeFromData(data);
    }

    return () => {
      destroyFormulaEngine();
    };
  }, []);

  // Re-initialize formula engine when data changes
  useEffect(() => {
    formulaEngineRef.current?.initializeFromData(data);
  }, [data]);

  // =========================================================================
  // Virtualization
  // =========================================================================

  const virtualizer = useVirtualizer({
    count: data.length,
    getScrollElement: () => containerRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  });

  // =========================================================================
  // Cell Edit Handler - FLAT structure
  // =========================================================================

  const handleCellValueChange = useCallback(
    (rowId: string, columnId: string, newValue: string | number) => {
      const rowIndex = data.findIndex((r) => r.id === rowId);
      if (rowIndex === -1) return;

      const row = data[rowIndex];

      // Get original value from flat structure
      const originalValue = (row as any)[columnId] ?? 0;

      // Record undo action
      const beforeState = { ...row };
      const afterState = { ...row, [columnId]: newValue };

      undoRedo.recordAction(
        createCellEditAction(
          rowId,
          columnId,
          originalValue,
          newValue,
          beforeState,
          afterState,
        ),
      );

      // Update state - simple flat array update
      setData((prev) =>
        prev.map((r) =>
          r.id === rowId ? { ...r, [columnId]: newValue, isEdited: true } : r,
        ),
      );
      setIsDirty(true);

      // Check for reasoning prompt (only for numeric columns)
      if (
        typeof originalValue === "number" &&
        typeof newValue === "number" &&
        originalValue !== 0
      ) {
        const changePercent = Math.abs(
          (newValue - originalValue) / originalValue,
        );
        if (changePercent >= REASONING_THRESHOLD) {
          setPendingReasoningPrompt({
            rowId,
            columnName: columnId,
            originalValue,
            newValue: newValue as number,
            changePercent: changePercent * 100,
            confidence: row.confidence,
          });
        }
      }

      // Immediate save for demo - call onDataChange directly with updated data
      const updatedData = data.map((r) =>
        r.id === rowId ? { ...r, [columnId]: newValue, isEdited: true } : r,
      );
      onDataChange(updatedData);
    },
    [data, undoRedo, onDataChange],
  );

  // =========================================================================
  // Row Operations
  // =========================================================================

  const handleToggleReasoning = useCallback((rowId: string) => {
    setExpandedReasoningRows((prev) => {
      const next = new Set(prev);
      if (next.has(rowId)) {
        next.delete(rowId);
      } else {
        next.add(rowId);
      }
      return next;
    });
  }, []);

  const handleAddRow = useCallback(() => {
    const newRow: RDTableRow = {
      id: crypto.randomUUID(),
      functionId: "",
      peFunction: "",
      mainActivitiesDescription: "",
      manpower: 0,
      benchDev: 0,
      benchSpecial: 0,
      benchDur: 0,
      vehicleTests: 0,
      investmentKEur: 0,
      confidence: 0.5,
    };
    setData((prev) => [...prev, newRow]);
    setIsDirty(true);
  }, []);

  // Calculate totals for all numeric columns (PE02 format)
  const totals = useMemo(
    () => ({
      manpower: data.reduce((sum, row) => sum + (row.manpower || 0), 0),
      benchDev: data.reduce((sum, row) => sum + (row.benchDev || 0), 0),
      benchSpecial: data.reduce((sum, row) => sum + (row.benchSpecial || 0), 0),
      benchDur: data.reduce((sum, row) => sum + (row.benchDur || 0), 0),
      vehicleTests: data.reduce((sum, row) => sum + (row.vehicleTests || 0), 0),
      investmentKEur: data.reduce(
        (sum, row) => sum + (row.investmentKEur || 0),
        0,
      ),
    }),
    [data],
  );

  const handleDeleteSelectedRows = useCallback(() => {
    const selectedCell = selection.state.activeCell;
    if (!selectedCell) return;

    setData((prev) => prev.filter((r) => r.id !== selectedCell.rowId));
    setIsDirty(true);
  }, [selection.state.activeCell]);

  // =========================================================================
  // Save & Version Operations
  // =========================================================================

  const handleSave = useCallback(async () => {
    if (!isDirty || isSaving) return;

    setIsSaving(true);
    try {
      await onDataChange(data);
      setIsDirty(false);
    } catch (error) {
      console.error("Failed to save:", error);
    } finally {
      setIsSaving(false);
    }
  }, [data, isDirty, isSaving, onDataChange]);

  const handleFinalize = useCallback(() => {
    if (isDirty) {
      handleSave().then(() => onFinalize());
    } else {
      onFinalize();
    }
  }, [isDirty, handleSave, onFinalize]);

  // =========================================================================
  // Clipboard Operations
  // =========================================================================

  const handleCut = useCallback(() => {
    handleCopy();
    // Clear selected cells
  }, []);

  const handleCopy = useCallback(() => {
    const selectedCells = selection.getSelectedCells();
    if (selectedCells.length === 0) return;

    // Build clipboard data
    const _clipboardData: string[][] = [];
    // ... implementation for Excel-compatible clipboard
  }, [selection]);

  const handlePaste = useCallback(() => {
    // ... paste implementation
  }, []);

  // =========================================================================
  // Reasoning Operations
  // =========================================================================

  const handleSaveReasoning = useCallback(
    (reasoning: Omit<RowReasoning, "id" | "createdAt">) => {
      const fullReasoning: RowReasoning = {
        ...reasoning,
        id: crypto.randomUUID(),
        createdAt: new Date(),
      };

      setReasonings((prev) => {
        const next = new Map(prev);
        next.set(`${reasoning.rowId}-${reasoning.columnName}`, fullReasoning);
        return next;
      });

      setPendingReasoningPrompt(null);
    },
    [],
  );

  const handleDismissReasoning = useCallback(() => {
    setPendingReasoningPrompt(null);
  }, []);

  // =========================================================================
  // Version History Operations
  // =========================================================================

  const handleViewVersion = useCallback((_version: TableVersion) => {
    // Show version in read-only mode
  }, []);

  const handleRestoreVersion = useCallback((version: TableVersion) => {
    setData(version.data);
    setCurrentVersionId(version.id);
    setIsDirty(true);
  }, []);

  const handleForkVersion = useCallback((_version: TableVersion) => {
    // Create new branch from version
  }, []);

  const handleDownloadVersion = useCallback(
    (_version: TableVersion, _format: "xlsx" | "json") => {
      // Download version data
    },
    [],
  );

  // =========================================================================
  // Column Resize
  // =========================================================================

  const handleColumnResize = useCallback(
    (columnId: string, newWidth: number) => {
      setColumns((prev) =>
        prev.map((col) =>
          col.id === columnId
            ? { ...col, width: Math.max(col.minWidth || 50, newWidth) }
            : col,
        ),
      );
    },
    [],
  );

  // =========================================================================
  // Keyboard Handler
  // =========================================================================

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Let selection manager handle navigation
      if (selection.handleKeyDown(e)) {
        e.preventDefault();
      }
    };

    const container = containerRef.current;
    container?.addEventListener("keydown", handleKeyDown);

    return () => {
      container?.removeEventListener("keydown", handleKeyDown);
    };
  }, [selection.handleKeyDown]);

  // =========================================================================
  // Render
  // =========================================================================

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full border border-gray-200 rounded-lg overflow-hidden bg-white">
      {/* Toolbar */}
      <RDCostTableToolbar
        canUndo={undoRedo.state.canUndo}
        canRedo={undoRedo.state.canRedo}
        undoDescription={undoRedo.state.undoDescription}
        redoDescription={undoRedo.state.redoDescription}
        hasSelection={selection.state.activeCell !== null}
        onUndo={undoRedo.undo}
        onRedo={undoRedo.redo}
        onCut={handleCut}
        onCopy={handleCopy}
        onPaste={handlePaste}
        onAddRow={handleAddRow}
        onDeleteRows={handleDeleteSelectedRows}
        onShowHistory={() => setShowVersionHistory(true)}
        onResetToAI={() => setData(initialData)}
        onSave={handleSave}
        onFinalize={handleFinalize}
        onExport={onExport}
        isDirty={isDirty}
        isSaving={isSaving}
        isReadOnly={isReadOnly}
      />

      {/* Table Container */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto"
        role="grid"
        aria-rowcount={data.length + 1}
        aria-colcount={columns.length}
        tabIndex={0}
      >
        {/* Header */}
        <RDCostTableHeader
          columns={columns}
          onColumnResize={handleColumnResize}
        />

        {/* Virtualized Rows */}
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            width: "100%",
            position: "relative",
          }}
        >
          {virtualizer.getVirtualItems().map((virtualRow) => {
            const row = data[virtualRow.index];
            const hasReasoning = reasonings.has(`${row.id}-manpower`); // Check any reasoning
            const isReasoningExpanded = expandedReasoningRows.has(row.id);

            return (
              <div
                key={row.id}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: `${virtualRow.size}px`,
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <RDCostTableRow
                  row={row}
                  columns={columns}
                  rowIndex={virtualRow.index}
                  hasReasoning={hasReasoning}
                  isReasoningExpanded={isReasoningExpanded}
                  formulaEngine={formulaEngineRef.current}
                  selectionState={{
                    isCellSelected: selection.isCellSelected,
                    isCellInRange: selection.isCellInRange,
                    editingCell:
                      selection.state.isEditing && selection.state.activeCell
                        ? selection.state.activeCell
                        : null,
                  }}
                  onToggleReasoning={handleToggleReasoning}
                  onCellSelect={selection.selectCell}
                  onCellRangeStart={selection.startRangeSelection}
                  onCellRangeExtend={selection.extendRangeSelection}
                  onCellDoubleClick={(rowId, columnId) => {
                    selection.selectCell(rowId, columnId);
                    selection.enterEditMode();
                  }}
                  onCellValueChange={handleCellValueChange}
                  onEditComplete={selection.exitEditMode}
                  onEditCancel={selection.exitEditMode}
                />
              </div>
            );
          })}
        </div>

        {/* Total Row (PE02 format) - Excel-style footer */}
        <div
          className="flex items-stretch sticky bottom-0 z-10 bg-gray-100 font-semibold border-t-2 border-gray-400"
          role="row"
          aria-rowindex={data.length + 3}
        >
          {/* Row number column */}
          <div
            className="flex items-center justify-center shrink-0 border-r border-gray-300 bg-gray-200"
            style={{ width: 40 }}
          />
          {/* TOTAL label and values */}
          {columns.map((column, colIndex) => {
            const isIdentifierColumn = [
              "functionId",
              "peFunction",
              "mainActivitiesDescription",
            ].includes(column.id);
            const isNumericColumn = [
              "manpower",
              "benchDev",
              "benchSpecial",
              "benchDur",
              "vehicleTests",
              "investmentKEur",
            ].includes(column.id);
            const isLastColumn = colIndex === columns.length - 1;

            if (isIdentifierColumn && column.id === "functionId") {
              return (
                <div
                  key={column.id}
                  className="flex items-center justify-center border-r border-gray-300 text-sm font-bold text-gray-700 h-8"
                  style={{ width: column.width }}
                />
              );
            }
            if (isIdentifierColumn && column.id === "peFunction") {
              return (
                <div
                  key={column.id}
                  className="flex items-center border-r border-gray-300 h-8"
                  style={{ width: column.width }}
                />
              );
            }
            if (
              isIdentifierColumn &&
              column.id === "mainActivitiesDescription"
            ) {
              return (
                <div
                  key={column.id}
                  className="flex items-center border-r border-gray-300 h-8"
                  style={{ width: column.width }}
                />
              );
            }
            if (isNumericColumn) {
              const value = totals[column.id as keyof typeof totals];
              const isInvestment = column.id === "investmentKEur";
              return (
                <div
                  key={column.id}
                  className={cn(
                    "flex items-center justify-end px-2 text-sm font-mono font-bold text-gray-900 h-8",
                    !isLastColumn && "border-r border-gray-300",
                  )}
                  style={{ width: column.width }}
                >
                  {isInvestment
                    ? value.toLocaleString("en-US", {
                        minimumFractionDigits: 0,
                        maximumFractionDigits: 0,
                      })
                    : value.toLocaleString("en-US")}
                </div>
              );
            }
            return (
              <div
                key={column.id}
                className={cn(
                  "flex items-center h-8",
                  !isLastColumn && "border-r border-gray-300",
                )}
                style={{ width: column.width }}
              />
            );
          })}
        </div>
      </div>

      {/* Reasoning Toast (non-blocking) */}
      {pendingReasoningPrompt && (
        <ReasoningToast
          rowId={pendingReasoningPrompt.rowId}
          changePercent={pendingReasoningPrompt.changePercent}
          onOpen={() => {
            setExpandedReasoningRows((prev) => {
              const next = new Set(prev);
              next.add(pendingReasoningPrompt.rowId);
              return next;
            });
          }}
          onDismiss={handleDismissReasoning}
        />
      )}

      {/* Version History Panel */}
      <VersionHistory
        isOpen={showVersionHistory}
        onClose={() => setShowVersionHistory(false)}
        versions={versions}
        currentVersionId={currentVersionId}
        onViewVersion={handleViewVersion}
        onRestoreVersion={handleRestoreVersion}
        onForkVersion={handleForkVersion}
        onDownloadVersion={handleDownloadVersion}
      />
    </div>
  );
});

// ============================================================================
// Export Index
// ============================================================================

export * from "./types";
export * from "./HyperFormulaEngine";
export * from "./SelectionManager";
export * from "./UndoRedoManager";
export * from "./RDCostTableCell";
export * from "./RDCostTableRow";
export * from "./RDCostTableHeader";
export * from "./RDCostTableToolbar";
export * from "./ReasoningPanel";
export * from "./ChangeTagSelector";
export * from "./VersionHistory";

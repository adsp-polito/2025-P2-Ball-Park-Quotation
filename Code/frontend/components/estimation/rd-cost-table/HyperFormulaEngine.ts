/**
 * HyperFormula Engine Wrapper
 *
 * Provides Excel-like formula capabilities for the R&D Cost Table.
 * Wraps HyperFormula library with domain-specific validation and
 * same-row reference enforcement.
 *
 * NOTE: HyperFormula requires browser APIs and cannot run during SSR.
 * This module uses dynamic import to load HyperFormula only on the client.
 */

import type { RDTableRow } from "./types";

// Type definitions for SSR compatibility
// These match HyperFormula's types without requiring the actual import
type CellValue = string | number | boolean | null | undefined | object;
type SimpleCellAddress = { sheet: number; row: number; col: number };
type ExportedChange = { address?: SimpleCellAddress; newValue?: CellValue };

// HyperFormula instance type (will be dynamically imported)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type HyperFormulaInstance = any;

// ============================================================================
// Configuration
// ============================================================================

// HyperFormula configuration
// Note: Cast separators to their literal types to match HyperFormula's strict typing
// IMPORTANT: functionArgSeparator and thousandSeparator MUST be different characters!
const HF_CONFIG = {
  licenseKey: "gpl-v3",
  useColumnIndex: true,
  useStats: false,
  evaluateNullToZero: true,
  precisionRounding: 2,
  smartRounding: true,
  leapYear1900: false,
  nullYear: 30, // Must be <= 100, represents 1930 as 30
  dateFormats: ["DD/MM/YYYY", "YYYY-MM-DD"],
  functionArgSeparator: "," as "," | ";", // Comma for function arguments: =SUM(1,2,3)
  thousandSeparator: " " as "," | " " | "" | ".", // Space for thousands: 1 000 000 (avoids conflict with comma)
  decimalSeparator: "." as "." | ",", // Period for decimals: 1.5
  language: "enGB",
};

// Column mapping for FLAT table (matches PE02 template columns)
// Order: Function ID | PE Function | Main Activities | Manpower | Bench(Dev) | Bench(Special) | Bench(Dur) | Vehicle tests | Investment [k€]
const COLUMN_MAP = {
  functionId: 0, // A1, A2, B1, B2, C, D1, D2, D3, E, F, G
  peFunction: 1, // PE Function name
  mainActivitiesDescription: 2,
  manpower: 3, // Hours
  benchDev: 4, // Bench Development hours
  benchSpecial: 5, // Bench Special hours (NEW)
  benchDur: 6, // Bench Durability hours
  vehicleTests: 7, // Vehicle tests hours
  investmentKEur: 8, // Investment in k€
} as const;

type ColumnName = keyof typeof COLUMN_MAP;

// ============================================================================
// Types
// ============================================================================

export interface FormulaValidationResult {
  isValid: boolean;
  error?: string;
  referencedCells?: string[];
  hasCrossRowReference?: boolean;
}

export interface CellChangeEvent {
  rowId: string;
  columnName: string;
  oldValue: CellValue;
  newValue: CellValue;
  isFormula: boolean;
}

export interface FormulaEngineState {
  sheetId: number;
  rowMapping: Map<string, number>; // rowId -> sheet row
  reverseRowMapping: Map<number, string>; // sheet row -> rowId
}

// ============================================================================
// Browser Check Helper
// ============================================================================

const isBrowser = typeof window !== "undefined";

// ============================================================================
// HyperFormula Engine Class
// ============================================================================

export class HyperFormulaEngine {
  private hf: HyperFormulaInstance | null = null;
  private state: FormulaEngineState;
  private changeListeners: Set<(changes: CellChangeEvent[]) => void>;
  private isUpdating: boolean = false;
  private isInitialized: boolean = false;
  private initPromise: Promise<void> | null = null;

  constructor() {
    this.state = {
      sheetId: 0,
      rowMapping: new Map(),
      reverseRowMapping: new Map(),
    };
    this.changeListeners = new Set();

    // Only initialize on browser
    if (isBrowser) {
      this.initPromise = this.initializeHyperFormula();
    }
  }

  /**
   * Dynamically import and initialize HyperFormula.
   * Only runs on the client side.
   */
  private async initializeHyperFormula(): Promise<void> {
    if (!isBrowser || this.isInitialized) return;

    try {
      const HyperFormula = (await import("hyperformula")).default;
      this.hf = HyperFormula.buildEmpty(HF_CONFIG);
      const sheetName = this.hf.addSheet("RDCostTable");
      const sheetId = this.hf.getSheetId(sheetName ?? "RDCostTable");
      this.state.sheetId = sheetId ?? 0;

      // Subscribe to value changes - filter to cell changes only
      this.hf.on("valuesUpdated", (changes: ExportedChange[]) => {
        if (this.isUpdating) return;
        // Filter to only cell changes (exclude named expression changes and array values)
        const cellChanges = changes.filter(
          (
            c,
          ): c is ExportedChange & {
            address: SimpleCellAddress;
            newValue: CellValue;
          } => "address" in c && !Array.isArray(c.newValue),
        );
        if (cellChanges.length > 0) {
          this.handleValueChanges(cellChanges);
        }
      });

      this.isInitialized = true;
    } catch (error) {
      console.error("Failed to initialize HyperFormula:", error);
    }
  }

  /**
   * Wait for HyperFormula to be initialized.
   */
  async waitForInit(): Promise<boolean> {
    if (!isBrowser) return false;
    if (this.isInitialized) return true;
    if (this.initPromise) {
      await this.initPromise;
    }
    return this.isInitialized;
  }

  /**
   * Check if engine is ready to use.
   */
  isReady(): boolean {
    return this.isInitialized && this.hf !== null;
  }

  // ==========================================================================
  // Initialization & Data Sync
  // ==========================================================================

  /**
   * Initialize the engine with flattened table data.
   * Builds the HyperFormula sheet from hierarchical data.
   */
  initializeFromData(flattenedRows: RDTableRow[]): void {
    if (!this.hf) {
      // Store data to initialize when ready
      this.initPromise?.then(() => {
        if (this.hf) {
          this.initializeFromData(flattenedRows);
        }
      });
      return;
    }

    this.isUpdating = true;

    try {
      // Clear existing data
      this.hf.clearSheet(this.state.sheetId);
      this.state.rowMapping.clear();
      this.state.reverseRowMapping.clear();

      // Build row mapping and populate sheet
      flattenedRows.forEach((row, index) => {
        this.state.rowMapping.set(row.id, index);
        this.state.reverseRowMapping.set(index, row.id);
        this.setRowData(index, row);
      });
    } finally {
      this.isUpdating = false;
    }
  }

  /**
   * Set data for a single row in the HyperFormula sheet.
   * Matches PE02 column structure.
   */
  private setRowData(sheetRow: number, row: RDTableRow): void {
    if (!this.hf) return;

    const address = (col: number): SimpleCellAddress => ({
      sheet: this.state.sheetId,
      row: sheetRow,
      col,
    });

    // Set text columns (PE02 format)
    this.hf.setCellContents(
      address(COLUMN_MAP.functionId),
      row.functionId ?? "",
    );
    this.hf.setCellContents(
      address(COLUMN_MAP.peFunction),
      row.peFunction ?? "",
    );
    this.hf.setCellContents(
      address(COLUMN_MAP.mainActivitiesDescription),
      row.mainActivitiesDescription ?? "",
    );

    // Set numeric effort columns (may contain formulas)
    const numericColumns: (keyof typeof COLUMN_MAP)[] = [
      "manpower",
      "benchDev",
      "benchSpecial",
      "benchDur",
      "vehicleTests",
    ];

    numericColumns.forEach((col) => {
      const formula = row.formulas?.[col];
      const value = formula || row[col as keyof RDTableRow] || 0;
      this.hf.setCellContents(address(COLUMN_MAP[col]), value);
    });

    // Set investment k€ column (may have formula)
    const investmentFormula = row.formulas?.investmentKEur;
    this.hf.setCellContents(
      address(COLUMN_MAP.investmentKEur),
      investmentFormula || row.investmentKEur || 0,
    );
  }

  // ==========================================================================
  // Cell Operations
  // ==========================================================================

  /**
   * Get the current value of a cell.
   */
  getCellValue(rowId: string, columnName: ColumnName): CellValue {
    if (!this.hf) return null;

    const sheetRow = this.state.rowMapping.get(rowId);
    if (sheetRow === undefined) return null;

    const col = COLUMN_MAP[columnName];
    const address: SimpleCellAddress = {
      sheet: this.state.sheetId,
      row: sheetRow,
      col,
    };

    return this.hf.getCellValue(address);
  }

  /**
   * Get the raw cell content (formula string if applicable).
   */
  getCellFormula(rowId: string, columnName: ColumnName): string | null {
    if (!this.hf) return null;

    const sheetRow = this.state.rowMapping.get(rowId);
    if (sheetRow === undefined) return null;

    const col = COLUMN_MAP[columnName];
    const address: SimpleCellAddress = {
      sheet: this.state.sheetId,
      row: sheetRow,
      col,
    };

    const serialized = this.hf.getCellSerialized(address);
    if (typeof serialized === "string" && serialized.startsWith("=")) {
      return serialized;
    }
    return null;
  }

  /**
   * Set cell content (value or formula).
   * Validates same-row references before applying.
   */
  setCellContent(
    rowId: string,
    columnName: ColumnName,
    content: string | number,
  ): FormulaValidationResult {
    if (!this.hf) {
      return { isValid: false, error: "Engine not initialized" };
    }

    const sheetRow = this.state.rowMapping.get(rowId);
    if (sheetRow === undefined) {
      return { isValid: false, error: "Row not found" };
    }

    // Validate formula if applicable
    const contentStr = String(content);
    if (contentStr.startsWith("=")) {
      const validation = this.validateFormula(contentStr, sheetRow);
      if (!validation.isValid) {
        return validation;
      }
    }

    const col = COLUMN_MAP[columnName];
    const address: SimpleCellAddress = {
      sheet: this.state.sheetId,
      row: sheetRow,
      col,
    };

    try {
      this.hf.setCellContents(address, content);
      return { isValid: true };
    } catch (error) {
      return {
        isValid: false,
        error: error instanceof Error ? error.message : "Unknown error",
      };
    }
  }

  // ==========================================================================
  // Formula Validation
  // ==========================================================================

  /**
   * Validate a formula string.
   * Enforces same-row-only references as per spec.
   */
  validateFormula(formula: string, sourceRow: number): FormulaValidationResult {
    if (!formula.startsWith("=")) {
      return { isValid: true };
    }

    // Parse the formula to extract cell references
    const cellRefPattern = /\$?([A-Z]+)\$?(\d+)/gi;
    const matches = formula.matchAll(cellRefPattern);
    const referencedCells: string[] = [];
    let hasCrossRowReference = false;

    for (const match of matches) {
      const refRow = parseInt(match[2], 10) - 1; // Convert to 0-indexed
      referencedCells.push(match[0]);

      if (refRow !== sourceRow) {
        hasCrossRowReference = true;
      }
    }

    if (hasCrossRowReference) {
      return {
        isValid: false,
        error:
          "Cross-row references are not allowed. Formulas must reference cells in the same row only.",
        referencedCells,
        hasCrossRowReference: true,
      };
    }

    // Let HyperFormula validate the formula syntax
    try {
      // Create a temporary address for validation
      const tempAddress: SimpleCellAddress = {
        sheet: this.state.sheetId,
        row: sourceRow,
        col: 0,
      };
      // Use calculateFormula to check if formula is valid
      // (validateFormula is not available in this HyperFormula version)
      const testResult = this.hf.calculateFormula(formula, tempAddress.sheet);
      // If it throws or returns an error type, formula is invalid
      if (testResult === null || typeof testResult === "object") {
        // Check if it's a detailed error
        const errorTypes = [
          "#DIV/0!",
          "#VALUE!",
          "#REF!",
          "#NAME?",
          "#NUM!",
          "#N/A",
          "#ERROR!",
        ];
        if (
          typeof testResult === "object" &&
          testResult !== null &&
          "type" in testResult
        ) {
          const errorType = (testResult as { type: string }).type;
          if (errorTypes.some((e) => errorType?.includes(e))) {
            return {
              isValid: false,
              error: `Formula error: ${errorType}`,
              referencedCells,
            };
          }
        }
      }
    } catch (error) {
      return {
        isValid: false,
        error:
          error instanceof Error ? error.message : "Formula validation failed",
        referencedCells,
      };
    }

    return {
      isValid: true,
      referencedCells,
      hasCrossRowReference: false,
    };
  }

  /**
   * Check if a cell contains a formula.
   */
  isFormula(rowId: string, columnName: ColumnName): boolean {
    if (!this.hf) return false;

    const sheetRow = this.state.rowMapping.get(rowId);
    if (sheetRow === undefined) return false;

    const col = COLUMN_MAP[columnName];
    const address: SimpleCellAddress = {
      sheet: this.state.sheetId,
      row: sheetRow,
      col,
    };

    return this.hf.doesCellHaveFormula(address);
  }

  /**
   * Check if a cell has an error.
   */
  hasError(rowId: string, columnName: ColumnName): boolean {
    const value = this.getCellValue(rowId, columnName);
    return this.isDetailedError(value);
  }

  /**
   * Check if a value is a HyperFormula detailed error.
   */
  private isDetailedError(value: CellValue): boolean {
    if (typeof value === "object" && value !== null && "type" in value) {
      const errorTypes = [
        "ERROR",
        "DIV_BY_ZERO",
        "VALUE",
        "REF",
        "NAME",
        "NUM",
        "NA",
      ];
      return errorTypes.some((t) =>
        (value as { type: string }).type?.includes(t),
      );
    }
    return false;
  }

  /**
   * Get error message for a cell.
   */
  getErrorMessage(rowId: string, columnName: ColumnName): string | null {
    const value = this.getCellValue(rowId, columnName);
    if (this.isDetailedError(value)) {
      // Extract error message from the detailed error object
      if (typeof value === "object" && value !== null) {
        const errorObj = value as { type?: string; value?: string };
        return errorObj.value ?? errorObj.type ?? "Unknown error";
      }
    }
    return null;
  }

  // ==========================================================================
  // Utility Functions
  // ==========================================================================

  /**
   * Convert column index to Excel-style letter (0 -> A, 1 -> B, etc.).
   */
  static columnIndexToLetter(index: number): string {
    let result = "";
    let temp = index;
    while (temp >= 0) {
      result = String.fromCharCode((temp % 26) + 65) + result;
      temp = Math.floor(temp / 26) - 1;
    }
    return result;
  }

  /**
   * Convert Excel-style column letter to index (A -> 0, B -> 1, etc.).
   */
  static columnLetterToIndex(letter: string): number {
    let result = 0;
    for (let i = 0; i < letter.length; i++) {
      result = result * 26 + (letter.charCodeAt(i) - 64);
    }
    return result - 1;
  }

  /**
   * Get cell address in A1 notation.
   */
  getCellAddress(rowId: string, columnName: ColumnName): string | null {
    const sheetRow = this.state.rowMapping.get(rowId);
    if (sheetRow === undefined) return null;

    const col = COLUMN_MAP[columnName];
    const letter = HyperFormulaEngine.columnIndexToLetter(col);
    return `${letter}${sheetRow + 1}`;
  }

  /**
   * Build a SUM formula for effort columns in the same row.
   * Sums: manpower, benchDev, benchSpecial, benchDur, vehicleTests
   */
  buildResourceSumFormula(rowNumber: number): string {
    // Columns D through H (manpower to vehicleTests)
    const startCol = HyperFormulaEngine.columnIndexToLetter(
      COLUMN_MAP.manpower,
    );
    const endCol = HyperFormulaEngine.columnIndexToLetter(
      COLUMN_MAP.vehicleTests,
    );
    return `=SUM(${startCol}${rowNumber}:${endCol}${rowNumber})`;
  }

  // ==========================================================================
  // Event Handling
  // ==========================================================================

  /**
   * Handle value changes from HyperFormula.
   * Note: ExportedChange only provides newValue, not oldValue in newer versions.
   */
  private handleValueChanges(
    changes: Array<
      ExportedChange & { address: SimpleCellAddress; newValue: CellValue }
    >,
  ): void {
    const cellChanges: CellChangeEvent[] = [];

    for (const change of changes) {
      const rowId = this.state.reverseRowMapping.get(change.address.row);
      if (!rowId) continue;

      const columnName = this.getColumnNameByIndex(change.address.col);
      if (!columnName) continue;

      cellChanges.push({
        rowId,
        columnName,
        oldValue: null, // HyperFormula ExportedChange doesn't provide oldValue
        newValue: change.newValue,
        isFormula: this.isFormula(rowId, columnName),
      });
    }

    if (cellChanges.length > 0) {
      this.notifyListeners(cellChanges);
    }
  }

  /**
   * Get column name from column index.
   */
  private getColumnNameByIndex(index: number): ColumnName | null {
    for (const [name, col] of Object.entries(COLUMN_MAP)) {
      if (col === index) {
        return name as ColumnName;
      }
    }
    return null;
  }

  /**
   * Subscribe to cell change events.
   */
  onCellChange(listener: (changes: CellChangeEvent[]) => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }

  /**
   * Notify all listeners of cell changes.
   */
  private notifyListeners(changes: CellChangeEvent[]): void {
    this.changeListeners.forEach((listener) => listener(changes));
  }

  // ==========================================================================
  // Batch Operations
  // ==========================================================================

  /**
   * Begin a batch operation (groups multiple changes).
   */
  beginBatch(): void {
    if (!this.hf) return;
    this.hf.suspendEvaluation();
  }

  /**
   * End a batch operation and recalculate.
   */
  endBatch(): void {
    if (!this.hf) return;
    this.hf.resumeEvaluation();
  }

  /**
   * Perform operations within a batch.
   */
  batch<T>(operation: () => T): T {
    this.beginBatch();
    try {
      return operation();
    } finally {
      this.endBatch();
    }
  }

  // ==========================================================================
  // Row Operations
  // ==========================================================================

  /**
   * Add a new row to the sheet.
   */
  addRow(rowId: string, atIndex: number, rowData: RDTableRow): void {
    if (!this.hf) return;

    this.isUpdating = true;
    try {
      // Shift existing mappings
      const newRowMapping = new Map<string, number>();
      const newReverseMapping = new Map<number, string>();

      this.state.rowMapping.forEach((row, id) => {
        if (row >= atIndex) {
          newRowMapping.set(id, row + 1);
          newReverseMapping.set(row + 1, id);
        } else {
          newRowMapping.set(id, row);
          newReverseMapping.set(row, id);
        }
      });

      // Add new row mapping
      newRowMapping.set(rowId, atIndex);
      newReverseMapping.set(atIndex, rowId);

      this.state.rowMapping = newRowMapping;
      this.state.reverseRowMapping = newReverseMapping;

      // Insert row in HyperFormula
      this.hf.addRows(this.state.sheetId, [atIndex, 1]);
      this.setRowData(atIndex, rowData);
    } finally {
      this.isUpdating = false;
    }
  }

  /**
   * Remove a row from the sheet.
   */
  removeRow(rowId: string): void {
    if (!this.hf) return;

    const sheetRow = this.state.rowMapping.get(rowId);
    if (sheetRow === undefined) return;

    this.isUpdating = true;
    try {
      // Remove row in HyperFormula
      this.hf.removeRows(this.state.sheetId, [sheetRow, 1]);

      // Update mappings
      const newRowMapping = new Map<string, number>();
      const newReverseMapping = new Map<number, string>();

      this.state.rowMapping.forEach((row, id) => {
        if (id === rowId) return; // Skip removed row
        if (row > sheetRow) {
          newRowMapping.set(id, row - 1);
          newReverseMapping.set(row - 1, id);
        } else {
          newRowMapping.set(id, row);
          newReverseMapping.set(row, id);
        }
      });

      this.state.rowMapping = newRowMapping;
      this.state.reverseRowMapping = newReverseMapping;
    } finally {
      this.isUpdating = false;
    }
  }

  // ==========================================================================
  // Export & Cleanup
  // ==========================================================================

  /**
   * Export all data from the sheet.
   */
  exportData(): (CellValue | null)[][] {
    if (!this.hf) return [];
    return this.hf.getSheetValues(this.state.sheetId);
  }

  /**
   * Get all formulas in the sheet.
   * Converts undefined values to null for consistent typing.
   */
  exportFormulas(): (string | number | boolean | Date | null)[][] {
    if (!this.hf) return [];
    const raw = this.hf.getSheetSerialized(this.state.sheetId);
    return raw.map((row: (string | number | boolean | Date | undefined)[]) =>
      row.map((cell: string | number | boolean | Date | undefined) =>
        cell === undefined ? null : cell,
      ),
    );
  }

  /**
   * Destroy the engine and clean up resources.
   */
  destroy(): void {
    this.changeListeners.clear();
    if (this.hf) {
      this.hf.destroy();
      this.hf = null;
    }
    this.isInitialized = false;
  }
}

// ============================================================================
// Singleton Instance (optional - can also create per-component)
// ============================================================================

let engineInstance: HyperFormulaEngine | null = null;

/**
 * Get the formula engine singleton.
 * Only creates an instance on the client side.
 */
export function getFormulaEngine(): HyperFormulaEngine | null {
  if (!isBrowser) {
    return null; // Return null during SSR
  }
  if (!engineInstance) {
    engineInstance = new HyperFormulaEngine();
  }
  return engineInstance;
}

export function destroyFormulaEngine(): void {
  if (engineInstance) {
    engineInstance.destroy();
    engineInstance = null;
  }
}

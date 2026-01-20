"use client";

/**
 * R&D Cost Table Header Component
 *
 * Excel-style two-tier header with grouped columns.
 * Matches FPT PE02 template format exactly.
 */

import React, { memo, useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { type ColumnDefinition, FPT_COLORS } from "./types";

// ============================================================================
// Types
// ============================================================================

export interface RDCostTableHeaderProps {
  columns: ColumnDefinition[];
  onColumnResize: (columnId: string, newWidth: number) => void;
  leftOffset?: number;
}

// Column group definitions for the two-tier header
const EFFORT_COLUMNS = ["manpower", "benchDev", "benchSpecial", "benchDur"];
const VEHICLE_TESTS_COLUMN = "vehicleTests";
const IDENTIFIER_COLUMNS = [
  "functionId",
  "peFunction",
  "mainActivitiesDescription",
];

// ============================================================================
// Header Component - Excel-style two-tier structure
// ============================================================================

export const RDCostTableHeader = memo(function RDCostTableHeader({
  columns,
  onColumnResize,
  leftOffset = 40,
}: RDCostTableHeaderProps) {
  // Calculate widths for grouped headers
  const effortColumnsWidth = columns
    .filter((c) => EFFORT_COLUMNS.includes(c.id))
    .reduce((sum, c) => sum + c.width, 0);

  const vehicleTestsColumn = columns.find((c) => c.id === VEHICLE_TESTS_COLUMN);
  const investmentColumn = columns.find((c) => c.id === "investmentKEur");

  return (
    <div className="sticky top-0 z-20 bg-white" role="rowgroup">
      {/* Row 1: Grouped headers */}
      <div className="flex items-stretch" role="row" aria-rowindex={1}>
        {/* Empty cell above row number */}
        <div
          className="flex items-center justify-center h-6 shrink-0 border-r border-b"
          style={{
            width: leftOffset,
            backgroundColor: FPT_COLORS.headerBg,
            borderColor: FPT_COLORS.headerBg,
          }}
        />

        {/* Empty cells above identifier columns (functionId, peFunction, mainActivitiesDescription) */}
        {columns
          .filter((c) => IDENTIFIER_COLUMNS.includes(c.id))
          .map((column) => (
            <div
              key={column.id}
              className="flex items-center justify-center h-6 shrink-0 border-r border-b"
              style={{
                width: column.width,
                minWidth: column.width,
                backgroundColor: FPT_COLORS.headerBg,
                borderColor: FPT_COLORS.headerBg,
              }}
            />
          ))}

        {/* "Effort [hrs]" spanning effort columns */}
        <div
          className="flex items-center justify-center h-6 shrink-0 border-r border-b text-white font-semibold text-xs"
          style={{
            width: effortColumnsWidth,
            minWidth: effortColumnsWidth,
            backgroundColor: FPT_COLORS.headerBg,
            borderColor: "#5c0000",
          }}
        >
          Effort [hrs]
        </div>

        {/* "Vehicle tests" header spanning its column */}
        <div
          className="flex items-center justify-center h-6 shrink-0 border-r border-b text-white font-semibold text-xs"
          style={{
            width: vehicleTestsColumn?.width || 100,
            minWidth: vehicleTestsColumn?.width || 100,
            backgroundColor: FPT_COLORS.headerBg,
            borderColor: "#5c0000",
          }}
        >
          Vehicle tests
        </div>

        {/* "Investment [k€]" header */}
        <div
          className="flex items-center justify-center h-6 shrink-0 border-b text-white font-semibold text-xs"
          style={{
            width: investmentColumn?.width || 110,
            minWidth: investmentColumn?.width || 110,
            backgroundColor: FPT_COLORS.headerBg,
            borderColor: "#5c0000",
          }}
        >
          Investment [k€]
        </div>
      </div>

      {/* Row 2: Individual column headers */}
      <div className="flex items-stretch" role="row" aria-rowindex={2}>
        {/* Row number header */}
        <div
          className="flex items-center justify-center h-8 shrink-0 border-r border-b text-white font-semibold text-xs sticky left-0 z-30"
          style={{
            width: leftOffset,
            backgroundColor: FPT_COLORS.headerBg,
            borderColor: "#5c0000",
          }}
          role="columnheader"
        />

        {/* PE Function header - spans functionId column with label */}
        <HeaderCell
          columnId="functionId"
          label=""
          width={columns.find((c) => c.id === "functionId")?.width || 70}
          onResize={onColumnResize}
        />

        {/* PE Function name column */}
        <HeaderCell
          columnId="peFunction"
          label="PE Function"
          width={columns.find((c) => c.id === "peFunction")?.width || 160}
          onResize={onColumnResize}
          showLabel
        />

        {/* Main Activities Description */}
        <HeaderCell
          columnId="mainActivitiesDescription"
          label="Main Activities Description"
          width={
            columns.find((c) => c.id === "mainActivitiesDescription")?.width ||
            280
          }
          onResize={onColumnResize}
        />

        {/* Effort sub-columns */}
        <HeaderCell
          columnId="manpower"
          label="Manpower"
          width={columns.find((c) => c.id === "manpower")?.width || 90}
          onResize={onColumnResize}
        />
        <HeaderCell
          columnId="benchDev"
          label="Bench (Dev)"
          width={columns.find((c) => c.id === "benchDev")?.width || 90}
          onResize={onColumnResize}
        />
        <HeaderCell
          columnId="benchSpecial"
          label="Bench (Special)"
          width={columns.find((c) => c.id === "benchSpecial")?.width || 100}
          onResize={onColumnResize}
        />
        <HeaderCell
          columnId="benchDur"
          label="Bench (Dur)"
          width={columns.find((c) => c.id === "benchDur")?.width || 90}
          onResize={onColumnResize}
        />

        {/* Vehicle tests sub-header */}
        <HeaderCell
          columnId="vehicleTests"
          label="(roller, PEMS)"
          width={vehicleTestsColumn?.width || 100}
          onResize={onColumnResize}
        />

        {/* Investment - no sub-label needed */}
        <HeaderCell
          columnId="investmentKEur"
          label=""
          width={investmentColumn?.width || 110}
          onResize={onColumnResize}
          isLast
        />
      </div>
    </div>
  );
});

// ============================================================================
// Header Cell Sub-component
// ============================================================================

interface HeaderCellProps {
  columnId: string;
  label: string;
  width: number;
  onResize: (columnId: string, newWidth: number) => void;
  showLabel?: boolean;
  isLast?: boolean;
}

const HeaderCell = memo(function HeaderCell({
  columnId,
  label,
  width,
  onResize,
  showLabel = false,
  isLast = false,
}: HeaderCellProps) {
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
        onResize(columnId, newWidth);
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
        "relative flex items-center justify-center h-8 px-1 text-xs font-semibold text-white border-b select-none",
        !isLast && "border-r",
      )}
      style={{
        width,
        minWidth: width,
        maxWidth: width,
        backgroundColor: FPT_COLORS.headerBg,
        borderColor: "#5c0000",
      }}
      role="columnheader"
    >
      <span className="truncate text-center">{label}</span>

      {/* Resize handle */}
      <div
        className={cn(
          "absolute right-0 top-0 bottom-0 w-1 cursor-col-resize",
          "hover:bg-white/30",
          { "bg-white/50": isResizing },
        )}
        onMouseDown={handleResizeStart}
      />
    </div>
  );
});

// ============================================================================
// Column Definitions Factory
// ============================================================================

/**
 * Create column definitions matching FPT PE02 template.
 *
 * PE02 Format:
 * Function ID | PE Function | Main Activities Description |
 * Manpower | Bench(Dev) | Bench(Special) | Bench(Dur) | Vehicle tests | Investment [k€]
 */
export function createDefaultColumns(): ColumnDefinition[] {
  return [
    {
      id: "functionId",
      header: "Function ID",
      type: "text",
      width: 70,
      minWidth: 60,
      frozen: true,
      editable: false, // PE02 code from AI
    },
    {
      id: "peFunction",
      header: "PE Function",
      type: "text",
      width: 160,
      minWidth: 120,
      frozen: true,
      editable: false, // PE Function name from AI
    },
    {
      id: "mainActivitiesDescription",
      header: "Main Activities Description",
      type: "text",
      width: 280,
      minWidth: 200,
      editable: true,
    },
    {
      id: "manpower",
      header: "Manpower",
      type: "number",
      width: 90,
      minWidth: 70,
      editable: true,
    },
    {
      id: "benchDev",
      header: "Bench (Dev)",
      type: "number",
      width: 90,
      minWidth: 70,
      editable: true,
    },
    {
      id: "benchSpecial",
      header: "Bench (Special)",
      type: "number",
      width: 100,
      minWidth: 80,
      editable: true,
    },
    {
      id: "benchDur",
      header: "Bench (Dur)",
      type: "number",
      width: 90,
      minWidth: 70,
      editable: true,
    },
    {
      id: "vehicleTests",
      header: "Vehicle tests",
      type: "number",
      width: 100,
      minWidth: 80,
      editable: true,
    },
    {
      id: "investmentKEur",
      header: "Investment [k€]",
      type: "currency",
      width: 110,
      minWidth: 90,
      editable: true, // Allow user to edit total cost
    },
  ];
}

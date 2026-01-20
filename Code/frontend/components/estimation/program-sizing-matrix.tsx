"use client";

import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Check, Sparkles, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

export type ProgramSize = "full" | "large" | "medium" | "small" | "x_small";

// Column domains for the matrix
export type ColumnDomain =
  | "basePWT"
  | "systemAssembly"
  | "installation"
  | "plantBasePWT"
  | "plantATSPG"
  | "sourcing"
  | "supplierQuality";

// Per-column prediction and selection
export interface ColumnPrediction {
  aiPredictedSize: ProgramSize;
  selectedSize: ProgramSize;
  confidence: number;
}

// Enhanced data structure with per-column predictions
export interface ProgramSizingData {
  // Overall Program Size - INDEPENDENT selection (not derived from columns)
  overallSize: ProgramSize;
  aiPredictedOverallSize: ProgramSize;
  overallConfidence: number;

  // Per-column predictions and selections
  columns: Record<ColumnDomain, ColumnPrediction>;
}

interface ProgramSizingMatrixProps {
  data: ProgramSizingData;
  onDataChange?: (data: ProgramSizingData) => void;
  readOnly?: boolean;
  className?: string;
}

// Column definitions
const columnDefinitions: Record<
  ColumnDomain,
  {
    header: string;
    subHeader?: string;
    category: "productEngineering" | "manufacturing" | "purchasing";
  }
> = {
  basePWT: {
    header: "BASE PWT",
    subHeader: "as delivered from Manufacturing Plant",
    category: "productEngineering",
  },
  systemAssembly: {
    header: "System assembly",
    subHeader: "Engine + ATS/PG as delivered from Assembly Plant",
    category: "productEngineering",
  },
  installation: {
    header: "Installation/Application/Homologation",
    category: "manufacturing",
  },
  plantBasePWT: {
    header: "Plant (Base PWT)",
    category: "manufacturing",
  },
  plantATSPG: {
    header: "Plant (ATS/PG assembly)",
    category: "manufacturing",
  },
  sourcing: {
    header: "SOURCING",
    category: "purchasing",
  },
  supplierQuality: {
    header: "SUPPLIER QUALITY",
    category: "purchasing",
  },
};

// Cell content for each size and column
const cellContent: Record<ProgramSize, Record<ColumnDomain, string>> = {
  full: {
    basePWT:
      "New concept required; High level of New Content (NC); New serviceability req.s; High validation effort.",
    systemAssembly:
      "New concept required; High level of New Content (NC); New serviceability req.s; High validation effort.",
    installation:
      "First installation; New SW & Cals; New Emission Stage; 250-1000 RGT",
    plantBasePWT: "Manuf. Class: AA; Manuf. Project Score: 250-1000",
    plantATSPG: "Manuf. Class: AA; Manuf. Project Score: 250-1000",
    sourcing: "# Parts in New Sourcing: High; Tooling Lead Time: High.",
    supplierQuality: "# of new parts in APQP (4,5): High",
  },
  large: {
    basePWT:
      "Heavy modification of existing concepts or serviceability req.s with impact on manufacturing process; High/Medium level of NC; High/Medium validation effort.",
    systemAssembly:
      "Heavy modification of existing concepts or serviceability req.s w/impact on manufacturing process; High/Medium level of NC; High/Medium validation effort.",
    installation:
      "Medium Installation effort; Medium Cals Score: 50-250; Homologation; RGT.",
    plantBasePWT: "Manuf. Class: A; Manuf. Project Score: 50-250",
    plantATSPG: "Manuf. Class: A; Manuf. Project Score: 50-250",
    sourcing:
      "# Parts in New Sourcing: Medium; Tooling Lead Time: High/Medium.",
    supplierQuality: "# of new parts in APQP (4,5): High/Medium",
  },
  medium: {
    basePWT:
      "Medium modification of existing concepts or serviceability req.s (no impact on manufacturing process); Medium level of NC; Medium validation effort.",
    systemAssembly:
      "Medium modification of existing concepts or serviceability req.s (no impact on manufacturing process); Medium level of NC; Medium validation effort.",
    installation:
      "Medium installation effort; Medium Cals Review; Homologation; RGT",
    plantBasePWT: "Manuf. Class: B; Manuf. Project Score: 5-50",
    plantATSPG: "Manuf. Class: B; Manuf. Project Score: 5-50",
    sourcing: "# Parts in New Sourcing: Medium/Low; Tooling Lead Time: Low",
    supplierQuality: "# of new parts in APQP (4,5): Medium/Low",
  },
  small: {
    basePWT:
      "Light modification of existing product; Low level of NC; Low validation effort.",
    systemAssembly:
      "Light modification of existing product; Low level of NC; Low validation effort.",
    installation: "Low installation effort; Limited Cals Review; Homologation;",
    plantBasePWT: "Manuf. Class: B; Manuf. Project Score: 5-50",
    plantATSPG: "Manuf. Class: B; Manuf. Project Score: 5-50",
    sourcing: "No parts in new sourcing/ only modified: No Tooling",
    supplierQuality: "No parts in APQP (4,5)",
  },
  x_small: {
    basePWT:
      "Minimum modification of existing product (only adaptation); Minimum level of NC; No validation effort.",
    systemAssembly:
      "Minimum modification of existing product (only adaptation); Minimum level of NC; No validation effort.",
    installation: "Minimum installation and Cals. effort; No homologation",
    plantBasePWT: "Manuf. Class: C; Manuf. Project Score: <5",
    plantATSPG: "Manuf. Class: C; Manuf. Project Score: <5",
    sourcing: "Minimum number of modified buy parts; No Tooling",
    supplierQuality: "No parts in APQP(4,5).",
  },
};

// Program size labels and build stage requirements
const programSizeInfo: Record<
  ProgramSize,
  { label: string; buildStages: string }
> = {
  full: { label: "FULL", buildStages: "ALL Build Stages required" },
  large: { label: "LARGE", buildStages: "Beta/Gamma/PP/Pilot required" },
  medium: { label: "MEDIUM", buildStages: "Beta/PP/Pilot required" },
  small: { label: "SMALL", buildStages: "PP/Pilot required" },
  x_small: { label: "X SMALL", buildStages: "Only Pilot required" },
};

const sizeOrder: ProgramSize[] = [
  "full",
  "large",
  "medium",
  "small",
  "x_small",
];
const columnOrder: ColumnDomain[] = [
  "basePWT",
  "systemAssembly",
  "installation",
  "plantBasePWT",
  "plantATSPG",
  "sourcing",
  "supplierQuality",
];

export function ProgramSizingMatrix({
  data,
  onDataChange,
  readOnly = false,
  className,
}: ProgramSizingMatrixProps) {
  const [localData, setLocalData] = useState<ProgramSizingData>(data);

  // Handle cell selection for domain columns (PE, Manufacturing, Purchasing)
  const handleCellSelect = useCallback(
    (column: ColumnDomain, size: ProgramSize) => {
      if (readOnly) return;

      setLocalData((prev) => {
        const newColumns = {
          ...prev.columns,
          [column]: {
            ...prev.columns[column],
            selectedSize: size,
          },
        };

        const newData = {
          ...prev,
          columns: newColumns,
          // Keep overallSize independent - don't change it here
        };

        onDataChange?.(newData);
        return newData;
      });
    },
    [readOnly, onDataChange],
  );

  // Handle Program Size selection - INDEPENDENT from domain columns
  const handleProgramSizeSelect = useCallback(
    (size: ProgramSize) => {
      if (readOnly) return;

      setLocalData((prev) => {
        const newData = {
          ...prev,
          overallSize: size,
        };

        onDataChange?.(newData);
        return newData;
      });
    },
    [readOnly, onDataChange],
  );

  const handleResetToAI = useCallback(() => {
    setLocalData((prev) => {
      const newColumns = { ...prev.columns };
      (Object.keys(newColumns) as ColumnDomain[]).forEach((col) => {
        newColumns[col] = {
          ...newColumns[col],
          selectedSize: newColumns[col].aiPredictedSize,
        };
      });

      const newData = {
        ...prev,
        columns: newColumns,
        // Also reset overall size to AI predicted
        overallSize: prev.aiPredictedOverallSize,
      };

      onDataChange?.(newData);
      return newData;
    });
  }, [onDataChange]);

  // Check if any column has user override OR program size is changed
  const hasColumnOverrides = Object.values(localData.columns).some(
    (col) => col.selectedSize !== col.aiPredictedSize,
  );
  const hasProgramSizeOverride =
    localData.overallSize !== localData.aiPredictedOverallSize;
  const hasUserOverrides = hasColumnOverrides || hasProgramSizeOverride;

  // Count adjustments per category
  const categoryAdjustments = {
    productEngineering: columnOrder
      .filter((col) => columnDefinitions[col].category === "productEngineering")
      .filter(
        (col) =>
          localData.columns[col].selectedSize !==
          localData.columns[col].aiPredictedSize,
      ).length,
    manufacturing: columnOrder
      .filter((col) => columnDefinitions[col].category === "manufacturing")
      .filter(
        (col) =>
          localData.columns[col].selectedSize !==
          localData.columns[col].aiPredictedSize,
      ).length,
    purchasing: columnOrder
      .filter((col) => columnDefinitions[col].category === "purchasing")
      .filter(
        (col) =>
          localData.columns[col].selectedSize !==
          localData.columns[col].aiPredictedSize,
      ).length,
  };

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="bg-blue-600 text-white py-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg font-bold">PROGRAM SIZING</CardTitle>
            <p className="text-blue-100 text-sm">
              Platform – Functional Consolidation (Per-Domain AI Prediction)
            </p>
          </div>
          {!readOnly && hasUserOverrides && (
            <Button
              size="sm"
              variant="outline"
              className="bg-white/10 hover:bg-white/20 text-white border-white/30"
              onClick={handleResetToAI}
            >
              <RotateCcw className="h-4 w-4 mr-1" />
              Reset to AI
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* Overall AI Prediction Summary */}
        <div className="bg-blue-50 px-4 py-3 border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium text-gray-700">
                  Overall Size:
                </span>
                <span className="font-bold text-blue-700 text-lg">
                  {programSizeInfo[localData.overallSize].label}
                </span>
              </div>
              <div className="text-xs text-gray-500">
                ({Math.round(localData.overallConfidence * 100)}% avg
                confidence)
              </div>
            </div>
            {hasUserOverrides && (
              <div className="flex items-center gap-2 text-xs text-amber-600">
                <span className="font-medium">User adjustments:</span>
                {categoryAdjustments.productEngineering > 0 && (
                  <span className="bg-amber-100 px-2 py-0.5 rounded">
                    PE: {categoryAdjustments.productEngineering}
                  </span>
                )}
                {categoryAdjustments.manufacturing > 0 && (
                  <span className="bg-amber-100 px-2 py-0.5 rounded">
                    Manuf: {categoryAdjustments.manufacturing}
                  </span>
                )}
                {categoryAdjustments.purchasing > 0 && (
                  <span className="bg-amber-100 px-2 py-0.5 rounded">
                    Purch: {categoryAdjustments.purchasing}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Legend */}
        <div className="bg-gray-50 px-4 py-2 border-b flex items-center gap-6 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded border-2 border-blue-500 bg-blue-100 flex items-center justify-center">
              <Sparkles className="w-2.5 h-2.5 text-blue-600" />
            </div>
            <span className="text-gray-600">AI Prediction</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-blue-600 flex items-center justify-center">
              <Check className="w-3 h-3 text-white" />
            </div>
            <span className="text-gray-600">Selected</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-amber-100 border border-amber-300"></div>
            <span className="text-gray-600">User Override</span>
          </div>
          <span className="ml-auto text-gray-500 italic">
            Click any cell to select
          </span>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse min-w-[800px]">
            {/* Header Row 1 - Main Categories */}
            <thead>
              <tr>
                <th
                  colSpan={2}
                  className="border border-blue-600 px-3 py-3 text-center font-bold text-white bg-blue-700"
                >
                  PRODUCT ENGINEERING
                </th>
                <th
                  colSpan={3}
                  className="border border-blue-600 px-3 py-3 text-center font-bold text-white bg-blue-800"
                >
                  MANUFACTURING
                </th>
                <th
                  colSpan={2}
                  className="border border-blue-600 px-3 py-3 text-center font-bold text-white bg-blue-600"
                >
                  PURCHASING
                </th>
                <th
                  colSpan={2}
                  className="border border-blue-600 px-3 py-3 text-center font-bold text-white bg-gray-700"
                >
                  PROGRAM SIZE
                </th>
              </tr>
              {/* Header Row 2 - Sub Categories with AI predictions per column */}
              <tr className="bg-blue-50">
                {columnOrder.map((col) => {
                  const def = columnDefinitions[col];
                  const prediction = localData.columns[col];
                  const isManufacturing = def.category === "manufacturing";

                  return (
                    <th
                      key={col}
                      className={cn(
                        "border border-gray-300 px-1.5 py-2 text-left font-semibold text-gray-800 min-w-[100px] max-w-[140px]",
                        isManufacturing && "bg-blue-100",
                      )}
                    >
                      <div className="space-y-1">
                        <div className="text-xs font-bold">{def.header}</div>
                        {def.subHeader && (
                          <div className="text-[10px] text-gray-500 font-normal leading-tight">
                            {def.subHeader}
                          </div>
                        )}
                        {/* AI prediction badge for this column */}
                        <div className="flex items-center gap-1 mt-1">
                          <Sparkles className="w-3 h-3 text-blue-500" />
                          <span className="text-[10px] text-blue-600 font-medium">
                            AI:{" "}
                            {programSizeInfo[prediction.aiPredictedSize].label}
                          </span>
                          <span className="text-[9px] text-gray-400">
                            ({Math.round(prediction.confidence * 100)}%)
                          </span>
                        </div>
                      </div>
                    </th>
                  );
                })}
                {/* Program Size header */}
                <th className="border border-gray-300 px-2 py-2 text-center font-semibold text-gray-800 bg-gray-100 w-24">
                  Size
                </th>
                <th className="border border-gray-300 px-2 py-2 text-center font-semibold text-gray-800 bg-gray-100 w-36">
                  Build Stages
                </th>
              </tr>
            </thead>
            <tbody>
              {sizeOrder.map((size, rowIndex) => {
                const sizeInfo = programSizeInfo[size];
                const isEven = rowIndex % 2 === 0;

                return (
                  <tr
                    key={size}
                    className={cn(
                      "transition-all",
                      isEven ? "bg-white" : "bg-gray-50/50",
                    )}
                  >
                    {columnOrder.map((col) => {
                      const prediction = localData.columns[col];
                      const isSelected = prediction.selectedSize === size;
                      const isAIPredicted = prediction.aiPredictedSize === size;
                      const isUserOverride = isSelected && !isAIPredicted;
                      const def = columnDefinitions[col];
                      const isManufacturing = def.category === "manufacturing";

                      return (
                        <td
                          key={`${size}-${col}`}
                          className={cn(
                            "border border-gray-300 px-1.5 py-1.5 text-[11px] leading-snug cursor-pointer transition-all relative",
                            isManufacturing && "bg-blue-50/30",
                            // Selection states
                            isSelected &&
                              !isUserOverride &&
                              "bg-blue-100 ring-2 ring-inset ring-blue-500",
                            isSelected &&
                              isUserOverride &&
                              "bg-amber-100 ring-2 ring-inset ring-amber-500",
                            // AI predicted (not selected)
                            isAIPredicted &&
                              !isSelected &&
                              "bg-blue-50 border-blue-300",
                            // Hover
                            !readOnly && "hover:bg-blue-100/70",
                          )}
                          onClick={() => handleCellSelect(col, size)}
                        >
                          {/* Cell content */}
                          <div className="text-gray-800">
                            {cellContent[size][col]}
                          </div>

                          {/* Selection indicator */}
                          {isSelected && (
                            <div className="absolute top-1 right-1">
                              <div
                                className={cn(
                                  "w-5 h-5 rounded-full flex items-center justify-center",
                                  isUserOverride
                                    ? "bg-amber-500"
                                    : "bg-blue-600",
                                )}
                              >
                                <Check className="w-3 h-3 text-white" />
                              </div>
                            </div>
                          )}

                          {/* AI prediction indicator (when not selected) */}
                          {isAIPredicted && !isSelected && (
                            <div className="absolute top-1 right-1">
                              <div className="w-5 h-5 rounded-full border-2 border-blue-400 bg-white flex items-center justify-center">
                                <Sparkles className="w-2.5 h-2.5 text-blue-500" />
                              </div>
                            </div>
                          )}
                        </td>
                      );
                    })}

                    {/* Program Size column - INDEPENDENTLY SELECTABLE */}
                    <td
                      className={cn(
                        "border border-gray-300 px-3 py-3 text-center cursor-pointer transition-all",
                        localData.overallSize === size
                          ? "bg-blue-100 ring-2 ring-inset ring-blue-500"
                          : "bg-gray-100 hover:bg-blue-50",
                        // AI prediction highlight
                        localData.aiPredictedOverallSize === size &&
                          localData.overallSize !== size &&
                          "bg-blue-50/70",
                      )}
                      onClick={() => handleProgramSizeSelect(size)}
                    >
                      <div className="flex flex-col items-center gap-1">
                        <span
                          className={cn(
                            "font-bold text-lg",
                            localData.overallSize === size
                              ? "text-blue-700"
                              : "text-gray-800",
                          )}
                        >
                          {sizeInfo.label}
                        </span>
                        {/* Radio button indicator */}
                        <div
                          className={cn(
                            "w-6 h-6 rounded-full border-2 flex items-center justify-center transition-all",
                            localData.overallSize === size
                              ? "border-blue-600 bg-blue-600"
                              : "border-gray-400 bg-white",
                          )}
                        >
                          {localData.overallSize === size && (
                            <Check className="w-4 h-4 text-white" />
                          )}
                        </div>
                        {/* AI Pick indicator */}
                        {localData.aiPredictedOverallSize === size && (
                          <span className="text-xs text-blue-600 font-semibold flex items-center gap-1">
                            <Sparkles className="w-3 h-3" />
                            AI Pick
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="border border-gray-300 px-2 py-3 text-center text-sm text-gray-800 bg-gray-100 font-medium">
                      {sizeInfo.buildStages}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Summary footer */}
        <div className="bg-gray-100 px-4 py-3 border-t">
          <div className="flex items-center justify-between text-sm">
            <div className="text-gray-600">
              <span className="font-medium">Selected sizes per domain:</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">
                  Product Engineering:
                </span>
                <span className="font-semibold text-blue-700">
                  {
                    programSizeInfo[localData.columns.basePWT.selectedSize]
                      .label
                  }{" "}
                  /{" "}
                  {
                    programSizeInfo[
                      localData.columns.systemAssembly.selectedSize
                    ].label
                  }
                </span>
              </div>
              <div className="h-4 w-px bg-gray-300" />
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Manufacturing:</span>
                <span className="font-semibold text-blue-800">
                  {
                    programSizeInfo[localData.columns.installation.selectedSize]
                      .label
                  }
                </span>
              </div>
              <div className="h-4 w-px bg-gray-300" />
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Purchasing:</span>
                <span className="font-semibold text-blue-600">
                  {
                    programSizeInfo[localData.columns.sourcing.selectedSize]
                      .label
                  }
                </span>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper function to create default ProgramSizingData from AI predictions
export function createProgramSizingData(
  aiPredictions: Partial<
    Record<ColumnDomain, { size: ProgramSize; confidence: number }>
  >,
  defaultSize: ProgramSize = "medium",
  defaultConfidence: number = 0.7,
  overallPredictedSize?: ProgramSize,
): ProgramSizingData {
  const columns: Record<ColumnDomain, ColumnPrediction> = {} as Record<
    ColumnDomain,
    ColumnPrediction
  >;

  columnOrder.forEach((col) => {
    const prediction = aiPredictions[col];
    columns[col] = {
      aiPredictedSize: prediction?.size || defaultSize,
      selectedSize: prediction?.size || defaultSize,
      confidence: prediction?.confidence || defaultConfidence,
    };
  });

  // Calculate average confidence
  const avgConfidence =
    Object.values(columns).reduce((sum, col) => sum + col.confidence, 0) /
    columnOrder.length;

  // Use provided overall size or default
  const predictedOverall = overallPredictedSize || defaultSize;

  return {
    overallSize: predictedOverall,
    aiPredictedOverallSize: predictedOverall,
    overallConfidence: avgConfidence,
    columns,
  };
}

// Legacy compatibility - convert old ProgramSizingData to new format
export function convertLegacyProgramSizingData(legacy: {
  selectedSize: ProgramSize;
  aiPredictedSize: ProgramSize;
  confidence: number;
}): ProgramSizingData {
  const columns: Record<ColumnDomain, ColumnPrediction> = {} as Record<
    ColumnDomain,
    ColumnPrediction
  >;

  columnOrder.forEach((col) => {
    columns[col] = {
      aiPredictedSize: legacy.aiPredictedSize,
      selectedSize: legacy.selectedSize,
      confidence: legacy.confidence,
    };
  });

  return {
    overallSize: legacy.selectedSize,
    aiPredictedOverallSize: legacy.aiPredictedSize,
    overallConfidence: legacy.confidence,
    columns,
  };
}

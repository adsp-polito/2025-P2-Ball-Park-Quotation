"use client";

import { useState, useMemo, useCallback, Fragment } from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ChevronDown,
  ChevronRight,
  Edit2,
  Check,
  X,
  RotateCcw,
  Save,
} from "lucide-react";

// Data types matching PE02 format
export interface ActivityItem {
  id: string;
  code?: string; // PE02 function code (A1, B1, C, etc.)
  description: string;
  effort: {
    manpower: number;
    benchDev: number; // Bench Development hours
    benchSpecial: number; // Bench Special hours (NVH, climatic) - NEW
    benchDur: number; // Bench Durability hours
    vehicle: number; // Vehicle tests hours
  };
  costKEur: number;
  confidence?: number; // AI confidence score
  isEdited?: boolean;
  originalValues?: {
    manpower: number;
    benchDev: number;
    benchSpecial: number;
    benchDur: number;
    vehicle: number;
    costKEur: number;
  };
}

export interface PEFunction {
  id: string;
  code?: string; // PE02 function code
  name: string;
  activities: ActivityItem[];
}

export interface MainCategory {
  id: string;
  code?: string; // PE02 category code (A, B, C, etc.)
  name: string;
  peFunctions: PEFunction[];
}

export interface ActivitySummaryData {
  categories: MainCategory[];
  totalCostKEur: number;
  aiGenerated: boolean;
}

interface ActivitySummaryTableProps {
  data: ActivitySummaryData;
  onDataChange?: (data: ActivitySummaryData) => void;
  readOnly?: boolean;
  className?: string;
}

// Calculate totals for effort columns (PE02 format with benchSpecial)
function calculateCategoryTotals(category: MainCategory) {
  let manpower = 0;
  let benchDev = 0;
  let benchSpecial = 0;
  let benchDur = 0;
  let vehicle = 0;
  let cost = 0;

  category.peFunctions.forEach((fn) => {
    fn.activities.forEach((act) => {
      manpower += act.effort.manpower || 0;
      benchDev += act.effort.benchDev || 0;
      benchSpecial += act.effort.benchSpecial || 0;
      benchDur += act.effort.benchDur || 0;
      vehicle += act.effort.vehicle || 0;
      cost += act.costKEur || 0;
    });
  });

  return { manpower, benchDev, benchSpecial, benchDur, vehicle, cost };
}

function calculateGrandTotals(categories: MainCategory[]) {
  let manpower = 0;
  let benchDev = 0;
  let benchSpecial = 0;
  let benchDur = 0;
  let vehicle = 0;
  let cost = 0;

  categories.forEach((cat) => {
    const totals = calculateCategoryTotals(cat);
    manpower += totals.manpower;
    benchDev += totals.benchDev;
    benchSpecial += totals.benchSpecial;
    benchDur += totals.benchDur;
    vehicle += totals.vehicle;
    cost += totals.cost;
  });

  return { manpower, benchDev, benchSpecial, benchDur, vehicle, cost };
}

// Editable cell component - uses explicit light backgrounds to avoid dark mode issues
function EditableCell({
  value,
  onChange,
  readOnly,
  type = "number",
}: {
  value: number;
  onChange: (val: number) => void;
  readOnly?: boolean;
  type?: "number" | "currency";
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [tempValue, setTempValue] = useState(value.toString());

  const handleSave = () => {
    const numVal = parseFloat(tempValue) || 0;
    onChange(numVal);
    setIsEditing(false);
  };

  const handleCancel = () => {
    setTempValue(value.toString());
    setIsEditing(false);
  };

  if (readOnly) {
    return (
      <span className="block text-right text-gray-800 font-medium text-xs">
        {type === "currency" ? value.toFixed(1) : value.toLocaleString()}
      </span>
    );
  }

  if (isEditing) {
    return (
      <div className="flex items-center gap-0.5">
        <Input
          type="number"
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
          className="h-5 w-14 text-xs text-right p-0.5 text-gray-900 bg-white border-gray-300 focus:bg-white"
          autoFocus
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSave();
            if (e.key === "Escape") handleCancel();
          }}
        />
        <Button
          size="icon"
          variant="ghost"
          className="h-4 w-4 bg-white hover:bg-green-50"
          onClick={handleSave}
        >
          <Check className="h-2.5 w-2.5 text-green-600" />
        </Button>
        <Button
          size="icon"
          variant="ghost"
          className="h-4 w-4 bg-white hover:bg-red-50"
          onClick={handleCancel}
        >
          <X className="h-2.5 w-2.5 text-red-600" />
        </Button>
      </div>
    );
  }

  return (
    <div
      className="flex items-center justify-end gap-0.5 cursor-pointer hover:bg-amber-50 rounded px-0.5 -mx-0.5 group"
      onClick={() => setIsEditing(true)}
    >
      <span className="text-gray-800 font-medium text-xs">
        {type === "currency" ? value.toFixed(1) : value.toLocaleString()}
      </span>
      <Edit2 className="h-2.5 w-2.5 text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity" />
    </div>
  );
}

export function ActivitySummaryTable({
  data,
  onDataChange,
  readOnly = false,
  className,
}: ActivitySummaryTableProps) {
  const [expandedCategories, setExpandedCategories] = useState<string[]>(
    data.categories.map((c) => c.id),
  );
  const [_expandedFunctions, _setExpandedFunctions] = useState<string[]>([]);
  const [localData, setLocalData] = useState(data);

  const grandTotals = useMemo(
    () => calculateGrandTotals(localData.categories),
    [localData.categories],
  );

  const hasEdits = useMemo(() => {
    return localData.categories.some((cat) =>
      cat.peFunctions.some((fn) => fn.activities.some((act) => act.isEdited)),
    );
  }, [localData.categories]);

  const toggleCategory = (catId: string) => {
    setExpandedCategories((prev) =>
      prev.includes(catId)
        ? prev.filter((id) => id !== catId)
        : [...prev, catId],
    );
  };

  const _toggleFunction = (fnId: string) => {
    _setExpandedFunctions((prev) =>
      prev.includes(fnId) ? prev.filter((id) => id !== fnId) : [...prev, fnId],
    );
  };

  const updateActivityValue = useCallback(
    (
      catId: string,
      fnId: string,
      actId: string,
      field: keyof ActivityItem["effort"] | "costKEur",
      value: number,
    ) => {
      setLocalData((prev) => {
        const newData = { ...prev };
        newData.categories = prev.categories.map((cat) => {
          if (cat.id !== catId) return cat;
          return {
            ...cat,
            peFunctions: cat.peFunctions.map((fn) => {
              if (fn.id !== fnId) return fn;
              return {
                ...fn,
                activities: fn.activities.map((act) => {
                  if (act.id !== actId) return act;

                  // Store original values if first edit
                  const originalValues = act.originalValues || {
                    manpower: act.effort.manpower,
                    benchDev: act.effort.benchDev,
                    benchSpecial: act.effort.benchSpecial || 0,
                    benchDur: act.effort.benchDur,
                    vehicle: act.effort.vehicle,
                    costKEur: act.costKEur,
                  };

                  if (field === "costKEur") {
                    return {
                      ...act,
                      costKEur: value,
                      isEdited: true,
                      originalValues,
                    };
                  }

                  return {
                    ...act,
                    effort: {
                      ...act.effort,
                      [field]: value,
                    },
                    isEdited: true,
                    originalValues,
                  };
                }),
              };
            }),
          };
        });

        // Recalculate total
        newData.totalCostKEur = calculateGrandTotals(newData.categories).cost;
        return newData;
      });
    },
    [],
  );

  const handleSaveAll = () => {
    onDataChange?.(localData);
  };

  const handleResetAll = () => {
    setLocalData((prev) => ({
      ...prev,
      categories: prev.categories.map((cat) => ({
        ...cat,
        peFunctions: cat.peFunctions.map((fn) => ({
          ...fn,
          activities: fn.activities.map((act) => {
            if (!act.originalValues) return act;
            return {
              ...act,
              effort: {
                manpower: act.originalValues.manpower,
                benchDev: act.originalValues.benchDev,
                benchSpecial: act.originalValues.benchSpecial || 0,
                benchDur: act.originalValues.benchDur,
                vehicle: act.originalValues.vehicle,
              },
              costKEur: act.originalValues.costKEur,
              isEdited: false,
              originalValues: undefined,
            };
          }),
        })),
      })),
    }));
  };

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardHeader className="bg-red-700 text-white py-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lg font-bold">
              Engineering Activity Summary and R&D expenses forecast (PE.02)
            </CardTitle>
            <p className="text-red-100 text-sm">By Function</p>
          </div>
          {!readOnly && hasEdits && (
            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                className="bg-white/10 hover:bg-white/20 text-white border-white/30"
                onClick={handleResetAll}
              >
                <RotateCcw className="h-4 w-4 mr-1" />
                Reset
              </Button>
              <Button
                size="sm"
                className="bg-white text-red-700 hover:bg-red-50"
                onClick={handleSaveAll}
              >
                <Save className="h-4 w-4 mr-1" />
                Save Changes
              </Button>
            </div>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        {/* AI Generated Badge */}
        {localData.aiGenerated && (
          <div className="bg-red-50 px-4 py-2 border-b flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">
                AI Predicted Breakdown
              </span>
              <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded">
                Generated by ML Models
              </span>
            </div>
            {hasEdits && (
              <span className="text-xs text-amber-600 font-medium">
                You have unsaved changes
              </span>
            )}
          </div>
        )}

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse min-w-[700px]">
            {/* Header */}
            <thead>
              {/* Header Row 1 */}
              <tr>
                <th
                  rowSpan={2}
                  className="border-2 border-gray-400 bg-gray-300 px-1 py-2 text-left font-bold w-8"
                >
                  {/* Main activities vertical text handled by row spans */}
                </th>
                <th
                  rowSpan={2}
                  className="border-2 border-red-700 bg-red-700 text-white px-2 py-2 text-left text-sm font-bold w-32"
                >
                  PE Function
                </th>
                <th
                  rowSpan={2}
                  className="border-2 border-red-700 bg-red-700 text-white px-2 py-2 text-left text-sm font-bold"
                >
                  Activities Description
                </th>
                <th
                  colSpan={4}
                  className="border-2 border-red-700 bg-red-700 text-white px-2 py-1.5 text-center text-sm font-bold"
                >
                  Effort [hrs]
                </th>
                <th
                  rowSpan={2}
                  className="border-2 border-red-700 bg-red-700 text-white px-2 py-2 text-center text-sm font-bold w-20"
                >
                  Cost [k€]
                </th>
              </tr>
              {/* Header Row 2 - Effort sub-columns */}
              <tr>
                <th className="border-2 border-red-600 bg-red-600 text-white px-1.5 py-1.5 text-center text-xs font-bold w-16">
                  Manpower
                </th>
                <th className="border-2 border-red-600 bg-red-600 text-white px-1.5 py-1.5 text-center text-xs font-bold w-16">
                  Bench
                  <br />
                  (Dur)
                </th>
                <th className="border-2 border-red-600 bg-red-600 text-white px-1.5 py-1.5 text-center text-xs font-bold w-16">
                  Bench
                  <br />
                  (Dev)
                </th>
                <th className="border-2 border-red-600 bg-red-600 text-white px-1.5 py-1.5 text-center text-xs font-bold w-20">
                  Vehicle
                </th>
              </tr>
            </thead>
            <tbody>
              {localData.categories.map((category, catIndex) => {
                const isExpanded = expandedCategories.includes(category.id);
                const catTotals = calculateCategoryTotals(category);
                const _totalRows = category.peFunctions.reduce(
                  (sum, fn) => sum + fn.activities.length,
                  0,
                );

                return (
                  <Fragment key={`category-${category.id}`}>
                    {/* Category header row */}
                    <tr
                      className="bg-gray-200 cursor-pointer hover:bg-gray-300 transition-colors"
                      onClick={() => toggleCategory(category.id)}
                    >
                      {/* Main activities label - spans all rows in category */}
                      {catIndex === 0 && (
                        <td
                          rowSpan={
                            localData.categories.reduce((sum, c) => {
                              if (expandedCategories.includes(c.id)) {
                                return (
                                  sum +
                                  1 +
                                  c.peFunctions.reduce(
                                    (s, f) => s + f.activities.length,
                                    0,
                                  )
                                );
                              }
                              return sum + 1;
                            }, 0) + 1
                          }
                          className="border border-gray-300 bg-gray-200 text-center w-8 relative"
                        >
                          <span
                            className="absolute font-bold text-gray-600 text-xs whitespace-nowrap"
                            style={{
                              transform: "rotate(-90deg)",
                              transformOrigin: "center center",
                              top: "50%",
                              left: "50%",
                              marginLeft: "-40px",
                              marginTop: "-8px",
                            }}
                          >
                            Main activities
                          </span>
                        </td>
                      )}
                      <td
                        colSpan={2}
                        className="border-2 border-gray-400 px-2 py-2 font-bold text-gray-900 text-sm"
                      >
                        <div className="flex items-center gap-1.5">
                          {isExpanded ? (
                            <ChevronDown className="h-4 w-4 text-gray-700" />
                          ) : (
                            <ChevronRight className="h-4 w-4 text-gray-700" />
                          )}
                          {category.name}
                        </div>
                      </td>
                      <td className="border-2 border-gray-400 px-1.5 py-2 text-right font-bold bg-amber-100 text-gray-900 text-xs">
                        {catTotals.manpower.toLocaleString()}
                      </td>
                      <td className="border-2 border-gray-400 px-1.5 py-2 text-right font-bold bg-amber-100 text-gray-900 text-xs">
                        {catTotals.benchDur.toLocaleString()}
                      </td>
                      <td className="border-2 border-gray-400 px-1.5 py-2 text-right font-bold bg-amber-100 text-gray-900 text-xs">
                        {catTotals.benchDev.toLocaleString()}
                      </td>
                      <td className="border-2 border-gray-400 px-1.5 py-2 text-right font-bold bg-amber-100 text-gray-900 text-xs">
                        {catTotals.vehicle.toLocaleString()}
                      </td>
                      <td className="border-2 border-gray-400 px-1.5 py-2 text-right font-bold bg-amber-200 text-gray-900 text-sm">
                        {catTotals.cost.toFixed(1)}
                      </td>
                    </tr>

                    {/* PE Functions and Activities */}
                    {isExpanded &&
                      category.peFunctions.map((peFunction) =>
                        peFunction.activities.map((activity, actIndex) => (
                          <tr
                            key={activity.id}
                            className={cn(
                              "hover:bg-blue-50 transition-colors",
                              activity.isEdited && "bg-amber-50",
                            )}
                          >
                            {/* PE Function name - only on first activity */}
                            {actIndex === 0 ? (
                              <td
                                rowSpan={peFunction.activities.length}
                                className="border-2 border-gray-300 px-2 py-2 font-bold bg-gray-100 align-top text-gray-900 text-xs"
                              >
                                {peFunction.name}
                              </td>
                            ) : null}
                            {/* Activity description */}
                            <td className="border-2 border-gray-300 px-2 py-2 text-gray-900 text-xs">
                              <span className="text-red-600 font-bold mr-1">
                                •
                              </span>
                              {activity.description}
                            </td>
                            {/* Effort columns - editable with explicit light backgrounds */}
                            <td className="border-2 border-gray-300 px-1.5 py-1.5 text-right !bg-white">
                              <EditableCell
                                value={activity.effort.manpower}
                                onChange={(val) =>
                                  updateActivityValue(
                                    category.id,
                                    peFunction.id,
                                    activity.id,
                                    "manpower",
                                    val,
                                  )
                                }
                                readOnly={readOnly}
                              />
                            </td>
                            <td className="border-2 border-gray-300 px-1.5 py-1.5 text-right !bg-white">
                              <EditableCell
                                value={activity.effort.benchDur}
                                onChange={(val) =>
                                  updateActivityValue(
                                    category.id,
                                    peFunction.id,
                                    activity.id,
                                    "benchDur",
                                    val,
                                  )
                                }
                                readOnly={readOnly}
                              />
                            </td>
                            <td className="border-2 border-gray-300 px-1.5 py-1.5 text-right !bg-white">
                              <EditableCell
                                value={activity.effort.benchDev}
                                onChange={(val) =>
                                  updateActivityValue(
                                    category.id,
                                    peFunction.id,
                                    activity.id,
                                    "benchDev",
                                    val,
                                  )
                                }
                                readOnly={readOnly}
                              />
                            </td>
                            <td className="border-2 border-gray-300 px-1.5 py-1.5 text-right !bg-white">
                              <EditableCell
                                value={activity.effort.vehicle}
                                onChange={(val) =>
                                  updateActivityValue(
                                    category.id,
                                    peFunction.id,
                                    activity.id,
                                    "vehicle",
                                    val,
                                  )
                                }
                                readOnly={readOnly}
                              />
                            </td>
                            {/* Cost column - editable with explicit light background */}
                            <td className="border-2 border-gray-300 px-1.5 py-1.5 text-right font-semibold !bg-gray-50">
                              <EditableCell
                                value={activity.costKEur}
                                onChange={(val) =>
                                  updateActivityValue(
                                    category.id,
                                    peFunction.id,
                                    activity.id,
                                    "costKEur",
                                    val,
                                  )
                                }
                                readOnly={readOnly}
                                type="currency"
                              />
                            </td>
                          </tr>
                        )),
                      )}
                  </Fragment>
                );
              })}

              {/* Grand Total Row */}
              <tr className="bg-red-200 font-bold sticky bottom-0">
                <td className="border-t-2 border-red-400"></td>
                <td
                  colSpan={2}
                  className="border-2 border-red-400 px-2 py-2.5 text-right text-sm text-red-900 font-bold"
                >
                  TOTAL
                </td>
                <td className="border-2 border-red-400 px-1.5 py-2.5 text-right text-sm bg-red-300 text-red-900 font-bold">
                  {grandTotals.manpower.toLocaleString()}
                </td>
                <td className="border-2 border-red-400 px-1.5 py-2.5 text-right text-sm bg-red-300 text-red-900 font-bold">
                  {grandTotals.benchDur.toLocaleString()}
                </td>
                <td className="border-2 border-red-400 px-1.5 py-2.5 text-right text-sm bg-red-300 text-red-900 font-bold">
                  {grandTotals.benchDev.toLocaleString()}
                </td>
                <td className="border-2 border-red-400 px-1.5 py-2.5 text-right text-sm bg-red-300 text-red-900 font-bold">
                  {grandTotals.vehicle.toLocaleString()}
                </td>
                <td className="border-2 border-red-400 px-1.5 py-2.5 text-right text-base bg-red-400 text-white font-bold">
                  {grandTotals.cost.toFixed(0)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper to convert backend breakdown format to ActivitySummaryData
// Uses PE02 effort fields from backend (effort_manpower, effort_bench_dev, etc.)
export function convertBreakdownToActivitySummary(
  breakdown: Array<{
    id: string;
    activity_code: string;
    activity_name: string;
    hours: number;
    confidence_score: number;
    reasoning: string;
    // PE02 effort fields from backend
    code?: string;
    function?: string;
    effort_manpower?: number;
    effort_bench_dev?: number;
    effort_bench_special?: number;
    effort_bench_dur?: number;
    effort_vehicle?: number;
    investment_keur?: number;
    cost_eur?: number;
    hourly_rate_eur?: number;
  }>,
): ActivitySummaryData {
  // Group by PE Function (based on activity_code prefix)
  const functionGroups: Record<string, PEFunction> = {};
  const categoryGroups: Record<string, MainCategory> = {
    design: {
      id: "design",
      name: "Design",
      peFunctions: [],
    },
    devrel: {
      id: "devrel",
      name: "Dev&Rel",
      peFunctions: [],
    },
    application: {
      id: "application",
      name: "Application",
      peFunctions: [],
    },
    contracts: {
      id: "contracts",
      name: "Contracts",
      peFunctions: [],
    },
    other: {
      id: "other",
      name: "Other",
      peFunctions: [],
    },
  };

  // Calculate total hours for distribution
  const _totalProjectHours = breakdown.reduce(
    (sum, item) => sum + (item.hours || 0),
    0,
  );
  const _activityCount = breakdown.length;

  // PE02 Standard Function Code Mapping (FPT standard)
  const PE02_CODE_MAP: Record<string, { category: string; name: string }> = {
    A1: { category: "design", name: "Project Management" },
    A2: { category: "design", name: "Design & Release" },
    A3: { category: "design", name: "Virtual Validation" },
    B1: { category: "devrel", name: "Calibration" },
    B2: { category: "devrel", name: "Reliability & Durability" },
    B3: { category: "devrel", name: "System Integration" },
    C: { category: "application", name: "Application Engineering" },
    C1: { category: "application", name: "Application Testing" },
    C2: { category: "application", name: "Technical Certification" },
    D1: { category: "contracts", name: "Bench Testing" },
    D2: { category: "contracts", name: "Vehicle Testing" },
    D3: { category: "contracts", name: "Laboratories (CRF)" },
    E1: { category: "contracts", name: "Materials & Prototypes" },
    E2: { category: "contracts", name: "Supplier_B" },
  };

  // Map activity codes to categories and functions
  breakdown.forEach((item, _index) => {
    const legacyCode = (item.activity_code || "").toUpperCase();
    const actName = (item.activity_name || "").toLowerCase();
    const pe02Code = item.code?.toUpperCase(); // PE02 code from backend (A1, B1, etc.)
    let categoryId = "other";
    let functionName = "General";

    // PRIORITY 1: Use PE02 code if available from backend
    if (pe02Code && PE02_CODE_MAP[pe02Code]) {
      categoryId = PE02_CODE_MAP[pe02Code].category;
      functionName = PE02_CODE_MAP[pe02Code].name;
    }
    // PRIORITY 2: Use backend function name if PE02 code not mapped
    else if (item.function) {
      functionName = item.function;
      // Try to determine category from function name
      const fnLower = item.function.toLowerCase();
      if (
        fnLower.includes("project") ||
        fnLower.includes("design") ||
        fnLower.includes("management")
      ) {
        categoryId = "design";
      } else if (
        fnLower.includes("calibr") ||
        fnLower.includes("reliab") ||
        fnLower.includes("integr")
      ) {
        categoryId = "devrel";
      } else if (
        fnLower.includes("applic") ||
        fnLower.includes("certif") ||
        fnLower.includes("homolog")
      ) {
        categoryId = "application";
      } else if (
        fnLower.includes("test") ||
        fnLower.includes("lab") ||
        fnLower.includes("material") ||
        fnLower.includes("supplier")
      ) {
        categoryId = "contracts";
      }
    }
    // PRIORITY 3: Legacy pattern matching fallback
    else if (
      legacyCode.includes("DES") ||
      legacyCode.includes("DESIGN") ||
      actName.includes("design")
    ) {
      categoryId = "design";
      if (legacyCode.includes("BASE") || actName.includes("base"))
        functionName = "Base Design";
      else if (
        legacyCode.includes("VIRTUAL") ||
        legacyCode.includes("VAL") ||
        actName.includes("valid")
      )
        functionName = "Virtual Validation";
      else if (legacyCode.includes("ATS") || actName.includes("ats"))
        functionName = "ATS";
      else if (legacyCode.includes("EMS") || actName.includes("ems"))
        functionName = "EMS";
      else if (
        legacyCode.includes("OBD") ||
        actName.includes("obd") ||
        actName.includes("diagnostic")
      )
        functionName = "OBD & Diagnostics";
      else functionName = "Design Engineering";
    } else if (
      legacyCode.includes("DEV") ||
      legacyCode.includes("REL") ||
      actName.includes("develop")
    ) {
      categoryId = "devrel";
      functionName = "Dev&Rel";
    } else if (legacyCode.includes("APP") || actName.includes("application")) {
      categoryId = "application";
      functionName = "Application";
    } else if (
      legacyCode.includes("CERT") ||
      actName.includes("certif") ||
      actName.includes("homolog")
    ) {
      categoryId = "application";
      functionName = "Technical Certification";
    } else if (legacyCode.includes("PROTO") || actName.includes("prototype")) {
      categoryId = "application";
      functionName = "Prototype";
    } else if (legacyCode.includes("MAT") || actName.includes("material")) {
      categoryId = "contracts";
      functionName = "Materials";
    } else if (
      legacyCode.includes("LAB") ||
      actName.includes("lab") ||
      actName.includes("test")
    ) {
      categoryId = "contracts";
      functionName = "Laboratories (CRF)";
    } else if (legacyCode.includes("SUPP") || actName.includes("supplier")) {
      categoryId = "contracts";
      functionName = "Supplier_B";
    } else if (
      legacyCode.includes("PM") ||
      legacyCode.includes("MGMT") ||
      actName.includes("manag")
    ) {
      categoryId = "design";
      functionName = "Project Management";
    } else if (actName.includes("stress") || actName.includes("analysis")) {
      categoryId = "design";
      functionName = "Stress Analysis";
    } else if (actName.includes("integration") || actName.includes("integr")) {
      categoryId = "devrel";
      functionName = "System Integration";
    }

    const fnKey = `${categoryId}-${functionName}`;
    if (!functionGroups[fnKey]) {
      functionGroups[fnKey] = {
        id: fnKey,
        name: functionName,
        activities: [],
      };
    }

    // USE BACKEND PE02 EFFORT VALUES if available (from estimation_node.py)
    // Otherwise fall back to simple distribution based on total hours
    const hasBackendEffort =
      item.effort_manpower !== undefined ||
      item.effort_bench_dev !== undefined ||
      item.effort_bench_dur !== undefined ||
      item.effort_vehicle !== undefined;

    let manpower: number;
    let benchDur: number;
    let benchDev: number;
    let benchSpecial: number;
    let vehicle: number;
    let costKEur: number;

    if (hasBackendEffort) {
      // Use backend-calculated PE02 effort distribution
      manpower = Math.round(item.effort_manpower || 0);
      benchDev = Math.round(item.effort_bench_dev || 0);
      benchSpecial = Math.round(item.effort_bench_special || 0);
      benchDur = Math.round(item.effort_bench_dur || 0);
      vehicle = Math.round(item.effort_vehicle || 0);

      // Use backend cost if available
      if (item.investment_keur !== undefined) {
        costKEur = item.investment_keur;
      } else if (item.cost_eur !== undefined) {
        costKEur = item.cost_eur / 1000;
      } else {
        // Fallback: calculate from hours and rate
        const hourlyRate = item.hourly_rate_eur || 75;
        const totalHours =
          manpower + benchDev + benchSpecial + benchDur + vehicle;
        costKEur = (totalHours * hourlyRate) / 1000;
      }
    } else {
      // LEGACY FALLBACK: Calculate effort distribution from total hours
      const baseHours = item.hours || 0;

      // Simple default distribution
      manpower = Math.round(baseHours * 0.7);
      benchDev = Math.round(baseHours * 0.1);
      benchSpecial = 0;
      benchDur = Math.round(baseHours * 0.15);
      vehicle = Math.round(baseHours * 0.05);

      // Calculate cost with default rate
      const totalEffortHours = manpower + benchDur + benchDev + vehicle;
      costKEur = (totalEffortHours * 75) / 1000;
    }

    functionGroups[fnKey].activities.push({
      id: item.id,
      code: item.code || item.activity_code, // PE02 function code (A1, B1, etc.)
      description: item.activity_name,
      effort: {
        manpower,
        benchDev,
        benchSpecial, // Now populated from backend
        benchDur,
        vehicle,
      },
      costKEur: Math.round(costKEur * 10) / 10, // Round to 1 decimal
      confidence: item.confidence_score || 0.75,
    });
  });

  // Assign functions to categories
  Object.entries(functionGroups).forEach(([key, fn]) => {
    const categoryId = key.split("-")[0];
    if (categoryGroups[categoryId]) {
      categoryGroups[categoryId].peFunctions.push(fn);
    }
  });

  // Filter out empty categories
  const categories = Object.values(categoryGroups).filter(
    (cat) => cat.peFunctions.length > 0,
  );

  // Calculate total cost
  const totalCostKEur = categories.reduce(
    (sum, cat) =>
      sum +
      cat.peFunctions.reduce(
        (s, fn) => s + fn.activities.reduce((a, act) => a + act.costKEur, 0),
        0,
      ),
    0,
  );

  return {
    categories,
    totalCostKEur: Math.round(totalCostKEur * 10) / 10,
    aiGenerated: true,
  };
}

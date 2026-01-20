/**
 * FPT Cost Brain 2.0 - Client-side Export Utilities
 * Fallback export when backend is unavailable
 */

import * as XLSX from "xlsx";
import type { RDTableRow } from "@/components/estimation/rd-cost-table";
import type { ProgramSizingData, ActivitySummaryData } from "@/stores/estimationStore";
import type { BreakdownItem } from "@/lib/api";

interface ExportData {
  sessionId: string;
  prCode?: string;
  prTitle?: string;
  programSizing: ProgramSizingData;
  rdTableRows: RDTableRow[];
  totalHours: number;
  totalCost: number;
  // Fallback data sources
  breakdown?: BreakdownItem[];
  activitySummary?: ActivitySummaryData | null;
}

/**
 * Export estimation data to Excel using client-side data
 */
export function exportToExcel(data: ExportData): void {
  console.log("[EXPORT] exportToExcel called with data:", {
    sessionId: data.sessionId,
    rdTableRowsCount: data.rdTableRows?.length || 0,
    breakdownCount: data.breakdown?.length || 0,
    totalHours: data.totalHours,
    totalCost: data.totalCost,
    programSizingOverall: data.programSizing?.overallSize,
  });

  // Use rdTableRows if available, otherwise fall back to breakdown
  const hasRdData = data.rdTableRows && data.rdTableRows.length > 0;
  const hasBreakdown = data.breakdown && data.breakdown.length > 0;

  if (!hasRdData && !hasBreakdown) {
    console.warn("[EXPORT] No data to export! Both rdTableRows and breakdown are empty.");
    throw new Error("No estimation data available to export");
  }

  const workbook = XLSX.utils.book_new();

  // Sheet 1: Summary
  const summaryData = [
    ["FPT Cost Brain - Estimation Export"],
    [],
    ["Session ID", data.sessionId],
    ["PR Code", data.prCode || "N/A"],
    ["PR Title", data.prTitle || "N/A"],
    [],
    ["Overall Program Size", data.programSizing.overallSize.toUpperCase()],
    ["AI Confidence", `${(data.programSizing.overallConfidence * 100).toFixed(0)}%`],
    [],
    ["Total Hours", data.totalHours.toLocaleString()],
    ["Total Cost (k€)", data.totalCost.toLocaleString()],
  ];
  const summarySheet = XLSX.utils.aoa_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(workbook, summarySheet, "Summary");

  // Sheet 2: R&D Cost Breakdown (PE02 format)
  // Use rdTableRows if available, otherwise use breakdown directly
  let rdRows: (string | number)[][] = [];
  let totals = { manpower: 0, benchDev: 0, benchSpecial: 0, benchDur: 0, vehicle: 0, investment: 0 };

  if (hasRdData) {
    // Use rdTableRows (preferred - PE02 format)
    rdRows = data.rdTableRows.map((row) => [
      row.functionId,
      row.peFunction,
      row.mainActivitiesDescription,
      row.manpower || 0,
      row.benchDev || 0,
      row.benchSpecial || 0,
      row.benchDur || 0,
      row.vehicleTests || 0,
      row.investmentKEur || 0,
    ]);

    totals = data.rdTableRows.reduce(
      (acc, row) => ({
        manpower: acc.manpower + (row.manpower || 0),
        benchDev: acc.benchDev + (row.benchDev || 0),
        benchSpecial: acc.benchSpecial + (row.benchSpecial || 0),
        benchDur: acc.benchDur + (row.benchDur || 0),
        vehicle: acc.vehicle + (row.vehicleTests || 0),
        investment: acc.investment + (row.investmentKEur || 0),
      }),
      totals
    );
  } else if (hasBreakdown && data.breakdown) {
    // Fallback: use breakdown data directly
    console.log("[EXPORT] Using breakdown fallback for export");
    rdRows = data.breakdown.map((item) => [
      item.code || item.activity_code || "",
      item.function || item.activity_name || "",
      item.reasoning || item.activity_name || "",
      item.effort_manpower || item.hours || 0,
      item.effort_bench_dev || 0,
      item.effort_bench_special || 0,
      item.effort_bench_dur || 0,
      item.effort_vehicle || 0,
      item.investment_keur || (item.cost_eur ? item.cost_eur / 1000 : 0),
    ]);

    totals = data.breakdown.reduce(
      (acc, item) => ({
        manpower: acc.manpower + (item.effort_manpower || item.hours || 0),
        benchDev: acc.benchDev + (item.effort_bench_dev || 0),
        benchSpecial: acc.benchSpecial + (item.effort_bench_special || 0),
        benchDur: acc.benchDur + (item.effort_bench_dur || 0),
        vehicle: acc.vehicle + (item.effort_vehicle || 0),
        investment: acc.investment + (item.investment_keur || (item.cost_eur ? item.cost_eur / 1000 : 0)),
      }),
      totals
    );
  }

  const rdHeaders = [
    "PE Function",
    "Function Name",
    "Activities",
    "Manpower [hrs]",
    "Bench Dev [hrs]",
    "Bench Special [hrs]",
    "Bench Dur [hrs]",
    "Vehicle [hrs]",
    "Investment [k€]",
  ];

  rdRows.push([
    "TOTAL",
    "",
    "",
    totals.manpower,
    totals.benchDev,
    totals.benchSpecial,
    totals.benchDur,
    totals.vehicle,
    totals.investment,
  ]);

  const rdSheet = XLSX.utils.aoa_to_sheet([rdHeaders, ...rdRows]);

  // Set column widths
  rdSheet["!cols"] = [
    { wch: 12 }, // PE Function
    { wch: 25 }, // Function Name
    { wch: 30 }, // Activities
    { wch: 14 }, // Manpower
    { wch: 14 }, // Bench Dev
    { wch: 14 }, // Bench Special
    { wch: 14 }, // Bench Dur
    { wch: 12 }, // Vehicle
    { wch: 14 }, // Investment
  ];

  XLSX.utils.book_append_sheet(workbook, rdSheet, "R&D Cost Breakdown");

  // Sheet 3: Program Sizing
  const sizingHeaders = ["Domain", "AI Prediction", "Selected Size", "Confidence"];
  const sizingRows = Object.entries(data.programSizing.columns).map(([domain, col]) => [
    domain,
    col.aiPredictedSize.toUpperCase(),
    col.selectedSize.toUpperCase(),
    `${(col.confidence * 100).toFixed(0)}%`,
  ]);

  const sizingSheet = XLSX.utils.aoa_to_sheet([sizingHeaders, ...sizingRows]);
  sizingSheet["!cols"] = [
    { wch: 20 },
    { wch: 15 },
    { wch: 15 },
    { wch: 12 },
  ];

  XLSX.utils.book_append_sheet(workbook, sizingSheet, "Program Sizing");

  // Generate and download
  const filename = `estimation_${data.sessionId.slice(0, 8)}_${new Date().toISOString().slice(0, 10)}.xlsx`;
  XLSX.writeFile(workbook, filename);
}

/**
 * Export estimation data to CSV (simpler format)
 */
export function exportToCSV(data: ExportData): void {
  const csvRows: string[] = [];
  const hasRdData = data.rdTableRows && data.rdTableRows.length > 0;
  const hasBreakdown = data.breakdown && data.breakdown.length > 0;

  if (!hasRdData && !hasBreakdown) {
    throw new Error("No estimation data available to export");
  }

  // Headers
  csvRows.push([
    "PE Function",
    "Function Name",
    "Activities",
    "Manpower [hrs]",
    "Bench Dev [hrs]",
    "Bench Special [hrs]",
    "Bench Dur [hrs]",
    "Vehicle [hrs]",
    "Investment [k€]",
  ].join(","));

  // Data rows - use rdTableRows or fallback to breakdown
  if (hasRdData) {
    data.rdTableRows.forEach((row) => {
      csvRows.push([
        row.functionId,
        `"${(row.peFunction || "").replace(/"/g, '""')}"`,
        `"${(row.mainActivitiesDescription || "").replace(/"/g, '""')}"`,
        row.manpower || 0,
        row.benchDev || 0,
        row.benchSpecial || 0,
        row.benchDur || 0,
        row.vehicleTests || 0,
        row.investmentKEur || 0,
      ].join(","));
    });
  } else if (hasBreakdown && data.breakdown) {
    data.breakdown.forEach((item) => {
      csvRows.push([
        item.code || item.activity_code || "",
        `"${(item.function || item.activity_name || "").replace(/"/g, '""')}"`,
        `"${(item.reasoning || item.activity_name || "").replace(/"/g, '""')}"`,
        item.effort_manpower || item.hours || 0,
        item.effort_bench_dev || 0,
        item.effort_bench_special || 0,
        item.effort_bench_dur || 0,
        item.effort_vehicle || 0,
        item.investment_keur || (item.cost_eur ? item.cost_eur / 1000 : 0),
      ].join(","));
    });
  }

  // Create and download
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `estimation_${data.sessionId.slice(0, 8)}_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
}

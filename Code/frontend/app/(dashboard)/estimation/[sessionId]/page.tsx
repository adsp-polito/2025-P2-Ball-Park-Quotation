"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { useParams, useRouter } from "next/navigation";
// import { useTranslations } from "next-intl";
import {
  ArrowLeft,
  ArrowRight,
  Download,
  Loader2,
  RotateCcw,
  AlertCircle,
  MessageSquare,
  PanelRightClose,
  PanelRightOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  StepIndicator,
  StepIndicatorCompact,
  type Step,
} from "@/components/estimation/step-indicator";
import {
  QuestionList,
  type Question,
} from "@/components/estimation/question-list";
import {
  PRSummaryCard,
  type PRSummary,
} from "@/components/estimation/pr-summary-card";
import { type QuotationData } from "@/components/estimation/quotation-table";
import { type EditedItem } from "@/components/estimation/editable-quotation-table";
import { StepLoading } from "@/components/estimation/step-loading";
import { ProgramSizingMatrix } from "@/components/estimation/program-sizing-matrix";
import {
  ActivitySummaryTable,
  convertBreakdownToActivitySummary,
  type ActivitySummaryData,
} from "@/components/estimation/activity-summary-table";
import {
  RDCostTable,
  type RDTableRow,
} from "@/components/estimation/rd-cost-table";
import {
  AdaptiveRAGChat,
  type ChatMessage,
} from "@/components/chat/adaptive-rag-chat";
import { useEstimationStore, type ProgramSize } from "@/stores/estimationStore";
import { chatApi, exportApi, extractErrorMessage } from "@/lib/api";
import { exportToExcel, exportToCSV } from "@/lib/export-utils";

/**
 * Convert ActivitySummaryData to RDTableRow[] format for PE02-style R&D Cost Table.
 *
 * PE02 Format:
 * Function ID (A1, A2, B1...) | PE Function | Main Activities | Manpower |
 * Bench(Dev) | Bench(Special) | Bench(Dur) | Vehicle tests | Investment [k€]
 */
function convertActivitySummaryToRDTableRows(
  summary: ActivitySummaryData | null,
  _parsedPr: Record<string, unknown> | null // Kept for backward compat
): RDTableRow[] {
  if (!summary) return [];

  const rows: RDTableRow[] = [];

  summary.categories.forEach(
    (category: ActivitySummaryData["categories"][0], catIdx: number) => {
      category.peFunctions.forEach(
        (
          fn: ActivitySummaryData["categories"][0]["peFunctions"][0],
          fnIdx: number
        ) => {
          fn.activities.forEach(
            (
              activity: ActivitySummaryData["categories"][0]["peFunctions"][0]["activities"][0],
              actIdx: number
            ) => {
              // Create FLAT row for each activity (PE02 format)
              const rowId = `row-${catIdx}-${fnIdx}-${actIdx}`;

              // Extract PE02 function ID from activity code or category
              // Expected format from backend: A1, A2, B1, B2, C, D1, D2, D3, E, F, G
              const functionId =
                activity.code || fn.code || category.code || `A${fnIdx + 1}`;

              rows.push({
                id: rowId,
                // PE02 identifier columns
                functionId: functionId,
                peFunction: fn.name || category.name,
                // Activities description
                mainActivitiesDescription: activity.description,
                // PE02 effort columns
                manpower: activity.effort.manpower || 0,
                benchDev: activity.effort.benchDev || 0,
                benchSpecial: activity.effort.benchSpecial || 0, // NEW column
                benchDur: activity.effort.benchDur || 0,
                vehicleTests: activity.effort.vehicle || 0,
                // Cost in k€
                investmentKEur: activity.costKEur || 0,
                // AI metadata
                confidence: activity.confidence || 0.75,
              });
            }
          );
        }
      );
    }
  );

  return rows;
}

export default function EstimationSessionPage() {
  const params = useParams();
  const router = useRouter();
  const sessionId = params.sessionId as string;

  // Get all state from the store
  const {
    currentStep,
    advanceStep,
    isLoading,
    setCurrentStep,
    completedSteps, // From store for step navigation persistence
    goToStep, // Navigate to completed steps via API
    loadSession,
    setAnswer,
    updateBreakdownItem,
    generateQuestions, // LAZY LOADING: Generate questions on Q&A entry
    questionsReady, // LAZY LOADING: True when questions generated
    questionsGenerating, // LAZY LOADING: True while generating
    error,
    // Data from API
    parsedPr,
    questions: storeQuestions,
    answers,
    prSummary,
    breakdown,
    totalHours,
    totalCost,
    overallConfidence,
    similarPrs,
    // PE02 Tables (from store for chat agent modification)
    programSizing,
    setProgramSizing,
    activitySummary,
    setActivitySummary,
    // Backend sizing predictions (from ref_sizing.json rules)
    mlSizing,
    sizingPredictions,
    sizingConfidence,
  } = useEstimationStore();

  const [editedItems, setEditedItems] = useState<EditedItem[]>([]);
  const [_expandedCategories, _setExpandedCategories] = useState<string[]>([
    "Management",
  ]);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [_isChatLoading, setIsChatLoading] = useState(false);
  // completedSteps now comes from store (persisted via backend step_status)
  const [sessionLoaded, setSessionLoaded] = useState(false);
  const [rdTableData, setRdTableData] = useState<RDTableRow[]>([]);

  // Derive activitySummary from breakdown if not already set
  // This bridges the gap between backend BreakdownItem[] and frontend ActivitySummaryData
  const derivedActivitySummary = useMemo(() => {
    // Use store's activitySummary if available (e.g., from chat agent modifications)
    if (activitySummary) return activitySummary;
    // Otherwise derive from breakdown data (from backend estimation node)
    if (breakdown && breakdown.length > 0) {
      return convertBreakdownToActivitySummary(breakdown);
    }
    return null;
  }, [activitySummary, breakdown]);

  // Convert activitySummary to RD table format using memoization
  // Passes parsedPr to extract actual PR code (e.g., "PR_24019")
  const rdTableRows = useMemo(() => {
    if (rdTableData.length > 0) return rdTableData;
    return convertActivitySummaryToRDTableRows(
      derivedActivitySummary,
      parsedPr
    );
  }, [derivedActivitySummary, rdTableData, parsedPr]);

  // Handlers for RDCostTable - sync edits back to activitySummary store
  const handleRdTableChange = useCallback(
    (newData: RDTableRow[]) => {
      console.log(
        "[EDIT] handleRdTableChange called, edited rows:",
        newData
          .filter((r) => r.isEdited)
          .map((r) => ({ id: r.id, invested: r.investmentKEur }))
      );
      // 1. Update local table state immediately (for smooth UX)
      setRdTableData(newData);

      // 2. Sync edited rows back to activitySummary store
      // This ensures data persists when navigating between steps
      if (!derivedActivitySummary) return;

      // Build updated activitySummary from edited rows
      const updatedCategories = derivedActivitySummary.categories.map(
        (category) => ({
          ...category,
          peFunctions: category.peFunctions.map((fn) => ({
            ...fn,
            activities: fn.activities.map((activity) => {
              // Find matching row by ID
              const editedRow = newData.find((row) => row.id === activity.id);
              if (!editedRow || !editedRow.isEdited) return activity;

              // Update activity with edited values
              return {
                ...activity,
                effort: {
                  manpower: editedRow.manpower ?? activity.effort.manpower,
                  benchDev: editedRow.benchDev ?? activity.effort.benchDev,
                  benchSpecial:
                    editedRow.benchSpecial ?? activity.effort.benchSpecial,
                  benchDur: editedRow.benchDur ?? activity.effort.benchDur,
                  vehicle: editedRow.vehicleTests ?? activity.effort.vehicle,
                },
                costKEur: editedRow.investmentKEur ?? activity.costKEur,
                isEdited: true,
              };
            }),
          })),
        })
      );

      // Recalculate total cost
      let totalCost = 0;
      updatedCategories.forEach((cat) => {
        cat.peFunctions.forEach((fn) => {
          fn.activities.forEach((act) => {
            totalCost += act.costKEur;
          });
        });
      });

      // Update store (persists to Zustand state)
      setActivitySummary({
        ...derivedActivitySummary,
        categories: updatedCategories,
        totalCostKEur: totalCost,
      });

      console.log(
        "[RDCostTable] Synced",
        newData.filter((r) => r.isEdited).length,
        "edited rows to store"
      );
    },
    [derivedActivitySummary, setActivitySummary]
  );

  const handleRdTableFinalize = useCallback(() => {
    console.log("RD Table finalized");
    // TODO: Call API to finalize the table
  }, []);

  const handleRdTableExport = useCallback(
    (format: "pptx" | "xlsx" | "pdf" | "csv") => {
      console.log("Export RD table as:", format);
      // TODO: Call export API
    },
    []
  );

  // Chat panel collapse state with localStorage persistence
  const CHAT_COLLAPSED_KEY = "estimation_chat_collapsed";
  const [isChatCollapsed, setIsChatCollapsed] = useState(false);
  const [isChatMounted, setIsChatMounted] = useState(false);

  // Load chat collapsed state from localStorage on mount
  useEffect(() => {
    setIsChatMounted(true);
    const stored = localStorage.getItem(CHAT_COLLAPSED_KEY);
    if (stored !== null) {
      setIsChatCollapsed(stored === "true");
    }
  }, []);

  // Toggle chat collapsed state with persistence
  const toggleChatCollapsed = useCallback(() => {
    const newState = !isChatCollapsed;
    setIsChatCollapsed(newState);
    localStorage.setItem(CHAT_COLLAPSED_KEY, String(newState));
  }, [isChatCollapsed]);

  // Load session on mount
  useEffect(() => {
    console.log("[DEBUG] Load session effect triggered:", {
      sessionId,
      sessionLoaded,
      isLoading,
    });
    // Guard: only load if we have sessionId, haven't loaded yet, and not currently loading
    if (sessionId && !sessionLoaded && !isLoading) {
      console.log("[DEBUG] Actually calling loadSession...");
      loadSession(sessionId)
        .then(() => {
          console.log("[DEBUG] loadSession completed successfully");
          setSessionLoaded(true);
        })
        .catch((err) => console.error("Failed to load session:", err));
    }
  }, [sessionId, sessionLoaded, isLoading, loadSession]);

  // LAZY LOADING: Generate questions when entering Q&A step
  // Track if we've already attempted generation to prevent loops
  const [generationAttempted, setGenerationAttempted] = useState(false);

  useEffect(() => {
    // Reset generation attempt flag when session changes
    if (!sessionLoaded) {
      setGenerationAttempted(false);
      return;
    }

    // Extra guards to prevent unnecessary regeneration:
    // 1. Must be in Q&A step
    // 2. Questions not already ready
    // 3. Not currently generating
    // 4. Actually have no questions (double-check using storeQuestions from Zustand)
    // 5. Haven't already attempted (prevents loops)
    if (
      sessionLoaded &&
      currentStep === "qa" &&
      !questionsReady &&
      !questionsGenerating &&
      storeQuestions.length === 0 &&
      !generationAttempted
    ) {
      console.log("[LAZY LOADING] Entering Q&A step, generating questions...");
      setGenerationAttempted(true);
      generateQuestions().catch((err) => {
        console.error("Failed to generate questions:", err);
        // Don't reset generationAttempted on error - prevents infinite loop
      });
    }
  }, [
    sessionLoaded,
    currentStep,
    questionsReady,
    questionsGenerating,
    storeQuestions.length,
    generateQuestions,
    generationAttempted,
  ]);

  // Convert breakdown to Activity Summary format when available
  useEffect(() => {
    if (breakdown && breakdown.length > 0 && !activitySummary) {
      const converted = convertBreakdownToActivitySummary(breakdown);
      setActivitySummary(converted);
    }
  }, [breakdown, activitySummary, setActivitySummary]);

  // Derive program sizing from backend predictions (uses ref_sizing.json rules)
  // The backend now provides per-domain sizing via LLM + knowledge base
  useEffect(() => {
    // Wait for backend sizing data or fallback to prSummary
    if (mlSizing || sizingPredictions || prSummary) {
      // Map backend sizing to frontend ProgramSize type
      const mapSize = (s: string): ProgramSize => {
        const normalized = s.toLowerCase().replace("-", "_").replace(" ", "_");
        if (normalized.includes("full") || normalized.includes("extra_large"))
          return "full";
        if (normalized.includes("large")) return "large";
        if (normalized.includes("x_small") || normalized.includes("xsmall"))
          return "x_small";
        if (normalized.includes("small")) return "small";
        return "medium";
      };

      // Get overall sizing from backend (LLM-predicted using ref_sizing.json rules)
      const summaryData = (prSummary || {}) as { program_size?: string };
      const overallSizeStr = (
        mlSizing ||
        summaryData.program_size ||
        "medium"
      ).toLowerCase();

      const overallSize = mapSize(overallSizeStr);
      const backendConfidence = sizingConfidence || 0.75;

      // Column domains matching backend's sizing_predictions keys
      const columnDomains = [
        "basePWT",
        "systemAssembly",
        "installation",
        "plantBasePWT",
        "plantATSPG",
        "sourcing",
        "supplierQuality",
      ] as const;

      const columns: Record<
        string,
        {
          aiPredictedSize: ProgramSize;
          selectedSize: ProgramSize;
          confidence: number;
        }
      > = {};

      // Use backend per-domain sizing predictions if available
      const backendPredictions = sizingPredictions || {};

      columnDomains.forEach((col) => {
        const colPred = backendPredictions[col];
        if (colPred) {
          const colSize = mapSize(colPred.size || overallSizeStr);
          columns[col] = {
            aiPredictedSize: colSize,
            selectedSize: colSize,
            confidence: colPred.confidence || backendConfidence,
          };
        } else {
          // Fallback: use overall sizing for domains without predictions
          columns[col] = {
            aiPredictedSize: overallSize,
            selectedSize: overallSize,
            confidence: backendConfidence * 0.9, // Slightly lower confidence for fallback
          };
        }
      });

      setProgramSizing({
        overallSize: overallSize,
        aiPredictedOverallSize: overallSize,
        overallConfidence: backendConfidence,
        columns: columns as any,
      });
    }
  }, [
    mlSizing,
    sizingPredictions,
    sizingConfidence,
    prSummary,
    setProgramSizing,
  ]);

  // Transform store questions to component format
  const questions: Question[] = useMemo(() => {
    if (!storeQuestions || storeQuestions.length === 0) {
      return [];
    }
    return storeQuestions.map((q) => ({
      id: q.id,
      question: q.text || q.question || "",
      category: q.category || "General",
      importance: (q.priority as "high" | "medium" | "low") || "medium",
      suggestedAnswers: q.suggested_answers || [],
      answer: answers[q.id] || undefined,
      relatedPRs: [],
    }));
  }, [storeQuestions, answers]);

  // Transform store summary to component format
  const summary: PRSummary | null = useMemo(() => {
    if (!parsedPr) return null;

    // Type-safe access to parsedPr properties
    const pr = parsedPr as {
      pr_code?: string;
      title?: string;
      description?: string;
      customer?: string;
      program_family?: string;
      raw_activities?: Array<{ code: string; name: string; hours: number }>;
    };

    // Type-safe access to prSummary with all LLM-generated fields
    const summaryData = prSummary as {
      complexity_score?: number;
      program_size?: string;
      activity_count?: number;
      // LLM-generated narrative fields
      summary_text?: string;
      key_features?: string[];
      dependencies?: string[];
      risk_factors?: string[];
      special_requirements?: string[];
    } | null;

    return {
      prCode: pr.pr_code || "Unknown",
      title: pr.title || "Untitled PR",
      description: pr.description || "",
      // LLM-generated narrative summary for customer manager
      summaryText: summaryData?.summary_text || undefined,
      keyFeatures: summaryData?.key_features || [],
      dependencies: summaryData?.dependencies || [],
      riskFactors: summaryData?.risk_factors || [],
      specialRequirements: summaryData?.special_requirements || [],
      customer: pr.customer || "Unknown",
      program: pr.program_family || "",
      programFamily: pr.program_family || "Unknown",
      createdDate: new Date().toISOString().split("T")[0],
      targetDate: "",
      complexity:
        (summaryData?.complexity_score ?? 0) > 0.7
          ? "high"
          : (summaryData?.complexity_score ?? 0) > 0.4
            ? "medium"
            : "low",
      programSize: (summaryData?.program_size || "medium") as
        | "small"
        | "medium"
        | "large"
        | "extra_large",
      activityCount: summaryData?.activity_count || 0,
      features: [
        { name: "Activity Count", value: summaryData?.activity_count || 0 },
        {
          name: "Complexity Score",
          value: summaryData?.complexity_score || 0.5,
          confidence: 0.8,
        },
        { name: "Program Family", value: pr.program_family || "Unknown" },
      ],
      similarPRs: similarPrs.map((sp) => ({
        prCode: sp.pr_code,
        title: sp.title,
        similarity: sp.similarity_score,
        totalHours: sp.total_hours,
        accuracy: 90,
      })),
    };
  }, [parsedPr, prSummary, similarPrs]);

  // Transform store breakdown to component format
  const quotation: QuotationData = useMemo(() => {
    return {
      totalHours: totalHours || 0,
      totalCost: totalCost || 0,
      currency: "EUR",
      confidence: overallConfidence || 0.5,
      breakdown: breakdown.map((item) => ({
        id: item.id,
        activityCode: item.activity_code,
        activityName: item.activity_name,
        category: getCategoryFromCode(item.activity_code),
        hours: item.hours,
        confidence: item.confidence_score,
        reasoning: item.reasoning,
      })),
      rulesApplied: [],
    };
  }, [breakdown, totalHours, totalCost, overallConfidence]);

  // Helper to get category from activity code
  function getCategoryFromCode(code: string): string {
    const prefix = code.split(".")[0];
    const categories: Record<string, string> = {
      PE: "Management",
      SE: "Engineering",
      DE: "Engineering",
      TE: "Testing",
      CE: "Certification",
      MF: "Manufacturing",
      QA: "Quality",
      SP: "Supply Chain",
      DO: "Documentation",
    };
    return categories[prefix] || "Other";
  }

  const handleAnswerChange = useCallback(
    (questionId: string, answer: string) => {
      setAnswer(questionId, answer);
    },
    [setAnswer]
  );

  const handleQASubmit = useCallback(async () => {
    // completedSteps auto-updated via store when advanceStep returns step_status
    await advanceStep();
  }, [advanceStep]);

  const handleSummaryNext = useCallback(async () => {
    // completedSteps auto-updated via store when advanceStep returns step_status
    await advanceStep();
  }, [advanceStep]);

  const handleEstimationNext = useCallback(async () => {
    // DEMO FIX: Don't call advanceStep (which reloads from backend and loses edits)
    // Just change step locally to preserve edited values
    setCurrentStep("review");
    console.log("[DEMO] Moving to review step, preserving local edits");
  }, [setCurrentStep]);

  const _handleItemEdit = useCallback(
    (itemId: string, newHours: number, reason: string) => {
      const item = quotation.breakdown.find((i) => i.id === itemId);
      if (!item) return;

      setEditedItems((prev) => {
        const existing = prev.find((e) => e.itemId === itemId);
        if (existing) {
          return prev.map((e) =>
            e.itemId === itemId ? { ...e, newHours, reason } : e
          );
        }
        return [
          ...prev,
          { itemId, originalHours: item.hours, newHours, reason },
        ];
      });
    },
    [quotation.breakdown]
  );

  const _handleSaveAll = useCallback(async () => {
    // Save each edit to the API
    for (const edit of editedItems) {
      try {
        await updateBreakdownItem(edit.itemId, edit.newHours, edit.reason);
      } catch (_e) {
        console.error("Failed to save edit:", _e);
      }
    }
    setEditedItems([]);
  }, [editedItems, updateBreakdownItem]);

  const handleResetAll = useCallback(() => {
    setEditedItems([]);
  }, []);

  const _handleCategoryToggle = useCallback((category: string) => {
    _setExpandedCategories((prev) =>
      prev.includes(category)
        ? prev.filter((c) => c !== category)
        : [...prev, category]
    );
  }, []);

  const handleChatMessage = useCallback(
    async (message: string): Promise<ChatMessage> => {
      setIsChatLoading(true);

      // Add user message
      const userMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        role: "user",
        content: message,
        timestamp: new Date(),
      };
      setChatMessages((prev) => [...prev, userMessage]);

      try {
        // Call real chat API
        const response = await chatApi.sendMessage(
          sessionId,
          message,
          chatMessages.map((m) => ({ role: m.role, content: m.content }))
        );

        const assistantMessage: ChatMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: response.response,
          timestamp: new Date(),
          sources: response.tool_calls?.map((tc) => ({
            type: "knowledge" as const,
            title: tc.tool,
          })),
          suggestions: response.suggestions?.map((s) => s.text),
        };

        setChatMessages((prev) => [...prev, assistantMessage]);
        setIsChatLoading(false);
        return assistantMessage;
      } catch {
        // Fallback response on error
        const errorMessage: ChatMessage = {
          id: `msg-${Date.now() + 1}`,
          role: "assistant",
          content: `I understand you're asking about "${message}". I'm currently processing your request. The system is being configured for full AI chat capabilities.`,
          timestamp: new Date(),
          suggestions: ["Tell me more", "Show examples"],
        };

        setChatMessages((prev) => [...prev, errorMessage]);
        setIsChatLoading(false);
        return errorMessage;
      }
    },
    [sessionId, chatMessages]
  );

  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExport = useCallback(
    async (format: "pptx" | "xlsx" | "bundle" | "csv") => {
      if (!sessionId) return;

      setIsExporting(true);
      setExportError(null);

      // Debug: log available data
      console.log("[EXPORT] Data available for export:", {
        sessionId,
        rdTableRowsCount: rdTableRows?.length || 0,
        breakdownCount: breakdown?.length || 0,
        activitySummaryCategories: activitySummary?.categories?.length || 0,
        derivedActivitySummaryCategories:
          derivedActivitySummary?.categories?.length || 0,
        totalHours,
        totalCost,
      });

      // For CSV, use client-side export directly
      if (format === "csv") {
        try {
          exportToCSV({
            sessionId,
            prCode: (parsedPr as { pr_code?: string } | null)?.pr_code,
            prTitle: (parsedPr as { title?: string } | null)?.title,
            programSizing,
            rdTableRows,
            totalHours: totalHours || 0,
            totalCost: totalCost || 0,
            breakdown,
            activitySummary: derivedActivitySummary,
          });
        } catch (e) {
          console.error("CSV export failed:", e);
          setExportError("Failed to export CSV");
        } finally {
          setIsExporting(false);
        }
        return;
      }

      try {
        console.log(`Exporting as ${format} for session ${sessionId}`);

        switch (format) {
          case "pptx":
            await exportApi.exportPptx(sessionId);
            break;
          case "xlsx":
            await exportApi.exportXlsx(sessionId);
            break;
          case "bundle":
            await exportApi.exportBundle(sessionId);
            break;
        }
      } catch (e) {
        console.error("Backend export failed, trying client-side fallback:", e);

        // Fallback to client-side Excel export
        if (format === "xlsx" || format === "bundle") {
          try {
            console.log("Using client-side Excel export as fallback");
            exportToExcel({
              sessionId,
              prCode: (parsedPr as { pr_code?: string } | null)?.pr_code,
              prTitle: (parsedPr as { title?: string } | null)?.title,
              programSizing,
              rdTableRows,
              totalHours: totalHours || 0,
              totalCost: totalCost || 0,
              breakdown,
              activitySummary: derivedActivitySummary,
            });
            setExportError(null); // Clear error since fallback worked
          } catch (fallbackError) {
            console.error("Fallback export also failed:", fallbackError);
            setExportError("Export failed. Please try again.");
          }
        } else {
          // For PPTX, show error (no client-side fallback for PowerPoint)
          const errorMsg = extractErrorMessage(e, "Export failed");
          setExportError(`${errorMsg}. Try Excel export instead.`);
        }
      } finally {
        setIsExporting(false);
      }
    },
    [
      sessionId,
      parsedPr,
      programSizing,
      rdTableRows,
      totalHours,
      totalCost,
      breakdown,
      derivedActivitySummary,
    ]
  );

  const handleStepClick = useCallback(
    async (step: Step) => {
      console.log("[STEP_NAV] handleStepClick called:", {
        clickedStep: step,
        currentStep,
        completedSteps,
        isIncluded: completedSteps.includes(step),
        isCurrent: currentStep === step,
      });
      if (completedSteps.includes(step) || currentStep === step) {
        console.log("[STEP_NAV] Calling goToStep for:", step);
        await goToStep(step); // Navigate via API to sync backend state
      } else {
        console.log("[STEP_NAV] Step not clickable - not in completedSteps");
      }
    },
    [completedSteps, currentStep, goToStep]
  );

  // Show error state
  if (error) {
    // Safely extract error message from any format (handles nested objects)
    const errorMessage = extractErrorMessage(
      error,
      "An unknown error occurred"
    );

    return (
      <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error Loading Session</AlertTitle>
          <AlertDescription>{errorMessage}</AlertDescription>
          <Button
            variant="outline"
            className="mt-4"
            onClick={() => router.push("/estimation/new")}
          >
            Start New Estimation
          </Button>
        </Alert>
      </div>
    );
  }

  // Show loading state
  if (!sessionLoaded || isLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
        <div className="text-center space-y-4">
          <Loader2 className="h-12 w-12 animate-spin mx-auto text-primary" />
          <p className="text-muted-foreground">Loading estimation session...</p>
        </div>
      </div>
    );
  }

  const renderStepContent = () => {
    switch (currentStep) {
      case "qa":
        // LAZY LOADING: Show animation while generating questions
        if (questionsGenerating) {
          return <StepLoading variant="qa" />;
        }

        // Show error if generation failed and no questions
        if (!questionsReady && questions.length === 0 && error) {
          return (
            <div className="flex flex-col items-center justify-center py-12 space-y-4">
              <AlertCircle className="h-12 w-12 text-destructive" />
              <h3 className="text-lg font-semibold">
                Failed to Generate Questions
              </h3>
              <p className="text-muted-foreground text-center max-w-md">
                {error}
              </p>
              <Button
                onClick={() => {
                  setGenerationAttempted(false);
                  generateQuestions().catch((err) =>
                    console.error("Failed to generate questions:", err)
                  );
                }}
                className="gap-2"
              >
                <RotateCcw className="h-4 w-4" />
                Retry Generation
              </Button>
            </div>
          );
        }

        // Still loading - show loading state
        if (!questionsReady && questions.length === 0) {
          return <StepLoading variant="qa" />;
        }

        return (
          <QuestionList
            questions={questions}
            onAnswerChange={handleAnswerChange}
            onSubmitAll={handleQASubmit}
            isSubmitting={isLoading}
          />
        );

      case "summary":
        // Show beautiful loading while LLM summary is being generated
        // Check if prSummary (LLM-generated) is available, not just parsedPr
        if (!prSummary || !summary || isLoading) {
          return <StepLoading variant="summary" />;
        }

        return (
          <div className="space-y-4">
            <PRSummaryCard summary={summary} />
            <div className="flex justify-end pt-4">
              <Button
                onClick={handleSummaryNext}
                disabled={isLoading}
                size="lg"
                className="gap-2"
              >
                {isLoading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : null}
                Continue to Estimation
                <ArrowRight className="h-5 w-5" />
              </Button>
            </div>
          </div>
        );

      case "estimation":
        // Show beautiful loading while ML prediction is being generated
        // Check breakdown from API and activitySummary (converted format)
        if (breakdown.length === 0 || !activitySummary || isLoading) {
          return <StepLoading variant="estimation" />;
        }

        return (
          <div className="space-y-6">
            {/* Table 1: Program Sizing Matrix (Blue) - Per-Column AI Predictions */}
            <ProgramSizingMatrix
              data={programSizing}
              onDataChange={(newData) => setProgramSizing(newData)}
              readOnly={false}
            />

            {/* Table 2: R&D Cost Table (Excel-like with DPO reasoning) */}
            <RDCostTable
              sessionId={sessionId}
              initialData={rdTableRows}
              onDataChange={handleRdTableChange}
              onFinalize={handleRdTableFinalize}
              onExport={handleRdTableExport}
              isLoading={isLoading}
              isReadOnly={false}
            />

            {/* Summary Stats */}
            <Card>
              <CardContent className="py-4">
                <div className="flex items-center justify-between">
                  <div className="flex gap-8">
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Total Hours
                      </p>
                      <p className="text-2xl font-bold">
                        {activitySummary.categories
                          .reduce(
                            (sum, cat) =>
                              sum +
                              cat.peFunctions.reduce(
                                (s, fn) =>
                                  s +
                                  fn.activities.reduce(
                                    (a, act) =>
                                      a +
                                      act.effort.manpower +
                                      act.effort.benchDur +
                                      act.effort.benchDev +
                                      act.effort.vehicle,
                                    0
                                  ),
                                0
                              ),
                            0
                          )
                          .toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Total Cost
                      </p>
                      <p className="text-2xl font-bold text-primary">
                        €{activitySummary.totalCostKEur.toLocaleString()}K
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        AI Confidence
                      </p>
                      <p className="text-2xl font-bold">
                        {Math.round((overallConfidence || 0.75) * 100)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Program Size
                      </p>
                      <p className="text-2xl font-bold capitalize">
                        {programSizing.overallSize.replace("_", " ")}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={handleResetAll}
                      className="gap-2"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Reset to AI Values
                    </Button>
                    <Button
                      onClick={handleEstimationNext}
                      disabled={isLoading}
                      size="lg"
                      className="gap-2"
                    >
                      {isLoading ? (
                        <Loader2 className="h-5 w-5 animate-spin" />
                      ) : null}
                      Finalize & Export
                      <ArrowRight className="h-5 w-5" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case "review":
        return (
          <div className="space-y-6">
            {/* Final Summary Card */}
            <Card className="border-2 border-primary/20 bg-primary/5">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    ✓
                  </div>
                  Estimation Complete
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div>
                    <p className="text-sm text-muted-foreground">PR Code</p>
                    <p className="text-lg font-semibold">
                      {summary?.prCode || "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">
                      Program Size
                    </p>
                    <p className="text-lg font-semibold capitalize">
                      {programSizing.overallSize.replace("_", " ")}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Hours</p>
                    <p className="text-lg font-semibold">
                      {activitySummary?.categories
                        .reduce(
                          (sum, cat) =>
                            sum +
                            cat.peFunctions.reduce(
                              (s, fn) =>
                                s +
                                fn.activities.reduce(
                                  (a, act) =>
                                    a +
                                    act.effort.manpower +
                                    act.effort.benchDur +
                                    act.effort.benchDev +
                                    act.effort.vehicle,
                                  0
                                ),
                              0
                            ),
                          0
                        )
                        .toLocaleString() || "0"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Total Cost</p>
                    <p className="text-lg font-semibold text-primary">
                      €{activitySummary?.totalCostKEur.toLocaleString() || "0"}K
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Read-only Tables Preview */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">
                Final Estimation Preview
              </h3>

              <ProgramSizingMatrix
                data={programSizing}
                onDataChange={() => {}}
                readOnly={true}
              />

              {activitySummary && (
                <ActivitySummaryTable
                  data={activitySummary}
                  onDataChange={() => {}}
                  readOnly={true}
                />
              )}
            </div>

            {/* Export options */}
            <Card>
              <CardHeader>
                <CardTitle>Export to PE02 Format</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Download your estimation in FPT standard PE02 format for
                  presentation and approval.
                </p>

                {exportError && (
                  <Alert variant="destructive" className="mb-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{exportError}</AlertDescription>
                  </Alert>
                )}

                <div className="flex flex-wrap gap-3">
                  <Button
                    onClick={() => handleExport("pptx")}
                    variant="outline"
                    size="lg"
                    className="gap-2"
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Download className="h-5 w-5" />
                    )}
                    PowerPoint (PE02)
                  </Button>
                  <Button
                    onClick={() => handleExport("xlsx")}
                    variant="outline"
                    size="lg"
                    className="gap-2"
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Download className="h-5 w-5" />
                    )}
                    Excel Breakdown
                  </Button>
                  <Button
                    onClick={() => handleExport("csv")}
                    variant="outline"
                    size="lg"
                    className="gap-2"
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Download className="h-5 w-5" />
                    )}
                    CSV (Quick)
                  </Button>
                  <Button
                    onClick={() => handleExport("bundle")}
                    size="lg"
                    className="gap-2"
                    disabled={isExporting}
                  >
                    {isExporting ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      <Download className="h-5 w-5" />
                    )}
                    Download All
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Back to Edit */}
            <div className="flex justify-start">
              <Button
                variant="ghost"
                onClick={() => setCurrentStep("estimation")}
                className="gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Edit Estimation
              </Button>
            </div>
          </div>
        );

      default:
        return (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        );
    }
  };

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex h-[calc(100vh-8rem)] gap-4 xl:gap-6">
        {/* Main Content - expands when chat is collapsed */}
        <div
          className={cn(
            "flex-1 min-w-0 overflow-y-auto transition-all duration-300 ease-in-out"
          )}
        >
          <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => router.push("/estimation/new")}
                    className="gap-1 p-0 h-auto"
                  >
                    <ArrowLeft className="h-4 w-4" />
                    New Estimation
                  </Button>
                  <span>/</span>
                  <span>Session {sessionId.slice(0, 8)}...</span>
                </div>
                <h1 className="mt-1 text-2xl font-bold tracking-tight">
                  {summary?.prCode || "Loading..."} - {summary?.title || ""}
                </h1>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => router.push("/estimation/new")}
                className="gap-1"
              >
                <RotateCcw className="h-4 w-4" />
                Start Over
              </Button>
            </div>

            {/* Step Indicator */}
            <StepIndicator
              currentStep={currentStep}
              completedSteps={completedSteps}
              onStepClick={handleStepClick}
            />

            {/* Step Content */}
            <div className="pb-6">{renderStepContent()}</div>
          </div>
        </div>

        {/* Collapsible Chat Sidebar */}
        <div
          className={cn(
            "hidden lg:flex flex-col flex-shrink-0 transition-all duration-300 ease-in-out",
            isChatMounted
              ? isChatCollapsed
                ? "w-12"
                : "w-72 xl:w-80"
              : "w-72 xl:w-80"
          )}
        >
          {/* Chat Toggle Button (visible when collapsed) */}
          {isChatCollapsed ? (
            <div className="h-full flex flex-col items-center pt-2 border-l bg-card/50">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={toggleChatCollapsed}
                    className="w-10 h-10 p-0 mb-2"
                  >
                    <PanelRightOpen className="h-5 w-5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="left">Expand AI Assistant</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex flex-col items-center gap-1 text-muted-foreground">
                    <MessageSquare className="h-5 w-5" />
                    <span
                      className="text-[10px] writing-mode-vertical rotate-180"
                      style={{ writingMode: "vertical-rl" }}
                    >
                      AI Chat
                    </span>
                  </div>
                </TooltipTrigger>
                <TooltipContent side="left">
                  Click to expand chat
                </TooltipContent>
              </Tooltip>
            </div>
          ) : (
            <div className="h-full flex flex-col border-l">
              {/* Chat Header with Collapse Button */}
              <div className="flex items-center justify-between px-3 py-2 border-b bg-card/50">
                <div className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4 text-primary" />
                  <span className="text-sm font-medium">AI Assistant</span>
                </div>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={toggleChatCollapsed}
                      className="h-8 w-8 p-0"
                    >
                      <PanelRightClose className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="left">Collapse chat</TooltipContent>
                </Tooltip>
              </div>
              {/* Chat Content */}
              <div className="flex-1 overflow-hidden">
                <AdaptiveRAGChat
                  sessionId={sessionId}
                  currentStep={currentStep}
                  onSendMessage={handleChatMessage}
                  className="h-full"
                  pageContext={{
                    questions: storeQuestions.map((q) => ({
                      id: q.id,
                      question_text: q.question || q.text,
                      answer: answers[q.id] || "",
                      category: q.category,
                    })),
                    breakdown: breakdown,
                    parsed_pr: parsedPr || undefined,
                    pr_summary: prSummary || undefined,
                    current_step: currentStep,
                    program_sizing: programSizing,
                    activity_summary: activitySummary || undefined,
                  }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Mobile Step Indicator */}
        <div className="fixed bottom-0 left-0 right-0 border-t bg-background p-4 lg:hidden">
          <StepIndicatorCompact
            currentStep={currentStep}
            completedSteps={completedSteps}
          />
        </div>
      </div>
    </TooltipProvider>
  );
}

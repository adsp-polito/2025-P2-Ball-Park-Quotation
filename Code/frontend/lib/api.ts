/**
 * FPT Cost Brain 2.0 - API Client
 * Typed fetch wrapper for backend communication
 */

import { logger } from "@/lib/logger";

// Use empty string to make relative requests that go through Next.js rewrites
// This ensures the requests work both in development and production
const API_BASE_URL = "";

interface ApiError {
  detail: string;
  status: number;
}

/**
 * Safely extract a string error message from any error value.
 * Handles nested objects, Error instances, and various error formats.
 */
export function extractErrorMessage(error: unknown, fallback = "An error occurred"): string {
  if (!error) return fallback;

  // Already a string
  if (typeof error === "string") return error;

  // Error instance
  if (error instanceof Error) return error.message;

  // Object with message/detail properties
  if (typeof error === "object" && error !== null) {
    const obj = error as Record<string, unknown>;

    // Try common error property names, ensuring they're strings
    for (const key of ["message", "detail", "error", "msg", "errorMessage"]) {
      const value = obj[key];
      if (typeof value === "string" && value.length > 0) {
        return value;
      }
      // Recursively extract if nested object
      if (typeof value === "object" && value !== null) {
        const nested = extractErrorMessage(value, "");
        if (nested) return nested;
      }
    }

    // Try JSON stringify as last resort
    try {
      const jsonStr = JSON.stringify(error);
      // Don't return empty objects or arrays
      if (jsonStr !== "{}" && jsonStr !== "[]") {
        return jsonStr;
      }
    } catch {
      // Stringify failed
    }
  }

  return fallback;
}

interface RequestOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

/**
 * Get stored auth token from cookie or localStorage
 */
function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;

  // Try cookie first (for SSR compatibility)
  const cookieMatch = document.cookie.match(/auth_token=([^;]+)/);
  if (cookieMatch) return cookieMatch[1];

  // Fallback to localStorage
  return localStorage.getItem("auth_token");
}

/**
 * Store auth token in both cookie and localStorage
 */
export function setAuthToken(token: string): void {
  if (typeof window === "undefined") return;
  localStorage.setItem("auth_token", token);
  // Set cookie with SameSite=Lax for security, expires in 7 days
  document.cookie = `auth_token=${token}; path=/; max-age=${7 * 24 * 60 * 60}; SameSite=Lax`;
}

/**
 * Clear auth token from both cookie and localStorage
 */
export function clearAuthToken(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem("auth_token");
  document.cookie = "auth_token=; path=/; max-age=0";
}

/**
 * Build URL with query parameters
 */
function buildUrl(
  path: string,
  params?: Record<string, string | number | boolean | undefined>,
): string {
  const url = new URL(path, API_BASE_URL || window.location.origin);

  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        url.searchParams.append(key, String(value));
      }
    });
  }

  return url.toString();
}

/**
 * Base fetch function with auth and error handling
 */
async function apiFetch<T>(
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { params, ...fetchOptions } = options;

  const token = getAuthToken();

  // Check if body is FormData - if so, don't set Content-Type (browser will set it with boundary)
  const isFormData = fetchOptions.body instanceof FormData;

  const headers: HeadersInit = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...fetchOptions.headers,
  };

  const url = buildUrl(`/api/v1${path}`, params);

  // Log the request
  const logBody = isFormData
    ? "[FormData]"
    : typeof fetchOptions.body === "string"
      ? fetchOptions.body
      : undefined;
  const ctx = logger.request(fetchOptions.method || "GET", url, {
    headers,
    body: logBody,
  });

  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  });

  if (!response.ok) {
    const error: ApiError = {
      detail: "An error occurred",
      status: response.status,
    };

    let responseData: unknown;
    try {
      responseData = await response.json();
      // Use extractErrorMessage to safely get a string from any format
      error.detail = extractErrorMessage(responseData, error.detail);
    } catch {
      // Ignore JSON parse errors
    }

    // Log the error response
    logger.response(ctx, {
      status: response.status,
      body: responseData || error.detail,
    });

    // Handle 401 - redirect to login without throwing
    if (response.status === 401) {
      clearAuthToken();
      if (typeof window !== "undefined") {
        window.location.href = "/login";
        // Return a never-resolving promise to prevent further execution
        return new Promise(() => {}) as T;
      }
    }

    throw error;
  }

  // Handle 204 No Content
  if (response.status === 204) {
    logger.response(ctx, { status: 204, body: "[No Content]" });
    return {} as T;
  }

  const data = await response.json();

  // Log the successful response
  logger.response(ctx, {
    status: response.status,
    body: data,
  });

  return data as T;
}

/**
 * Download a file from the API (handles binary responses)
 */
async function apiDownload(
  path: string,
  options: RequestOptions = {},
  defaultFilename = "download",
): Promise<void> {
  const { params, ...fetchOptions } = options;

  const token = getAuthToken();

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...fetchOptions.headers,
  };

  const url = buildUrl(`/api/v1${path}`, params);

  const response = await fetch(url, {
    ...fetchOptions,
    headers,
  });

  if (!response.ok) {
    let errorDetail = "Download failed";
    try {
      const data = await response.json();
      errorDetail = extractErrorMessage(data, errorDetail);
    } catch {
      // Response might not be JSON
    }
    throw new Error(errorDetail);
  }

  // Get filename from Content-Disposition header if available
  const contentDisposition = response.headers.get("Content-Disposition");
  let filename = defaultFilename;
  if (contentDisposition) {
    const match = contentDisposition.match(/filename="?([^";\n]+)"?/);
    if (match) filename = match[1];
  }

  // Convert response to blob and trigger download
  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(downloadUrl);
}

// ===== Auth API =====

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user: {
    id: string;
    email: string;
    full_name: string;
    role: string;
  };
}

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
}

export const authApi = {
  login: (data: LoginRequest) =>
    apiFetch<LoginResponse>("/auth/login/json", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  logout: () =>
    apiFetch<void>("/auth/logout", {
      method: "POST",
    }),

  me: () => apiFetch<User>("/auth/me"),

  changePassword: (data: { current_password: string; new_password: string }) =>
    apiFetch<void>("/auth/change-password", {
      method: "POST",
      body: JSON.stringify(data),
    }),
};

// ===== Estimation API =====

export interface EstimationSession {
  session_id: string;
  current_step: string;
  created_at: string;
  updated_at: string;
}

export interface BreakdownItem {
  id: string;
  activity_code: string;
  activity_name: string;
  hours: number;
  hourly_rate_eur: number;
  cost_eur: number;
  confidence_score: number;
  reasoning: string;
  source: string;
  user_edited: boolean;
  edit_reason?: string;
  // PE02 effort breakdown fields (from backend estimation node)
  code?: string; // PE02 function code (A1, B1, C, etc.)
  function?: string; // PE Function name
  effort_manpower?: number; // Manpower hours
  effort_bench_dev?: number; // Bench Development hours
  effort_bench_special?: number; // Bench Special hours (NVH, climatic)
  effort_bench_dur?: number; // Bench Durability hours
  effort_vehicle?: number; // Vehicle tests hours
  investment_keur?: number; // Investment in k€
}

export interface EstimationState {
  session_id: string;
  current_step: string;
  // Step completion status (for navigation)
  step_status?: Record<string, string>; // e.g., {"intake": "COMPLETED", "qa": "WAITING_INPUT"}
  parsed_pr?: Record<string, unknown>;
  questions?: Array<{
    id: string;
    text: string;
    category: string;
    required: boolean;
  }>;
  // LAZY LOADING: Question generation state
  questions_ready?: boolean; // True when questions have been generated
  questions_generating?: boolean; // True while generation in progress
  answers?: Record<string, string>;
  pr_summary?: Record<string, unknown>;
  breakdown?: BreakdownItem[];
  total_hours?: number;
  total_cost_eur?: number;
  overall_confidence?: number;
  similar_prs?: Array<Record<string, unknown>>;
  error_message?: string;
  // Program sizing predictions (from ref_sizing.json rules via LLM)
  ml_sizing?: string; // Overall sizing: Full, Large, Medium, Small, X-small
  sizing_predictions?: Record<
    string,
    { size: string; confidence: number; reason: string }
  >; // Per-domain sizing
  sizing_confidence?: number; // Overall confidence 0-1
}

// ===== HCQE Prediction Types =====

export interface HCQEPredictionRequest {
  features: Record<string, unknown>;
}

export interface HCQEPrediction {
  predicted_cost_keur: number;
  predicted_hours: number;
  confidence: number;
  method: string;
  prediction_interval?: {
    lower_keur: number;
    upper_keur: number;
    lower_hours: number;
    upper_hours: number;
  };
  quantiles?: {
    q10_keur: number;
    q50_keur: number;
    q90_keur: number;
  };
  sizing?: {
    predicted: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  cluster_estimates: Record<string, number>;
  reasoning: string;
  recommendations: string[];
}

export interface HCQEHealth {
  status: "healthy" | "degraded";
  model: string;
  version?: string;
  accuracy?: string;
  interval_coverage?: string;
  message?: string;
}

export const estimationApi = {
  start: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    // FormData Content-Type is automatically handled by apiFetch
    return apiFetch<EstimationSession>("/estimation/start", {
      method: "POST",
      body: formData,
    });
  },

  getSession: (sessionId: string) =>
    apiFetch<EstimationState>(`/estimation/${sessionId}`),

  /**
   * Generate Q&A questions (LAZY LOADING)
   * Call this when entering the Q&A step - questions are NOT generated during upload
   * to provide faster initial upload experience.
   */
  generateQuestions: (sessionId: string) =>
    apiFetch<EstimationState>(`/estimation/${sessionId}/generate-questions`, {
      method: "POST",
    }),

  nextStep: (sessionId: string, data?: Record<string, unknown>) =>
    apiFetch<EstimationState>(`/estimation/${sessionId}/next`, {
      method: "POST",
      body: JSON.stringify(data || {}),
    }),

  /**
   * Navigate to a specific completed step (view only, no re-processing)
   * Allows users to go back and review previous steps without restarting.
   */
  goToStep: (sessionId: string, step: string) =>
    apiFetch<EstimationState>(`/estimation/${sessionId}/go-to-step`, {
      method: "POST",
      body: JSON.stringify({ step }),
    }),

  getStep: (sessionId: string, step: string) =>
    apiFetch<Record<string, unknown>>(`/estimation/${sessionId}/step/${step}`),

  updateBreakdown: (
    sessionId: string,
    itemId: string,
    data: { hours: number; reason: string },
  ) =>
    apiFetch<BreakdownItem>(`/estimation/${sessionId}/breakdown/${itemId}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  // HCQE Prediction endpoints
  predict: (features: Record<string, unknown>) =>
    apiFetch<HCQEPrediction>("/estimation/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  getPredictionHealth: () => apiFetch<HCQEHealth>("/estimation/predict/health"),
};

// ===== Chat API =====

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

/**
 * Chat mode - determines agent capabilities
 * - chat: Read-only assistant mode (default)
 * - agent: Full agent mode with write capabilities (GOD MODE)
 */
export type ChatMode = "chat" | "agent";

/**
 * Page context for chat - what's currently visible on the screen
 * This ensures the chat sees exactly what the user sees, even if Redis state is stale
 */
export interface PageContext {
  questions?: Array<{
    id?: string;
    question_text?: string;
    text?: string;
    answer?: string;
    category?: string;
  }>;
  breakdown?: BreakdownItem[];
  parsed_pr?: Record<string, unknown>;
  pr_summary?: Record<string, unknown>;
  user_edits?: Array<{
    breakdown_id?: string;
    original_hours?: number;
    new_hours?: number;
    reason?: string;
  }>;
  current_step?: string;
  // PE02 Tables (for chat agent to read/modify)
  // New per-column prediction format with independent program size
  program_sizing?: {
    overallSize: string;
    aiPredictedOverallSize: string;
    overallConfidence: number;
    columns: Record<
      string,
      {
        aiPredictedSize: string;
        selectedSize: string;
        confidence: number;
      }
    >;
  };
  activity_summary?: {
    categories: Array<{
      id: string;
      name: string;
      peFunctions: Array<{
        id: string;
        name: string;
        activities: Array<{
          id: string;
          description: string;
          effort: {
            manpower: number;
            benchDur: number;
            benchDev: number;
            vehicle: number;
          };
          costKEur: number;
        }>;
      }>;
    }>;
    totalCostKEur: number;
    aiGenerated: boolean;
  };
}

// GOD MODE: Action result from state modifications
export interface ActionResult {
  status: "success" | "error" | "pending_reprocess" | "no_action";
  action_type: string;
  details: string;
}

export interface ChatResponse {
  response: string;
  suggestions: Array<{
    text: string;
    action: string;
    icon: string;
  }>;
  tool_calls?: Array<{
    tool: string;
    result: string;
  }>;
  step: string;
  // GOD MODE fields
  intent?: string;
  action_executed?: boolean;
  action_result?: ActionResult;
  updated_state?: Record<string, unknown>;
}

// Streaming response types
export interface StreamChunk {
  type: "chunk" | "done" | "error" | "status";
  content?: string;
  suggestions?: Array<{ text: string; action: string; icon: string }>;
  step?: string;
  message?: string;
  // Status event fields
  status?: "thinking" | "generating";
  // GOD MODE fields (included in 'done' event)
  intent?: string;
  action_executed?: boolean;
  action_result?: ActionResult;
  updated_state?: Record<string, unknown>;
}

export interface ChatHistoryItem {
  id: string;
  content: string;
  role: string;
  created_at: string;
  tool_calls?: Array<{ tool: string; result: string }>;
  sources?: Record<string, unknown>;
}

export const chatApi = {
  /**
   * Send a message to the chat API
   * @param sessionId - The estimation session ID
   * @param message - The user's message
   * @param history - Previous messages for context
   * @param pageContext - Optional current page context (questions, breakdown, etc.)
   * @param mode - Chat mode: 'chat' (read-only) or 'agent' (GOD MODE with write capabilities)
   */
  sendMessage: (
    sessionId: string,
    message: string,
    history: ChatMessage[],
    pageContext?: PageContext,
    mode: ChatMode = "chat",
  ) =>
    apiFetch<ChatResponse>(`/chat/${sessionId}/message`, {
      method: "POST",
      body: JSON.stringify({
        message,
        history,
        page_context: pageContext || null,
        mode,
      }),
    }),

  /**
   * Send message with streaming response using Server-Sent Events
   * Returns an async generator that yields StreamChunk objects
   *
   * Uses a dedicated Next.js Route Handler to bypass proxy buffering
   * and enable true real-time streaming.
   *
   * @param sessionId - The estimation session ID
   * @param message - The user's message
   * @param history - Previous messages for context
   * @param pageContext - Optional current page context (questions, breakdown, etc.)
   * @param mode - Chat mode: 'chat' (read-only) or 'agent' (GOD MODE with write capabilities)
   */
  sendMessageStream: async function* (
    sessionId: string,
    message: string,
    history: ChatMessage[],
    pageContext?: PageContext,
    mode: ChatMode = "chat",
  ): AsyncGenerator<StreamChunk> {
    // Use dedicated streaming route that bypasses Next.js rewrite buffering
    const url = `/api/chat/${sessionId}/stream`;

    logger.info("stream", `Starting stream: ${sessionId}`, {
      mode,
      messageLength: message.length,
      historyLength: history.length,
      hasPageContext: !!pageContext,
    });

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        history,
        page_context: pageContext || null,
        mode,
      }),
    });

    if (!response.ok) {
      logger.error("stream", `Stream failed: ${response.status}`, { sessionId });
      if (response.status === 401) {
        clearAuthToken();
        window.location.href = "/login";
        return;
      }
      throw new Error(`Stream request failed: ${response.status}`);
    }

    logger.debug("stream", `Stream connected: ${sessionId}`);

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("No response body");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let chunkCount = 0;

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data) {
              try {
                const chunk: StreamChunk = JSON.parse(data);
                chunkCount++;
                logger.streamChunk(sessionId, chunkCount, chunk);
                yield chunk;
              } catch {
                // Not JSON, might be raw text
                chunkCount++;
                const rawChunk = { type: "chunk" as const, content: data };
                logger.streamChunk(sessionId, chunkCount, rawChunk);
                yield rawChunk;
              }
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
      logger.info("stream", `Stream complete: ${sessionId}`, { totalChunks: chunkCount });
    }
  },

  getSuggestions: (sessionId: string) =>
    apiFetch<Array<{ text: string; action: string; icon: string }>>(
      `/chat/${sessionId}/suggestions`,
    ),

  getHistory: (sessionId: string) =>
    apiFetch<ChatHistoryItem[]>(`/chat/${sessionId}/history`),

  clearHistory: (sessionId: string) =>
    apiFetch<{ message: string; session_id: string }>(
      `/chat/${sessionId}/history`,
      { method: "DELETE" },
    ),

  executeTool: (
    toolName: string,
    sessionId?: string,
    params?: Record<string, unknown>,
  ) =>
    apiFetch<{ tool: string; result: string; data: Record<string, unknown> }>(
      `/chat/tool/${toolName}`,
      {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, ...params }),
      },
    ),
};

// ===== Export API =====

export const exportApi = {
  /**
   * Download estimation as PowerPoint (PE02 format)
   */
  exportPptx: (sessionId: string, language = "en") =>
    apiDownload(
      `/export/pptx`,
      {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, language }),
      },
      `estimation_${sessionId}.pptx`,
    ),

  /**
   * Download estimation as Excel
   */
  exportXlsx: (sessionId: string, language = "en") =>
    apiDownload(
      `/export/xlsx`,
      {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, language }),
      },
      `estimation_${sessionId}.xlsx`,
    ),

  /**
   * Download both PowerPoint and Excel as ZIP bundle
   */
  exportBundle: (sessionId: string, language = "en") =>
    apiDownload(
      `/export/bundle`,
      {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, language }),
      },
      `estimation_${sessionId}_bundle.zip`,
    ),
};

// ===== History API =====

export interface PRHistory {
  id: string;
  pr_code: string;
  title: string;
  program_family: string | null;
  customer: string | null;
  total_hours: number | null;
  total_cost_eur: number | null;
  created_at: string;
  status: string;
}

export interface QuotationSummary {
  id: string;
  version: number;
  total_hours: number;
  total_cost_eur: number;
  confidence_score: number | null;
  is_finalized: boolean;
  created_at: string;
}

export interface PRDetail {
  id: string;
  pr_code: string;
  title: string;
  description: string | null;
  program_family: string | null;
  customer: string | null;
  status: string;
  created_at: string;
  updated_at: string | null;
  raw_data: Record<string, unknown>;
  quotations: QuotationSummary[];
}

export interface QuotationBreakdown {
  id: string;
  activity_code: string;
  activity_name: string;
  hours: number;
  hourly_rate_eur: number;
  cost_eur: number;
  confidence_score: number | null;
  reasoning: string | null;
  source: string | null;
  user_edited: boolean;
}

export interface QuotationDetail {
  id: string;
  pr_id: string;
  version: number;
  total_hours: number;
  total_cost_eur: number;
  confidence_score: number | null;
  estimation_method: string | null;
  is_finalized: boolean;
  created_at: string;
  breakdowns: QuotationBreakdown[];
}

export const historyApi = {
  listPRs: (params?: {
    page?: number;
    limit?: number;
    search?: string;
    status?: string;
    program_family?: string;
  }) =>
    apiFetch<{ items: PRHistory[]; total: number }>("/history/prs", { params }),

  getPR: (prId: string) => apiFetch<PRDetail>(`/history/prs/${prId}`),

  getQuotation: (prId: string, version: number) =>
    apiFetch<QuotationDetail>(`/history/prs/${prId}/quotation/${version}`),

  compare: (prIds: string[]) =>
    apiFetch<Record<string, unknown>>("/history/compare", {
      method: "POST",
      body: JSON.stringify({ pr_ids: prIds }),
    }),

  getSimilar: (prId: string, limit?: number) =>
    apiFetch<{
      source_pr: { id: string; pr_code: string; program_family: string | null };
      similar: Array<{
        id: string;
        pr_code: string;
        title: string;
        program_family: string | null;
        similarity_reason: string;
      }>;
    }>(`/history/similar/${prId}`, { params: limit ? { limit } : undefined }),

  getStatistics: () =>
    apiFetch<{
      product_requests: {
        total: number;
        by_status: Record<string, number>;
        this_month: number;
        last_month: number;
      };
      quotations: {
        total: number;
        avg_hours: number;
        avg_cost_eur: number;
        avg_confidence: number;
      };
      top_activities: Array<{
        activity: string;
        total_hours: number;
        count: number;
      }>;
    }>("/history/statistics"),
};

// ===== Dashboard API =====

export interface DashboardStats {
  totalEstimations: number;
  completedThisMonth: number;
  averageAccuracy: number;
  modelVersion: string;
  pendingCorrections: number;
  averageProcessingTime: string;
}

export interface RecentEstimation {
  id: string;
  prCode: string;
  title: string;
  status: string;
  hours: number | null;
  accuracy: number | null;
  date: string;
}

export const dashboardApi = {
  getStats: async (): Promise<DashboardStats> => {
    const historyStats = await historyApi.getStatistics();

    return {
      totalEstimations: historyStats.product_requests.total,
      completedThisMonth: historyStats.product_requests.this_month,
      averageAccuracy: historyStats.quotations.avg_confidence
        ? Math.round(historyStats.quotations.avg_confidence * 100)
        : 0,
      modelVersion: "2.1.0", // TODO: Get from model status API
      pendingCorrections: 0, // TODO: Get from admin API if user is admin
      averageProcessingTime: "~2 min",
    };
  },

  getRecentEstimations: async (limit = 5): Promise<RecentEstimation[]> => {
    const response = await historyApi.listPRs({ limit });
    return response.items.map((item) => ({
      id: item.id,
      prCode: item.pr_code,
      title: item.title,
      status: item.status,
      hours: item.total_hours,
      accuracy: null, // Confidence could be added if available
      date: item.created_at.split("T")[0],
    }));
  },
};

// ===== Knowledge API =====

// ===== Knowledge API Types =====

export interface Acronym {
  id: string;
  acronym: string;
  full_form: string;
  description: string | null;
  category: string | null;
}

export interface AcronymCreate {
  acronym: string;
  full_form: string;
  description?: string;
  category?: string;
}

export interface AcronymUpdate {
  acronym?: string;
  full_form?: string;
  description?: string;
  category?: string;
}

export interface AcronymListResponse {
  items: Acronym[];
  total: number;
}

export interface KnowledgeDocument {
  id: string;
  title: string;
  content: string;
  doc_type: string | null;
  category: string | null;
  source_file_path: string | null;
  is_indexed: boolean;
  chunk_count: number | null;
  created_at: string;
}

export interface DocumentListResponse {
  items: KnowledgeDocument[];
  total: number;
}

export const knowledgeApi = {
  // Acronyms
  listAcronyms: (params?: {
    search?: string;
    domain?: string;
    limit?: number;
  }) => apiFetch<AcronymListResponse>("/knowledge/acronyms", { params }),

  getAcronym: (id: string) => apiFetch<Acronym>(`/knowledge/acronyms/${id}`),

  createAcronym: (data: AcronymCreate) =>
    apiFetch<Acronym>("/knowledge/acronyms", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  updateAcronym: (id: string, data: AcronymUpdate) =>
    apiFetch<Acronym>(`/knowledge/acronyms/${id}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    }),

  deleteAcronym: (id: string) =>
    apiFetch<{ message: string }>(`/knowledge/acronyms/${id}`, {
      method: "DELETE",
    }),

  // Documents
  listDocuments: (params?: {
    search?: string;
    document_type?: string;
    is_indexed?: boolean;
  }) => apiFetch<DocumentListResponse>("/knowledge/documents", { params }),

  getDocument: (id: string) =>
    apiFetch<KnowledgeDocument>(`/knowledge/documents/${id}`),

  deleteDocument: (id: string) =>
    apiFetch<{ message: string }>(`/knowledge/documents/${id}`, {
      method: "DELETE",
    }),

  uploadDocument: (file: File, documentType: string = "uploaded") => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("document_type", documentType);
    return apiFetch<{ id: string; title: string; message: string }>(
      "/knowledge/documents/upload",
      {
        method: "POST",
        body: formData,
      },
    );
  },

  // Statistics
  getStatistics: () =>
    apiFetch<{
      total_acronyms: number;
      total_documents: number;
      indexed_documents: number;
    }>("/knowledge/statistics"),
};

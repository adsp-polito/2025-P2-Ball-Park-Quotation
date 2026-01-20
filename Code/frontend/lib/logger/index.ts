/**
 * FPT Cost Brain 2.0 - Structured Logger
 *
 * Provides comprehensive logging for debugging frontend-backend communication.
 * SSR-safe, configurable via localStorage or env vars.
 */

// Log categories for filtering
export type LogCategory =
  | "api"
  | "store"
  | "route"
  | "stream"
  | "export"
  | "chat"
  | "navigation";

// Log levels
export type LogLevel = "debug" | "info" | "warn" | "error";

// Request context for correlation
export interface RequestContext {
  id: string;
  method: string;
  url: string;
  startTime: number;
}

// Color map for categories (console styling)
const CATEGORY_COLORS: Record<LogCategory, string> = {
  api: "#4CAF50", // Green
  store: "#2196F3", // Blue
  route: "#FF9800", // Orange
  stream: "#9C27B0", // Purple
  export: "#795548", // Brown
  chat: "#E91E63", // Pink
  navigation: "#607D8B", // Blue Grey
};

// Level styles
const LEVEL_STYLES: Record<LogLevel, { color: string; badge: string }> = {
  debug: { color: "#9E9E9E", badge: "DEBUG" },
  info: { color: "#4CAF50", badge: "INFO" },
  warn: { color: "#FF9800", badge: "WARN" },
  error: { color: "#F44336", badge: "ERROR" },
};

// Generate short unique ID for request correlation
function generateRequestId(): string {
  return Math.random().toString(36).substring(2, 8);
}

// Get current timestamp in HH:MM:SS format
function getTimestamp(): string {
  const now = new Date();
  return now.toTimeString().split(" ")[0];
}

// Check if logging is enabled
function isLoggingEnabled(): boolean {
  // Always check on each call for dynamic enable/disable

  // Server-side: check env var
  if (typeof window === "undefined") {
    return process.env.NEXT_PUBLIC_DEBUG_LOGGING === "true";
  }

  // Client-side: check localStorage first, then env
  const localSetting = localStorage.getItem("fpt_debug");
  if (localSetting !== null) {
    return localSetting === "true";
  }

  return process.env.NEXT_PUBLIC_DEBUG_LOGGING === "true";
}

// Get configured log level
function getLogLevel(): LogLevel {
  if (typeof window === "undefined") {
    return (process.env.NEXT_PUBLIC_DEBUG_LEVEL as LogLevel) || "info";
  }

  const localLevel = localStorage.getItem("fpt_debug_level");
  return (localLevel as LogLevel) || "info";
}

// Get filtered categories (empty = all)
function getFilteredCategories(): LogCategory[] | null {
  if (typeof window === "undefined") {
    return null; // No filtering on server
  }

  const filter = localStorage.getItem("fpt_debug_categories");
  if (!filter) return null;

  return filter.split(",").map((c) => c.trim() as LogCategory);
}

// Check if a log should be output
function shouldLog(level: LogLevel, category: LogCategory): boolean {
  if (!isLoggingEnabled()) return false;

  const levelOrder: LogLevel[] = ["debug", "info", "warn", "error"];
  const configuredLevel = getLogLevel();

  if (levelOrder.indexOf(level) < levelOrder.indexOf(configuredLevel)) {
    return false;
  }

  const categories = getFilteredCategories();
  if (categories && !categories.includes(category)) {
    return false;
  }

  return true;
}

// Mask sensitive headers
function maskHeaders(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};

  const masked: Record<string, string> = {};
  const headersObj =
    headers instanceof Headers
      ? Object.fromEntries(headers.entries())
      : (headers as Record<string, string>);

  for (const [key, value] of Object.entries(headersObj)) {
    if (key.toLowerCase() === "authorization") {
      masked[key] = value ? `Bearer ***${value.slice(-6)}` : "[empty]";
    } else {
      masked[key] = value;
    }
  }

  return masked;
}

// Truncate large body for logging
function truncateBody(body: unknown, maxLength = 2000): string {
  if (!body) return "[empty]";

  let str: string;
  if (typeof body === "string") {
    str = body;
  } else {
    try {
      str = JSON.stringify(body, null, 2);
    } catch {
      str = String(body);
    }
  }

  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength) + `\n... [truncated ${str.length - maxLength} chars]`;
}

// Safe JSON parse for response preview
function safeParseJson(data: unknown): unknown {
  if (typeof data === "string") {
    try {
      return JSON.parse(data);
    } catch {
      return data;
    }
  }
  return data;
}

// Format log output
function formatLog(
  level: LogLevel,
  category: LogCategory,
  message: string,
  data?: unknown
): void {
  if (typeof window === "undefined") {
    // Server-side: plain console output
    const prefix = `[${LEVEL_STYLES[level].badge}][${category.toUpperCase()}]`;
    const timestamp = getTimestamp();
    if (data !== undefined) {
      console.log(`${prefix} ${timestamp} ${message}`, data);
    } else {
      console.log(`${prefix} ${timestamp} ${message}`);
    }
    return;
  }

  // Client-side: styled console output
  const categoryColor = CATEGORY_COLORS[category];
  const levelStyle = LEVEL_STYLES[level];
  const timestamp = getTimestamp();

  const baseStyle = `
    background: ${categoryColor};
    color: white;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: bold;
  `;

  const timeStyle = `
    color: #888;
    font-size: 11px;
  `;

  const messageStyle = `
    color: ${levelStyle.color};
  `;

  if (data !== undefined) {
    console.groupCollapsed(
      `%c${category.toUpperCase()}%c ${timestamp} %c${message}`,
      baseStyle,
      timeStyle,
      messageStyle
    );
    console.log(data);
    console.groupEnd();
  } else {
    console.log(
      `%c${category.toUpperCase()}%c ${timestamp} %c${message}`,
      baseStyle,
      timeStyle,
      messageStyle
    );
  }
}

// Main logger interface
export interface Logger {
  debug(category: LogCategory, message: string, data?: unknown): void;
  info(category: LogCategory, message: string, data?: unknown): void;
  warn(category: LogCategory, message: string, data?: unknown): void;
  error(category: LogCategory, message: string, data?: unknown): void;

  // Specialized methods
  request(
    method: string,
    url: string,
    options?: { headers?: HeadersInit; body?: unknown }
  ): RequestContext;
  response(
    ctx: RequestContext,
    response: {
      status: number;
      duration?: string;
      body?: unknown;
      headers?: HeadersInit;
    }
  ): void;
  stateChange(action: string, before: unknown, after: unknown): void;
  streamChunk(
    sessionId: string,
    chunkNumber: number,
    chunk: { type: string; content?: string }
  ): void;
}

// Logger implementation
export const logger: Logger = {
  debug(category: LogCategory, message: string, data?: unknown) {
    if (shouldLog("debug", category)) {
      formatLog("debug", category, message, data);
    }
  },

  info(category: LogCategory, message: string, data?: unknown) {
    if (shouldLog("info", category)) {
      formatLog("info", category, message, data);
    }
  },

  warn(category: LogCategory, message: string, data?: unknown) {
    if (shouldLog("warn", category)) {
      formatLog("warn", category, message, data);
    }
  },

  error(category: LogCategory, message: string, data?: unknown) {
    if (shouldLog("error", category)) {
      formatLog("error", category, message, data);
    }
  },

  request(
    method: string,
    url: string,
    options?: { headers?: HeadersInit; body?: unknown }
  ): RequestContext {
    const ctx: RequestContext = {
      id: generateRequestId(),
      method,
      url,
      startTime: performance.now(),
    };

    if (shouldLog("info", "api")) {
      formatLog("info", "api", `→ ${method} ${url} [${ctx.id}]`, {
        headers: maskHeaders(options?.headers),
        body: options?.body ? truncateBody(options.body, 500) : undefined,
      });
    }

    return ctx;
  },

  response(
    ctx: RequestContext,
    response: {
      status: number;
      duration?: string;
      body?: unknown;
      headers?: HeadersInit;
    }
  ): void {
    const duration =
      response.duration || `${(performance.now() - ctx.startTime).toFixed(0)}ms`;
    const statusEmoji = response.status >= 400 ? "✗" : "✓";

    if (shouldLog("info", "api")) {
      formatLog(
        response.status >= 400 ? "error" : "info",
        "api",
        `← ${response.status} ${statusEmoji} ${ctx.method} ${ctx.url} (${duration}) [${ctx.id}]`,
        {
          body: response.body
            ? truncateBody(safeParseJson(response.body), 2000)
            : undefined,
        }
      );
    }
  },

  stateChange(action: string, before: unknown, after: unknown): void {
    if (shouldLog("debug", "store")) {
      formatLog("debug", "store", `State change: ${action}`, {
        before: truncateBody(before, 500),
        after: truncateBody(after, 500),
      });
    }
  },

  streamChunk(
    sessionId: string,
    chunkNumber: number,
    chunk: { type: string; content?: string }
  ): void {
    if (shouldLog("debug", "stream")) {
      formatLog("debug", "stream", `Chunk #${chunkNumber} [${sessionId}]`, {
        type: chunk.type,
        contentPreview: chunk.content?.slice(0, 100),
      });
    }
  },
};

// Helper for route logging (server-side only)
export function logRoute(
  method: string,
  path: string,
  options?: { body?: unknown; status?: number; duration?: string; error?: unknown }
): void {
  const prefix = options?.status ? "←" : "→";
  const statusStr = options?.status ? ` ${options.status}` : "";
  const durationStr = options?.duration ? ` (${options.duration})` : "";

  console.log(`[ROUTE] ${prefix} ${method}${statusStr} ${path}${durationStr}`);

  if (options?.body) {
    console.log("  Body:", truncateBody(options.body, 500));
  }
  if (options?.error) {
    console.error("  Error:", options.error);
  }
}

// Export utility functions
export { truncateBody, maskHeaders, safeParseJson };

import { NextRequest, NextResponse } from "next/server";
import { logRoute } from "@/lib/logger";

// Server-side: use NEXT_PUBLIC_API_URL for Docker or fallback to localhost
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Extend timeout for file upload and parsing (2 minutes)
const UPLOAD_TIMEOUT_MS = 120_000;

// Next.js route segment config - extend max duration for file operations
export const maxDuration = 120; // 2 minutes

// Check if debug logging is enabled
const isDebugEnabled = () => process.env.NEXT_PUBLIC_DEBUG_LOGGING === "true";

export async function POST(request: NextRequest) {
  // Use AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);
  const startTime = performance.now();

  // Log the incoming request
  if (isDebugEnabled()) {
    logRoute("POST", "/estimation/start");
  }

  try {
    // Get the authorization header
    const authHeader = request.headers.get("authorization");

    if (!authHeader) {
      clearTimeout(timeoutId);
      if (isDebugEnabled()) {
        logRoute("POST", "/estimation/start", {
          status: 401,
          error: "No auth header",
        });
      }
      return NextResponse.json(
        { detail: "Authorization header required" },
        { status: 401 },
      );
    }

    // Get the form data from the request
    const incomingFormData = await request.formData();
    const file = incomingFormData.get("file") as File | null;

    if (!file) {
      clearTimeout(timeoutId);
      if (isDebugEnabled()) {
        logRoute("POST", "/estimation/start", {
          status: 400,
          error: "No file provided",
        });
      }
      return NextResponse.json({ detail: "No file provided" }, { status: 400 });
    }

    if (isDebugEnabled()) {
      logRoute("POST", "/estimation/start", {
        body: `File: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`,
      });
    }

    // Create new FormData with the file
    const outgoingFormData = new FormData();
    outgoingFormData.append("file", file, file.name);

    // Forward the request to the backend
    const response = await fetch(`${BACKEND_URL}/api/v1/estimation/start`, {
      method: "POST",
      headers: {
        Authorization: authHeader,
      },
      body: outgoingFormData,
      signal: controller.signal,
    });

    // Clear timeout on success
    clearTimeout(timeoutId);
    const duration = `${(performance.now() - startTime).toFixed(0)}ms`;

    // Handle response based on content type
    const contentType = response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      const data = await response.json();
      if (isDebugEnabled()) {
        logRoute("POST", "/estimation/start", {
          status: response.status,
          duration,
          body: JSON.stringify(data).slice(0, 500),
        });
      }
      return NextResponse.json(data, { status: response.status });
    } else {
      // If not JSON, return the raw text with the error
      const text = await response.text();
      logRoute("POST", "/estimation/start", {
        status: response.status,
        duration,
        error: text,
      });
      return NextResponse.json(
        { detail: text || "Backend error" },
        { status: response.status },
      );
    }
  } catch (error) {
    clearTimeout(timeoutId);
    const duration = `${(performance.now() - startTime).toFixed(0)}ms`;

    // Handle abort (timeout) specifically
    if (error instanceof Error && error.name === "AbortError") {
      logRoute("POST", "/estimation/start", {
        status: 504,
        duration,
        error: `Timeout after ${UPLOAD_TIMEOUT_MS}ms`,
      });
      return NextResponse.json(
        {
          detail:
            "Upload timeout - file processing is taking longer than expected. Please try again.",
        },
        { status: 504 },
      );
    }

    logRoute("POST", "/estimation/start", {
      status: 500,
      duration,
      error,
    });
    return NextResponse.json(
      { detail: "Internal server error" },
      { status: 500 },
    );
  }
}

import { NextRequest, NextResponse } from "next/server";
import { logRoute } from "@/lib/logger";

// Server-side: use NEXT_PUBLIC_API_URL for Docker or fallback to localhost
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Extend timeout for long-running ML predictions (5 minutes)
const PROXY_TIMEOUT_MS = 300_000;

// Next.js route segment config - extend max duration for ML operations
export const maxDuration = 300; // 5 minutes (Vercel/serverless)
export const dynamic = "force-dynamic";

// Check if debug logging is enabled
const isDebugEnabled = () => process.env.NEXT_PUBLIC_DEBUG_LOGGING === "true";

/**
 * Catch-all route to proxy estimation API requests to the backend.
 * This handles all paths like /api/v1/estimation/{session_id},
 * /api/v1/estimation/{session_id}/next, etc.
 */
async function proxyToBackend(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const { path } = await params;
  const pathString = path.join("/");
  const url = new URL(request.url);
  const queryString = url.search;

  const backendUrl = `${BACKEND_URL}/api/v1/estimation/${pathString}${queryString}`;

  // Log the incoming request
  if (isDebugEnabled()) {
    logRoute(request.method, `/estimation/${pathString}`);
  }

  // Use AbortController for timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), PROXY_TIMEOUT_MS);
  const startTime = performance.now();

  try {
    // Get headers to forward
    const headers: HeadersInit = {};
    const authHeader = request.headers.get("authorization");
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }
    const contentType = request.headers.get("content-type");
    if (contentType) {
      headers["Content-Type"] = contentType;
    }

    // Get body for non-GET requests
    let body: BodyInit | undefined;
    let bodyPreview: string | undefined;
    if (request.method !== "GET" && request.method !== "HEAD") {
      const contentType = request.headers.get("content-type") || "";
      if (contentType.includes("application/json")) {
        body = await request.text();
        bodyPreview = body.slice(0, 200);
        if (isDebugEnabled() && bodyPreview) {
          logRoute(request.method, `/estimation/${pathString}`, {
            body: bodyPreview,
          });
        }
      } else if (contentType.includes("multipart/form-data")) {
        // For form data, pass through the request body
        body = await request.arrayBuffer();
        bodyPreview = "[FormData]";
      }
    }

    const response = await fetch(backendUrl, {
      method: request.method,
      headers,
      body,
      signal: controller.signal,
    });

    // Clear timeout on success
    clearTimeout(timeoutId);

    // Forward the response
    const responseData = await response.text();
    const duration = `${(performance.now() - startTime).toFixed(0)}ms`;

    // Log the response
    if (isDebugEnabled()) {
      logRoute(request.method, `/estimation/${pathString}`, {
        status: response.status,
        duration,
        body: responseData.slice(0, 500),
      });
    }

    return new NextResponse(responseData, {
      status: response.status,
      headers: {
        "Content-Type":
          response.headers.get("Content-Type") || "application/json",
      },
    });
  } catch (error) {
    clearTimeout(timeoutId);
    const duration = `${(performance.now() - startTime).toFixed(0)}ms`;

    // Handle abort (timeout) specifically
    if (error instanceof Error && error.name === "AbortError") {
      logRoute(request.method, `/estimation/${pathString}`, {
        status: 504,
        duration,
        error: `Timeout after ${PROXY_TIMEOUT_MS}ms`,
      });
      return NextResponse.json(
        {
          detail:
            "Request timeout - estimation is taking longer than expected. Please refresh to check status.",
        },
        { status: 504 },
      );
    }

    logRoute(request.method, `/estimation/${pathString}`, {
      status: 502,
      duration,
      error,
    });
    return NextResponse.json(
      { detail: "Failed to connect to backend" },
      { status: 502 },
    );
  }
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return proxyToBackend(request, context);
}

export async function POST(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return proxyToBackend(request, context);
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return proxyToBackend(request, context);
}

export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return proxyToBackend(request, context);
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  return proxyToBackend(request, context);
}

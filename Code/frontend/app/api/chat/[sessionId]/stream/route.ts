/**
 * SSE Streaming Route Handler for Chat
 *
 * This bypasses Next.js rewrite buffering to enable true real-time streaming.
 * The route proxies SSE from the backend with proper streaming headers.
 */

import { cookies } from "next/headers";
import { NextRequest } from "next/server";
import { logRoute } from "@/lib/logger";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Check if debug logging is enabled
const isDebugEnabled = () => process.env.NEXT_PUBLIC_DEBUG_LOGGING === "true";

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> },
) {
  const { sessionId } = await params;
  const startTime = performance.now();

  if (isDebugEnabled()) {
    logRoute("POST", `/chat/${sessionId}/stream`);
  }

  // Get auth token from cookies
  const cookieStore = await cookies();
  const authToken = cookieStore.get("auth_token")?.value;

  if (!authToken) {
    if (isDebugEnabled()) {
      logRoute("POST", `/chat/${sessionId}/stream`, {
        status: 401,
        error: "No auth token",
      });
    }
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Get request body
  const body = await request.json();

  if (isDebugEnabled()) {
    logRoute("POST", `/chat/${sessionId}/stream`, {
      body: `message: "${(body.message || "").slice(0, 50)}...", mode: ${body.mode}, historyCount: ${(body.history || []).length}`,
    });
  }

  // Create the upstream request to backend
  const backendResponse = await fetch(
    `${BACKEND_URL}/api/v1/chat/${sessionId}/message/stream`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${authToken}`,
      },
      body: JSON.stringify(body),
    },
  );

  if (!backendResponse.ok) {
    const duration = `${(performance.now() - startTime).toFixed(0)}ms`;
    logRoute("POST", `/chat/${sessionId}/stream`, {
      status: backendResponse.status,
      duration,
      error: "Backend error",
    });
    return new Response(JSON.stringify({ error: "Backend error" }), {
      status: backendResponse.status,
      headers: { "Content-Type": "application/json" },
    });
  }

  if (isDebugEnabled()) {
    const connectTime = `${(performance.now() - startTime).toFixed(0)}ms`;
    logRoute("POST", `/chat/${sessionId}/stream`, {
      status: 200,
      duration: `connected in ${connectTime}`,
    });
  }

  // Create a TransformStream to pass through the SSE data
  const { readable, writable } = new TransformStream();

  // Pipe backend response to our transform stream
  const reader = backendResponse.body?.getReader();
  const writer = writable.getWriter();

  // Process the stream in background
  (async () => {
    if (!reader) {
      await writer.close();
      return;
    }

    let chunkCount = 0;
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunkCount++;
        await writer.write(value);
      }
      if (isDebugEnabled()) {
        const totalDuration = `${(performance.now() - startTime).toFixed(0)}ms`;
        console.log(
          `[STREAM] Session ${sessionId} completed: ${chunkCount} chunks in ${totalDuration}`,
        );
      }
    } catch (error) {
      console.error(`[STREAM] Error for session ${sessionId}:`, error);
    } finally {
      await writer.close();
    }
  })();

  // Return streaming response with proper SSE headers
  return new Response(readable, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

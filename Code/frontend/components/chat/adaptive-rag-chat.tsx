"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useTranslations } from "next-intl";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Bot,
  User,
  Sparkles,
  MessageCircle,
  Minimize2,
  Maximize2,
  BookOpen,
  HelpCircle,
  Calculator,
  FileCheck,
  Lightbulb,
  Zap,
  Trash2,
  Brain,
  Wand2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "@/components/ui/tooltip";
import type { Step } from "@/stores/estimationStore";
import {
  chatApi,
  // type StreamChunk,
  type ChatHistoryItem,
  type PageContext,
  type ActionResult,
  type ChatMode,
} from "@/lib/api";
import { useEstimationStore } from "@/stores/estimationStore";

// ===== Animated Components =====

/**
 * Animated thinking indicator with bouncing dots and dynamic status message
 */
function ThinkingIndicator({
  status,
  statusMessage,
}: {
  status: "thinking" | "generating";
  statusMessage?: string;
}) {
  // Default messages based on status
  const defaultMessage =
    status === "thinking"
      ? "Analyzing your question..."
      : "Writing response...";
  const displayMessage = statusMessage || defaultMessage;

  return (
    <div className="flex items-start gap-3 animate-in fade-in slide-in-from-bottom-2 duration-300">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-primary/10 animate-pulse">
        {status === "thinking" ? (
          <Brain className="h-4 w-4 text-primary animate-pulse" />
        ) : (
          <Wand2 className="h-4 w-4 text-primary" />
        )}
      </div>
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2 rounded-2xl bg-muted/80 backdrop-blur-sm px-4 py-3 shadow-sm">
          <span className="text-sm text-muted-foreground">
            {status === "thinking" ? "Thinking" : "Generating response"}
          </span>
          <div className="flex gap-1">
            <span
              className="h-2 w-2 rounded-full bg-primary/60 animate-bounce"
              style={{ animationDelay: "0ms" }}
            />
            <span
              className="h-2 w-2 rounded-full bg-primary/60 animate-bounce"
              style={{ animationDelay: "150ms" }}
            />
            <span
              className="h-2 w-2 rounded-full bg-primary/60 animate-bounce"
              style={{ animationDelay: "300ms" }}
            />
          </div>
        </div>
        <span className="text-[10px] text-muted-foreground/60 ml-1 animate-in fade-in duration-200">
          {displayMessage}
        </span>
      </div>
    </div>
  );
}

/**
 * Animated typing cursor for streaming text
 */
function TypingCursor() {
  return (
    <span className="inline-block w-0.5 h-4 ml-0.5 bg-primary animate-pulse" />
  );
}

/**
 * GOD MODE: Action result indicator
 */
function ActionIndicator({ actionResult }: { actionResult: ActionResult }) {
  const statusConfig = {
    success: {
      icon: "✅",
      bg: "bg-green-500/10",
      border: "border-green-500/30",
      text: "text-green-700 dark:text-green-400",
    },
    error: {
      icon: "❌",
      bg: "bg-red-500/10",
      border: "border-red-500/30",
      text: "text-red-700 dark:text-red-400",
    },
    pending_reprocess: {
      icon: "⏳",
      bg: "bg-yellow-500/10",
      border: "border-yellow-500/30",
      text: "text-yellow-700 dark:text-yellow-400",
    },
    no_action: {
      icon: "❓",
      bg: "bg-gray-500/10",
      border: "border-gray-500/30",
      text: "text-gray-700 dark:text-gray-400",
    },
  };

  const config = statusConfig[actionResult.status] || statusConfig.no_action;

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-3 py-2 rounded-lg border text-xs animate-in fade-in slide-in-from-top-2 duration-300",
        config.bg,
        config.border,
      )}
    >
      <span>{config.icon}</span>
      <div className="flex flex-col">
        <span className={cn("font-medium", config.text)}>
          {actionResult.action_type.replace(/_/g, " ")}
        </span>
        <span className="text-muted-foreground">{actionResult.details}</span>
      </div>
    </div>
  );
}

/**
 * Message bubble with animations
 */
function MessageBubble({
  message,
  onSuggestionClick,
}: {
  message: ChatMessage;
  onSuggestionClick?: (suggestion: string) => void;
}) {
  const isUser = message.role === "user";
  const isStreaming = message.isStreaming;

  return (
    <div
      className={cn(
        "flex gap-3 animate-in fade-in duration-300",
        isUser
          ? "justify-end slide-in-from-right-2"
          : "justify-start slide-in-from-left-2",
      )}
    >
      {/* Assistant avatar */}
      {!isUser && (
        <div
          className={cn(
            "flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full transition-all duration-300",
            isStreaming
              ? "bg-gradient-to-br from-primary/30 to-primary/10 animate-pulse"
              : "bg-gradient-to-br from-primary/20 to-primary/10",
          )}
        >
          <Bot
            className={cn(
              "h-4 w-4 text-primary transition-transform",
              isStreaming && "animate-pulse",
            )}
          />
        </div>
      )}

      {/* Message content */}
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-2.5 transition-all duration-200",
          isUser
            ? "bg-primary text-primary-foreground shadow-md"
            : "bg-muted/80 backdrop-blur-sm shadow-sm",
          isStreaming && !isUser && "ring-1 ring-primary/20",
        )}
      >
        <div className="text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:my-2 prose-headings:font-semibold">
          <ReactMarkdown>{message.content}</ReactMarkdown>
          {isStreaming && <TypingCursor />}
        </div>

        {/* GOD MODE: Action Result */}
        {message.actionResult && !isStreaming && (
          <div className="mt-2 animate-in fade-in duration-500 delay-100">
            <ActionIndicator actionResult={message.actionResult} />
          </div>
        )}

        {/* Sources */}
        {message.sources && message.sources.length > 0 && !isStreaming && (
          <div className="mt-2 flex flex-wrap gap-1 animate-in fade-in duration-500 delay-200">
            {message.sources.map((source, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1 rounded-full bg-background/50 px-2 py-0.5 text-[10px] font-medium"
              >
                {source.type === "pr" && "📄"}
                {source.type === "knowledge" && "📚"}
                {source.type === "rule" && "⚙️"}
                {source.title}
              </span>
            ))}
          </div>
        )}

        {/* Suggestions */}
        {message.suggestions &&
          message.suggestions.length > 0 &&
          !isStreaming && (
            <div className="mt-3 flex flex-wrap gap-1.5 animate-in fade-in slide-in-from-bottom-1 duration-500 delay-300">
              {message.suggestions.map((suggestion, i) => (
                <Button
                  key={i}
                  variant="ghost"
                  size="sm"
                  className="h-auto py-1 px-2 text-xs bg-background/50 hover:bg-background/80 rounded-full"
                  onClick={() => onSuggestionClick?.(suggestion)}
                >
                  <Lightbulb className="mr-1 h-3 w-3" />
                  {suggestion}
                </Button>
              ))}
            </div>
          )}
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary shadow-md">
          <User className="h-4 w-4 text-primary-foreground" />
        </div>
      )}
    </div>
  );
}

// ===== Types =====

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: ChatSource[];
  suggestions?: string[];
  isStreaming?: boolean;
  // GOD MODE fields
  actionResult?: ActionResult;
}

export interface ChatSource {
  type: "pr" | "knowledge" | "rule";
  title: string;
  reference?: string;
}

interface StepContext {
  step: Step;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  placeholder: string;
  suggestions: string[];
}

const stepContexts: Record<Step, StepContext> = {
  upload: {
    step: "upload",
    title: "Upload Assistant",
    description: "Help with PR file format and requirements",
    icon: HelpCircle,
    placeholder: "Ask about file requirements...",
    suggestions: [
      "What file formats are supported?",
      "What fields should the PR contain?",
      "How do I fix validation errors?",
    ],
  },
  qa: {
    step: "qa",
    title: "Q&A Assistant",
    description: "Help refine questions and find similar answers",
    icon: MessageCircle,
    placeholder: "Ask for help with questions...",
    suggestions: [
      "Why is this question asked?",
      "What did similar PRs answer?",
      "Can you rephrase this question?",
    ],
  },
  summary: {
    step: "summary",
    title: "Summary Explorer",
    description: "Understand features and compare with similar PRs",
    icon: BookOpen,
    placeholder: "Ask about PR analysis...",
    suggestions: [
      "Explain this feature",
      "Why is this complexity high?",
      "Compare with similar PRs",
    ],
  },
  estimation: {
    step: "estimation",
    title: "Estimate Explainer",
    description: "Understand cost breakdown and reasoning",
    icon: Calculator,
    placeholder: "Ask about cost estimates...",
    suggestions: [
      "Why is this activity high?",
      "What rules were applied?",
      "Show historical comparison",
    ],
  },
  review: {
    step: "review",
    title: "Review Helper",
    description: "Validate changes and finalize estimates",
    icon: FileCheck,
    placeholder: "Ask about final review...",
    suggestions: [
      "Suggest a correction reason",
      "Preview what system will learn",
      "Generate summary report",
    ],
  },
};

// ===== Main Component =====

interface AdaptiveRAGChatProps {
  sessionId: string;
  currentStep: Step;
  onSendMessage?: (message: string) => Promise<ChatMessage>;
  className?: string;
  enableStreaming?: boolean;
  pageContext?: PageContext;
}

export function AdaptiveRAGChat({
  sessionId,
  currentStep,
  onSendMessage,
  className,
  enableStreaming: initialEnableStreaming = true,
  pageContext,
}: AdaptiveRAGChatProps) {
  const _t = useTranslations("chat");
  const [input, setInput] = useState("");
  const [isExpanded, setIsExpanded] = useState(true);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [enableStreaming, setEnableStreaming] = useState(
    initialEnableStreaming,
  );

  // GOD MODE: Agent mode toggle - 'chat' for read-only, 'agent' for full capabilities
  const [chatMode, setChatMode] = useState<ChatMode>("chat");

  // GOD MODE: Get the store update function
  const applyPartialUpdate = useEstimationStore(
    (state) => state.applyPartialUpdate,
  );

  // Enhanced loading states
  const [loadingState, setLoadingState] = useState<
    "idle" | "thinking" | "generating" | "done"
  >("idle");

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const stepContext = stepContexts[currentStep];
  const StepIcon = stepContext.icon;

  const isLoading = loadingState !== "idle" && loadingState !== "done";

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loadingState]);

  // Load chat history on mount
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const history = await chatApi.getHistory(sessionId);
        if (history && history.length > 0) {
          const loadedMessages: ChatMessage[] = history.map(
            (item: ChatHistoryItem) => ({
              id: item.id,
              role: item.role as "user" | "assistant",
              content: item.content,
              timestamp: new Date(item.created_at),
            }),
          );
          setMessages(loadedMessages);
        }
      } catch (error) {
        console.error("Failed to load chat history:", error);
      }
    };

    if (sessionId) {
      loadHistory();
    }
  }, [sessionId]);

  // Status message for dynamic display
  const [statusMessage, setStatusMessage] = useState<string>("");

  // Handle streaming message with better UX - uses backend status events
  const handleStreamingMessage = useCallback(
    async (userMessage: string) => {
      // Add user message immediately
      const userMsgId = `user-${Date.now()}`;
      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        content: userMessage,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);

      // Show thinking state immediately
      setLoadingState("thinking");
      setStatusMessage("Connecting...");

      // Prepare assistant message placeholder (will be added when generating starts)
      const assistantMsgId = `assistant-${Date.now()}`;
      let assistantMsgAdded = false;

      try {
        // Build page context with current step
        const contextWithStep: PageContext = {
          ...pageContext,
          current_step: currentStep,
        };

        // Convert messages to API format
        const history = messages.map((m) => ({
          role: m.role,
          content: m.content,
        }));

        // Stream response - pass chatMode for GOD MODE support
        let fullContent = "";
        const stream = chatApi.sendMessageStream(
          sessionId,
          userMessage,
          history,
          contextWithStep,
          chatMode,
        );

        for await (const chunk of stream) {
          // Handle status events from backend
          if (chunk.type === "status") {
            if (chunk.status === "thinking") {
              setLoadingState("thinking");
              setStatusMessage(chunk.message || "Analyzing...");
            } else if (chunk.status === "generating") {
              setLoadingState("generating");
              setStatusMessage(chunk.message || "Writing response...");
            }
          } else if (chunk.type === "chunk" && chunk.content) {
            // First chunk - add assistant message bubble
            if (!assistantMsgAdded) {
              const assistantMsg: ChatMessage = {
                id: assistantMsgId,
                role: "assistant",
                content: "",
                timestamp: new Date(),
                isStreaming: true,
              };
              setMessages((prev) => [...prev, assistantMsg]);
              assistantMsgAdded = true;
              setLoadingState("generating");
            }

            fullContent += chunk.content;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, content: fullContent, isStreaming: true }
                  : m,
              ),
            );
          } else if (chunk.type === "done") {
            // GOD MODE: Apply state updates if present
            if (chunk.updated_state) {
              console.log("[GOD MODE] Applying updated state from chat");
              applyPartialUpdate(chunk.updated_state);
            }

            // Finalize message with action result
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: fullContent,
                      isStreaming: false,
                      suggestions: chunk.suggestions?.map((s) => s.text),
                      actionResult: chunk.action_result,
                    }
                  : m,
              ),
            );
            setLoadingState("done");
            setStatusMessage("");
          } else if (chunk.type === "error") {
            // Add error message if not already added
            if (!assistantMsgAdded) {
              const assistantMsg: ChatMessage = {
                id: assistantMsgId,
                role: "assistant",
                content:
                  chunk.message ||
                  "Sorry, an error occurred. Please try again.",
                timestamp: new Date(),
                isStreaming: false,
              };
              setMessages((prev) => [...prev, assistantMsg]);
            } else {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsgId
                    ? {
                        ...m,
                        content:
                          chunk.message ||
                          "Sorry, an error occurred. Please try again.",
                        isStreaming: false,
                      }
                    : m,
                ),
              );
            }
            setLoadingState("idle");
            setStatusMessage("");
          }
        }
      } catch (error) {
        console.error("Streaming error:", error);
        // Add error message
        const errorMsg: ChatMessage = {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: "Sorry, an error occurred. Please try again.",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
        setLoadingState("idle");
        setStatusMessage("");
      }

      // Reset loading state after a brief delay
      setTimeout(() => {
        setLoadingState("idle");
        setStatusMessage("");
      }, 500);
    },
    [
      sessionId,
      messages,
      pageContext,
      currentStep,
      chatMode,
      applyPartialUpdate,
    ],
  );

  // Handle regular (non-streaming) message with better UX
  const handleRegularMessage = useCallback(
    async (userMessage: string) => {
      // Add user message
      const userMsgId = `user-${Date.now()}`;
      const userMsg: ChatMessage = {
        id: userMsgId,
        role: "user",
        content: userMessage,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, userMsg]);

      // Show thinking state
      setLoadingState("thinking");

      try {
        // Build page context with current step
        const contextWithStep: PageContext = {
          ...pageContext,
          current_step: currentStep,
        };

        // Convert messages to API format
        const history = messages.map((m) => ({
          role: m.role,
          content: m.content,
        }));

        // Call API - pass chatMode for GOD MODE support
        const response = await chatApi.sendMessage(
          sessionId,
          userMessage,
          history,
          contextWithStep,
          chatMode,
        );

        // Switch to generating briefly for animation
        setLoadingState("generating");
        await new Promise((resolve) => setTimeout(resolve, 300));

        // GOD MODE: Apply state updates if present
        if (response.updated_state) {
          console.log("[GOD MODE] Applying updated state from chat");
          applyPartialUpdate(response.updated_state);
        }

        // Add assistant response with action result
        const assistantMsg: ChatMessage = {
          id: `assistant-${Date.now()}`,
          role: "assistant",
          content: response.response,
          timestamp: new Date(),
          suggestions: response.suggestions?.map((s) => s.text),
          actionResult: response.action_result,
        };

        setMessages((prev) => [...prev, assistantMsg]);
        setLoadingState("done");

        // If custom handler provided, call it
        if (onSendMessage) {
          return onSendMessage(userMessage);
        }

        return assistantMsg;
      } catch (error) {
        console.error("Message error:", error);
        const errorMsg: ChatMessage = {
          id: `error-${Date.now()}`,
          role: "assistant",
          content: "Sorry, an error occurred. Please try again.",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
        setLoadingState("idle");
        return errorMsg;
      } finally {
        // Reset loading state after a brief delay
        setTimeout(() => setLoadingState("idle"), 500);
      }
    },
    [
      sessionId,
      messages,
      onSendMessage,
      pageContext,
      currentStep,
      chatMode,
      applyPartialUpdate,
    ],
  );

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const message = input.trim();
    setInput("");

    if (enableStreaming) {
      await handleStreamingMessage(message);
    } else {
      await handleRegularMessage(message);
    }
  }, [
    input,
    isLoading,
    enableStreaming,
    handleStreamingMessage,
    handleRegularMessage,
  ]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  const handleClearHistory = async () => {
    try {
      await chatApi.clearHistory(sessionId);
      setMessages([]);
    } catch (error) {
      console.error("Failed to clear history:", error);
    }
  };

  // Collapsed state - show floating button
  if (!isExpanded) {
    return (
      <Button
        onClick={() => setIsExpanded(true)}
        className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
        size="icon"
      >
        <MessageCircle className="h-6 w-6" />
      </Button>
    );
  }

  return (
    <TooltipProvider>
      <Card className={cn("flex h-full flex-col overflow-hidden", className)}>
        {/* Header */}
        <CardHeader className="flex-none border-b py-3 bg-gradient-to-r from-background to-muted/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-primary/5 shadow-inner">
                <StepIcon className="h-4 w-4 text-primary" />
              </div>
              <div>
                <CardTitle className="text-sm font-semibold">
                  {stepContext.title}
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  {stepContext.description}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              {/* Agent mode toggle */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 mr-1 px-2 py-1 rounded-full bg-muted/50">
                    <Sparkles
                      className={cn(
                        "h-3 w-3 transition-colors",
                        chatMode === "agent"
                          ? "text-purple-500"
                          : "text-muted-foreground",
                      )}
                    />
                    <Switch
                      id="agent-mode"
                      checked={chatMode === "agent"}
                      onCheckedChange={(checked) =>
                        setChatMode(checked ? "agent" : "chat")
                      }
                      className="scale-75"
                    />
                  </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-[200px]">
                  <p className="font-medium">
                    {chatMode === "agent" ? "Agent Mode" : "Chat Mode"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">
                    {chatMode === "agent"
                      ? "Full capabilities: regenerate questions, modify estimates, update state"
                      : "Read-only assistant: explanations and suggestions only"}
                  </p>
                </TooltipContent>
              </Tooltip>

              {/* Streaming toggle */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="flex items-center gap-1.5 mr-2 px-2 py-1 rounded-full bg-muted/50">
                    <Zap
                      className={cn(
                        "h-3 w-3 transition-colors",
                        enableStreaming
                          ? "text-yellow-500"
                          : "text-muted-foreground",
                      )}
                    />
                    <Switch
                      id="streaming"
                      checked={enableStreaming}
                      onCheckedChange={setEnableStreaming}
                      className="scale-75"
                    />
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    {enableStreaming ? "Real-time streaming" : "Standard mode"}
                  </p>
                </TooltipContent>
              </Tooltip>

              {/* Clear history button */}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 hover:bg-destructive/10 hover:text-destructive"
                    onClick={handleClearHistory}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Clear chat history</p>
                </TooltipContent>
              </Tooltip>

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setIsExpanded(false)}
              >
                <Minimize2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        {/* Messages */}
        <CardContent className="flex-1 overflow-y-auto p-4 bg-gradient-to-b from-transparent to-muted/10">
          <div className="space-y-4">
            {/* Welcome message */}
            {messages.length === 0 && loadingState === "idle" && (
              <div className="flex flex-col items-center justify-center py-8 text-center animate-in fade-in zoom-in-95 duration-500">
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-primary/20 to-primary/5 shadow-lg">
                  <Sparkles className="h-7 w-7 text-primary" />
                </div>
                <p className="font-semibold text-lg">How can I help?</p>
                <p className="mt-1 text-sm text-muted-foreground max-w-[280px]">
                  I&apos;m here to assist you during{" "}
                  <span className="text-primary font-medium">
                    {stepContext.title.toLowerCase()}
                  </span>
                </p>

                {/* Suggestions */}
                <div className="mt-6 flex flex-wrap justify-center gap-2">
                  {stepContext.suggestions.map((suggestion, i) => (
                    <Button
                      key={i}
                      variant="outline"
                      size="sm"
                      className="h-auto py-2 px-3 text-xs rounded-full hover:bg-primary/5 hover:border-primary/30 transition-all duration-200 animate-in fade-in slide-in-from-bottom-2"
                      style={{ animationDelay: `${i * 100}ms` }}
                      onClick={() => handleSuggestionClick(suggestion)}
                    >
                      <Lightbulb className="mr-1.5 h-3 w-3 text-yellow-500" />
                      {suggestion}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages list */}
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onSuggestionClick={handleSuggestionClick}
              />
            ))}

            {/* Thinking/Generating indicator */}
            {(loadingState === "thinking" || loadingState === "generating") &&
              !messages.some((m) => m.isStreaming) && (
                <ThinkingIndicator
                  status={loadingState as "thinking" | "generating"}
                  statusMessage={statusMessage}
                />
              )}

            <div ref={messagesEndRef} />
          </div>
        </CardContent>

        {/* Input */}
        <div className="flex-none border-t p-4 bg-gradient-to-t from-muted/20 to-transparent">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  isLoading
                    ? loadingState === "thinking"
                      ? "Thinking..."
                      : "Generating..."
                    : stepContext.placeholder
                }
                disabled={isLoading}
                className="pr-10 rounded-full border-muted-foreground/20 focus:border-primary/50 transition-colors"
              />
              {enableStreaming && !isLoading && (
                <Zap className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-yellow-500/50" />
              )}
            </div>
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              size="icon"
              className={cn(
                "rounded-full transition-all duration-300",
                isLoading && "animate-pulse",
              )}
            >
              <Send
                className={cn(
                  "h-4 w-4 transition-transform",
                  !isLoading && input.trim() && "translate-x-0.5",
                )}
              />
            </Button>
          </div>

          {/* Status indicator */}
          <div className="h-5 mt-1.5 flex items-center justify-center">
            {loadingState === "idle" && chatMode === "agent" && (
              <p className="text-[10px] text-purple-600/70 dark:text-purple-400/70 flex items-center gap-1 animate-in fade-in duration-300">
                <Sparkles className="h-3 w-3 text-purple-500/70" />
                Agent Mode — I can modify questions & estimates
              </p>
            )}
            {loadingState === "idle" &&
              chatMode === "chat" &&
              enableStreaming && (
                <p className="text-[10px] text-muted-foreground/60 flex items-center gap-1 animate-in fade-in duration-300">
                  <Zap className="h-3 w-3 text-yellow-500/70" />
                  Real-time streaming enabled
                </p>
              )}
            {loadingState === "thinking" && (
              <p className="text-[10px] text-primary/70 flex items-center gap-1 animate-in fade-in duration-300">
                <Brain className="h-3 w-3 animate-pulse" />
                {statusMessage || "Analyzing your question..."}
              </p>
            )}
            {loadingState === "generating" && (
              <p className="text-[10px] text-primary/70 flex items-center gap-1 animate-in fade-in duration-300">
                <Wand2 className="h-3 w-3 animate-pulse" />
                {statusMessage || "Writing response..."}
              </p>
            )}
          </div>
        </div>
      </Card>
    </TooltipProvider>
  );
}

// ===== Floating Chat Button =====

export function FloatingChatButton({
  currentStep,
  onClick,
  hasUnread = false,
}: {
  currentStep: Step;
  onClick: () => void;
  hasUnread?: boolean;
}) {
  const stepContext = stepContexts[currentStep];
  const StepIcon = stepContext.icon;

  return (
    <Button
      onClick={onClick}
      className="fixed bottom-6 right-6 h-14 gap-2 rounded-full px-4 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-105"
    >
      <StepIcon className="h-5 w-5" />
      <span className="hidden sm:inline">{stepContext.title}</span>
      {hasUnread && (
        <span className="absolute -right-1 -top-1 flex h-3 w-3">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75"></span>
          <span className="relative inline-flex h-3 w-3 rounded-full bg-red-500"></span>
        </span>
      )}
    </Button>
  );
}

// ===== Compact Chat Panel =====

export function ChatPanelCompact({
  sessionId,
  currentStep,
  onExpand,
  className,
  pageContext,
}: {
  sessionId: string;
  currentStep: Step;
  onExpand: () => void;
  className?: string;
  pageContext?: PageContext;
}) {
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const stepContext = stepContexts[currentStep];

  const handleSend = useCallback(async () => {
    if (!input.trim() || isLoading) return;

    const message = input.trim();
    setInput("");
    setIsLoading(true);

    try {
      const contextWithStep: PageContext = {
        ...pageContext,
        current_step: currentStep,
      };
      const response = await chatApi.sendMessage(
        sessionId,
        message,
        [],
        contextWithStep,
      );
      setLastMessage(response.response);
    } catch (error) {
      console.error("Chat error:", error);
      setLastMessage("Error: Could not send message");
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading, sessionId, pageContext, currentStep]);

  return (
    <Card className={cn("flex flex-col", className)}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">{stepContext.title}</CardTitle>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={onExpand}
          >
            <Maximize2 className="h-3 w-3" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="pb-3">
        {lastMessage && (
          <p className="mb-3 line-clamp-3 text-xs text-muted-foreground">
            {lastMessage}
          </p>
        )}
        <div className="flex gap-1">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder={isLoading ? "Thinking..." : "Ask..."}
            className="h-8 text-xs rounded-full"
            disabled={isLoading}
          />
          <Button
            size="icon"
            className="h-8 w-8 rounded-full"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
          >
            {isLoading ? (
              <Brain className="h-3 w-3 animate-pulse" />
            ) : (
              <Send className="h-3 w-3" />
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

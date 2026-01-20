"use client";

import { Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export type StepLoadingVariant = "qa" | "summary" | "estimation" | "review";

interface LoadingStep {
  label: string;
  status: "completed" | "active" | "pending";
}

interface StepLoadingProps {
  variant: StepLoadingVariant;
  className?: string;
}

const stepConfigs: Record<
  StepLoadingVariant,
  {
    title: string;
    description: string;
    icon: React.ReactNode;
    steps: LoadingStep[];
    estimatedTime: string;
  }
> = {
  qa: {
    title: "Analyzing Your PR Document",
    description:
      "Our AI is generating smart clarifying questions based on your Product Request and similar historical projects...",
    icon: (
      <svg
        className="h-10 w-10 animate-pulse text-primary"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
        />
      </svg>
    ),
    steps: [
      { label: "PR document parsed", status: "completed" },
      { label: "Finding similar historical PRs...", status: "active" },
      { label: "Generating questions", status: "pending" },
    ],
    estimatedTime: "5-10 seconds",
  },
  summary: {
    title: "Creating Executive Summary",
    description:
      "Our AI is analyzing the PR scope, extracting key features, and preparing a comprehensive summary for management review...",
    icon: (
      <svg
        className="h-10 w-10 animate-pulse text-primary"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
    steps: [
      { label: "Extracting project features", status: "completed" },
      { label: "Analyzing technical scope...", status: "active" },
      { label: "Finding similar projects", status: "pending" },
      { label: "Generating executive summary", status: "pending" },
    ],
    estimatedTime: "8-15 seconds",
  },
  estimation: {
    title: "Predicting Engineering Effort",
    description:
      "Our ML models are analyzing historical patterns, applying learned rules, and generating detailed cost breakdown predictions...",
    icon: (
      <svg
        className="h-10 w-10 animate-pulse text-primary"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
        />
      </svg>
    ),
    steps: [
      { label: "Loading ML prediction models", status: "completed" },
      { label: "Analyzing Q&A context...", status: "active" },
      { label: "Retrieving similar project costs", status: "pending" },
      { label: "Applying learned rules", status: "pending" },
      { label: "Generating activity breakdown", status: "pending" },
    ],
    estimatedTime: "10-20 seconds",
  },
  review: {
    title: "Preparing Review Dashboard",
    description:
      "Setting up the interactive review interface for finalizing your cost estimation...",
    icon: (
      <svg
        className="h-10 w-10 animate-pulse text-primary"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4"
        />
      </svg>
    ),
    steps: [
      { label: "Loading estimation data", status: "completed" },
      { label: "Preparing edit interface...", status: "active" },
      { label: "Loading export templates", status: "pending" },
    ],
    estimatedTime: "3-5 seconds",
  },
};

export function StepLoading({ variant, className }: StepLoadingProps) {
  const config = stepConfigs[variant];

  return (
    <Card className={cn("overflow-hidden", className)}>
      <CardContent className="py-16">
        <div className="flex flex-col items-center justify-center space-y-6">
          {/* Animated icon with ping effect */}
          <div className="relative">
            <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
            <div className="relative flex h-20 w-20 items-center justify-center rounded-full bg-primary/10">
              {config.icon}
            </div>
          </div>

          {/* Title and description */}
          <div className="text-center space-y-2">
            <h3 className="text-xl font-semibold">{config.title}</h3>
            <p className="text-muted-foreground max-w-md">
              {config.description}
            </p>
          </div>

          {/* Progress indicator */}
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>This usually takes {config.estimatedTime}</span>
          </div>

          {/* Animated progress steps */}
          <div className="w-full max-w-sm space-y-3 pt-4">
            {config.steps.map((step, index) => (
              <div key={index} className="flex items-center gap-3">
                <div
                  className={cn(
                    "h-2 w-2 rounded-full transition-all duration-300",
                    step.status === "completed" && "bg-green-500",
                    step.status === "active" && "bg-primary animate-pulse",
                    step.status === "pending" && "bg-muted"
                  )}
                />
                <span
                  className={cn(
                    "text-sm transition-colors",
                    step.status === "completed" && "text-muted-foreground",
                    step.status === "active" && "text-foreground font-medium",
                    step.status === "pending" && "text-muted-foreground"
                  )}
                >
                  {step.label}
                </span>
              </div>
            ))}
          </div>

          {/* Fun facts / tips while waiting */}
          <div className="mt-6 pt-6 border-t w-full max-w-md">
            <div className="text-center">
              <p className="text-xs text-muted-foreground mb-2">Did you know?</p>
              <p className="text-sm text-muted-foreground italic">
                {variant === "qa" &&
                  "FPT Cost Brain uses historical data from 37+ similar PRs to generate targeted questions."}
                {variant === "summary" &&
                  "Our ML models analyze over 3,100 features to understand your project complexity."}
                {variant === "estimation" &&
                  "The ensemble prediction combines 5 ML models achieving 78% accuracy within 30% margin."}
                {variant === "review" &&
                  "Every correction you make helps improve future predictions through online learning."}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Alternative: Compact inline loader for smaller spaces
export function StepLoadingCompact({
  variant,
  className,
}: StepLoadingProps) {
  const config = stepConfigs[variant];

  return (
    <div
      className={cn(
        "flex items-center gap-4 p-4 rounded-lg bg-muted/50",
        className
      )}
    >
      <div className="relative flex-shrink-0">
        <div className="absolute inset-0 animate-ping rounded-full bg-primary/20" />
        <div className="relative flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
          <Loader2 className="h-6 w-6 animate-spin text-primary" />
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-medium truncate">{config.title}</p>
        <p className="text-sm text-muted-foreground truncate">
          {config.estimatedTime}
        </p>
      </div>
    </div>
  );
}

// Skeleton loader for content areas
export function StepLoadingSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn("space-y-4 animate-pulse", className)}>
      <div className="h-8 bg-muted rounded w-1/3" />
      <div className="space-y-2">
        <div className="h-4 bg-muted rounded w-full" />
        <div className="h-4 bg-muted rounded w-5/6" />
        <div className="h-4 bg-muted rounded w-4/6" />
      </div>
      <div className="grid grid-cols-3 gap-4 pt-4">
        <div className="h-24 bg-muted rounded" />
        <div className="h-24 bg-muted rounded" />
        <div className="h-24 bg-muted rounded" />
      </div>
    </div>
  );
}

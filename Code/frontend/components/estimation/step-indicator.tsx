"use client";

import { cn } from "@/lib/utils";
import {
  Check,
  Upload,
  HelpCircle,
  FileText,
  Calculator,
  Download,
} from "lucide-react";

export type Step = "upload" | "qa" | "summary" | "estimation" | "review";

interface StepConfig {
  id: Step;
  number: number;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const steps: StepConfig[] = [
  { id: "upload", number: 1, label: "Upload PR", icon: Upload },
  { id: "qa", number: 2, label: "Q&A", icon: HelpCircle },
  { id: "summary", number: 3, label: "Summary", icon: FileText },
  { id: "estimation", number: 4, label: "Estimation", icon: Calculator },
  { id: "review", number: 5, label: "Review & Export", icon: Download },
];

interface StepIndicatorProps {
  currentStep: Step;
  completedSteps?: Step[];
  onStepClick?: (step: Step) => void;
  className?: string;
}

export function StepIndicator({
  currentStep,
  completedSteps = [],
  onStepClick,
  className,
}: StepIndicatorProps) {
  const currentIndex = steps.findIndex((s) => s.id === currentStep);

  return (
    <nav aria-label="Progress" className={cn("w-full", className)}>
      <ol className="flex items-center justify-between">
        {steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = step.id === currentStep;
          const isPast = index < currentIndex;
          const isClickable = onStepClick && (isCompleted || isPast);
          const Icon = step.icon;

          return (
            <li key={step.id} className="relative flex-1">
              {/* Connector line */}
              {index !== 0 && (
                <div
                  className={cn(
                    "absolute left-0 top-5 -ml-px h-0.5 w-full -translate-x-1/2",
                    isPast || isCompleted
                      ? "bg-primary"
                      : "bg-muted-foreground/20",
                  )}
                  aria-hidden="true"
                />
              )}

              <button
                type="button"
                onClick={() => isClickable && onStepClick?.(step.id)}
                disabled={!isClickable}
                className={cn(
                  "group relative flex flex-col items-center",
                  isClickable && "cursor-pointer",
                  !isClickable && "cursor-default",
                )}
              >
                {/* Step circle */}
                <span
                  className={cn(
                    "relative z-10 flex h-10 w-10 items-center justify-center rounded-full border-2 transition-colors",
                    isCompleted &&
                      "border-primary bg-primary text-primary-foreground",
                    isCurrent &&
                      !isCompleted &&
                      "border-primary bg-background text-primary",
                    !isCurrent &&
                      !isCompleted &&
                      "border-muted-foreground/30 bg-background text-muted-foreground",
                    isClickable && "group-hover:border-primary/80",
                  )}
                >
                  {isCompleted ? (
                    <Check className="h-5 w-5" />
                  ) : (
                    <Icon className="h-5 w-5" />
                  )}
                </span>

                {/* Step label */}
                <span
                  className={cn(
                    "mt-2 text-xs font-medium transition-colors",
                    isCurrent && "text-primary",
                    isCompleted && "text-primary",
                    !isCurrent && !isCompleted && "text-muted-foreground",
                  )}
                >
                  {step.label}
                </span>

                {/* Step number badge */}
                <span
                  className={cn(
                    "absolute -top-1 -right-1 z-20 flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold",
                    isCurrent && "bg-primary text-primary-foreground",
                    isCompleted && "bg-green-500 text-white",
                    !isCurrent &&
                      !isCompleted &&
                      "bg-muted text-muted-foreground",
                  )}
                >
                  {step.number}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

// Compact horizontal version for mobile
export function StepIndicatorCompact({
  currentStep,
  completedSteps = [],
  className,
}: Omit<StepIndicatorProps, "onStepClick">) {
  const currentIndex = steps.findIndex((s) => s.id === currentStep);
  const currentStepConfig = steps[currentIndex];

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <div className="flex items-center gap-1">
        {steps.map((step, index) => {
          const isCompleted = completedSteps.includes(step.id);
          const isCurrent = step.id === currentStep;
          const isPast = index < currentIndex;

          return (
            <div
              key={step.id}
              className={cn(
                "h-2 w-8 rounded-full transition-colors",
                isCompleted || isPast
                  ? "bg-primary"
                  : isCurrent
                    ? "bg-primary/50"
                    : "bg-muted",
              )}
            />
          );
        })}
      </div>
      <span className="text-sm font-medium text-muted-foreground">
        Step {currentStepConfig.number}: {currentStepConfig.label}
      </span>
    </div>
  );
}

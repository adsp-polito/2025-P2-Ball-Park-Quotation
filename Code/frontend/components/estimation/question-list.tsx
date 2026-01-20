"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import {
  HelpCircle,
  CheckCircle,
  MessageSquare,
  Lightbulb,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export interface Question {
  id: string;
  question: string;
  answer?: string;
  category: string;
  importance: "high" | "medium" | "low";
  suggestedAnswers?: string[];
  context?: string;
  relatedPRs?: Array<{
    prCode: string;
    answer: string;
  }>;
}

interface QuestionListProps {
  questions: Question[];
  onAnswerChange: (questionId: string, answer: string) => void;
  onSubmitAll: () => void;
  isSubmitting?: boolean;
  className?: string;
}

export function QuestionList({
  questions,
  onAnswerChange,
  onSubmitAll,
  isSubmitting = false,
  className,
}: QuestionListProps) {
  const _t = useTranslations("estimation");
  const [expandedId, setExpandedId] = useState<string | null>(
    questions[0]?.id || null,
  );

  const answeredCount = questions.filter((q) => q.answer?.trim()).length;
  const requiredCount = questions.filter((q) => q.importance === "high").length;
  const answeredRequired = questions.filter(
    (q) => q.importance === "high" && q.answer?.trim(),
  ).length;

  const isComplete = answeredRequired >= requiredCount;

  const importanceColors = {
    high: "text-red-500 dark:text-red-400",
    medium: "text-yellow-500 dark:text-yellow-400",
    low: "text-green-500 dark:text-green-400",
  };

  const importanceBg = {
    high: "bg-red-100 dark:bg-red-900/30",
    medium: "bg-yellow-100 dark:bg-yellow-900/30",
    low: "bg-green-100 dark:bg-green-900/30",
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Progress header */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground">
            {answeredCount} of {questions.length} questions answered
          </p>
          <div className="mt-1 h-2 w-48 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all"
              style={{ width: `${(answeredCount / questions.length) * 100}%` }}
            />
          </div>
        </div>
        <div className="text-right">
          <p className="text-sm font-medium">
            Required: {answeredRequired}/{requiredCount}
          </p>
          <p className="text-xs text-muted-foreground">
            {isComplete ? (
              <span className="text-green-600 dark:text-green-400">
                ✓ Ready to proceed
              </span>
            ) : (
              <span className="text-yellow-600 dark:text-yellow-400">
                Answer required questions
              </span>
            )}
          </p>
        </div>
      </div>

      {/* Questions */}
      <div className="space-y-3">
        {questions.map((question, index) => (
          <Card
            key={question.id}
            className={cn(
              "transition-shadow",
              expandedId === question.id && "ring-2 ring-primary/20",
            )}
          >
            <CardHeader
              className="cursor-pointer pb-2"
              onClick={() =>
                setExpandedId(expandedId === question.id ? null : question.id)
              }
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-start gap-3">
                  <div
                    className={cn(
                      "flex h-8 w-8 items-center justify-center rounded-full",
                      question.answer?.trim()
                        ? "bg-green-100 dark:bg-green-900/30"
                        : importanceBg[question.importance],
                    )}
                  >
                    {question.answer?.trim() ? (
                      <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" />
                    ) : (
                      <span
                        className={cn(
                          "text-sm font-medium",
                          importanceColors[question.importance],
                        )}
                      >
                        {index + 1}
                      </span>
                    )}
                  </div>
                  <div className="flex-1">
                    <CardTitle className="text-base font-medium">
                      {question.question}
                    </CardTitle>
                    <div className="mt-1 flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {question.category}
                      </span>
                      <span
                        className={cn(
                          "text-xs font-medium",
                          importanceColors[question.importance],
                        )}
                      >
                        {question.importance === "high" && "Required"}
                        {question.importance === "medium" && "Recommended"}
                        {question.importance === "low" && "Optional"}
                      </span>
                    </div>
                  </div>
                </div>
                {expandedId === question.id ? (
                  <ChevronUp className="h-5 w-5 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-5 w-5 text-muted-foreground" />
                )}
              </div>
            </CardHeader>

            {expandedId === question.id && (
              <CardContent className="pt-0">
                {/* Context */}
                {question.context && (
                  <div className="mb-4 flex items-start gap-2 rounded-lg bg-muted/50 p-3 text-sm">
                    <HelpCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-muted-foreground" />
                    <p className="text-muted-foreground">{question.context}</p>
                  </div>
                )}

                {/* Answer input */}
                <div className="space-y-3">
                  <Input
                    value={question.answer || ""}
                    onChange={(e) =>
                      onAnswerChange(question.id, e.target.value)
                    }
                    placeholder="Type your answer..."
                    className="w-full"
                  />

                  {/* Suggested answers */}
                  {question.suggestedAnswers &&
                    question.suggestedAnswers.length > 0 && (
                      <div className="space-y-2">
                        <p className="flex items-center gap-1 text-xs text-muted-foreground">
                          <Lightbulb className="h-3 w-3" />
                          Suggestions:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {question.suggestedAnswers.map((suggestion, i) => (
                            <Button
                              key={i}
                              variant="outline"
                              size="sm"
                              className="h-auto py-1 text-xs"
                              onClick={() =>
                                onAnswerChange(question.id, suggestion)
                              }
                            >
                              {suggestion}
                            </Button>
                          ))}
                        </div>
                      </div>
                    )}

                  {/* Related PR answers */}
                  {question.relatedPRs && question.relatedPRs.length > 0 && (
                    <div className="mt-3 space-y-2 border-t pt-3">
                      <p className="flex items-center gap-1 text-xs text-muted-foreground">
                        <MessageSquare className="h-3 w-3" />
                        From similar PRs:
                      </p>
                      <div className="space-y-2">
                        {question.relatedPRs.slice(0, 3).map((related, i) => (
                          <div
                            key={i}
                            className="cursor-pointer rounded-md bg-muted/30 p-2 text-sm hover:bg-muted/50"
                            onClick={() =>
                              onAnswerChange(question.id, related.answer)
                            }
                          >
                            <span className="font-medium text-primary">
                              {related.prCode}:
                            </span>{" "}
                            <span className="text-muted-foreground">
                              {related.answer}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            )}
          </Card>
        ))}
      </div>

      {/* Submit button */}
      <div className="flex justify-end pt-4">
        <Button
          onClick={onSubmitAll}
          disabled={!isComplete || isSubmitting}
          size="lg"
          className="gap-2"
        >
          {isSubmitting ? (
            <>
              <span className="animate-spin">⏳</span>
              Processing...
            </>
          ) : (
            <>
              <CheckCircle className="h-5 w-5" />
              Continue to Summary
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

// Compact Q&A panel for sidebar
export function QuestionListCompact({
  questions,
  className,
}: {
  questions: Question[];
  className?: string;
}) {
  const answeredCount = questions.filter((q) => q.answer?.trim()).length;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">Q&A Progress</span>
        <span className="font-medium">
          {answeredCount}/{questions.length}
        </span>
      </div>
      <div className="space-y-1">
        {questions.map((q, i) => (
          <div key={q.id} className="flex items-center gap-2">
            <div
              className={cn(
                "h-2 w-2 rounded-full",
                q.answer?.trim() ? "bg-green-500" : "bg-muted",
              )}
            />
            <span className="truncate text-xs text-muted-foreground">
              Q{i + 1}: {q.question.slice(0, 40)}...
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

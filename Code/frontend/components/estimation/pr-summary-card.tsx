"use client";

import { useTranslations } from "next-intl";
import {
  FileText,
  Tag,
  Calendar,
  Building,
  Plane,
  Target,
  TrendingUp,
  Info,
  ChevronRight,
  ExternalLink,
  Sparkles,
  AlertTriangle,
  Link2,
  ClipboardList,
  BookOpen,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export interface PRSummary {
  prCode: string;
  title: string;
  description?: string;
  // LLM-generated narrative summary for customer manager
  summaryText?: string;
  keyFeatures?: string[];
  dependencies?: string[];
  riskFactors?: string[];
  specialRequirements?: string[];
  customer: string;
  program: string;
  programFamily?: string;
  createdDate: string;
  targetDate?: string;
  complexity: "low" | "medium" | "high";
  programSize: "small" | "medium" | "large" | "extra_large";
  activityCount: number;
  features: MLFeature[];
  similarPRs?: SimilarPR[];
}

export interface MLFeature {
  name: string;
  value: string | number;
  unit?: string;
  confidence?: number;
}

export interface SimilarPR {
  prCode: string;
  title: string;
  similarity: number;
  totalHours: number;
  accuracy?: number;
}

interface PRSummaryCardProps {
  summary: PRSummary;
  onViewSimilarPR?: (prCode: string) => void;
  className?: string;
}

export function PRSummaryCard({
  summary,
  onViewSimilarPR,
  className,
}: PRSummaryCardProps) {
  const _t = useTranslations("estimation");

  const complexityColors = {
    low: "text-green-600 bg-green-100 dark:text-green-400 dark:bg-green-900/30",
    medium:
      "text-yellow-600 bg-yellow-100 dark:text-yellow-400 dark:bg-yellow-900/30",
    high: "text-red-600 bg-red-100 dark:text-red-400 dark:bg-red-900/30",
  };

  const programSizeLabels = {
    small: "Small (<1000 hrs)",
    medium: "Medium (1000-5000 hrs)",
    large: "Large (5000-15000 hrs)",
    extra_large: "Extra Large (>15000 hrs)",
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Main PR Info */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-xl">
                <FileText className="h-5 w-5 text-primary" />
                {summary.prCode}
              </CardTitle>
              <CardDescription className="mt-1 text-base">
                {summary.title}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <span
                className={cn(
                  "rounded-full px-3 py-1 text-xs font-medium",
                  complexityColors[summary.complexity],
                )}
              >
                {summary.complexity.toUpperCase()}
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {summary.description && (
            <p className="mb-4 text-sm text-muted-foreground">
              {summary.description}
            </p>
          )}

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="flex items-center gap-2">
              <Building className="h-4 w-4 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Customer</p>
                <p className="font-medium">{summary.customer}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Plane className="h-4 w-4 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Program</p>
                <p className="font-medium">{summary.program}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Created</p>
                <p className="font-medium">{summary.createdDate}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Target className="h-4 w-4 text-muted-foreground" />
              <div>
                <p className="text-xs text-muted-foreground">Target</p>
                <p className="font-medium">{summary.targetDate || "TBD"}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Executive Summary - LLM Generated Narrative */}
      {summary.summaryText && (
        <Card className="border-primary/20 bg-gradient-to-br from-primary/5 to-transparent">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-base">
                <Sparkles className="h-4 w-4 text-primary" />
                Executive Summary
              </CardTitle>
              <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground">
                AI Generated
              </span>
            </div>
            <CardDescription>
              Comprehensive overview for customer manager review
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Main narrative summary */}
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <p className="text-sm leading-relaxed whitespace-pre-wrap">
                {summary.summaryText}
              </p>
            </div>

            {/* Key Features */}
            {summary.keyFeatures && summary.keyFeatures.length > 0 && (
              <div className="pt-2 border-t">
                <h4 className="flex items-center gap-2 text-sm font-medium mb-2">
                  <ClipboardList className="h-4 w-4 text-blue-500" />
                  Key Features
                </h4>
                <ul className="space-y-1">
                  {summary.keyFeatures.map((feature, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-primary mt-1">•</span>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Dependencies */}
            {summary.dependencies && summary.dependencies.length > 0 && (
              <div className="pt-2 border-t">
                <h4 className="flex items-center gap-2 text-sm font-medium mb-2">
                  <Link2 className="h-4 w-4 text-purple-500" />
                  Dependencies
                </h4>
                <ul className="space-y-1">
                  {summary.dependencies.map((dep, i) => (
                    <li
                      key={i}
                      className="text-sm text-muted-foreground flex items-start gap-2"
                    >
                      <span className="text-purple-500 mt-1">→</span>
                      {dep}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Risk Factors */}
            {summary.riskFactors && summary.riskFactors.length > 0 && (
              <div className="pt-2 border-t">
                <h4 className="flex items-center gap-2 text-sm font-medium mb-2">
                  <AlertTriangle className="h-4 w-4 text-yellow-500" />
                  Risk Factors
                </h4>
                <ul className="space-y-1">
                  {summary.riskFactors.map((risk, i) => (
                    <li
                      key={i}
                      className="text-sm text-yellow-700 dark:text-yellow-400 flex items-start gap-2"
                    >
                      <span className="mt-1">⚠</span>
                      {risk}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Special Requirements */}
            {summary.specialRequirements &&
              summary.specialRequirements.length > 0 && (
                <div className="pt-2 border-t">
                  <h4 className="flex items-center gap-2 text-sm font-medium mb-2">
                    <BookOpen className="h-4 w-4 text-green-500" />
                    Special Requirements
                  </h4>
                  <ul className="space-y-1">
                    {summary.specialRequirements.map((req, i) => (
                      <li
                        key={i}
                        className="text-sm text-muted-foreground flex items-start gap-2"
                      >
                        <span className="text-green-500 mt-1">✓</span>
                        {req}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

            {/* Hint to use Agent mode */}
            <div className="pt-3 border-t">
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <Sparkles className="h-3 w-3" />
                Enable Agent mode in chat to edit or regenerate this summary
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Program Size & Features */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Program Size Classification */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-primary" />
              Program Classification
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Size</span>
                <span className="font-medium">
                  {programSizeLabels[summary.programSize]}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Activity Count
                </span>
                <span className="font-medium">{summary.activityCount}</span>
              </div>
              {summary.programFamily && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Family</span>
                  <span className="font-medium">{summary.programFamily}</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ML Features */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Tag className="h-4 w-4 text-primary" />
              Extracted Features
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {summary.features.slice(0, 5).map((feature, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between text-sm"
                >
                  <span className="text-muted-foreground">{feature.name}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">
                      {feature.value}
                      {feature.unit && ` ${feature.unit}`}
                    </span>
                    {feature.confidence && (
                      <span className="text-xs text-muted-foreground">
                        ({Math.round(feature.confidence * 100)}%)
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
            {summary.features.length > 5 && (
              <Button variant="ghost" size="sm" className="mt-2 w-full text-xs">
                View all {summary.features.length} features
                <ChevronRight className="ml-1 h-3 w-3" />
              </Button>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Similar PRs */}
      {summary.similarPRs && summary.similarPRs.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Info className="h-4 w-4 text-primary" />
              Similar Product Requests
            </CardTitle>
            <CardDescription>
              Historical PRs with similar characteristics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {summary.similarPRs.map((pr, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between rounded-lg border p-3"
                >
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-primary">
                        {pr.prCode}
                      </span>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-xs",
                          pr.similarity >= 0.8
                            ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                            : pr.similarity >= 0.6
                              ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400"
                              : "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400",
                        )}
                      >
                        {Math.round(pr.similarity * 100)}% match
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {pr.title}
                    </p>
                  </div>
                  <div className="ml-4 text-right">
                    <p className="text-lg font-bold">
                      {pr.totalHours.toLocaleString()}
                    </p>
                    <p className="text-xs text-muted-foreground">hours</p>
                    {pr.accuracy && (
                      <p className="mt-1 text-xs text-green-600 dark:text-green-400">
                        {pr.accuracy}% accuracy
                      </p>
                    )}
                  </div>
                  {onViewSimilarPR && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="ml-2"
                      onClick={() => onViewSimilarPR(pr.prCode)}
                    >
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

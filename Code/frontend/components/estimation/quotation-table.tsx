"use client";

import { useMemo, Fragment } from "react";
import { useTranslations } from "next-intl";
import {
  Info,
  TrendingUp,
  TrendingDown,
  Minus,
  HelpCircle,
  ChevronDown,
  ChevronRight,
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

export interface BreakdownItem {
  id: string;
  activityCode: string;
  activityName: string;
  category: string;
  hours: number;
  confidence: number;
  reasoning?: string;
  comparisonToSimilar?: number; // percentage difference
  children?: BreakdownItem[];
}

export interface QuotationData {
  totalHours: number;
  totalCost?: number;
  currency?: string;
  confidence: number;
  breakdown: BreakdownItem[];
  rulesApplied?: string[];
}

interface QuotationTableProps {
  data: QuotationData;
  showComparison?: boolean;
  expandedCategories?: string[];
  onCategoryToggle?: (category: string) => void;
  onItemClick?: (item: BreakdownItem) => void;
  className?: string;
}

export function QuotationTable({
  data,
  showComparison = false,
  expandedCategories = [],
  onCategoryToggle,
  onItemClick,
  className,
}: QuotationTableProps) {
  const _t = useTranslations("estimation");

  // Group items by category
  const groupedItems = useMemo(() => {
    const groups: Record<string, BreakdownItem[]> = {};
    data.breakdown.forEach((item) => {
      if (!groups[item.category]) {
        groups[item.category] = [];
      }
      groups[item.category].push(item);
    });
    return groups;
  }, [data.breakdown]);

  const categoryTotals = useMemo(() => {
    const totals: Record<string, number> = {};
    Object.entries(groupedItems).forEach(([category, items]) => {
      totals[category] = items.reduce((sum, item) => sum + item.hours, 0);
    });
    return totals;
  }, [groupedItems]);

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat("en-US").format(num);
  };

  const formatCurrency = (num: number, currency = "EUR") => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(num);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600 dark:text-green-400";
    if (confidence >= 0.6) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Medium";
    return "Low";
  };

  const getComparisonIcon = (diff: number) => {
    if (diff > 5) return <TrendingUp className="h-4 w-4 text-red-500" />;
    if (diff < -5) return <TrendingDown className="h-4 w-4 text-green-500" />;
    return <Minus className="h-4 w-4 text-muted-foreground" />;
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">Total Hours</p>
              <p className="mt-1 text-3xl font-bold">
                {formatNumber(data.totalHours)}
              </p>
            </div>
          </CardContent>
        </Card>

        {data.totalCost && (
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-sm text-muted-foreground">Total Cost</p>
                <p className="mt-1 text-3xl font-bold">
                  {formatCurrency(data.totalCost, data.currency)}
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardContent className="pt-6">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">
                Overall Confidence
              </p>
              <p
                className={cn(
                  "mt-1 text-3xl font-bold",
                  getConfidenceColor(data.confidence),
                )}
              >
                {Math.round(data.confidence * 100)}%
              </p>
              <p className="text-xs text-muted-foreground">
                {getConfidenceLabel(data.confidence)}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Rules Applied */}
      {data.rulesApplied && data.rulesApplied.length > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-2 text-sm">
              <Info className="h-4 w-4 text-primary" />
              <span className="font-medium">Rules applied:</span>
              <span className="text-muted-foreground">
                {data.rulesApplied.join(", ")}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Breakdown Table */}
      <Card>
        <CardHeader>
          <CardTitle>Cost Breakdown</CardTitle>
          <CardDescription>
            Detailed breakdown by activity category
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left">
                  <th className="pb-3 font-medium">Activity</th>
                  <th className="pb-3 text-right font-medium">Hours</th>
                  <th className="pb-3 text-right font-medium">Confidence</th>
                  {showComparison && (
                    <th className="pb-3 text-right font-medium">vs Similar</th>
                  )}
                  <th className="pb-3 text-right font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(groupedItems).map(([category, items]) => {
                  const isExpanded = expandedCategories.includes(category);
                  return (
                    <Fragment key={`category-${category}`}>
                      {/* Category Row */}
                      <tr
                        className="cursor-pointer bg-muted/50 transition-colors hover:bg-muted"
                        onClick={() => onCategoryToggle?.(category)}
                      >
                        <td className="py-3">
                          <div className="flex items-center gap-2 font-medium">
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                            {category}
                            <span className="text-xs text-muted-foreground">
                              ({items.length} items)
                            </span>
                          </div>
                        </td>
                        <td className="py-3 text-right font-medium">
                          {formatNumber(categoryTotals[category])}
                        </td>
                        <td className="py-3 text-right">
                          <span
                            className={getConfidenceColor(
                              items.reduce((s, i) => s + i.confidence, 0) /
                                items.length,
                            )}
                          >
                            {Math.round(
                              (items.reduce((s, i) => s + i.confidence, 0) /
                                items.length) *
                                100,
                            )}
                            %
                          </span>
                        </td>
                        {showComparison && <td className="py-3"></td>}
                        <td className="py-3"></td>
                      </tr>

                      {/* Activity Rows */}
                      {isExpanded &&
                        items.map((item) => (
                          <tr
                            key={item.id}
                            className="border-b border-dashed transition-colors hover:bg-muted/30"
                            onClick={() => onItemClick?.(item)}
                          >
                            <td className="py-2 pl-8">
                              <div className="flex items-center gap-2">
                                <span className="text-xs text-muted-foreground">
                                  {item.activityCode}
                                </span>
                                <span>{item.activityName}</span>
                                {item.reasoning && (
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-5 w-5"
                                    title={item.reasoning}
                                  >
                                    <HelpCircle className="h-3 w-3" />
                                  </Button>
                                )}
                              </div>
                            </td>
                            <td className="py-2 text-right">
                              {formatNumber(item.hours)}
                            </td>
                            <td className="py-2 text-right">
                              <span
                                className={getConfidenceColor(item.confidence)}
                              >
                                {Math.round(item.confidence * 100)}%
                              </span>
                            </td>
                            {showComparison && (
                              <td className="py-2 text-right">
                                {item.comparisonToSimilar !== undefined && (
                                  <div className="flex items-center justify-end gap-1">
                                    {getComparisonIcon(
                                      item.comparisonToSimilar,
                                    )}
                                    <span
                                      className={cn(
                                        "text-xs",
                                        item.comparisonToSimilar > 5 &&
                                          "text-red-500",
                                        item.comparisonToSimilar < -5 &&
                                          "text-green-500",
                                        Math.abs(item.comparisonToSimilar) <=
                                          5 && "text-muted-foreground",
                                      )}
                                    >
                                      {item.comparisonToSimilar > 0 ? "+" : ""}
                                      {item.comparisonToSimilar}%
                                    </span>
                                  </div>
                                )}
                              </td>
                            )}
                            <td className="py-2"></td>
                          </tr>
                        ))}
                    </Fragment>
                  );
                })}

                {/* Total Row */}
                <tr className="border-t-2 bg-primary/5 font-bold">
                  <td className="py-3">TOTAL</td>
                  <td className="py-3 text-right">
                    {formatNumber(data.totalHours)}
                  </td>
                  <td className="py-3 text-right">
                    <span className={getConfidenceColor(data.confidence)}>
                      {Math.round(data.confidence * 100)}%
                    </span>
                  </td>
                  {showComparison && <td className="py-3"></td>}
                  <td className="py-3"></td>
                </tr>
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

// Confidence indicator component
export function ConfidenceIndicator({
  confidence,
  size = "md",
  showLabel = true,
  className,
}: {
  confidence: number;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}) {
  const sizeClasses = {
    sm: "h-2 w-16",
    md: "h-3 w-24",
    lg: "h-4 w-32",
  };

  const getColor = (c: number) => {
    if (c >= 0.8) return "bg-green-500";
    if (c >= 0.6) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div
        className={cn(
          "overflow-hidden rounded-full bg-muted",
          sizeClasses[size],
        )}
      >
        <div
          className={cn("h-full transition-all", getColor(confidence))}
          style={{ width: `${confidence * 100}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs text-muted-foreground">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </div>
  );
}

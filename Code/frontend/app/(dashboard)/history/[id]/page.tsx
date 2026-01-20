"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  ArrowLeft,
  Calendar,
  Building2,
  Layers,
  Clock,
  Euro,
  ChevronDown,
  ChevronUp,
  Loader2,
  AlertCircle,
  FileText,
  CheckCircle2,
  Edit3,
  Download,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  historyApi,
  type PRDetail,
  type QuotationDetail,
  type QuotationSummary,
} from "@/lib/api";

export default function HistoryDetailPage() {
  const params = useParams();
  const router = useRouter();
  const prId = params.id as string;

  const [prDetail, setPrDetail] = useState<PRDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Expanded quotation state
  const [expandedQuotation, setExpandedQuotation] = useState<number | null>(
    null,
  );
  const [quotationDetail, setQuotationDetail] =
    useState<QuotationDetail | null>(null);
  const [loadingQuotation, setLoadingQuotation] = useState(false);

  // Fetch PR detail
  const fetchPRDetail = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await historyApi.getPR(prId);
      setPrDetail(data);
      // Auto-expand the latest quotation if exists
      if (data.quotations.length > 0) {
        const latestVersion = Math.max(
          ...data.quotations.map((q) => q.version),
        );
        setExpandedQuotation(latestVersion);
      }
    } catch (err) {
      console.error("Failed to fetch PR detail:", err);
      setError("Failed to load product request details");
    } finally {
      setLoading(false);
    }
  }, [prId]);

  // Fetch quotation breakdown when expanded
  const fetchQuotationDetail = useCallback(
    async (version: number) => {
      setLoadingQuotation(true);
      try {
        const data = await historyApi.getQuotation(prId, version);
        setQuotationDetail(data);
      } catch (err) {
        console.error("Failed to fetch quotation detail:", err);
      } finally {
        setLoadingQuotation(false);
      }
    },
    [prId],
  );

  // Initial load
  useEffect(() => {
    fetchPRDetail();
  }, [fetchPRDetail]);

  // Fetch quotation detail when expanded changes
  useEffect(() => {
    if (expandedQuotation !== null) {
      fetchQuotationDetail(expandedQuotation);
    }
  }, [expandedQuotation, fetchQuotationDetail]);

  const toggleQuotation = (version: number) => {
    if (expandedQuotation === version) {
      setExpandedQuotation(null);
      setQuotationDetail(null);
    } else {
      setExpandedQuotation(version);
    }
  };

  const statusColors: Record<string, string> = {
    completed:
      "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    draft:
      "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    exported:
      "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    processing:
      "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  };

  if (loading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error || !prDetail) {
    return (
      <div className="space-y-6">
        <Button variant="ghost" onClick={() => router.back()}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back
        </Button>
        <Card>
          <CardContent className="py-12 text-center">
            <AlertCircle className="mx-auto h-12 w-12 text-destructive" />
            <p className="mt-4 font-medium text-destructive">
              {error || "Product request not found"}
            </p>
            <div className="mt-4 flex justify-center gap-2">
              <Button variant="outline" onClick={fetchPRDetail}>
                Try Again
              </Button>
              <Link href="/history">
                <Button variant="outline">Back to History</Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.back()}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold">{prDetail.pr_code}</h1>
              <Badge
                className={cn(
                  statusColors[prDetail.status] || statusColors.draft,
                )}
              >
                {prDetail.status}
              </Badge>
            </div>
            <p className="mt-1 text-muted-foreground">{prDetail.title}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Download className="mr-2 h-4 w-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <Building2 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Customer</p>
                <p className="font-medium">{prDetail.customer || "N/A"}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Layers className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Program Family</p>
                <p className="font-medium">
                  {prDetail.program_family || "N/A"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
                <Calendar className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Created</p>
                <p className="font-medium">
                  {new Date(prDetail.created_at).toLocaleDateString()}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-yellow-100 dark:bg-yellow-900/30">
                <FileText className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Quotations</p>
                <p className="font-medium">{prDetail.quotations.length}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Description */}
      {prDetail.description && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Description</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="whitespace-pre-wrap text-sm text-muted-foreground">
              {prDetail.description}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Quotations */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Quotation Versions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {prDetail.quotations.length === 0 ? (
            <p className="py-8 text-center text-muted-foreground">
              No quotations available for this PR
            </p>
          ) : (
            prDetail.quotations
              .sort((a, b) => b.version - a.version)
              .map((quotation) => (
                <QuotationCard
                  key={quotation.id}
                  quotation={quotation}
                  isExpanded={expandedQuotation === quotation.version}
                  onToggle={() => toggleQuotation(quotation.version)}
                  detail={
                    expandedQuotation === quotation.version
                      ? quotationDetail
                      : null
                  }
                  loading={
                    expandedQuotation === quotation.version && loadingQuotation
                  }
                />
              ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}

interface QuotationCardProps {
  quotation: QuotationSummary;
  isExpanded: boolean;
  onToggle: () => void;
  detail: QuotationDetail | null;
  loading: boolean;
}

function QuotationCard({
  quotation,
  isExpanded,
  onToggle,
  detail,
  loading,
}: QuotationCardProps) {
  return (
    <Collapsible open={isExpanded} onOpenChange={onToggle}>
      <div className="rounded-lg border">
        <CollapsibleTrigger className="flex w-full items-center justify-between p-4 hover:bg-muted/50">
          <div className="flex items-center gap-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/10 text-sm font-bold text-primary">
              v{quotation.version}
            </div>
            <div className="text-left">
              <div className="flex items-center gap-2">
                <span className="font-medium">Version {quotation.version}</span>
                {quotation.is_finalized && (
                  <Badge
                    variant="outline"
                    className="bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
                  >
                    <CheckCircle2 className="mr-1 h-3 w-3" />
                    Finalized
                  </Badge>
                )}
              </div>
              <p className="text-sm text-muted-foreground">
                {new Date(quotation.created_at).toLocaleDateString()} at{" "}
                {new Date(quotation.created_at).toLocaleTimeString()}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <div className="flex items-center gap-1 text-sm text-muted-foreground">
                <Clock className="h-3 w-3" />
                <span>{quotation.total_hours.toLocaleString()} hours</span>
              </div>
              <div className="flex items-center gap-1 font-medium text-green-600 dark:text-green-400">
                <Euro className="h-3 w-3" />
                <span>{(quotation.total_cost_eur / 1000).toFixed(0)}K EUR</span>
              </div>
            </div>
            {quotation.confidence_score !== null && (
              <div className="text-right">
                <p className="text-sm text-muted-foreground">Confidence</p>
                <p className="font-medium">
                  {Math.round(quotation.confidence_score * 100)}%
                </p>
              </div>
            )}
            {isExpanded ? (
              <ChevronUp className="h-4 w-4 text-muted-foreground" />
            ) : (
              <ChevronDown className="h-4 w-4 text-muted-foreground" />
            )}
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="border-t p-4">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : detail ? (
              <div className="space-y-4">
                {detail.estimation_method && (
                  <div className="rounded-lg bg-muted/50 p-3">
                    <p className="text-sm">
                      <span className="font-medium">Estimation Method:</span>{" "}
                      {detail.estimation_method}
                    </p>
                  </div>
                )}

                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[100px]">Code</TableHead>
                      <TableHead>Activity</TableHead>
                      <TableHead className="text-right">Hours</TableHead>
                      <TableHead className="text-right">Rate (EUR)</TableHead>
                      <TableHead className="text-right">Cost (EUR)</TableHead>
                      <TableHead className="text-right">Confidence</TableHead>
                      <TableHead className="w-[50px]"></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {detail.breakdowns.map((item) => (
                      <TableRow key={item.id}>
                        <TableCell className="font-mono text-sm">
                          {item.activity_code}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-2">
                            {item.activity_name}
                            {item.user_edited && (
                              <Badge variant="outline" className="text-xs">
                                <Edit3 className="mr-1 h-3 w-3" />
                                Edited
                              </Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-right font-medium">
                          {item.hours.toLocaleString()}
                        </TableCell>
                        <TableCell className="text-right">
                          {item.hourly_rate_eur.toFixed(2)}
                        </TableCell>
                        <TableCell className="text-right">
                          {item.cost_eur.toLocaleString()}
                        </TableCell>
                        <TableCell className="text-right">
                          {item.confidence_score !== null
                            ? `${Math.round(item.confidence_score * 100)}%`
                            : "-"}
                        </TableCell>
                        <TableCell>
                          {item.reasoning && (
                            <span
                              className="cursor-help text-muted-foreground"
                              title={item.reasoning}
                            >
                              ?
                            </span>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                <div className="flex justify-end border-t pt-4">
                  <div className="grid grid-cols-2 gap-8 text-right">
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Total Hours
                      </p>
                      <p className="text-xl font-bold">
                        {detail.total_hours.toLocaleString()}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">
                        Total Cost
                      </p>
                      <p className="text-xl font-bold text-green-600 dark:text-green-400">
                        EUR {detail.total_cost_eur.toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <p className="py-4 text-center text-muted-foreground">
                Failed to load breakdown details
              </p>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import Link from "next/link";
import {
  Search,
  FileText,
  Calendar,
  Target,
  TrendingUp,
  ExternalLink,
  Download,
  Loader2,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { historyApi, type PRHistory } from "@/lib/api";

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<string | null>(null);
  const [history, setHistory] = useState<PRHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);

  // Fetch history from API
  const fetchHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: { search?: string; status?: string; limit?: number } = {
        limit: 50,
      };
      if (searchQuery) {
        params.search = searchQuery;
      }
      if (statusFilter) {
        params.status = statusFilter;
      }
      const response = await historyApi.listPRs(params);
      setHistory(response.items);
      setTotal(response.total);
    } catch (err) {
      console.error("Failed to fetch history:", err);
      setError("Failed to load estimation history");
    } finally {
      setLoading(false);
    }
  }, [searchQuery, statusFilter]);

  // Initial load
  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchHistory();
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery, statusFilter, fetchHistory]);

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

  const stats = useMemo(() => {
    const completed = history.filter(
      (i) => i.status === "completed" || i.status === "exported",
    ).length;
    const exported = history.filter((i) => i.status === "exported").length;
    const totalHours = history.reduce(
      (sum, i) => sum + (i.total_hours || 0),
      0,
    );
    return {
      total: total,
      completed,
      exported,
      totalHours,
    };
  }, [history, total]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Estimation History
          </h1>
          <p className="mt-1 text-muted-foreground">
            Browse and manage your past cost estimations
          </p>
        </div>
        <Button variant="outline" size="icon" onClick={fetchHistory}>
          <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <FileText className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.total}</p>
                <p className="text-sm text-muted-foreground">Total PRs</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
                <Target className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.completed}</p>
                <p className="text-sm text-muted-foreground">Completed</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <Download className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">{stats.exported}</p>
                <p className="text-sm text-muted-foreground">Exported</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-yellow-100 dark:bg-yellow-900/30">
                <TrendingUp className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />
              </div>
              <div>
                <p className="text-2xl font-bold">
                  {stats.totalHours.toLocaleString()}
                </p>
                <p className="text-sm text-muted-foreground">Total Hours</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card>
        <CardContent className="py-4">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="relative flex-1 md:max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by PR code, title, customer..."
                className="pl-9"
              />
            </div>
            <div className="flex flex-wrap gap-2">
              <Button
                variant={statusFilter === null ? "default" : "outline"}
                size="sm"
                onClick={() => setStatusFilter(null)}
              >
                All
              </Button>
              <Button
                variant={statusFilter === "completed" ? "default" : "outline"}
                size="sm"
                onClick={() => setStatusFilter("completed")}
              >
                Completed
              </Button>
              <Button
                variant={statusFilter === "exported" ? "default" : "outline"}
                size="sm"
                onClick={() => setStatusFilter("exported")}
              >
                Exported
              </Button>
              <Button
                variant={statusFilter === "draft" ? "default" : "outline"}
                size="sm"
                onClick={() => setStatusFilter("draft")}
              >
                Draft
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* History List */}
      <div className="space-y-3">
        {loading ? (
          <Card>
            <CardContent className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </CardContent>
          </Card>
        ) : error ? (
          <Card>
            <CardContent className="py-12 text-center">
              <AlertCircle className="mx-auto h-12 w-12 text-destructive" />
              <p className="mt-4 font-medium text-destructive">{error}</p>
              <Button variant="outline" className="mt-4" onClick={fetchHistory}>
                Try Again
              </Button>
            </CardContent>
          </Card>
        ) : history.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <FileText className="mx-auto h-12 w-12 text-muted-foreground/50" />
              <p className="mt-4 font-medium">No estimations found</p>
              <p className="mt-1 text-sm text-muted-foreground">
                {searchQuery || statusFilter
                  ? "Try adjusting your search or filters"
                  : "Start a new estimation to see it here"}
              </p>
            </CardContent>
          </Card>
        ) : (
          history.map((item) => (
            <Card key={item.id} className="transition-shadow hover:shadow-md">
              <CardContent className="py-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <Link
                        href={`/history/${item.id}`}
                        className="font-medium text-primary hover:underline"
                      >
                        {item.pr_code}
                      </Link>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-xs font-medium",
                          statusColors[item.status] || statusColors.draft,
                        )}
                      >
                        {item.status}
                      </span>
                    </div>
                    <p className="mt-1 text-sm">{item.title}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
                      {item.customer && <span>{item.customer}</span>}
                      {item.customer && item.program_family && <span>•</span>}
                      {item.program_family && (
                        <span>{item.program_family}</span>
                      )}
                      {(item.customer || item.program_family) && <span>•</span>}
                      <span className="flex items-center gap-1">
                        <Calendar className="h-3 w-3" />
                        {new Date(item.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-6 text-right">
                    {item.total_hours !== null &&
                      item.total_hours !== undefined && (
                        <div>
                          <p className="text-lg font-bold">
                            {item.total_hours.toLocaleString()}
                          </p>
                          <p className="text-xs text-muted-foreground">hours</p>
                        </div>
                      )}
                    {item.total_cost_eur !== null &&
                      item.total_cost_eur !== undefined && (
                        <div>
                          <p className="text-lg font-bold text-green-600 dark:text-green-400">
                            €{(item.total_cost_eur / 1000).toFixed(0)}K
                          </p>
                          <p className="text-xs text-muted-foreground">cost</p>
                        </div>
                      )}
                    <Link href={`/history/${item.id}`}>
                      <Button variant="ghost" size="icon">
                        <ExternalLink className="h-4 w-4" />
                      </Button>
                    </Link>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
}

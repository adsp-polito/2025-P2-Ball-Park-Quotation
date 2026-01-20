"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import {
  FileText,
  TrendingUp,
  Target,
  Clock,
  ArrowRight,
  Brain,
  Calendar,
  Plus,
  Loader2,
  AlertCircle,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { useAuthStore } from "@/stores/authStore";
import {
  dashboardApi,
  type DashboardStats,
  type RecentEstimation,
} from "@/lib/api";

export default function DashboardPage() {
  const user = useAuthStore((state) => state.user);

  // State for real data
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [recentEstimations, setRecentEstimations] = useState<
    RecentEstimation[]
  >([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [statsData, estimationsData] = await Promise.all([
        dashboardApi.getStats(),
        dashboardApi.getRecentEstimations(5),
      ]);

      setStats(statsData);
      setRecentEstimations(estimationsData);
    } catch (err) {
      console.error("Failed to fetch dashboard data:", err);
      setError("Failed to load dashboard data");
      // Set default values on error (all zeros/empty)
      setStats({
        totalEstimations: 0,
        completedThisMonth: 0,
        averageAccuracy: 0,
        modelVersion: "—",
        pendingCorrections: 0,
        averageProcessingTime: "—",
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboardData();
  }, [fetchDashboardData]);

  const statusColors: Record<string, string> = {
    completed:
      "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    in_progress:
      "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
    draft:
      "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
    exported:
      "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  };

  if (loading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Welcome back, {user?.full_name?.split(" ")[0] || "Engineer"}
          </h1>
          <p className="mt-1 text-muted-foreground">
            Here&apos;s an overview of your cost estimation activity
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchDashboardData}
            disabled={loading}
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
          <Link href="/estimation/new">
            <Button size="lg" className="gap-2">
              <Plus className="h-5 w-5" />
              New Estimation
            </Button>
          </Link>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive">
          <AlertCircle className="h-5 w-5" />
          <span>{error}</span>
          <Button
            variant="outline"
            size="sm"
            onClick={fetchDashboardData}
            className="ml-auto"
          >
            Retry
          </Button>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10">
                <FileText className="h-6 w-6 text-primary" />
              </div>
              <div>
                <p className="text-3xl font-bold">
                  {stats?.totalEstimations ?? 0}
                </p>
                <p className="text-sm text-muted-foreground">
                  Total Estimations
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
                <Target className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="text-3xl font-bold">
                  {stats?.averageAccuracy ?? 0}%
                </p>
                <p className="text-sm text-muted-foreground">Avg Accuracy</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
                <TrendingUp className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <p className="text-3xl font-bold">
                  {stats?.completedThisMonth ?? 0}
                </p>
                <p className="text-sm text-muted-foreground">This Month</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
                <Clock className="h-6 w-6 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <p className="text-3xl font-bold">
                  {stats?.averageProcessingTime ?? "—"}
                </p>
                <p className="text-sm text-muted-foreground">Avg Time</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent Estimations */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Recent Estimations</CardTitle>
                <CardDescription>
                  Your latest cost estimation work
                </CardDescription>
              </div>
              <Link href="/history">
                <Button variant="outline" size="sm" className="gap-1">
                  View All
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </CardHeader>
            <CardContent>
              {recentEstimations.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-8 text-center">
                  <FileText className="h-12 w-12 text-muted-foreground/50" />
                  <p className="mt-4 text-lg font-medium">No estimations yet</p>
                  <p className="text-sm text-muted-foreground">
                    Start by creating your first estimation
                  </p>
                  <Link href="/estimation/new" className="mt-4">
                    <Button>
                      <Plus className="mr-2 h-4 w-4" />
                      New Estimation
                    </Button>
                  </Link>
                </div>
              ) : (
                <div className="space-y-3">
                  {recentEstimations.map((item) => (
                    <div
                      key={item.id}
                      className="flex items-center justify-between rounded-lg border p-3 transition-colors hover:bg-muted/50"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                          <FileText className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <Link
                              href={`/history/${item.id}`}
                              className="font-medium text-primary hover:underline"
                            >
                              {item.prCode}
                            </Link>
                            <span
                              className={`rounded-full px-2 py-0.5 text-xs font-medium ${statusColors[item.status] || statusColors.draft}`}
                            >
                              {item.status.replace("_", " ")}
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground line-clamp-1">
                            {item.title}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        {item.hours ? (
                          <p className="font-medium">
                            {item.hours.toLocaleString()} hrs
                          </p>
                        ) : (
                          <p className="text-sm text-muted-foreground">—</p>
                        )}
                        <p className="text-xs text-muted-foreground">
                          {item.date}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Model Status & Quick Actions */}
        <div className="space-y-6">
          {/* ML Model Card */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                ML Model Status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Version</span>
                <span className="font-medium">
                  {stats?.modelVersion ? `v${stats.modelVersion}` : "—"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  Pending Corrections
                </span>
                <span className="font-medium">
                  {stats?.pendingCorrections ?? 0}
                </span>
              </div>
              <div className="rounded-lg bg-muted p-3">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-green-500" />
                  <span className="text-sm font-medium">Model Healthy</span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  HCQE predictor achieving 78.8% accuracy
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Link href="/estimation/new" className="block">
                <Button
                  variant="outline"
                  className="w-full justify-start gap-2"
                >
                  <Plus className="h-4 w-4" />
                  New Estimation
                </Button>
              </Link>
              <Link href="/history" className="block">
                <Button
                  variant="outline"
                  className="w-full justify-start gap-2"
                >
                  <Calendar className="h-4 w-4" />
                  View History
                </Button>
              </Link>
              <Link href="/knowledge" className="block">
                <Button
                  variant="outline"
                  className="w-full justify-start gap-2"
                >
                  <FileText className="h-4 w-4" />
                  Knowledge Base
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

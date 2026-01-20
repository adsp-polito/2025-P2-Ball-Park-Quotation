import { getTranslations } from "next-intl/server";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { PlusCircle, TrendingUp, Clock, Target, FileText } from "lucide-react";

export default async function DashboardPage() {
  const t = await getTranslations("dashboard");

  // Empty initial state - data comes from API
  const stats = {
    totalEstimates: 0,
    avgAccuracy: 0,
    thisMonth: 0,
    avgTime: "—",
    trend: 0,
  };

  const recentEstimates: Array<{
    id: string;
    prCode: string;
    title: string;
    hours: number;
    date: string;
  }> = [];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">{t("title")}</h1>
          <p className="text-muted-foreground">{t("welcome")}</p>
        </div>
        <Link href="/estimation/new">
          <Button size="lg" className="gap-2">
            <PlusCircle className="h-5 w-5" />
            New Estimation
          </Button>
        </Link>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {t("totalEstimates")}
            </CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalEstimates}</div>
            <p className="text-xs text-muted-foreground">
              +{stats.thisMonth} this month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {t("avgAccuracy")}
            </CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgAccuracy}%</div>
            <p className="text-xs text-muted-foreground">—</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Time</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.avgTime}</div>
            <p className="text-xs text-muted-foreground">Per estimation</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              {t("monthlyTrend")}
            </CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.trend}%</div>
            <p className="text-xs text-muted-foreground">Vs previous month</p>
          </CardContent>
        </Card>
      </div>

      {/* Recent Estimates */}
      <Card>
        <CardHeader>
          <CardTitle>{t("recentEstimates")}</CardTitle>
          <CardDescription>Your latest cost estimations</CardDescription>
        </CardHeader>
        <CardContent>
          {recentEstimates.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <FileText className="h-12 w-12 text-muted-foreground/50" />
              <p className="mt-4 text-lg font-medium">No estimations yet</p>
              <p className="text-sm text-muted-foreground">
                Start by creating your first estimation
              </p>
              <Link href="/estimation/new" className="mt-4">
                <Button>
                  <PlusCircle className="mr-2 h-4 w-4" />
                  New Estimation
                </Button>
              </Link>
            </div>
          ) : (
            <>
              <div className="space-y-4">
                {recentEstimates.map((estimate) => (
                  <div
                    key={estimate.id}
                    className="flex items-center justify-between rounded-lg border p-4"
                  >
                    <div className="space-y-1">
                      <p className="font-medium">{estimate.prCode}</p>
                      <p className="text-sm text-muted-foreground">
                        {estimate.title}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">
                        {estimate.hours.toLocaleString()} hrs
                      </p>
                      <p className="text-sm text-muted-foreground">
                        {estimate.date}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-4 text-center">
                <Link href="/history">
                  <Button variant="outline">View All History</Button>
                </Link>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

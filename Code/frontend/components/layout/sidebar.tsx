"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import {
  LayoutDashboard,
  PlusCircle,
  History,
  BookOpen,
  Settings,
  LogOut,
  Brain,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuthStore } from "@/stores/authStore";

const navItems = [
  { key: "dashboard", href: "/dashboard", icon: LayoutDashboard },
  { key: "newEstimation", href: "/estimation/new", icon: PlusCircle },
  { key: "history", href: "/history", icon: History },
  { key: "knowledge", href: "/knowledge", icon: BookOpen },
  { key: "settings", href: "/settings", icon: Settings },
];

const SIDEBAR_COLLAPSED_KEY = "sidebar_collapsed";

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const t = useTranslations("nav");
  const logout = useAuthStore((state) => state.logout);

  // Collapsed state with localStorage persistence
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // Load collapsed state from localStorage on mount
  useEffect(() => {
    setIsMounted(true);
    const stored = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
    if (stored !== null) {
      setIsCollapsed(stored === "true");
    }
  }, []);

  // Persist collapsed state
  const toggleCollapsed = () => {
    const newState = !isCollapsed;
    setIsCollapsed(newState);
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(newState));
    // Dispatch custom event for layout to listen
    window.dispatchEvent(
      new CustomEvent("sidebar-toggle", { detail: { collapsed: newState } }),
    );
  };

  const handleLogout = async () => {
    await logout();
    router.push("/login");
  };

  // Prevent hydration mismatch
  if (!isMounted) {
    return (
      <aside className="fixed inset-y-0 left-0 z-50 hidden w-56 flex-col border-r bg-card lg:flex" />
    );
  }

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 hidden flex-col border-r bg-card lg:flex",
          "transition-all duration-300 ease-in-out",
          isCollapsed ? "w-14" : "w-56",
        )}
      >
        {/* Logo */}
        <div
          className={cn(
            "flex h-14 items-center border-b transition-all duration-300",
            isCollapsed ? "justify-center px-2" : "gap-2 px-4",
          )}
        >
          <Brain
            className={cn(
              "text-primary transition-all duration-300",
              isCollapsed ? "h-8 w-8" : "h-7 w-7",
            )}
          />
          <div
            className={cn(
              "overflow-hidden transition-all duration-300",
              isCollapsed ? "w-0 opacity-0" : "w-auto opacity-100",
            )}
          >
            <h1 className="text-base font-bold text-foreground whitespace-nowrap">
              Cost Brain
            </h1>
            <p className="text-[10px] text-muted-foreground">v2.0</p>
          </div>
        </div>

        {/* Navigation */}
        <nav
          className={cn(
            "flex-1 space-y-0.5 transition-all duration-300",
            isCollapsed ? "p-2" : "p-3",
          )}
        >
          {navItems.map((item) => {
            const isActive =
              pathname === item.href || pathname.startsWith(`${item.href}/`);
            const Icon = item.icon;

            const button = (
              <Button
                variant={isActive ? "secondary" : "ghost"}
                size="sm"
                className={cn(
                  "w-full transition-all duration-300",
                  isCollapsed
                    ? "justify-center px-0"
                    : "justify-start gap-2 text-sm",
                  isActive && "bg-secondary",
                )}
              >
                <Icon className="h-4 w-4 flex-shrink-0" />
                <span
                  className={cn(
                    "overflow-hidden transition-all duration-300 whitespace-nowrap",
                    isCollapsed ? "w-0 opacity-0" : "w-auto opacity-100",
                  )}
                >
                  {t(item.key)}
                </span>
              </Button>
            );

            if (isCollapsed) {
              return (
                <Tooltip key={item.key}>
                  <TooltipTrigger asChild>
                    <Link href={item.href}>{button}</Link>
                  </TooltipTrigger>
                  <TooltipContent side="right" className="font-medium">
                    {t(item.key)}
                  </TooltipContent>
                </Tooltip>
              );
            }

            return (
              <Link key={item.key} href={item.href}>
                {button}
              </Link>
            );
          })}
        </nav>

        {/* Toggle Button */}
        <div
          className={cn(
            "border-t transition-all duration-300",
            isCollapsed ? "p-2" : "p-3",
          )}
        >
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleCollapsed}
                className={cn(
                  "w-full transition-all duration-300",
                  isCollapsed
                    ? "justify-center px-0"
                    : "justify-start gap-2 text-sm",
                )}
              >
                {isCollapsed ? (
                  <ChevronRight className="h-4 w-4" />
                ) : (
                  <>
                    <ChevronLeft className="h-4 w-4" />
                    <span className="overflow-hidden whitespace-nowrap">
                      Collapse
                    </span>
                  </>
                )}
              </Button>
            </TooltipTrigger>
            {isCollapsed && (
              <TooltipContent side="right" className="font-medium">
                Expand sidebar
              </TooltipContent>
            )}
          </Tooltip>
        </div>

        {/* Logout */}
        <div
          className={cn(
            "border-t transition-all duration-300",
            isCollapsed ? "p-2" : "p-3",
          )}
        >
          {isCollapsed ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="w-full justify-center px-0 text-destructive hover:bg-destructive/10 hover:text-destructive"
                  onClick={handleLogout}
                >
                  <LogOut className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right" className="font-medium">
                {t("logout")}
              </TooltipContent>
            </Tooltip>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start gap-2 text-sm text-destructive hover:bg-destructive/10 hover:text-destructive"
              onClick={handleLogout}
            >
              <LogOut className="h-4 w-4" />
              <span className="overflow-hidden whitespace-nowrap">
                {t("logout")}
              </span>
            </Button>
          )}
        </div>
      </aside>
    </TooltipProvider>
  );
}

// Export hook for other components to check sidebar state
export function useSidebarState() {
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
    if (stored !== null) {
      setIsCollapsed(stored === "true");
    }

    const handleToggle = (e: CustomEvent<{ collapsed: boolean }>) => {
      setIsCollapsed(e.detail.collapsed);
    };

    window.addEventListener("sidebar-toggle", handleToggle as EventListener);
    return () => {
      window.removeEventListener(
        "sidebar-toggle",
        handleToggle as EventListener,
      );
    };
  }, []);

  return isCollapsed;
}

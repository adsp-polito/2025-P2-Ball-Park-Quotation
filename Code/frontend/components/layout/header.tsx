"use client";

import { useTranslations } from "next-intl";
import { Moon, Sun, Globe, Bell, User } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useAuthStore } from "@/stores/authStore";

export function Header() {
  const _t = useTranslations("app");
  const { theme, setTheme } = useTheme();
  const user = useAuthStore((state) => state.user);

  return (
    <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b bg-background/95 px-6 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      {/* Page Title - filled by page content */}
      <div className="flex-1" />

      {/* Actions */}
      <div className="flex items-center gap-2">
        {/* Language Selector */}
        <Button variant="ghost" size="icon" title="Change Language">
          <Globe className="h-5 w-5" />
        </Button>

        {/* Theme Toggle */}
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title={theme === "dark" ? "Light mode" : "Dark mode"}
        >
          <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
          <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
        </Button>

        {/* Notifications */}
        <Button variant="ghost" size="icon" title="Notifications">
          <Bell className="h-5 w-5" />
        </Button>

        {/* User Menu */}
        <Button variant="ghost" className="gap-2">
          <User className="h-5 w-5" />
          <span className="hidden md:inline">{user?.full_name || "User"}</span>
        </Button>
      </div>
    </header>
  );
}

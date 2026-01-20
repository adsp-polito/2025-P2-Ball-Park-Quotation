"use client";

import { useSidebarState } from "./sidebar";
import { cn } from "@/lib/utils";

interface MainContentProps {
  children: React.ReactNode;
}

export function MainContent({ children }: MainContentProps) {
  const isCollapsed = useSidebarState();

  return (
    <div
      className={cn(
        "transition-all duration-300 ease-in-out",
        isCollapsed ? "lg:pl-14" : "lg:pl-56",
      )}
    >
      {children}
    </div>
  );
}

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind CSS classes with proper precedence
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format number according to locale
 */
export function formatNumber(
  value: number,
  locale: "en" | "it" = "en",
  options?: Intl.NumberFormatOptions
): string {
  const localeMap = {
    en: "en-US",
    it: "it-IT",
  };

  return new Intl.NumberFormat(localeMap[locale], {
    maximumFractionDigits: 2,
    ...options,
  }).format(value);
}

/**
 * Format currency (EUR) according to locale
 */
export function formatCurrency(
  value: number,
  locale: "en" | "it" = "en"
): string {
  const localeMap = {
    en: "en-US",
    it: "it-IT",
  };

  return new Intl.NumberFormat(localeMap[locale], {
    style: "currency",
    currency: "EUR",
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Format date according to locale
 */
export function formatDate(
  date: Date | string,
  locale: "en" | "it" = "en",
  options?: Intl.DateTimeFormatOptions
): string {
  const localeMap = {
    en: "en-US",
    it: "it-IT",
  };

  const d = typeof date === "string" ? new Date(date) : date;

  return new Intl.DateTimeFormat(localeMap[locale], {
    dateStyle: "medium",
    ...options,
  }).format(d);
}

/**
 * Format percentage
 */
export function formatPercent(value: number, decimals = 0): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Truncate text with ellipsis
 */
export function truncate(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 3)}...`;
}

/**
 * Generate unique ID
 */
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Debounce function
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;

  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

/**
 * Sleep for specified milliseconds
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

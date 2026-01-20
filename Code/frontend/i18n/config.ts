export const locales = ["en", "it"] as const;
export const defaultLocale = "en" as const;

export type Locale = (typeof locales)[number];

export const localeNames: Record<Locale, string> = {
  en: "English",
  it: "Italiano",
};

export const localeFlags: Record<Locale, string> = {
  en: "🇬🇧",
  it: "🇮🇹",
};

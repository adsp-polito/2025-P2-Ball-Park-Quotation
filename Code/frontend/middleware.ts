import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Supported locales
const locales = ["en", "it"];
const defaultLocale = "en";

// Public paths that don't require authentication
const publicPaths = ["/login", "/register", "/forgot-password"];

// Get locale from request
function getLocale(request: NextRequest): string {
  // Check cookie first
  const cookieLocale = request.cookies.get("NEXT_LOCALE")?.value;
  if (cookieLocale && locales.includes(cookieLocale)) {
    return cookieLocale;
  }

  // Check Accept-Language header
  const acceptLanguage = request.headers.get("accept-language");
  if (acceptLanguage) {
    const preferredLocale = acceptLanguage
      .split(",")
      .map((lang) => lang.split(";")[0].trim().substring(0, 2))
      .find((lang) => locales.includes(lang));
    if (preferredLocale) {
      return preferredLocale;
    }
  }

  return defaultLocale;
}

// Check if user is authenticated
function isAuthenticated(request: NextRequest): boolean {
  const token = request.cookies.get("auth_token")?.value;
  return !!token;
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Skip middleware for static files and API routes
  if (
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api") ||
    pathname.includes(".") ||
    pathname.startsWith("/favicon")
  ) {
    return NextResponse.next();
  }

  // Get locale
  const locale = getLocale(request);

  // Check authentication
  const authenticated = isAuthenticated(request);
  const isPublicPath = publicPaths.some((path) => pathname.startsWith(path));

  // Redirect unauthenticated users to login
  if (!authenticated && !isPublicPath) {
    const loginUrl = new URL("/login", request.url);
    loginUrl.searchParams.set("redirect", pathname);
    const response = NextResponse.redirect(loginUrl);
    response.cookies.set("NEXT_LOCALE", locale);
    return response;
  }

  // Redirect authenticated users away from auth pages
  if (authenticated && isPublicPath) {
    const response = NextResponse.redirect(new URL("/", request.url));
    response.cookies.set("NEXT_LOCALE", locale);
    return response;
  }

  // Add locale header for components
  const response = NextResponse.next();
  response.headers.set("x-locale", locale);
  response.cookies.set("NEXT_LOCALE", locale);

  return response;
}

export const config = {
  matcher: [
    // Match all paths except static files
    "/((?!_next/static|_next/image|favicon.ico|.*\\..*).*)",
  ],
};

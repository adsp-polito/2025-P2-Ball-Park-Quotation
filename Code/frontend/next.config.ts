import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin("./i18n/request.ts");

const nextConfig: NextConfig = {
  // API proxy to backend - only for /api/v1/* paths
  // Local Route Handlers (like /api/chat/*) are NOT rewritten
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/v1/:path*`,
      },
    ];
  },

  // Increase HTTP timeout for long-running ML predictions (5 minutes)
  httpAgentOptions: {
    keepAlive: true,
  },

  // Increase server timeout for long-running ML predictions
  // Default is 60s, estimation can take 90-120s on cold start
  serverExternalPackages: [],

  // Image domains
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
    ],
  },

  // Experimental features
  experimental: {
    // Enable React 19 features
    reactCompiler: false,
  },

  // Output standalone for Docker
  output: "standalone",
};

export default withNextIntl(nextConfig);

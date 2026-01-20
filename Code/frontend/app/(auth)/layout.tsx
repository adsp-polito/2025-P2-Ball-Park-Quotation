import { Brain } from "lucide-react";

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen flex">
      {/* Left Panel - Brand */}
      <div className="hidden lg:flex lg:w-1/2 bg-primary flex-col justify-between p-12 text-primary-foreground">
        <div className="flex items-center gap-2">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-foreground/10">
            <Brain className="h-6 w-6" />
          </div>
          <span className="text-xl font-bold">Cost Brain</span>
        </div>

        <div className="space-y-6">
          <h1 className="text-4xl font-bold leading-tight">
            AI-Powered R&D
            <br />
            Cost Estimation
          </h1>
          <p className="text-lg text-primary-foreground/80">
            Transform your Product Request analysis with intelligent cost
            predictions, adaptive learning, and enterprise-grade accuracy.
          </p>

          <div className="grid grid-cols-2 gap-6">
            <div>
              <p className="text-3xl font-bold">91%</p>
              <p className="text-sm text-primary-foreground/70">
                Average Accuracy
              </p>
            </div>
            <div>
              <p className="text-3xl font-bold">2.3min</p>
              <p className="text-sm text-primary-foreground/70">
                Avg Processing Time
              </p>
            </div>
            <div>
              <p className="text-3xl font-bold">127+</p>
              <p className="text-sm text-primary-foreground/70">
                PRs Processed
              </p>
            </div>
            <div>
              <p className="text-3xl font-bold">5-Step</p>
              <p className="text-sm text-primary-foreground/70">
                Guided Workflow
              </p>
            </div>
          </div>
        </div>

        <p className="text-sm text-primary-foreground/60">
          FPT Industrial - R&D Cost Estimation Platform
        </p>
      </div>

      {/* Right Panel - Auth Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">{children}</div>
      </div>
    </div>
  );
}

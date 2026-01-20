"use client";

import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { ArrowRight, Loader2, Eye } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileUploader } from "@/components/estimation/file-uploader";
import { StepIndicator } from "@/components/estimation/step-indicator";
import {
  PRPreviewModal,
  ParsedPRData,
} from "@/components/estimation/pr-preview-modal";
import { useEstimationStore } from "@/stores/estimationStore";
import { estimationApi } from "@/lib/api";

export default function NewEstimationPage() {
  const _t = useTranslations("estimation");
  const router = useRouter();
  const { loadSession, isLoading: storeLoading } = useEstimationStore();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<{
    valid: boolean;
    errors?: string[];
    warnings?: string[];
    parsedFields?: Record<string, string | number>;
  } | null>(null);

  // NEW: State for parsed PR data and preview modal
  const [parsedPR, setParsedPR] = useState<ParsedPRData | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);

  const handleFileSelect = useCallback(async (file: File) => {
    setSelectedFile(file);
    setValidationResult(null);
    setSessionId(null);
    setParsedPR(null); // Reset parsed PR
    setIsUploading(true);
    setUploadProgress(0);

    // Show progress while uploading
    const progressInterval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      // Call real API to upload and parse the file
      const session = await estimationApi.start(file);

      clearInterval(progressInterval);
      setUploadProgress(95);

      // Get full session state with parsed PR data
      const state = await estimationApi.getSession(session.session_id);

      setUploadProgress(100);
      setSessionId(session.session_id);

      // Store full parsed_pr for preview modal
      const parsedPrData = state.parsed_pr || {};
      setParsedPR(parsedPrData as ParsedPRData);

      // Extract summary fields for inline display
      const activities = (parsedPrData.raw_activities as Array<unknown>) || [];

      // Collect warnings from validation
      const warnings: string[] = [];
      if (!parsedPrData.pr_code) warnings.push("PR code not found");
      if (!parsedPrData.title) warnings.push("Title not found");
      if (activities.length === 0) {
        warnings.push("No activities extracted - will be predicted by AI");
      }

      setValidationResult({
        valid: true,
        warnings: warnings.length > 0 ? warnings : undefined,
        parsedFields: {
          "PR Code": (parsedPrData.pr_code as string) || "Not found",
          Title: (parsedPrData.title as string) || "Not found",
          Platform: (parsedPrData.platform as string) || "Not detected",
          Engine: (parsedPrData.engine as string) || "Not detected",
          Sector: (parsedPrData.sector as string) || "Not detected",
        },
      });
    } catch (error) {
      clearInterval(progressInterval);
      const errorMessage =
        error instanceof Error ? error.message : "Failed to parse file";
      setValidationResult({
        valid: false,
        errors: [errorMessage],
      });
      setParsedPR(null);
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleFileRemove = useCallback(() => {
    setSelectedFile(null);
    setValidationResult(null);
    setUploadProgress(0);
    setSessionId(null);
    setParsedPR(null); // Clear parsed PR
  }, []);

  const handleStartEstimation = useCallback(async () => {
    if (!sessionId || !validationResult?.valid) return;

    try {
      // Load the existing session into the store
      await loadSession(sessionId);
      // Navigate to the estimation session
      router.push(`/estimation/${sessionId}`);
    } catch {
      // Error handled in store
    }
  }, [sessionId, validationResult, loadSession, router]);

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">
          New Cost Estimation
        </h1>
        <p className="mt-1 text-muted-foreground">
          Upload a Product Request (PR) Excel file to begin the estimation
          process
        </p>
      </div>

      {/* Step Indicator */}
      <StepIndicator currentStep="upload" className="py-4" />

      {/* Main Content */}
      <Card>
        <CardHeader>
          <CardTitle>Step 1: Upload Product Request</CardTitle>
          <CardDescription>
            Upload your PR Excel file (.xls or .xlsx). The system will parse and
            validate the document to extract relevant information.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <FileUploader
            onFileSelect={handleFileSelect}
            onFileRemove={handleFileRemove}
            selectedFile={selectedFile}
            isUploading={isUploading}
            uploadProgress={uploadProgress}
            validationResult={validationResult ?? undefined}
          />

          {/* Action buttons */}
          {validationResult?.valid && (
            <div className="flex items-center justify-between pt-4">
              {/* View Parsed Details button */}
              <Button
                variant="outline"
                onClick={() => setPreviewOpen(true)}
                className="gap-2"
              >
                <Eye className="h-4 w-4" />
                View Parsed Details
              </Button>

              {/* Start Estimation button */}
              <Button
                onClick={handleStartEstimation}
                disabled={storeLoading}
                size="lg"
                className="gap-2"
              >
                {storeLoading ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    Start Estimation
                    <ArrowRight className="h-5 w-5" />
                  </>
                )}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Help Section */}
      <Card className="bg-muted/50">
        <CardHeader>
          <CardTitle className="text-base">What happens next?</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-xs text-primary-foreground">
                1
              </span>
              <span>
                <strong>Upload</strong> - Your PR file is parsed and validated
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted-foreground/30 text-xs">
                2
              </span>
              <span>
                <strong>Q&A</strong> - Answer smart questions to refine the
                estimate
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted-foreground/30 text-xs">
                3
              </span>
              <span>
                <strong>Summary</strong> - Review extracted features and similar
                PRs
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted-foreground/30 text-xs">
                4
              </span>
              <span>
                <strong>Estimation</strong> - Get AI-powered cost breakdown
              </span>
            </li>
            <li className="flex items-start gap-2">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted-foreground/30 text-xs">
                5
              </span>
              <span>
                <strong>Review & Export</strong> - Adjust, approve, and export
                PE02
              </span>
            </li>
          </ol>
        </CardContent>
      </Card>

      {/* PR Preview Modal */}
      <PRPreviewModal
        open={previewOpen}
        onOpenChange={setPreviewOpen}
        parsedPR={parsedPR}
        warnings={validationResult?.warnings}
      />
    </div>
  );
}

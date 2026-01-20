"use client";

import { useState, useCallback, useRef } from "react";
import { useTranslations } from "next-intl";
import {
  Upload,
  FileSpreadsheet,
  X,
  AlertCircle,
  CheckCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
  onFileRemove?: () => void;
  selectedFile?: File | null;
  isUploading?: boolean;
  uploadProgress?: number;
  validationResult?: {
    valid: boolean;
    errors?: string[];
    warnings?: string[];
    parsedFields?: Record<string, string | number>;
  };
  className?: string;
}

const ACCEPTED_TYPES = [
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
];
const ACCEPTED_EXTENSIONS = [".xls", ".xlsx"];
const MAX_SIZE_MB = 10;

export function FileUploader({
  onFileSelect,
  onFileRemove,
  selectedFile,
  isUploading = false,
  uploadProgress,
  validationResult,
  className,
}: FileUploaderProps) {
  const _t = useTranslations("estimation");
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): boolean => {
    setError(null);

    // Check file type
    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf("."));
    if (
      !ACCEPTED_EXTENSIONS.includes(extension) &&
      !ACCEPTED_TYPES.includes(file.type)
    ) {
      setError("Only Excel files (.xls, .xlsx) are accepted");
      return false;
    }

    // Check file size
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setError(`File size must be less than ${MAX_SIZE_MB}MB`);
      return false;
    }

    return true;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      if (validateFile(file)) {
        onFileSelect(file);
      }
    },
    [validateFile, onFileSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragOver(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile],
  );

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile],
  );

  const handleRemove = useCallback(() => {
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
    onFileRemove?.();
  }, [onFileRemove]);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Drop zone */}
      {!selectedFile && (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          className={cn(
            "relative cursor-pointer rounded-lg border-2 border-dashed p-8 transition-colors",
            isDragOver
              ? "border-primary bg-primary/5"
              : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/50",
            error && "border-destructive/50",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_EXTENSIONS.join(",")}
            onChange={handleInputChange}
            className="hidden"
          />

          <div className="flex flex-col items-center justify-center gap-4 text-center">
            <div
              className={cn(
                "flex h-16 w-16 items-center justify-center rounded-full",
                isDragOver ? "bg-primary/10" : "bg-muted",
              )}
            >
              <Upload
                className={cn(
                  "h-8 w-8",
                  isDragOver ? "text-primary" : "text-muted-foreground",
                )}
              />
            </div>

            <div>
              <p className="text-lg font-medium">
                {isDragOver
                  ? "Drop your file here"
                  : "Drag & drop your PR file"}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                or click to browse • Excel files only (.xls, .xlsx)
              </p>
            </div>

            <Button type="button" variant="outline" className="mt-2">
              Select File
            </Button>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="flex items-center gap-2 rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Selected file card */}
      {selectedFile && (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
                  <FileSpreadsheet className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
                <div>
                  <p className="font-medium">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>

              {!isUploading && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleRemove}
                  className="text-muted-foreground hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>

            {/* Upload progress */}
            {isUploading && uploadProgress !== undefined && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Uploading...</span>
                  <span className="font-medium">{uploadProgress}%</span>
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Validation result */}
            {validationResult && !isUploading && (
              <div className="mt-4 space-y-3">
                {validationResult.valid ? (
                  <div className="flex items-center gap-2 text-sm text-green-600 dark:text-green-400">
                    <CheckCircle className="h-4 w-4" />
                    <span>File validated successfully</span>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {validationResult.errors?.map((err, i) => (
                      <div
                        key={i}
                        className="flex items-center gap-2 text-sm text-destructive"
                      >
                        <AlertCircle className="h-4 w-4 flex-shrink-0" />
                        <span>{err}</span>
                      </div>
                    ))}
                  </div>
                )}

                {validationResult.warnings &&
                  validationResult.warnings.length > 0 && (
                    <div className="space-y-2">
                      {validationResult.warnings.map((warn, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400"
                        >
                          <AlertCircle className="h-4 w-4 flex-shrink-0" />
                          <span>{warn}</span>
                        </div>
                      ))}
                    </div>
                  )}

                {/* Parsed fields preview */}
                {validationResult.parsedFields && (
                  <div className="mt-4 rounded-lg bg-muted/50 p-3">
                    <p className="mb-2 text-xs font-medium uppercase text-muted-foreground">
                      Extracted Information
                    </p>
                    <dl className="grid grid-cols-2 gap-2 text-sm">
                      {Object.entries(validationResult.parsedFields)
                        .slice(0, 6)
                        .map(([key, value]) => (
                          <div key={key}>
                            <dt className="text-muted-foreground">{key}</dt>
                            <dd className="font-medium">{String(value)}</dd>
                          </div>
                        ))}
                    </dl>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

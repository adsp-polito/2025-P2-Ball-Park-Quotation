"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  FileText,
  Check,
  X,
  AlertTriangle,
  Cpu,
  Gauge,
  Factory,
  Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";

// Types matching backend ParsedPR structure
export interface ParsedPRData {
  // Header fields
  pr_code?: string;
  revision?: string;
  title?: string;
  platform?: string;
  engine?: string;
  customer?: string;
  tier?: string;
  description?: string;
  program_family?: string;
  project_phase?: string;

  // Detected features
  product_family?: string;
  emissions?: string;
  sector?: string;
  ats_tech?: string;

  // Boolean flags
  hardware_change?: boolean;
  calibration_change?: boolean;
  ats_change?: boolean;
  software_vcu_change?: boolean;

  // Raw text for LLM
  raw_text?: string;

  // Raw data info
  raw_data?: {
    filename?: string;
    shape?: [number, number];
    columns?: string[];
  };

  // Validation
  validation_errors?: string[];
}

interface PRPreviewModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  parsedPR: ParsedPRData | null;
  warnings?: string[];
}

// Helper to display value or placeholder
function DisplayValue({
  value,
  placeholder = "—",
}: {
  value?: string | number | null;
  placeholder?: string;
}) {
  if (value === undefined || value === null || value === "") {
    return <span className="text-muted-foreground">{placeholder}</span>;
  }
  return <span className="font-medium">{String(value)}</span>;
}

// Boolean flag indicator
function FlagIndicator({
  label,
  value,
}: {
  label: string;
  value?: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      {value ? (
        <Check className="h-4 w-4 text-green-600" />
      ) : (
        <X className="h-4 w-4 text-muted-foreground" />
      )}
      <span className={cn(value ? "text-foreground" : "text-muted-foreground")}>
        {label}
      </span>
    </div>
  );
}

// Field row component
function FieldRow({
  label,
  value,
}: {
  label: string;
  value?: string | number | null;
}) {
  return (
    <div className="flex justify-between py-1.5 border-b border-border/50 last:border-0">
      <span className="text-muted-foreground text-sm">{label}</span>
      <DisplayValue value={value} />
    </div>
  );
}

// Section header
function SectionHeader({
  icon: Icon,
  title,
}: {
  icon: React.ElementType;
  title: string;
}) {
  return (
    <div className="flex items-center gap-2 mb-3 pb-2 border-b border-border">
      <Icon className="h-4 w-4 text-primary" />
      <h4 className="font-semibold text-sm uppercase tracking-wide">{title}</h4>
    </div>
  );
}

export function PRPreviewModal({
  open,
  onOpenChange,
  parsedPR,
  warnings = [],
}: PRPreviewModalProps) {
  if (!parsedPR) return null;

  const allWarnings = [
    ...(warnings || []),
    ...(parsedPR.validation_errors || []),
  ];

  // Check if any activities were extracted (for FPT format, usually empty)
  const hasNoActivities = !parsedPR.raw_text?.includes("activity") &&
    !parsedPR.raw_text?.includes("Activity");

  if (hasNoActivities && !allWarnings.includes("No activities extracted - will be predicted by AI")) {
    allWarnings.push("No activities extracted - will be predicted by AI");
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[85vh] p-0 gap-0">
        {/* Header */}
        <DialogHeader className="px-6 py-4 border-b">
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Parsed PR Details
            </DialogTitle>
            {parsedPR.pr_code && (
              <Badge variant="secondary" className="font-mono">
                {parsedPR.pr_code}
                {parsedPR.revision && ` Rev ${parsedPR.revision}`}
              </Badge>
            )}
          </div>
        </DialogHeader>

        {/* Content - Side by Side */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Panel - Parsed Fields */}
          <div className="w-2/5 border-r">
            <ScrollArea className="h-[calc(85vh-8rem)]">
              <div className="p-4 space-y-6">
                {/* Header Information */}
                <div>
                  <SectionHeader icon={FileText} title="Header Information" />
                  <div className="space-y-0">
                    <FieldRow label="PR Code" value={parsedPR.pr_code} />
                    <FieldRow label="Revision" value={parsedPR.revision} />
                    <FieldRow label="Title" value={parsedPR.title} />
                    <FieldRow label="Platform" value={parsedPR.platform} />
                    <FieldRow label="Engine" value={parsedPR.engine} />
                    <FieldRow label="Plant" value={parsedPR.customer} />
                    <FieldRow label="Tier" value={parsedPR.tier} />
                  </div>
                </div>

                {/* Detected Features */}
                <div>
                  <SectionHeader icon={Cpu} title="Detected Features" />
                  <div className="space-y-0">
                    <FieldRow
                      label="Product Family"
                      value={parsedPR.product_family}
                    />
                    <FieldRow label="Emissions" value={parsedPR.emissions} />
                    <FieldRow label="Sector" value={parsedPR.sector} />
                    <FieldRow label="ATS Tech" value={parsedPR.ats_tech} />
                  </div>
                </div>

                {/* Change Flags */}
                <div>
                  <SectionHeader icon={Wrench} title="Change Flags" />
                  <div className="grid grid-cols-2 gap-2">
                    <FlagIndicator
                      label="Hardware"
                      value={parsedPR.hardware_change}
                    />
                    <FlagIndicator
                      label="Calibration"
                      value={parsedPR.calibration_change}
                    />
                    <FlagIndicator label="ATS" value={parsedPR.ats_change} />
                    <FlagIndicator
                      label="Software"
                      value={parsedPR.software_vcu_change}
                    />
                  </div>
                </div>

                {/* File Info */}
                <div>
                  <SectionHeader icon={Factory} title="File Info" />
                  <div className="space-y-0">
                    <FieldRow
                      label="Filename"
                      value={parsedPR.raw_data?.filename}
                    />
                    <FieldRow
                      label="Rows"
                      value={parsedPR.raw_data?.shape?.[0]}
                    />
                    <FieldRow
                      label="Columns"
                      value={parsedPR.raw_data?.shape?.[1]}
                    />
                  </div>
                </div>

                {/* Warnings */}
                {allWarnings.length > 0 && (
                  <div>
                    <SectionHeader icon={AlertTriangle} title="Warnings" />
                    <div className="space-y-2">
                      {allWarnings.map((warning, i) => (
                        <div
                          key={i}
                          className="flex items-start gap-2 text-sm text-yellow-600 dark:text-yellow-400"
                        >
                          <AlertTriangle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                          <span>{warning}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Right Panel - Raw Text */}
          <div className="w-3/5">
            <div className="px-4 py-3 border-b bg-muted/30">
              <div className="flex items-center gap-2">
                <Gauge className="h-4 w-4 text-primary" />
                <h4 className="font-semibold text-sm uppercase tracking-wide">
                  Raw Extracted Text
                </h4>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                This text will be analyzed by AI in the Q&A step
              </p>
            </div>
            <ScrollArea className="h-[calc(85vh-10rem)]">
              <div className="p-4">
                {parsedPR.raw_text ? (
                  <pre className="text-sm font-mono whitespace-pre-wrap text-muted-foreground leading-relaxed">
                    {parsedPR.raw_text}
                  </pre>
                ) : (
                  <div className="flex flex-col items-center justify-center h-40 text-muted-foreground">
                    <FileText className="h-8 w-8 mb-2 opacity-50" />
                    <p>No text extracted from file</p>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

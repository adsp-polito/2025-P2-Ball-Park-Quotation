"use client";

/**
 * R&D Cost Table Toolbar Component
 *
 * Segmented toolbar with Edit, Format, Insert, and View sections.
 */

import React, { memo } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Undo2,
  Redo2,
  Scissors,
  Copy,
  ClipboardPaste,
  Plus,
  Trash2,
  ChevronDown,
  History,
  RotateCcw,
  Save,
  FileDown,
  Download,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface RDCostTableToolbarProps {
  // Edit state
  canUndo: boolean;
  canRedo: boolean;
  undoDescription: string | null;
  redoDescription: string | null;
  hasSelection: boolean;
  // Handlers
  onUndo: () => void;
  onRedo: () => void;
  onCut: () => void;
  onCopy: () => void;
  onPaste: () => void;
  onAddRow: () => void; // FLAT structure - no type needed
  onDeleteRows: () => void;
  onShowHistory: () => void;
  onResetToAI: () => void;
  onSave: () => void;
  onFinalize: () => void;
  onExport: (format: "pptx" | "xlsx" | "pdf" | "csv") => void;
  // State
  isDirty: boolean;
  isSaving: boolean;
  isReadOnly: boolean;
}

// ============================================================================
// Toolbar Component
// ============================================================================

export const RDCostTableToolbar = memo(function RDCostTableToolbar({
  canUndo,
  canRedo,
  undoDescription,
  redoDescription,
  hasSelection,
  onUndo,
  onRedo,
  onCut,
  onCopy,
  onPaste,
  onAddRow,
  onDeleteRows,
  onShowHistory,
  onResetToAI,
  onSave,
  onFinalize,
  onExport,
  isDirty,
  isSaving,
  isReadOnly,
}: RDCostTableToolbarProps) {
  return (
    <TooltipProvider>
      <div className="flex items-center gap-1 p-2 border-b border-gray-200 bg-gray-50">
        {/* Edit Segment */}
        <ToolbarSegment label="Edit">
          <ToolbarButton
            icon={<Undo2 className="w-4 h-4" />}
            tooltip={undoDescription ? `Undo: ${undoDescription}` : "Undo"}
            onClick={onUndo}
            disabled={!canUndo || isReadOnly}
            shortcut="Ctrl+Z"
          />
          <ToolbarButton
            icon={<Redo2 className="w-4 h-4" />}
            tooltip={redoDescription ? `Redo: ${redoDescription}` : "Redo"}
            onClick={onRedo}
            disabled={!canRedo || isReadOnly}
            shortcut="Ctrl+Y"
          />
          <ToolbarDivider />
          <ToolbarButton
            icon={<Scissors className="w-4 h-4" />}
            tooltip="Cut"
            onClick={onCut}
            disabled={!hasSelection || isReadOnly}
            shortcut="Ctrl+X"
          />
          <ToolbarButton
            icon={<Copy className="w-4 h-4" />}
            tooltip="Copy"
            onClick={onCopy}
            disabled={!hasSelection}
            shortcut="Ctrl+C"
          />
          <ToolbarButton
            icon={<ClipboardPaste className="w-4 h-4" />}
            tooltip="Paste"
            onClick={onPaste}
            disabled={isReadOnly}
            shortcut="Ctrl+V"
          />
        </ToolbarSegment>

        <ToolbarSegmentDivider />

        {/* Insert Segment - FLAT structure (simple add row) */}
        <ToolbarSegment label="Insert">
          <Button
            variant="ghost"
            size="sm"
            className="h-8 gap-1 text-gray-700"
            disabled={isReadOnly}
            onClick={onAddRow}
          >
            <Plus className="w-4 h-4" />
            Add Row
          </Button>
          <ToolbarButton
            icon={<Trash2 className="w-4 h-4" />}
            tooltip="Delete selected rows"
            onClick={onDeleteRows}
            disabled={!hasSelection || isReadOnly}
            variant="destructive"
          />
        </ToolbarSegment>

        <ToolbarSegmentDivider />

        {/* View Segment - History only (no expand/collapse for flat table) */}
        <ToolbarSegment label="View">
          <ToolbarButton
            icon={<History className="w-4 h-4" />}
            tooltip="Version history"
            onClick={onShowHistory}
          />
        </ToolbarSegment>

        <ToolbarSegmentDivider />

        {/* AI Segment */}
        <ToolbarSegment label="AI">
          <ToolbarButton
            icon={<RotateCcw className="w-4 h-4" />}
            tooltip="Reset to AI predictions"
            onClick={onResetToAI}
            disabled={isReadOnly}
          />
        </ToolbarSegment>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Actions Segment */}
        <ToolbarSegment>
          {/* Save status indicator */}
          {isDirty && !isSaving && (
            <span className="text-xs text-amber-600 mr-2">Unsaved changes</span>
          )}
          {isSaving && (
            <span className="text-xs text-gray-500 mr-2">Saving...</span>
          )}

          <ToolbarButton
            icon={<Save className="w-4 h-4" />}
            tooltip="Save draft"
            onClick={onSave}
            disabled={!isDirty || isSaving || isReadOnly}
          />

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 gap-1">
                <Download className="w-4 h-4" />
                Export
                <ChevronDown className="w-3 h-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onExport("pptx")}>
                <FileDown className="w-4 h-4 mr-2" />
                PowerPoint (PE02)
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport("xlsx")}>
                <FileDown className="w-4 h-4 mr-2" />
                Excel
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => onExport("pdf")}>
                <FileDown className="w-4 h-4 mr-2" />
                PDF
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onExport("csv")}>
                <FileDown className="w-4 h-4 mr-2" />
                CSV
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <Button
            variant="default"
            size="sm"
            className="h-8 ml-2 bg-green-600 hover:bg-green-700"
            onClick={onFinalize}
            disabled={isReadOnly}
          >
            Finalize
          </Button>
        </ToolbarSegment>
      </div>
    </TooltipProvider>
  );
});

// ============================================================================
// Sub-components
// ============================================================================

interface ToolbarSegmentProps {
  label?: string;
  children: React.ReactNode;
}

const ToolbarSegment = memo(function ToolbarSegment({
  label,
  children,
}: ToolbarSegmentProps) {
  return (
    <div className="flex items-center gap-0.5">
      {label && (
        <span className="text-xs text-gray-700 font-medium mr-1 select-none">
          {label}:
        </span>
      )}
      {children}
    </div>
  );
});

const ToolbarSegmentDivider = memo(function ToolbarSegmentDivider() {
  return <div className="w-px h-6 bg-gray-300 mx-2" />;
});

const ToolbarDivider = memo(function ToolbarDivider() {
  return <div className="w-px h-4 bg-gray-200 mx-0.5" />;
});

interface ToolbarButtonProps {
  icon: React.ReactNode;
  tooltip: string;
  onClick: () => void;
  disabled?: boolean;
  shortcut?: string;
  variant?: "default" | "destructive";
}

const ToolbarButton = memo(function ToolbarButton({
  icon,
  tooltip,
  onClick,
  disabled = false,
  shortcut,
  variant = "default",
}: ToolbarButtonProps) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "h-8 w-8 p-0",
            // Explicit dark color for icon visibility
            "text-gray-700 hover:text-gray-900 hover:bg-gray-200",
            // Disabled state
            "disabled:text-gray-400 disabled:opacity-50",
            {
              "text-red-600 hover:text-red-700 hover:bg-red-50":
                variant === "destructive",
            },
          )}
          onClick={onClick}
          disabled={disabled}
        >
          {icon}
        </Button>
      </TooltipTrigger>
      <TooltipContent>
        <p>{tooltip}</p>
        {shortcut && <p className="text-xs text-gray-400 mt-0.5">{shortcut}</p>}
      </TooltipContent>
    </Tooltip>
  );
});

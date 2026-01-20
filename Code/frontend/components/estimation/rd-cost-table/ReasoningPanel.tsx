"use client";

/**
 * Reasoning Panel Component
 *
 * Expandable panel for capturing DPO training data - why users made changes.
 * Includes text area for reasoning and change tag selector.
 */

import React, { memo, useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { X, AlertCircle, Check, Lightbulb } from "lucide-react";
import { type RowReasoning, type ChangeTag } from "./types";
import { ChangeTagSelector } from "./ChangeTagSelector";

// ============================================================================
// Types
// ============================================================================

export interface ReasoningPanelProps {
  rowId: string;
  columnName: string;
  originalValue: number;
  newValue: number;
  changePercent: number;
  confidence: number;
  existingReasoning?: RowReasoning;
  isPrompted: boolean; // True if triggered by >15% change
  onSave: (reasoning: Omit<RowReasoning, "id" | "createdAt">) => void;
  onDismiss: () => void;
  onClose: () => void;
}

// ============================================================================
// Reasoning Panel Component
// ============================================================================

export const ReasoningPanel = memo(function ReasoningPanel({
  rowId,
  columnName,
  originalValue,
  newValue,
  changePercent,
  confidence,
  existingReasoning,
  isPrompted,
  onSave,
  onDismiss,
  onClose,
}: ReasoningPanelProps) {
  const [reasoningText, setReasoningText] = useState(
    existingReasoning?.reasoningText || "",
  );
  const [selectedTags, setSelectedTags] = useState<ChangeTag[]>(
    existingReasoning?.changeTags || [],
  );
  const [isDirty, setIsDirty] = useState(false);

  // Mark as dirty when content changes
  useEffect(() => {
    if (
      reasoningText !== (existingReasoning?.reasoningText || "") ||
      JSON.stringify(selectedTags) !==
        JSON.stringify(existingReasoning?.changeTags || [])
    ) {
      setIsDirty(true);
    } else {
      setIsDirty(false);
    }
  }, [reasoningText, selectedTags, existingReasoning]);

  // ==========================================================================
  // Handlers
  // ==========================================================================

  const handleTagToggle = useCallback((tag: ChangeTag) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag],
    );
  }, []);

  const handleSave = useCallback(() => {
    onSave({
      rowId,
      columnName,
      reasoningText,
      changeTags: selectedTags,
      originalValue,
      newValue,
      confidenceAtChange: confidence,
    });
    setIsDirty(false);
  }, [
    rowId,
    columnName,
    reasoningText,
    selectedTags,
    originalValue,
    newValue,
    confidence,
    onSave,
  ]);

  const handleDismiss = useCallback(() => {
    onDismiss();
  }, [onDismiss]);

  // ==========================================================================
  // Render
  // ==========================================================================

  const changeDirection = newValue > originalValue ? "increased" : "decreased";
  const changeColor =
    Math.abs(changePercent) > 20
      ? "text-red-600"
      : Math.abs(changePercent) > 10
        ? "text-amber-600"
        : "text-gray-600";

  return (
    <div className="border-l-4 border-blue-500 bg-blue-50/50 p-4 rounded-r-lg shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          {isPrompted && (
            <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
          )}
          <div>
            <h4 className="font-medium text-gray-900">
              {isPrompted ? "Significant Change Detected" : "Add Reasoning"}
            </h4>
            <p className="text-sm text-gray-600">
              <span className={changeColor}>
                {columnName} {changeDirection} by{" "}
                {Math.abs(changePercent).toFixed(1)}%
              </span>
              <span className="text-gray-400 mx-2">|</span>
              <span>
                {originalValue.toLocaleString()} → {newValue.toLocaleString()}
              </span>
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0"
          onClick={onClose}
        >
          <X className="w-4 h-4" />
        </Button>
      </div>

      {/* Prompt message */}
      {isPrompted && (
        <div className="flex items-start gap-2 p-3 mb-3 bg-amber-50 border border-amber-200 rounded-lg">
          <Lightbulb className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <p className="text-sm text-amber-800">
            This change is more than 15% from the AI prediction. Adding a brief
            explanation helps improve future predictions. What influenced this
            adjustment?
          </p>
        </div>
      )}

      {/* Reasoning text area */}
      <div className="mb-3">
        <label
          htmlFor={`reasoning-${rowId}`}
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Why did you make this change? (Optional)
        </label>
        <Textarea
          id={`reasoning-${rowId}`}
          value={reasoningText}
          onChange={(e) => setReasoningText(e.target.value)}
          placeholder="e.g., Previous similar project took longer due to regulatory requirements..."
          className="min-h-[80px] text-sm"
        />
      </div>

      {/* Change tags */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          What factors influenced this change?
        </label>
        <ChangeTagSelector
          selectedTags={selectedTags}
          onTagToggle={handleTagToggle}
        />
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleDismiss}
          className="text-gray-500"
        >
          Skip this time
        </Button>
        <div className="flex items-center gap-2">
          {isDirty && (
            <span className="text-xs text-gray-500">Unsaved changes</span>
          )}
          <Button
            variant="default"
            size="sm"
            onClick={handleSave}
            disabled={!reasoningText.trim() && selectedTags.length === 0}
            className="gap-1"
          >
            <Check className="w-4 h-4" />
            Save Reasoning
          </Button>
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Inline Reasoning Toast (for non-blocking notification)
// ============================================================================

export interface ReasoningToastProps {
  rowId: string;
  changePercent: number;
  onOpen: () => void;
  onDismiss: () => void;
}

export const ReasoningToast = memo(function ReasoningToast({
  rowId,
  changePercent,
  onOpen,
  onDismiss,
}: ReasoningToastProps) {
  return (
    <div className="fixed bottom-4 right-4 z-50 animate-in slide-in-from-bottom-2">
      <div className="flex items-center gap-3 p-3 bg-white border border-gray-200 rounded-lg shadow-lg">
        <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
        <div className="text-sm">
          <p className="font-medium text-gray-900">
            {Math.abs(changePercent).toFixed(0)}% change detected
          </p>
          <p className="text-gray-500">Help improve AI predictions</p>
        </div>
        <div className="flex items-center gap-1 ml-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onDismiss}
            className="h-8 text-gray-500"
          >
            Later
          </Button>
          <Button variant="default" size="sm" onClick={onOpen} className="h-8">
            Add Note
          </Button>
        </div>
      </div>
    </div>
  );
});

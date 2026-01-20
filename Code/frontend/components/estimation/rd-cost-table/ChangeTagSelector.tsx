"use client";

/**
 * Change Tag Selector Component
 *
 * Multi-select tag picker for DPO reasoning categories.
 * 8 predefined tags matching common reasons for cost adjustments.
 */

import React, { memo } from "react";
import { cn } from "@/lib/utils";
import { type ChangeTag, CHANGE_TAGS } from "./types";
import {
  Zap,
  Shield,
  Target,
  Users,
  Clock,
  FileText,
  Scale,
  Link2,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface ChangeTagSelectorProps {
  selectedTags: ChangeTag[];
  onTagToggle: (tag: ChangeTag) => void;
  disabled?: boolean;
}

// Icon mapping for tags
const TAG_ICONS: Record<
  ChangeTag,
  React.ComponentType<{ className?: string }>
> = {
  complexity: Zap,
  riskBuffer: Shield,
  scopeChange: Target,
  resourceConstraint: Users,
  historicalKnowledge: Clock,
  customerRequirement: FileText,
  regulatory: Scale,
  integration: Link2,
};

// ============================================================================
// Change Tag Selector Component
// ============================================================================

export const ChangeTagSelector = memo(function ChangeTagSelector({
  selectedTags,
  onTagToggle,
  disabled = false,
}: ChangeTagSelectorProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {CHANGE_TAGS.map((tag) => {
        const Icon = TAG_ICONS[tag.value];
        const isSelected = selectedTags.includes(tag.value);

        return (
          <button
            key={tag.value}
            type="button"
            onClick={() => onTagToggle(tag.value)}
            disabled={disabled}
            className={cn(
              "inline-flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-full",
              "border transition-all duration-150",
              "focus:outline-none focus:ring-2 focus:ring-offset-1 focus:ring-blue-500",
              {
                // Selected state
                "bg-blue-100 border-blue-300 text-blue-800": isSelected,
                // Unselected state
                "bg-white border-gray-200 text-gray-600 hover:border-gray-300 hover:bg-gray-50":
                  !isSelected,
                // Disabled state
                "opacity-50 cursor-not-allowed": disabled,
              },
            )}
            title={tag.description}
          >
            <Icon className="w-3.5 h-3.5" />
            <span>{tag.label}</span>
            {isSelected && <span className="ml-0.5 text-blue-600">✓</span>}
          </button>
        );
      })}
    </div>
  );
});

// ============================================================================
// Compact Tag Display (for read-only views)
// ============================================================================

export interface ChangeTagDisplayProps {
  tags: ChangeTag[];
  maxVisible?: number;
}

export const ChangeTagDisplay = memo(function ChangeTagDisplay({
  tags,
  maxVisible = 3,
}: ChangeTagDisplayProps) {
  if (tags.length === 0) return null;

  const visibleTags = tags.slice(0, maxVisible);
  const hiddenCount = tags.length - maxVisible;

  return (
    <div className="flex flex-wrap gap-1">
      {visibleTags.map((tagValue) => {
        const tag = CHANGE_TAGS.find((t) => t.value === tagValue);
        if (!tag) return null;

        const Icon = TAG_ICONS[tagValue];

        return (
          <span
            key={tagValue}
            className="inline-flex items-center gap-1 px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded-full"
            title={tag.description}
          >
            <Icon className="w-3 h-3" />
            {tag.label}
          </span>
        );
      })}
      {hiddenCount > 0 && (
        <span className="inline-flex items-center px-2 py-0.5 text-xs bg-gray-100 text-gray-500 rounded-full">
          +{hiddenCount} more
        </span>
      )}
    </div>
  );
});

// ============================================================================
// Tag Badge (single tag display)
// ============================================================================

export interface TagBadgeProps {
  tag: ChangeTag;
  size?: "sm" | "md";
}

export const TagBadge = memo(function TagBadge({
  tag,
  size = "sm",
}: TagBadgeProps) {
  const tagInfo = CHANGE_TAGS.find((t) => t.value === tag);
  if (!tagInfo) return null;

  const Icon = TAG_ICONS[tag];

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 bg-blue-50 text-blue-700 rounded-full",
        {
          "px-2 py-0.5 text-xs": size === "sm",
          "px-3 py-1 text-sm": size === "md",
        },
      )}
      title={tagInfo.description}
    >
      <Icon className={size === "sm" ? "w-3 h-3" : "w-4 h-4"} />
      {tagInfo.label}
    </span>
  );
});

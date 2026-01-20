"use client";

/**
 * Version History Panel
 *
 * Slide-out panel showing version timeline with restore/fork capabilities.
 */

import React, { memo, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  History,
  GitBranch,
  Download,
  Eye,
  RotateCcw,
  Check,
  Clock,
  User,
} from "lucide-react";
import { type TableVersion, type VersionDiff } from "./types";

// ============================================================================
// Types
// ============================================================================

export interface VersionHistoryProps {
  isOpen: boolean;
  onClose: () => void;
  versions: TableVersion[];
  currentVersionId: string | null;
  onViewVersion: (version: TableVersion) => void;
  onRestoreVersion: (version: TableVersion) => void;
  onForkVersion: (version: TableVersion) => void;
  onDownloadVersion: (version: TableVersion, format: "xlsx" | "json") => void;
}

// ============================================================================
// Version History Panel
// ============================================================================

export const VersionHistory = memo(function VersionHistory({
  isOpen,
  onClose,
  versions,
  currentVersionId,
  onViewVersion,
  onRestoreVersion,
  onForkVersion,
  onDownloadVersion,
}: VersionHistoryProps) {
  const [selectedVersion, setSelectedVersion] = useState<TableVersion | null>(
    null,
  );
  const [viewMode, setViewMode] = useState<"list" | "diff">("list");

  const handleVersionClick = useCallback((version: TableVersion) => {
    setSelectedVersion(version);
    setViewMode("diff");
  }, []);

  const handleBackToList = useCallback(() => {
    setSelectedVersion(null);
    setViewMode("list");
  }, []);

  return (
    <Sheet open={isOpen} onOpenChange={(open: boolean) => !open && onClose()}>
      <SheetContent side="right" className="w-[400px] sm:w-[450px] p-0">
        <SheetHeader className="p-4 border-b">
          <div className="flex items-center justify-between">
            <SheetTitle className="flex items-center gap-2">
              <History className="w-5 h-5" />
              Version History
            </SheetTitle>
          </div>
        </SheetHeader>

        {viewMode === "list" ? (
          <ScrollArea className="h-[calc(100vh-80px)]">
            <div className="p-4">
              {versions.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No versions yet</p>
                  <p className="text-sm">Changes will appear here when saved</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {versions.map((version, index) => (
                    <VersionCard
                      key={version.id}
                      version={version}
                      isCurrent={version.id === currentVersionId}
                      isLatest={index === 0}
                      onClick={() => handleVersionClick(version)}
                      onView={() => onViewVersion(version)}
                      onRestore={() => onRestoreVersion(version)}
                      onFork={() => onForkVersion(version)}
                      onDownload={(format) =>
                        onDownloadVersion(version, format)
                      }
                    />
                  ))}
                </div>
              )}
            </div>
          </ScrollArea>
        ) : selectedVersion ? (
          <VersionDiffView
            version={selectedVersion}
            onBack={handleBackToList}
            onRestore={() => onRestoreVersion(selectedVersion)}
            onFork={() => onForkVersion(selectedVersion)}
          />
        ) : null}
      </SheetContent>
    </Sheet>
  );
});

// ============================================================================
// Version Card
// ============================================================================

interface VersionCardProps {
  version: TableVersion;
  isCurrent: boolean;
  isLatest: boolean;
  onClick: () => void;
  onView: () => void;
  onRestore: () => void;
  onFork: () => void;
  onDownload: (format: "xlsx" | "json") => void;
}

const VersionCard = memo(function VersionCard({
  version,
  isCurrent,
  isLatest,
  onClick,
  onView,
  onRestore,
  onFork,
  onDownload,
}: VersionCardProps) {
  const formattedDate = new Date(version.createdAt).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div
      className={cn(
        "border rounded-lg p-3 cursor-pointer transition-all",
        "hover:border-blue-300 hover:bg-blue-50/50",
        {
          "border-blue-500 bg-blue-50": isCurrent,
          "border-gray-200": !isCurrent,
          "border-green-500 bg-green-50": version.isFinalized,
        },
      )}
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-900">
            v{version.versionNumber}
          </span>
          {isLatest && (
            <span className="px-1.5 py-0.5 text-xs bg-blue-100 text-blue-700 rounded">
              Latest
            </span>
          )}
          {isCurrent && (
            <span className="px-1.5 py-0.5 text-xs bg-green-100 text-green-700 rounded">
              Current
            </span>
          )}
          {version.isFinalized && (
            <span className="px-1.5 py-0.5 text-xs bg-purple-100 text-purple-700 rounded flex items-center gap-1">
              <Check className="w-3 h-3" />
              Finalized
            </span>
          )}
        </div>
      </div>

      {/* Meta info */}
      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formattedDate}
        </span>
        {version.createdByName && (
          <span className="flex items-center gap-1">
            <User className="w-3 h-3" />
            {version.createdByName}
          </span>
        )}
        {version.changeCount > 0 && (
          <span className="text-amber-600">{version.changeCount} changes</span>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1 mt-2 pt-2 border-t border-gray-100">
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={(e) => {
            e.stopPropagation();
            onView();
          }}
        >
          <Eye className="w-3 h-3 mr-1" />
          View
        </Button>
        {!isCurrent && !version.isFinalized && (
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs"
            onClick={(e) => {
              e.stopPropagation();
              onRestore();
            }}
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            Restore
          </Button>
        )}
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={(e) => {
            e.stopPropagation();
            onFork();
          }}
        >
          <GitBranch className="w-3 h-3 mr-1" />
          Fork
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs"
          onClick={(e) => {
            e.stopPropagation();
            onDownload("xlsx");
          }}
        >
          <Download className="w-3 h-3 mr-1" />
          Excel
        </Button>
      </div>
    </div>
  );
});

// ============================================================================
// Version Diff View
// ============================================================================

interface VersionDiffViewProps {
  version: TableVersion;
  onBack: () => void;
  onRestore: () => void;
  onFork: () => void;
}

const VersionDiffView = memo(function VersionDiffView({
  version,
  onBack,
  onRestore,
  onFork,
}: VersionDiffViewProps) {
  return (
    <div className="flex flex-col h-[calc(100vh-80px)]">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <button
          onClick={onBack}
          className="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
        >
          ← Back to list
        </button>
        <span className="font-medium">Version {version.versionNumber}</span>
      </div>

      {/* Changes list */}
      <ScrollArea className="flex-1 p-4">
        <h4 className="font-medium text-gray-900 mb-3">
          Changes from Previous Version
        </h4>
        {version.changesFromPrevious &&
        version.changesFromPrevious.length > 0 ? (
          <div className="space-y-2">
            {version.changesFromPrevious.map((diff, index) => (
              <DiffItem key={index} diff={diff} />
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-sm">
            No changes recorded for this version
          </p>
        )}
      </ScrollArea>

      {/* Actions */}
      <div className="flex items-center gap-2 p-4 border-t bg-gray-50">
        <Button variant="outline" size="sm" onClick={onFork} className="flex-1">
          <GitBranch className="w-4 h-4 mr-1" />
          Fork from here
        </Button>
        {!version.isFinalized && (
          <Button
            variant="default"
            size="sm"
            onClick={onRestore}
            className="flex-1"
          >
            <RotateCcw className="w-4 h-4 mr-1" />
            Restore this version
          </Button>
        )}
      </div>
    </div>
  );
});

// ============================================================================
// Diff Item
// ============================================================================

interface DiffItemProps {
  diff: VersionDiff;
}

const DiffItem = memo(function DiffItem({ diff }: DiffItemProps) {
  const changeTypeColors = {
    added: "text-green-600 bg-green-50",
    modified: "text-amber-600 bg-amber-50",
    deleted: "text-red-600 bg-red-50",
  };

  return (
    <div
      className={cn("p-2 rounded text-sm", changeTypeColors[diff.changeType])}
    >
      <div className="flex items-center justify-between">
        <span className="font-medium">{diff.columnName}</span>
        <span className="text-xs uppercase">{diff.changeType}</span>
      </div>
      {diff.changeType === "modified" && (
        <div className="mt-1 text-xs">
          <span className="line-through opacity-60">
            {String(diff.oldValue)}
          </span>
          <span className="mx-2">→</span>
          <span className="font-medium">{String(diff.newValue)}</span>
        </div>
      )}
      {diff.changeType === "added" && (
        <div className="mt-1 text-xs font-medium">{String(diff.newValue)}</div>
      )}
      {diff.changeType === "deleted" && (
        <div className="mt-1 text-xs line-through opacity-60">
          {String(diff.oldValue)}
        </div>
      )}
    </div>
  );
});

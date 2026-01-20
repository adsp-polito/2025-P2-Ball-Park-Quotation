"use client";

import { useState, useCallback } from "react";
import { useTranslations } from "next-intl";
import {
  Edit2,
  Check,
  X,
  MessageSquare,
  AlertCircle,
  Save,
  Undo2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { BreakdownItem, QuotationData } from "./quotation-table";

export interface EditedItem {
  itemId: string;
  originalHours: number;
  newHours: number;
  reason: string;
}

interface EditableQuotationTableProps {
  data: QuotationData;
  onItemEdit: (itemId: string, newHours: number, reason: string) => void;
  onSaveAll: () => void;
  onResetAll: () => void;
  editedItems: EditedItem[];
  isSaving?: boolean;
  className?: string;
}

export function EditableQuotationTable({
  data,
  onItemEdit,
  onSaveAll,
  onResetAll,
  editedItems,
  isSaving = false,
  className,
}: EditableQuotationTableProps) {
  const _t = useTranslations("estimation");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState<number>(0);
  const [editReason, setEditReason] = useState<string>("");

  const startEdit = useCallback((item: BreakdownItem) => {
    setEditingId(item.id);
    setEditValue(item.hours);
    setEditReason("");
  }, []);

  const cancelEdit = useCallback(() => {
    setEditingId(null);
    setEditValue(0);
    setEditReason("");
  }, []);

  const confirmEdit = useCallback(
    (item: BreakdownItem) => {
      if (editValue !== item.hours && editReason.trim()) {
        onItemEdit(item.id, editValue, editReason);
      }
      setEditingId(null);
      setEditValue(0);
      setEditReason("");
    },
    [editValue, editReason, onItemEdit],
  );

  const getEditedItem = (itemId: string) => {
    return editedItems.find((e) => e.itemId === itemId);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat("en-US").format(num);
  };

  const totalOriginalHours = data.breakdown.reduce(
    (sum, item) => sum + item.hours,
    0,
  );

  const totalEditedHours = data.breakdown.reduce((sum, item) => {
    const edited = getEditedItem(item.id);
    return sum + (edited ? edited.newHours : item.hours);
  }, 0);

  const hoursDifference = totalEditedHours - totalOriginalHours;
  const percentChange = ((hoursDifference / totalOriginalHours) * 100).toFixed(
    1,
  );

  return (
    <div className={cn("space-y-4", className)}>
      {/* Summary */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <div>
                <p className="text-sm text-muted-foreground">Original Total</p>
                <p className="text-xl font-bold">
                  {formatNumber(totalOriginalHours)} hrs
                </p>
              </div>
              {editedItems.length > 0 && (
                <>
                  <div className="text-2xl text-muted-foreground">→</div>
                  <div>
                    <p className="text-sm text-muted-foreground">New Total</p>
                    <p className="text-xl font-bold text-primary">
                      {formatNumber(totalEditedHours)} hrs
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Difference</p>
                    <p
                      className={cn(
                        "text-xl font-bold",
                        hoursDifference > 0 && "text-red-500",
                        hoursDifference < 0 && "text-green-500",
                      )}
                    >
                      {hoursDifference > 0 ? "+" : ""}
                      {formatNumber(hoursDifference)} ({percentChange}%)
                    </p>
                  </div>
                </>
              )}
            </div>

            <div className="flex items-center gap-2">
              {editedItems.length > 0 && (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onResetAll}
                    className="gap-1"
                  >
                    <Undo2 className="h-4 w-4" />
                    Reset All
                  </Button>
                  <Button
                    size="sm"
                    onClick={onSaveAll}
                    disabled={isSaving}
                    className="gap-1"
                  >
                    <Save className="h-4 w-4" />
                    {isSaving
                      ? "Saving..."
                      : `Save ${editedItems.length} Changes`}
                  </Button>
                </>
              )}
            </div>
          </div>

          {editedItems.length > 0 && (
            <div className="mt-4 rounded-lg bg-yellow-50 p-3 dark:bg-yellow-900/20">
              <div className="flex items-center gap-2 text-sm text-yellow-700 dark:text-yellow-400">
                <AlertCircle className="h-4 w-4" />
                <span>
                  You have {editedItems.length} pending changes. Save to apply
                  them and train the system.
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Editable Table */}
      <Card>
        <CardHeader>
          <CardTitle>Review & Adjust Estimates</CardTitle>
          <CardDescription>
            Click on any row to adjust the hours. Provide a reason to help the
            system learn from your corrections.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left">
                  <th className="pb-3 font-medium">Activity</th>
                  <th className="pb-3 text-right font-medium">Original</th>
                  <th className="pb-3 text-right font-medium">Adjusted</th>
                  <th className="pb-3 font-medium">Reason</th>
                  <th className="pb-3 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.breakdown.map((item) => {
                  const isEditing = editingId === item.id;
                  const editedItem = getEditedItem(item.id);
                  const hasChange = !!editedItem;

                  return (
                    <tr
                      key={item.id}
                      className={cn(
                        "border-b transition-colors",
                        hasChange && "bg-yellow-50 dark:bg-yellow-900/10",
                        isEditing && "bg-primary/5",
                      )}
                    >
                      <td className="py-3">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">
                            {item.activityCode}
                          </span>
                          <span className="font-medium">
                            {item.activityName}
                          </span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {item.category}
                        </span>
                      </td>

                      <td className="py-3 text-right">
                        <span
                          className={cn(
                            hasChange && "text-muted-foreground line-through",
                          )}
                        >
                          {formatNumber(item.hours)}
                        </span>
                      </td>

                      <td className="py-3 text-right">
                        {isEditing ? (
                          <Input
                            type="number"
                            value={editValue}
                            onChange={(e) =>
                              setEditValue(Number(e.target.value))
                            }
                            className="w-24 text-right"
                            autoFocus
                          />
                        ) : hasChange ? (
                          <span className="font-bold text-primary">
                            {formatNumber(editedItem.newHours)}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </td>

                      <td className="py-3">
                        {isEditing ? (
                          <Input
                            value={editReason}
                            onChange={(e) => setEditReason(e.target.value)}
                            placeholder="Why are you making this change?"
                            className="w-full"
                          />
                        ) : hasChange ? (
                          <div className="flex items-center gap-1 text-sm text-muted-foreground">
                            <MessageSquare className="h-3 w-3" />
                            {editedItem.reason}
                          </div>
                        ) : null}
                      </td>

                      <td className="py-3 text-right">
                        {isEditing ? (
                          <div className="flex items-center justify-end gap-1">
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-green-600 hover:text-green-700"
                              onClick={() => confirmEdit(item)}
                              disabled={
                                editValue === item.hours || !editReason.trim()
                              }
                            >
                              <Check className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-8 w-8 text-red-600 hover:text-red-700"
                              onClick={cancelEdit}
                            >
                              <X className="h-4 w-4" />
                            </Button>
                          </div>
                        ) : (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => startEdit(item)}
                          >
                            <Edit2 className="h-4 w-4" />
                          </Button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Changes Summary */}
      {editedItems.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Change Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {editedItems.map((edit) => {
                const item = data.breakdown.find((i) => i.id === edit.itemId);
                const diff = edit.newHours - edit.originalHours;
                const pct = ((diff / edit.originalHours) * 100).toFixed(1);

                return (
                  <div
                    key={edit.itemId}
                    className="flex items-center justify-between rounded-lg bg-muted/50 p-3"
                  >
                    <div>
                      <p className="font-medium">{item?.activityName}</p>
                      <p className="text-sm text-muted-foreground">
                        {edit.reason}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">
                        {formatNumber(edit.originalHours)} →{" "}
                        {formatNumber(edit.newHours)} hrs
                      </p>
                      <p
                        className={cn(
                          "text-sm",
                          diff > 0 && "text-red-500",
                          diff < 0 && "text-green-500",
                        )}
                      >
                        {diff > 0 ? "+" : ""}
                        {formatNumber(diff)} ({pct}%)
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Feedback reason selector
export function FeedbackReasonSelector({
  onSelect,
  className,
}: {
  onSelect: (reason: string) => void;
  className?: string;
}) {
  const commonReasons = [
    "Historical data suggests different estimate",
    "Complexity not captured by model",
    "Missing regulatory requirements",
    "Team availability constraints",
    "Similar project had different outcome",
    "Customer-specific requirements",
    "Technology change not reflected",
    "Risk factors not considered",
  ];

  return (
    <div className={cn("space-y-2", className)}>
      <p className="text-xs font-medium text-muted-foreground">
        Common reasons:
      </p>
      <div className="flex flex-wrap gap-2">
        {commonReasons.map((reason) => (
          <Button
            key={reason}
            variant="outline"
            size="sm"
            className="h-auto py-1 text-xs"
            onClick={() => onSelect(reason)}
          >
            {reason}
          </Button>
        ))}
      </div>
    </div>
  );
}

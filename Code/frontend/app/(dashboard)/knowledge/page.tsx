"use client";

import { useState, useEffect, useMemo, useCallback } from "react";
import {
  Search,
  Plus,
  BookOpen,
  FileText,
  Edit2,
  Trash2,
  Upload,
  ChevronRight,
  AlertCircle,
  Loader2,
  X,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  knowledgeApi,
  type Acronym,
  type AcronymCreate,
  type KnowledgeDocument,
} from "@/lib/api";

export default function KnowledgePage() {
  const [activeTab, setActiveTab] = useState<"acronyms" | "documents">(
    "acronyms",
  );
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedDomain, setSelectedDomain] = useState<string | null>(null);

  // Acronyms state
  const [acronyms, setAcronyms] = useState<Acronym[]>([]);
  const [acronymsLoading, setAcronymsLoading] = useState(true);
  const [acronymsError, setAcronymsError] = useState<string | null>(null);

  // Documents state
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  // Dialog states
  const [isAddAcronymOpen, setIsAddAcronymOpen] = useState(false);
  const [isEditAcronymOpen, setIsEditAcronymOpen] = useState(false);
  const [isDeleteAcronymOpen, setIsDeleteAcronymOpen] = useState(false);
  const [selectedAcronym, setSelectedAcronym] = useState<Acronym | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Form state for add/edit acronym
  const [acronymForm, setAcronymForm] = useState<AcronymCreate>({
    acronym: "",
    full_form: "",
    description: "",
    category: "general",
  });

  // Fetch acronyms
  const fetchAcronyms = useCallback(async () => {
    setAcronymsLoading(true);
    setAcronymsError(null);
    try {
      const params: { search?: string; domain?: string } = {};
      if (searchQuery && activeTab === "acronyms") {
        params.search = searchQuery;
      }
      if (selectedDomain) {
        params.domain = selectedDomain;
      }
      const response = await knowledgeApi.listAcronyms(params);
      setAcronyms(response.items);
    } catch (error) {
      console.error("Failed to fetch acronyms:", error);
      setAcronymsError("Failed to load acronyms");
    } finally {
      setAcronymsLoading(false);
    }
  }, [searchQuery, selectedDomain, activeTab]);

  // Fetch documents
  const fetchDocuments = useCallback(async () => {
    setDocumentsLoading(true);
    setDocumentsError(null);
    try {
      const params: { search?: string } = {};
      if (searchQuery && activeTab === "documents") {
        params.search = searchQuery;
      }
      const response = await knowledgeApi.listDocuments(params);
      setDocuments(response.items);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
      setDocumentsError("Failed to load documents");
    } finally {
      setDocumentsLoading(false);
    }
  }, [searchQuery, activeTab]);

  // Initial load and search effect
  useEffect(() => {
    if (activeTab === "acronyms") {
      fetchAcronyms();
    } else {
      fetchDocuments();
    }
  }, [activeTab, fetchAcronyms, fetchDocuments]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (activeTab === "acronyms") {
        fetchAcronyms();
      } else {
        fetchDocuments();
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchQuery, activeTab, fetchAcronyms, fetchDocuments]);

  // Get unique categories for filtering
  const categories = useMemo(() => {
    const categorySet = new Set(acronyms.map((a) => a.category || "general"));
    return Array.from(categorySet).sort();
  }, [acronyms]);

  // Filtered acronyms (client-side filtering for category when not using API filter)
  const filteredAcronyms = useMemo(() => {
    let result = [...acronyms];
    if (selectedDomain) {
      result = result.filter(
        (a) => (a.category || "general") === selectedDomain,
      );
    }
    return result.sort((a, b) => a.acronym.localeCompare(b.acronym));
  }, [acronyms, selectedDomain]);

  // Filtered documents
  const filteredDocuments = useMemo(() => {
    return documents;
  }, [documents]);

  // Handle add acronym
  const handleAddAcronym = async () => {
    setIsSubmitting(true);
    try {
      await knowledgeApi.createAcronym(acronymForm);
      setIsAddAcronymOpen(false);
      setAcronymForm({
        acronym: "",
        full_form: "",
        description: "",
        category: "general",
      });
      fetchAcronyms();
    } catch (error: unknown) {
      const err = error as { detail?: string };
      alert(err.detail || "Failed to create acronym");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle edit acronym
  const handleEditAcronym = async () => {
    if (!selectedAcronym) return;
    setIsSubmitting(true);
    try {
      await knowledgeApi.updateAcronym(selectedAcronym.id, {
        acronym: acronymForm.acronym,
        full_form: acronymForm.full_form,
        description: acronymForm.description,
        category: acronymForm.category,
      });
      setIsEditAcronymOpen(false);
      setSelectedAcronym(null);
      fetchAcronyms();
    } catch (error: unknown) {
      const err = error as { detail?: string };
      alert(err.detail || "Failed to update acronym");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle delete acronym
  const handleDeleteAcronym = async () => {
    if (!selectedAcronym) return;
    setIsSubmitting(true);
    try {
      await knowledgeApi.deleteAcronym(selectedAcronym.id);
      setIsDeleteAcronymOpen(false);
      setSelectedAcronym(null);
      fetchAcronyms();
    } catch (error: unknown) {
      const err = error as { detail?: string };
      alert(err.detail || "Failed to delete acronym");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle document upload
  const handleDocumentUpload = async (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      await knowledgeApi.uploadDocument(file);
      fetchDocuments();
    } catch (error: unknown) {
      const err = error as { detail?: string };
      alert(err.detail || "Failed to upload document");
    }
  };

  // Handle delete document
  const handleDeleteDocument = async (docId: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return;
    try {
      await knowledgeApi.deleteDocument(docId);
      fetchDocuments();
    } catch (error: unknown) {
      const err = error as { detail?: string };
      alert(err.detail || "Failed to delete document");
    }
  };

  // Open edit dialog
  const openEditDialog = (acronym: Acronym) => {
    setSelectedAcronym(acronym);
    setAcronymForm({
      acronym: acronym.acronym,
      full_form: acronym.full_form,
      description: acronym.description || "",
      category: acronym.category || "general",
    });
    setIsEditAcronymOpen(true);
  };

  // Open delete dialog
  const openDeleteDialog = (acronym: Acronym) => {
    setSelectedAcronym(acronym);
    setIsDeleteAcronymOpen(true);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Knowledge Base</h1>
          <p className="mt-1 text-muted-foreground">
            Manage acronyms, definitions, and reference documents
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="icon"
            onClick={() =>
              activeTab === "acronyms" ? fetchAcronyms() : fetchDocuments()
            }
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          {activeTab === "acronyms" ? (
            <Button
              className="gap-2"
              onClick={() => {
                setAcronymForm({
                  acronym: "",
                  full_form: "",
                  description: "",
                  category: "general",
                });
                setIsAddAcronymOpen(true);
              }}
            >
              <Plus className="h-4 w-4" />
              Add Acronym
            </Button>
          ) : (
            <label>
              <Button className="gap-2" asChild>
                <span>
                  <Upload className="h-4 w-4" />
                  Upload Document
                </span>
              </Button>
              <input
                type="file"
                className="hidden"
                accept=".txt,.pdf,.docx,.md"
                onChange={handleDocumentUpload}
              />
            </label>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b">
        <Button
          variant="ghost"
          className={cn(
            "rounded-none border-b-2 border-transparent px-4",
            activeTab === "acronyms" && "border-primary text-primary",
          )}
          onClick={() => setActiveTab("acronyms")}
        >
          <BookOpen className="mr-2 h-4 w-4" />
          Acronyms
          <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-xs">
            {acronyms.length}
          </span>
        </Button>
        <Button
          variant="ghost"
          className={cn(
            "rounded-none border-b-2 border-transparent px-4",
            activeTab === "documents" && "border-primary text-primary",
          )}
          onClick={() => setActiveTab("documents")}
        >
          <FileText className="mr-2 h-4 w-4" />
          Documents
          <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-xs">
            {documents.length}
          </span>
        </Button>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder={
            activeTab === "acronyms"
              ? "Search acronyms..."
              : "Search documents..."
          }
          className="pl-9"
        />
        {searchQuery && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-1 top-1/2 h-7 w-7 -translate-y-1/2"
            onClick={() => setSearchQuery("")}
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Content */}
      {activeTab === "acronyms" ? (
        <div className="grid gap-6 lg:grid-cols-4">
          {/* Category Filter */}
          <Card className="h-fit">
            <CardHeader>
              <CardTitle className="text-base">Categories</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1">
              <Button
                variant={selectedDomain === null ? "secondary" : "ghost"}
                className="w-full justify-start"
                onClick={() => setSelectedDomain(null)}
              >
                All
                <span className="ml-auto text-xs text-muted-foreground">
                  {acronyms.length}
                </span>
              </Button>
              {categories.map((category) => (
                <Button
                  key={category}
                  variant={selectedDomain === category ? "secondary" : "ghost"}
                  className="w-full justify-start"
                  onClick={() => setSelectedDomain(category)}
                >
                  {category}
                  <span className="ml-auto text-xs text-muted-foreground">
                    {
                      acronyms.filter(
                        (a) => (a.category || "general") === category,
                      ).length
                    }
                  </span>
                </Button>
              ))}
            </CardContent>
          </Card>

          {/* Acronyms List */}
          <div className="lg:col-span-3">
            {acronymsLoading ? (
              <Card>
                <CardContent className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </CardContent>
              </Card>
            ) : acronymsError ? (
              <Card>
                <CardContent className="py-12 text-center">
                  <AlertCircle className="mx-auto h-12 w-12 text-destructive" />
                  <p className="mt-4 font-medium text-destructive">
                    {acronymsError}
                  </p>
                  <Button
                    variant="outline"
                    className="mt-4"
                    onClick={fetchAcronyms}
                  >
                    Try Again
                  </Button>
                </CardContent>
              </Card>
            ) : filteredAcronyms.length === 0 ? (
              <Card>
                <CardContent className="py-12 text-center">
                  <BookOpen className="mx-auto h-12 w-12 text-muted-foreground/50" />
                  <p className="mt-4 font-medium">No acronyms found</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {searchQuery
                      ? "Try adjusting your search"
                      : "Add your first acronym to get started"}
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-2">
                {filteredAcronyms.map((acronym) => (
                  <Card
                    key={acronym.id}
                    className="transition-shadow hover:shadow-md"
                  >
                    <CardContent className="flex items-center justify-between py-4">
                      <div className="flex items-center gap-4">
                        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 font-mono text-sm font-bold text-primary">
                          {acronym.acronym}
                        </div>
                        <div>
                          <p className="font-medium">{acronym.full_form}</p>
                          {acronym.description && (
                            <p className="text-sm text-muted-foreground">
                              {acronym.description}
                            </p>
                          )}
                          <span className="mt-1 inline-block rounded-full bg-muted px-2 py-0.5 text-xs">
                            {acronym.category || "general"}
                          </span>
                        </div>
                      </div>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => openEditDialog(acronym)}
                        >
                          <Edit2 className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="text-destructive"
                          onClick={() => openDeleteDialog(acronym)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Upload Area */}
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-8">
              <Upload className="h-10 w-10 text-muted-foreground" />
              <p className="mt-4 font-medium">Upload a document</p>
              <p className="mt-1 text-sm text-muted-foreground">
                TXT, PDF, DOCX, or MD files
              </p>
              <label>
                <Button variant="outline" className="mt-4" asChild>
                  <span>Select File</span>
                </Button>
                <input
                  type="file"
                  className="hidden"
                  accept=".txt,.pdf,.docx,.md"
                  onChange={handleDocumentUpload}
                />
              </label>
            </CardContent>
          </Card>

          {/* Documents List */}
          {documentsLoading ? (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </CardContent>
            </Card>
          ) : documentsError ? (
            <Card>
              <CardContent className="py-12 text-center">
                <AlertCircle className="mx-auto h-12 w-12 text-destructive" />
                <p className="mt-4 font-medium text-destructive">
                  {documentsError}
                </p>
                <Button
                  variant="outline"
                  className="mt-4"
                  onClick={fetchDocuments}
                >
                  Try Again
                </Button>
              </CardContent>
            </Card>
          ) : filteredDocuments.length === 0 ? (
            <Card>
              <CardContent className="py-12 text-center">
                <FileText className="mx-auto h-12 w-12 text-muted-foreground/50" />
                <p className="mt-4 font-medium">No documents found</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Upload a document to get started
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-2">
              {filteredDocuments.map((doc) => (
                <Card
                  key={doc.id}
                  className="transition-shadow hover:shadow-md"
                >
                  <CardContent className="flex items-center justify-between py-4">
                    <div className="flex items-center gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
                        <FileText className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="font-medium">{doc.title}</p>
                        <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
                          <span>{doc.doc_type || "general"}</span>
                          <span>-</span>
                          <span>{doc.chunk_count ?? 0} chunks</span>
                          <span>-</span>
                          <span>
                            {new Date(doc.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {doc.is_indexed ? (
                        <span className="rounded-full bg-green-100 px-2 py-1 text-xs text-green-700 dark:bg-green-900/30 dark:text-green-400">
                          Indexed
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 rounded-full bg-yellow-100 px-2 py-1 text-xs text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400">
                          <AlertCircle className="h-3 w-3" />
                          Pending
                        </span>
                      )}
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-destructive"
                        onClick={() => handleDeleteDocument(doc.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Add Acronym Dialog */}
      <Dialog open={isAddAcronymOpen} onOpenChange={setIsAddAcronymOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Acronym</DialogTitle>
            <DialogDescription>
              Add a new acronym to the knowledge base.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="acronym">Acronym</Label>
              <Input
                id="acronym"
                placeholder="e.g., PR"
                value={acronymForm.acronym}
                onChange={(e) =>
                  setAcronymForm({
                    ...acronymForm,
                    acronym: e.target.value.toUpperCase(),
                  })
                }
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="full_form">Full Form</Label>
              <Input
                id="full_form"
                placeholder="e.g., Product Request"
                value={acronymForm.full_form}
                onChange={(e) =>
                  setAcronymForm({ ...acronymForm, full_form: e.target.value })
                }
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Textarea
                id="description"
                placeholder="Brief description or context..."
                value={acronymForm.description}
                onChange={(e) =>
                  setAcronymForm({
                    ...acronymForm,
                    description: e.target.value,
                  })
                }
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="category">Category</Label>
              <Input
                id="category"
                placeholder="e.g., general, certification, costs"
                value={acronymForm.category}
                onChange={(e) =>
                  setAcronymForm({ ...acronymForm, category: e.target.value })
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsAddAcronymOpen(false)}
            >
              Cancel
            </Button>
            <Button
              onClick={handleAddAcronym}
              disabled={
                isSubmitting || !acronymForm.acronym || !acronymForm.full_form
              }
            >
              {isSubmitting && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Add Acronym
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Acronym Dialog */}
      <Dialog open={isEditAcronymOpen} onOpenChange={setIsEditAcronymOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit Acronym</DialogTitle>
            <DialogDescription>Update the acronym details.</DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="edit-acronym">Acronym</Label>
              <Input
                id="edit-acronym"
                value={acronymForm.acronym}
                onChange={(e) =>
                  setAcronymForm({
                    ...acronymForm,
                    acronym: e.target.value.toUpperCase(),
                  })
                }
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-full_form">Full Form</Label>
              <Input
                id="edit-full_form"
                value={acronymForm.full_form}
                onChange={(e) =>
                  setAcronymForm({ ...acronymForm, full_form: e.target.value })
                }
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-description">Description</Label>
              <Textarea
                id="edit-description"
                value={acronymForm.description}
                onChange={(e) =>
                  setAcronymForm({
                    ...acronymForm,
                    description: e.target.value,
                  })
                }
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="edit-category">Category</Label>
              <Input
                id="edit-category"
                value={acronymForm.category}
                onChange={(e) =>
                  setAcronymForm({ ...acronymForm, category: e.target.value })
                }
              />
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsEditAcronymOpen(false)}
            >
              Cancel
            </Button>
            <Button onClick={handleEditAcronym} disabled={isSubmitting}>
              {isSubmitting && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Save Changes
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog
        open={isDeleteAcronymOpen}
        onOpenChange={setIsDeleteAcronymOpen}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Acronym</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{selectedAcronym?.acronym}
              &quot;? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteAcronym}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isSubmitting && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

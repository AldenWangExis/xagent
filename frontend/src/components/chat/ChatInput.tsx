import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Paperclip, X, File as FileIcon, Sparkles, Pause, Play, Loader2, ArrowUp, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn, getApiUrl } from "@/lib/utils";
import { useI18n } from "@/contexts/i18n-context";
import { ConfigDialog } from "@/components/config-dialog";
import { apiRequest } from "@/lib/api-wrapper";
import { toast } from "sonner";
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

interface FileItem {
  filename: string;
  file_size: number;
  modified_time: number;
  file_type?: string;
  workspace_id?: string;
  relative_path?: string;
}

interface ChatInputProps {
  onSend: (message: string, config?: any) => void | Promise<void>;
  isLoading?: boolean;
  files?: File[];
  onFilesChange?: (files: File[]) => void;
  showModeToggle?: boolean;
  mode?: "task" | "process";
  onModeChange?: (mode: "task" | "process") => void;
  inputValue?: string;
  onInputChange?: (value: string) => void;
  taskStatus?: "pending" | "running" | "completed" | "failed" | "paused";
  onPause?: () => void;
  onResume?: () => void;
  taskConfig?: {
    model?: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
  };
  hideConfig?: boolean;
  readOnlyConfig?: boolean;
}

export function ChatInput({
  onSend,
  isLoading,
  files = [],
  onFilesChange,
  inputValue,
  onInputChange,
  taskStatus,
  onPause,
  onResume,
  taskConfig,
  hideConfig = false,
  readOnlyConfig = false
}: ChatInputProps) {
  const router = useRouter();
  const [internalMessage, setInternalMessage] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [showNoModelAlert, setShowNoModelAlert] = useState(false);

  // File picker state
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [fileList, setFileList] = useState<FileItem[]>([]);
  const [filteredFiles, setFilteredFiles] = useState<FileItem[]>([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [triggerIndex, setTriggerIndex] = useState<number>(-1);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [downloadingFile, setDownloadingFile] = useState<string | null>(null);

  // Track files for async operations
  const filesRef = useRef(files);
  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  // Determine if controlled or uncontrolled
  const isControlled = inputValue !== undefined;
  const message = isControlled ? inputValue : internalMessage;

  const handleMessageChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const cursor = e.target.selectionStart;

    if (isControlled) {
      onInputChange?.(newValue);
    } else {
      setInternalMessage(newValue);
    }

    checkTrigger(newValue, cursor);
  };

  const fetchFiles = async () => {
    if (fileList.length > 0) return;
    setIsLoadingFiles(true);
    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/list`);
      if (response.ok) {
        const data = await response.json();
        if (data && data.files) {
          setFileList(data.files);
        }
      }
    } catch (error) {
      console.error(t("files.previewDialog.errors.loadFailed"), error);
      toast.error(t("files.previewDialog.errors.loadFailed"));
    } finally {
      setIsLoadingFiles(false);
    }
  };

  const checkTrigger = (text: string, cursor: number) => {
    const textBeforeCursor = text.slice(0, cursor);
    const lastHashIndex = textBeforeCursor.lastIndexOf("#");

    if (lastHashIndex !== -1) {
      // Allow trigger anywhere
      const query = textBeforeCursor.slice(lastHashIndex + 1);
      if (!query.includes(" ") && !query.includes("\n")) {
        setTriggerIndex(lastHashIndex);
        setShowFilePicker(true);
        fetchFiles();

        // Filter files
        const lowerQuery = query.toLowerCase();
        const filtered = fileList.filter(f =>
          (f.filename.toLowerCase().includes(lowerQuery) ||
           (f.relative_path && f.relative_path.toLowerCase().includes(lowerQuery)))
        );
        setFilteredFiles(filtered);
        setSelectedFileIndex(0);
        return;
      }
    }
    setShowFilePicker(false);
    setTriggerIndex(-1);
  };

  // Update filtered files when fileList changes (e.g. after fetch)
  useEffect(() => {
    if (showFilePicker && fileList.length > 0 && triggerIndex !== -1) {
       if (message.length > triggerIndex) {
         const query = message.slice(triggerIndex + 1).split(/[\s\n]/)[0];
         const lowerQuery = query.toLowerCase();
         const filtered = fileList.filter(f =>
            (f.filename.toLowerCase().includes(lowerQuery) ||
             (f.relative_path && f.relative_path.toLowerCase().includes(lowerQuery)))
          );
          setFilteredFiles(filtered);
       }
    }
  }, [fileList, showFilePicker, triggerIndex, message]);

  const insertFile = async (file: FileItem) => {
    const fileId = file.relative_path || file.filename;
    if (downloadingFile) return;

    setDownloadingFile(fileId);

    try {
      const filePath = file.relative_path || file.filename;
      const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(filePath)}`);

      if (response.ok) {
        const blob = await response.blob();

        const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
        if (blob.size > MAX_FILE_SIZE) {
          toast.error(t("files.fileTooLarge"));
          return;
        }

        const newFile = new File([blob], file.filename, {
          type: file.file_type || blob.type || 'application/octet-stream',
          lastModified: file.modified_time * 1000
        });

        if (onFilesChange) {
          // Use ref to get latest files to avoid stale closure
          onFilesChange([...filesRef.current, newFile]);
        }

        // Success: Remove query from text and close picker
        const currentText = textareaRef.current?.value || (isControlled ? inputValue || "" : internalMessage);

        if (triggerIndex !== -1) {
            let endIndex = currentText.indexOf(" ", triggerIndex);
            if (endIndex === -1) endIndex = currentText.indexOf("\n", triggerIndex);
            if (endIndex === -1) endIndex = currentText.length;

            const prefix = currentText.slice(0, triggerIndex);
            const suffix = currentText.slice(endIndex);
            const newText = prefix + suffix;

            if (isControlled) {
                onInputChange?.(newText);
            } else {
                setInternalMessage(newText);
            }
        }

        setShowFilePicker(false);
        setTriggerIndex(-1);
      } else {
        console.error("Failed to download file:", response.statusText);
        toast.error(t("files.downloadFailed") || "Failed to download file");
      }
    } catch (error) {
      console.error("Error fetching file:", error);
      toast.error(t("files.downloadFailed") || "Failed to download file");
    } finally {
      setDownloadingFile(null);
      // Restore focus
      setTimeout(() => textareaRef.current?.focus(), 0);
    }
  };


  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isSubmittingRef = useRef(false);
  const { t } = useI18n();
  const [agentConfig, setAgentConfig] = useState<{
    model: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
    memorySimilarityThreshold?: number;
  }>({ model: "", memorySimilarityThreshold: 1.5 });
  const [models, setModels] = useState<any[]>([]);

  // Fetch default models on mount
  useEffect(() => {
    const fetchDefaultModels = async () => {
      try {
        const apiUrl = getApiUrl();

        // Fetch all models first to have the list for display names
        const modelsResponse = await apiRequest(`${apiUrl}/api/models/?category=llm`, {
            headers: {}
        });

        let allModels: any[] = [];
        if (modelsResponse.ok) {
            allModels = await modelsResponse.json();
            if (Array.isArray(allModels)) {
                setModels(allModels);
            }
        }

        // Fetch user default models
        const defaultResponse = await apiRequest(`${apiUrl}/api/models/user-default`, {
          headers: {}
        });

        let defaultModels: Record<string, any> = {};
        if (defaultResponse.ok) {
          const defaults = await defaultResponse.json();
          if (Array.isArray(defaults)) {
            defaults.forEach((defaultConfig: any) => {
              if (defaultConfig && defaultConfig.config_type && defaultConfig.model) {
                defaultModels[defaultConfig.config_type] = defaultConfig.model;
              }
            });
          }
        }

        // Find default if no user preference
        if (!defaultModels.general && allModels.length > 0) {
            const defaultModel = allModels.find((m: any) => m.is_default) || allModels[0];
            if (defaultModel) {
            defaultModels.general = { model_id: defaultModel.model_id };
            }
        }

        setAgentConfig(prev => ({
          ...prev,
          model: prev.model || defaultModels.general?.model_id || "",
          smallFastModel: prev.smallFastModel || defaultModels.small_fast?.model_id,
          visualModel: prev.visualModel || defaultModels.visual?.model_id,
          compactModel: prev.compactModel || defaultModels.compact?.model_id
        }));
      } catch (error) {
        console.error('Failed to fetch default models:', error);
      }
    };

    fetchDefaultModels();
  }, []);

  // Update config when taskConfig changes
  useEffect(() => {
    if (taskConfig) {
      setAgentConfig(prev => ({
        ...prev,
        model: taskConfig.model || prev.model,
        smallFastModel: taskConfig.smallFastModel || prev.smallFastModel,
        visualModel: taskConfig.visualModel || prev.visualModel,
        compactModel: taskConfig.compactModel || prev.compactModel
      }));
    }
  }, [taskConfig]);

  const handleConfigChange = (config: {
    model: string;
    smallFastModel?: string;
    visualModel?: string;
    compactModel?: string;
    memorySimilarityThreshold?: number;
  }) => {
    setAgentConfig(config);
  };

  const canSubmit = () => {
    const hasText = message.trim().length > 0;
    const hasFiles = files.length > 0;
    return (hasText || hasFiles) && !isLoading;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!canSubmit() || isSubmittingRef.current) return;

    if (!agentConfig.model) {
      setShowNoModelAlert(true);
      return;
    }

    try {
      isSubmittingRef.current = true;
      const trimmed = message.trim();
      const messageToSend = trimmed;

      // Always send task mode config
      const configToSend = { ...agentConfig, vibeMode: { mode: "task" } };

      await onSend(messageToSend, configToSend);

      if (isControlled) {
        onInputChange?.("");
      } else {
        setInternalMessage("");
      }
    } finally {
      // Small delay to prevent double submission
      setTimeout(() => {
        isSubmittingRef.current = false;
      }, 500);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showFilePicker) {
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedFileIndex(prev => Math.max(0, prev - 1));
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedFileIndex(prev => Math.min(filteredFiles.length - 1, prev + 1));
        return;
      }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        if (filteredFiles.length > 0) {
          insertFile(filteredFiles[selectedFileIndex]);
        }
        return;
      }
      if (e.key === "Escape") {
        e.preventDefault();
        setShowFilePicker(false);
        return;
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      // Prevent triggering submit when using IME (e.g., Chinese input method)
      if (e.nativeEvent.isComposing) {
        return;
      }
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const handlePaste = (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
    const items = Array.from(e.clipboardData.items || []);
    const fileItems = items.filter(item => item.kind === 'file');

    if (fileItems.length > 0) {
      e.preventDefault();
      const pastedFiles: File[] = [];

      fileItems.forEach((item, index) => {
        const file = item.getAsFile();
        if (file) {
          const hasName = typeof (file as any).name === 'string' && (file as any).name.length > 0;
          // Handle default "image.png" name which causes conflicts when pasting multiple images
          if (hasName && file.name !== 'image.png') {
            pastedFiles.push(file);
          } else {
            const timestamp = Date.now();
            const mime = item.type || file.type || 'application/octet-stream';
            const ext = mime.split('/')[1] || 'bin';
            // If it was image.png, preserve extension but make unique. Otherwise default to pasted-file
            const baseName = file.name === 'image.png'
              ? `image-${timestamp}-${index}`
              : `pasted-file-${timestamp}-${index}`;

            const namedFile = new File([file], `${baseName}.${ext}`, {
              type: mime,
              lastModified: timestamp,
            });
            pastedFiles.push(namedFile);
          }
        }
      });

      if (pastedFiles.length > 0) {
        onFilesChange?.([...files, ...pastedFiles]);
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    onFilesChange?.([...files, ...selectedFiles]);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const removeFile = (index: number) => {
    onFilesChange?.(files.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className="space-y-3">
      {/* File list */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-3 py-2 bg-secondary/80 rounded-xl text-sm animate-fade-in-scale border border-border/50 hover:border-primary/30 transition-colors"
            >
              <div className="w-6 h-6 rounded-lg bg-primary/10 flex items-center justify-center">
                <FileIcon className="w-3.5 h-3.5 text-primary" />
              </div>
              <span className="truncate max-w-[120px] font-medium text-foreground/90">{file.name}</span>
              <span className="text-xs text-muted-foreground">
                {formatFileSize(file.size)}
              </span>
              <button
                type="button"
                onClick={() => removeFile(index)}
                className="ml-1 text-muted-foreground hover:text-destructive transition-colors p-0.5 rounded-md hover:bg-destructive/10"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input area */}
      <div className="relative">
        {showFilePicker && (
          <div className="absolute bottom-full left-0 mb-2 w-full max-w-sm rounded-lg border bg-popover shadow-md z-50 overflow-hidden">
            {isLoadingFiles ? (
              <div className="p-4 flex items-center justify-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {t("common.loading")}
              </div>
            ) : filteredFiles.length === 0 ? (
              <div className="p-4 text-sm text-muted-foreground text-center">
                {t("files.table.empty.noMatch")}
              </div>
            ) : (
              <div className="max-h-[200px] overflow-y-auto p-1">
                {filteredFiles.map((file, index) => (
                  <div
                    key={index}
                    className={cn(
                      "flex items-center gap-2 px-3 py-2 text-sm rounded-md cursor-pointer transition-colors overflow-scroll",
                      index === selectedFileIndex ? "bg-accent text-accent-foreground" : "hover:bg-muted",
                      downloadingFile === (file.relative_path || file.filename) && "opacity-70"
                    )}
                    onClick={() => insertFile(file)}
                  >
                    {downloadingFile === (file.relative_path || file.filename) ? (
                      <Loader2 className="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
                    ) : (
                      <FileIcon className="h-4 w-4 shrink-0 text-muted-foreground" />
                    )}
                    <div className="flex flex-col items-start">
                      <span className="truncate font-medium">{file.filename}</span>
                      {file.relative_path && file.relative_path !== file.filename && (
                        <span className="truncate text-xs text-muted-foreground">{file.relative_path}</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        <form
          onSubmit={handleSubmit}
        className={cn(
          "relative rounded-2xl bg-card overflow-hidden transition-all duration-300 border",
          isFocused
            ? "border-primary/50 shadow-md"
            : "border-border shadow-sm hover:border-border/80"
        )}
      >
        <Textarea
          ref={textareaRef}
          value={message}
          onClick={() => {
            if (showFilePicker) {
              setShowFilePicker(false);
              setTriggerIndex(-1);
            }
          }}
          onChange={handleMessageChange}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={t("chatPage.input.placeholder")}
          className="min-h-[130px] max-h-[300px] resize-none border-0 bg-transparent dark:bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 pb-14 text-[15px] placeholder:text-muted-foreground/60"
          disabled={isLoading || !!downloadingFile}
        />

        {/* Bottom toolbar */}
        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 py-3 bg-card">
          <div className="flex items-center gap-2">
            {/* Settings button - left of upload */}
            {!hideConfig && (
              <>
                {readOnlyConfig ? (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-9 px-3 text-muted-foreground rounded-xl gap-2 cursor-default hover:bg-transparent"
                    disabled={true}
                    title={models.find(m => m.model_id === agentConfig.model)?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                  >
                    <Globe className="h-4 w-4" />
                    <span className="text-xs font-normal max-w-[150px] truncate hidden sm:inline-block">
                      {models.find(m => m.model_id === agentConfig.model)?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                    </span>
                  </Button>
                ) : (
                  <ConfigDialog
                    onConfigChange={handleConfigChange}
                    currentConfig={agentConfig}
                    trigger={
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="h-9 px-3 text-muted-foreground hover:text-foreground hover:bg-secondary/80 rounded-xl gap-2"
                        disabled={isLoading}
                        title={t('agent.input.actions.config')}
                      >
                        <Globe className="h-4 w-4" />
                        <span className="text-xs font-normal max-w-[150px] truncate hidden sm:inline-block">
                          {models.find(m => m.model_id === agentConfig.model)?.model_name || agentConfig.model || t("chatPage.input.noModel")}
                        </span>
                      </Button>
                    }
                  />
                )}
              </>
            )}
            {/* Upload button - adjacent to bottom toolbar */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={handleFileSelect}
              className="hidden"
              accept=".pdf,.doc,.docx,.txt,.md,.csv,.json,.xlsx,.xls,.png,.jpg,.jpeg,.gif,.webp"
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-9 w-9 p-0 text-muted-foreground hover:text-foreground hover:bg-secondary/80 rounded-full"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              title={t("chatPage.input.actions.upload")}
            >
              <Paperclip className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex items-center gap-3">
            {taskStatus === 'running' ? (
              <Button
                type="button"
                size="icon"
                onClick={onPause}
                className="h-8 w-8 rounded-full transition-all duration-300 bg-yellow-500 hover:bg-yellow-600 text-white"
              >
                <Pause className="h-4 w-4" />
              </Button>
            ) : taskStatus === 'paused' ? (
              <Button
                type="button"
                size="icon"
                onClick={onResume}
                className="h-8 w-8 rounded-full transition-all duration-300 bg-green-500 hover:bg-green-600 text-white"
              >
                <Play className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                type="submit"
                size="icon"
                disabled={!canSubmit()}
                className={cn(
                  "h-8 w-8 rounded-lg transition-all duration-300",
                  !canSubmit() && "bg-muted text-muted-foreground/50"
                )}
              >
                {isLoading ? (
                  <Sparkles className="h-4 w-4 animate-pulse" />
                ) : (
                  <ArrowUp className="h-4 w-4" />
                )}
              </Button>
            )}
          </div>
        </div>
      </form>
      </div>

      <AlertDialog open={showNoModelAlert} onOpenChange={setShowNoModelAlert}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("common.notice")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("chatPage.input.noModelAlert")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction onClick={() => router.push("/models")}>
              {t("common.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

"use client"

import { useState, useEffect } from "react"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Eye, File as FileIcon, Loader2, RefreshCw, Upload, HardDrive, FolderOutput } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface FileItem {
  file_id: string
  filename: string
  file_size: number
  modified_time: number
  file_type?: string
  workspace_id?: string
  relative_path?: string
  category?: 'input' | 'output' | 'temp' | 'other'
}

interface TaskFileManagerProps {
  taskId: number | null
  children: React.ReactNode
  onPreview: (fileId: string, fileName: string) => void
}

export function TaskFileManager({ taskId, children, onPreview }: TaskFileManagerProps) {
  const { t } = useI18n()
  const [files, setFiles] = useState<FileItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(false)

  const loadFiles = async () => {
    if (!isOpen) return;  // Don't load if popover isn't open
    if (!taskId) return;  // Don't load if no task selected
    setIsLoading(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/task/${taskId}`)
      if (response.ok) {
        const data = await response.json()
        if (data && data.files) {
          setFiles(data.files)
        }
      }
    } catch (error) {
      console.error('Failed to load files:', error)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (isOpen) {
      loadFiles()
    }
  }, [taskId, isOpen])

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getInputFiles = () => {
    // Input files: files categorized as 'input'
    return files
      .filter(f => f.category === 'input')
      .sort((a, b) => b.modified_time - a.modified_time)
  }

  const getOutputFiles = () => {
    // Output files: files categorized as 'output'
    return files
      .filter(f => f.category === 'output')
      .sort((a, b) => b.modified_time - a.modified_time)
  }

  const renderFileList = (fileList: FileItem[], emptyMsg: string) => {
    if (isLoading) {
      return (
        <div className="w-full flex flex-col items-center justify-center py-8 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin mb-2" />
          <span className="text-sm">{t('common.loading')}</span>
        </div>
      )
    }

    if (fileList.length === 0) {
      return (
        <div className="w-full text-center text-sm text-muted-foreground py-8 border-2 border-dashed rounded-lg m-2">
          {emptyMsg}
        </div>
      )
    }

    return (
      <div className="space-y-1 p-2 w-full">
        {fileList.map((file, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-2 rounded-lg hover:bg-accent/50 transition-all group cursor-pointer"
            onClick={() => {
              onPreview(file.file_id, file.filename)
            }}
          >
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <div className="p-1.5 rounded-md bg-muted group-hover:bg-background transition-colors">
                <FileIcon className="h-3.5 w-3.5 text-muted-foreground" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium truncate" title={file.filename}>
                  {file.filename}
                </p>
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                  <span>{formatSize(file.file_size)}</span>
                  <span>•</span>
                  <span>{new Date(file.modified_time * 1000).toLocaleString()}</span>
                </div>
              </div>
            </div>

            <div className="opacity-0 group-hover:opacity-100">
               <Eye className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
          </div>
        ))}
      </div>
    )
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        {children}
      </PopoverTrigger>
      <PopoverContent
        className="w-[400px] p-0"
        align="start"
      >
        <div className="flex items-center justify-between p-3 border-b bg-muted/20">
          <h3 className="font-medium text-sm flex items-center gap-2">
            {t('files.header.title')}
          </h3>
          <Button
            variant="ghost"
            size="icon"
            onClick={loadFiles}
            disabled={isLoading}
            className="h-6 w-6"
            title={t('common.refresh')}
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        <Tabs defaultValue="output" className="w-full">
          <TabsList className="w-full justify-start rounded-none border-b bg-transparent p-0">
            <TabsTrigger
              value="input"
              className="relative h-9 rounded-none border-b-2 border-transparent bg-transparent px-4 pb-3 pt-2 font-semibold text-muted-foreground shadow-none transition-none data-[state=active]:border-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
            >
              <div className="flex items-center gap-2">
                <Upload className="h-3.5 w-3.5" />
                <span>{t('files.tabs.input')}</span>
                <span className="text-xs bg-muted px-1.5 py-0.5 rounded-full">{getInputFiles().length}</span>
              </div>
            </TabsTrigger>
            <TabsTrigger
              value="output"
              className="relative h-9 rounded-none border-b-2 border-transparent bg-transparent px-4 pb-3 pt-2 font-semibold text-muted-foreground shadow-none transition-none data-[state=active]:border-primary data-[state=active]:text-foreground data-[state=active]:shadow-none"
            >
              <div className="flex items-center gap-2">
                <FolderOutput className="h-3.5 w-3.5" />
                <span>{t('files.tabs.output')}</span>
                <span className="text-xs bg-muted px-1.5 py-0.5 rounded-full">{getOutputFiles().length}</span>
              </div>
            </TabsTrigger>
          </TabsList>

          <ScrollArea className="h-[300px]">
            <TabsContent value="input" className="m-0 border-0">
              {renderFileList(getInputFiles(), t('files.table.empty.noFiles'))}
            </TabsContent>
            <TabsContent value="output" className="m-0 border-0">
              {renderFileList(getOutputFiles(), t('files.table.empty.noFiles'))}
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </PopoverContent>
    </Popover>
  )
}

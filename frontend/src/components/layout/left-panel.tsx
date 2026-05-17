"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Bot, User, CheckCircle, XCircle, Clock, FileText } from "lucide-react"
import { cn } from "@/lib/utils"
import { formatTime } from "@/lib/time-utils"
import { AgentInput } from "@/components/agent-input"
import { useI18n } from "@/contexts/i18n-context";

interface Message {
  id: string
  role: "user" | "assistant"
  content: string | React.ReactNode
  timestamp: string | number
  status?: "pending" | "running" | "completed" | "failed"
  isResult?: boolean
  isFileOutput?: boolean
}

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused" | "waiting_for_user"
  description: string
  createdAt: string | number
  updatedAt: string | number
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
}

interface AgentConfig {
  model: string
  smallFastModel?: string
  visualModel?: string
  compactModel?: string
  memorySimilarityThreshold?: number
}

interface LeftPanelProps {
  onSendMessage: (message: string, config?: AgentConfig, files?: File[]) => Promise<void>
  onPauseTask: () => void
  onResumeTask: () => void
  messages: Message[]
  currentTask: Task | null
  isProcessing: boolean
  agentConfig?: AgentConfig
  onConfigChange?: (config: AgentConfig) => void
}

export function LeftPanel({ onSendMessage, onPauseTask, onResumeTask, messages, currentTask, isProcessing, agentConfig, onConfigChange }: LeftPanelProps) {
  const [inputMessage, setInputMessage] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  const { t } = useI18n();

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  const getStatusIcon = (status: Task["status"]) => {
    switch (status) {
      case "pending":
        return <Clock className="h-4 w-4 text-muted-foreground" />
      case "running":
        return <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      case "completed":
        return <CheckCircle className="h-4 w-4 text-muted-foreground" />
      case "failed":
        return <XCircle className="h-4 w-4 text-destructive" />
      case "paused":
      case "waiting_for_user":
        return <Clock className="h-4 w-4 text-secondary-foreground" />
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />
    }
  }

  const getStatusBadge = (status: Task["status"]) => {
    const variants = {
      pending: "secondary",
      running: "default",
      completed: "default",
      failed: "destructive",
      paused: "secondary",
      waiting_for_user: "secondary"
    } as const

    const labels = {
      pending: t("agent.layout.status.pending"),
      running: t("agent.layout.status.running"),
      completed: t("agent.layout.status.completed"),
      failed: t("agent.layout.status.failed"),
      paused: t("agent.layout.status.paused"),
      waiting_for_user: t("agent.layout.status.waitingForUser"),
    }

    const customStyles = {
      pending: "bg-muted/50 text-muted-foreground border-border",
      running: "bg-primary/10 text-primary border-primary/20",
      completed: "bg-green-500/10 text-green-500 border-green-500/20",
      failed: "bg-destructive/10 text-destructive border-destructive/20",
      paused: "bg-secondary text-secondary-foreground border-border",
      waiting_for_user: "bg-secondary text-secondary-foreground border-border"
    }

    return (
      <Badge
        variant={variants[status]}
        className={`text-sm border ${customStyles[status]}`}
      >
        {labels[status]}
      </Badge>
    )
  }


  return (
    <div className="flex flex-col h-full bg-card/30">
      {/* Header */}
      <div className="p-4 border-b border-border bg-card/50 flex-shrink-0">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-foreground">{t("agent.layout.left.title")}</h2>
          {currentTask && (
            <div className="flex items-center gap-2">
              {getStatusIcon(currentTask.status)}
              {getStatusBadge(currentTask.status)}
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4 overflow-hidden w-full">
        <div className="space-y-4 pb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-3 w-full",
                message.role === "user" ? "justify-end" : "justify-start"
              )}
            >
              {message.role === "assistant" && (
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center ring-1 ring-primary/20">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                </div>
              )}
              <div
                className={cn(
                  "max-w-[80%] rounded-xl p-3 shadow-sm overflow-hidden",
                  message.role === "user"
                    ? "bg-accent text-accent-foreground border-2 border-accent/80 user-message"
                    : message.isResult
                    ? "bg-green-500/20 border border-green-500/50 text-foreground backdrop-blur-sm"
                    : message.isFileOutput
                    ? "bg-blue-500/20 border border-blue-500/50 text-foreground backdrop-blur-sm"
                    : "bg-muted backdrop-blur-sm"
                )}
              >
                <div className="flex items-center gap-2 mb-1">
                  {message.role === "user" ? (
                    <User className="h-3 w-3" />
                  ) : message.isResult ? (
                    <CheckCircle className="h-3 w-3 text-green-400" />
                  ) : message.isFileOutput ? (
                    <FileText className="h-3 w-3 text-blue-400" />
                  ) : (
                    <Bot className="h-3 w-3" />
                  )}
                  <span className="text-sm opacity-70">
                    {formatTime(message.timestamp)}
                  </span>
                  {message.isResult && (
                    <span className="text-sm bg-green-500/20 text-green-600 dark:text-green-300 px-1.5 py-0.5 rounded">{t("agent.layout.left.messageTags.result")}</span>
                  )}
                  {message.isFileOutput && (
                    <span className="text-sm bg-blue-500/20 text-blue-600 dark:text-blue-300 px-1.5 py-0.5 rounded">{t("agent.layout.left.messageTags.file")}</span>
                  )}
                  {message.status && (
                    <div className="h-2 w-2 rounded-full bg-current opacity-70" />
                  )}
                </div>
                <div className="text-base whitespace-pre-wrap break-words prose prose-base max-w-none message-content overflow-x-hidden">
                  {typeof message.content === 'string' ? message.content : message.content}
                </div>
              </div>
              {message.role === "user" && (
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center ring-1 ring-primary/30">
                    <User className="h-4 w-4 text-primary-foreground" />
                  </div>
                </div>
              )}
            </div>
          ))}
          {/* Empty div for scrolling to bottom */}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="p-4 border-t border-border bg-card/50">
        <AgentInput
          value={inputMessage}
          onChange={(value) => setInputMessage(value)}
          onSend={async (files?: File[]) => {
            console.log('🚀 LeftPanel AgentInput onSend called:', {
              message: inputMessage.trim(),
              files: files?.map(f => f.name) || [],
              hasFiles: files && files.length > 0
            })
            if (inputMessage.trim() || files) {
              await onSendMessage(inputMessage.trim(), agentConfig, files)
              setInputMessage("")
              setSelectedFiles([])
            }
          }}
          isProcessing={isProcessing}
          agentConfig={agentConfig}
          onConfigChange={onConfigChange}
          currentTask={currentTask}
          onPauseTask={onPauseTask}
          onResumeTask={onResumeTask}
          variant="compact"
          selectedFiles={selectedFiles}
          setSelectedFiles={setSelectedFiles}
        />
      </div>
    </div>
  )
}

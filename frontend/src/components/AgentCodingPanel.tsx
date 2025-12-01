import React, { useState, useRef, useEffect, useCallback } from 'react';
import { DiffEditor } from '@monaco-editor/react';
import type { 
  AgentCodingSession, 
  AnyAgentAction, 
  ReasoningStep, 
  PendingChange,
  AgentMessage 
} from '../types/agentCoding';

// ============================================================================
// TYPES
// ============================================================================

interface AgentCodingPanelProps {
  socket: any;
  workspaceRoot: string;
  onFileOpen?: (path: string) => void;
}

interface DiffViewerProps {
  originalContent: string;
  modifiedContent: string;
  path: string;
  language: string;
  onAccept?: () => void;
  onReject?: () => void;
}

// ============================================================================
// DIFF VIEWER COMPONENT
// ============================================================================

const DiffViewer: React.FC<DiffViewerProps> = ({
  originalContent,
  modifiedContent,
  path,
  language,
  onAccept,
  onReject
}) => {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-2 bg-gray-800 border-b border-white/10">
        <div className="flex items-center gap-2">
          <span className="text-yellow-400">📄</span>
          <span className="text-sm font-mono text-gray-300">{path}</span>
        </div>
        <div className="flex gap-2">
          {onReject && (
            <button 
              onClick={onReject}
              className="px-3 py-1 text-xs bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 border border-red-500/30"
            >
              Reject
            </button>
          )}
          {onAccept && (
            <button 
              onClick={onAccept}
              className="px-3 py-1 text-xs bg-green-600/20 text-green-400 rounded hover:bg-green-600/30 border border-green-500/30"
            >
              Accept
            </button>
          )}
        </div>
      </div>
      <div className="flex-1">
        <DiffEditor
          original={originalContent}
          modified={modifiedContent}
          language={language}
          theme="vs-dark"
          options={{
            readOnly: true,
            renderSideBySide: true,
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            fontSize: 13,
            lineNumbers: 'on',
            glyphMargin: true,
            folding: true,
            renderIndicators: true,
            originalEditable: false,
          }}
        />
      </div>
    </div>
  );
};

// ============================================================================
// ACTION STEP COMPONENT
// ============================================================================

interface ActionStepProps {
  action: AnyAgentAction;
  index: number;
  isActive: boolean;
  isExpanded: boolean;
  onToggle: () => void;
  onViewDiff?: (path: string) => void;
}

const ActionStep: React.FC<ActionStepProps> = ({
  action,
  index,
  isActive,
  isExpanded,
  onToggle,
  onViewDiff
}) => {
  const getStatusIcon = () => {
    switch (action.status) {
      case 'completed': return '✅';
      case 'executing': return '⏳';
      case 'failed': return '❌';
      case 'awaiting_approval': return '⏸️';
      default: return '⚪';
    }
  };

  const getActionIcon = () => {
    switch (action.type) {
      case 'read_file': return '📖';
      case 'write_file': return '✏️';
      case 'edit_file': return '🔧';
      case 'create_file': return '📄';
      case 'delete_file': return '🗑️';
      case 'list_directory': return '📁';
      case 'search_files': return '🔍';
      case 'search_in_files': return '🔎';
      case 'run_command': return '⚡';
      case 'complete': return '🎉';
      case 'abort': return '🛑';
      default: return '❓';
    }
  };

  const getActionSummary = () => {
    const params = (action as any).params || {};
    switch (action.type) {
      case 'read_file':
      case 'write_file':
      case 'edit_file':
      case 'create_file':
      case 'delete_file':
        return params.path || 'unknown file';
      case 'rename_file':
        return params.oldPath && params.newPath
          ? `${params.oldPath} → ${params.newPath}`
          : params.newPath || params.oldPath || 'rename';
      case 'list_directory':
        return params.path || '.';
      case 'search_files':
        return params.pattern || 'pattern';
      case 'search_in_files':
        return `"${params.query?.slice(0, 30)}${params.query?.length > 30 ? '...' : ''}"`;
      case 'run_command':
        return params.command?.slice(0, 40) || 'command';
      case 'complete':
        return 'Task completed';
      case 'abort':
        return params.reason || 'Aborted';
      default:
        return action.type;
    }
  };

  return (
    <div 
      className={`border rounded-lg overflow-hidden transition-all ${
        isActive 
          ? 'border-blue-500/50 bg-blue-500/10' 
          : 'border-white/10 bg-gray-900/50'
      }`}
    >
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-3 text-left hover:bg-white/5"
      >
        <span className="text-lg">{getStatusIcon()}</span>
        <span className="text-lg">{getActionIcon()}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-white">
              {action.type.replace(/_/g, ' ')}
            </span>
            <span className="text-xs text-gray-500">#{index + 1}</span>
          </div>
          <p className="text-xs text-gray-400 truncate">
            {getActionSummary()}
          </p>
        </div>
        <svg 
          className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {isExpanded && (
        <div className="border-t border-white/10 p-3 bg-black/20">
          {/* Reasoning */}
          {action.reasoning && (
            <div className="mb-3">
              <h4 className="text-xs font-medium text-gray-400 mb-1">Reasoning</h4>
              <p className="text-sm text-gray-300">{action.reasoning}</p>
            </div>
          )}
          
          {/* Parameters */}
          {action.params && Object.keys(action.params).length > 0 && (
            <div className="mb-3">
              <h4 className="text-xs font-medium text-gray-400 mb-1">Parameters</h4>
              <pre className="text-xs text-gray-300 bg-gray-900/50 p-2 rounded overflow-x-auto">
                {JSON.stringify(action.params, null, 2)}
              </pre>
            </div>
          )}
          
          {/* Result */}
          {action.result && (
            <div>
              <h4 className="text-xs font-medium text-gray-400 mb-1">Result</h4>
              <pre className={`text-xs p-2 rounded overflow-x-auto ${
                action.result.success 
                  ? 'text-green-300 bg-green-900/20' 
                  : 'text-red-300 bg-red-900/20'
              }`}>
                {JSON.stringify(action.result, null, 2)}
              </pre>
            </div>
          )}
          
          {/* View Diff Button */}
          {action.result?.diff && onViewDiff && (
            <button
              onClick={() => {
                const params = (action as any).params || {};
                const path = params.path || params.oldPath || params.newPath || '';
                onViewDiff(path);
              }}
              className="mt-2 px-3 py-1 text-xs bg-blue-600/20 text-blue-400 rounded hover:bg-blue-600/30 border border-blue-500/30"
            >
              View Diff
            </button>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// REASONING PANEL
// ============================================================================

interface ReasoningPanelProps {
  reasoning: ReasoningStep[];
}

const ReasoningPanel: React.FC<ReasoningPanelProps> = ({ reasoning }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [reasoning]);

  return (
    <div ref={containerRef} className="flex-1 overflow-y-auto p-3 space-y-2">
      {reasoning.map((step, i) => (
        <div 
          key={i} 
          className="p-2 bg-gray-900/50 border border-white/5 rounded text-sm"
        >
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-2 py-0.5 text-xs rounded ${
              step.phase === 'planning' ? 'bg-purple-500/20 text-purple-300' :
              step.phase === 'executing' ? 'bg-blue-500/20 text-blue-300' :
              step.phase === 'reflecting' ? 'bg-yellow-500/20 text-yellow-300' :
              'bg-gray-500/20 text-gray-300'
            }`}>
              {step.phase}
            </span>
            <span className="text-xs text-gray-500">
              {new Date(step.timestamp * 1000).toLocaleTimeString()}
            </span>
          </div>
          <p className="text-gray-300">{step.thought}</p>
          {step.observation && (
            <p className="text-gray-400 text-xs mt-1">
              <span className="text-gray-500">Observed:</span> {step.observation}
            </p>
          )}
          {step.conclusion && (
            <p className="text-gray-400 text-xs mt-1">
              <span className="text-gray-500">Concluded:</span> {step.conclusion}
            </p>
          )}
        </div>
      ))}
    </div>
  );
};

// ============================================================================
// APPROVAL MODAL
// ============================================================================

interface ApprovalModalProps {
  changes: PendingChange[];
  message: string;
  onApprove: (changeIds?: string[]) => void;
  onReject: (reason: string) => void;
}

const ApprovalModal: React.FC<ApprovalModalProps> = ({
  changes,
  message,
  onApprove,
  onReject
}) => {
  const [selectedChanges, setSelectedChanges] = useState<Set<string>>(
    new Set(changes.map(c => c.id))
  );
  const [rejectReason, setRejectReason] = useState('');
  const [showRejectInput, setShowRejectInput] = useState(false);

  const toggleChange = (id: string) => {
    const newSelected = new Set(selectedChanges);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedChanges(newSelected);
  };

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-900 border border-white/20 rounded-lg w-full max-w-2xl max-h-[80vh] overflow-hidden">
        <div className="p-4 border-b border-white/10">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <span className="text-yellow-400">⚠️</span>
            Approval Required
          </h2>
          <p className="text-sm text-gray-400 mt-1">{message}</p>
        </div>
        
        <div className="p-4 max-h-[50vh] overflow-y-auto">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Pending Changes</h3>
          <div className="space-y-2">
            {changes.map(change => (
              <label 
                key={change.id}
                className="flex items-center gap-3 p-2 bg-gray-800/50 border border-white/5 rounded cursor-pointer hover:bg-gray-800"
              >
                <input
                  type="checkbox"
                  checked={selectedChanges.has(change.id)}
                  onChange={() => toggleChange(change.id)}
                  className="rounded bg-gray-700 border-gray-600"
                />
                <span className="text-lg">
                  {change.type === 'create' ? '📄' :
                   change.type === 'modify' ? '✏️' :
                   change.type === 'delete' ? '🗑️' :
                   change.type === 'rename' ? '📝' : '❓'}
                </span>
                <div className="flex-1">
                  <p className="text-sm text-white">{change.path}</p>
                  {change.newPath && (
                    <p className="text-xs text-gray-400">→ {change.newPath}</p>
                  )}
                </div>
                <span className={`px-2 py-0.5 text-xs rounded ${
                  change.type === 'create' ? 'bg-green-500/20 text-green-300' :
                  change.type === 'modify' ? 'bg-blue-500/20 text-blue-300' :
                  change.type === 'delete' ? 'bg-red-500/20 text-red-300' :
                  'bg-yellow-500/20 text-yellow-300'
                }`}>
                  {change.type}
                </span>
              </label>
            ))}
          </div>
        </div>
        
        <div className="p-4 border-t border-white/10 space-y-3">
          {showRejectInput ? (
            <div className="flex gap-2">
              <input
                type="text"
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Reason for rejection (optional)"
                className="flex-1 px-3 py-2 bg-gray-800 border border-white/10 rounded text-sm text-white"
              />
              <button
                onClick={() => onReject(rejectReason)}
                className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                Confirm Reject
              </button>
              <button
                onClick={() => setShowRejectInput(false)}
                className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowRejectInput(true)}
                className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600"
              >
                Reject All
              </button>
              <button
                onClick={() => onApprove(Array.from(selectedChanges))}
                disabled={selectedChanges.size === 0}
                className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Approve Selected ({selectedChanges.size})
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// MAIN AGENT CODING PANEL
// ============================================================================

const AgentCodingPanel: React.FC<AgentCodingPanelProps> = ({
  socket,
  workspaceRoot,
  onFileOpen
}) => {
  // Session state
  const [session, setSession] = useState<AgentCodingSession | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  
  // UI state
  const [inputValue, setInputValue] = useState('');
  const [activeTab, setActiveTab] = useState<'actions' | 'reasoning' | 'diff'>('actions');
  const [expandedActions, setExpandedActions] = useState<Set<string>>(new Set());
  
  // Diff viewer state
  const [diffView, setDiffView] = useState<{
    path: string;
    original: string;
    modified: string;
  } | null>(null);
  
  // Approval state
  const [approvalRequest, setApprovalRequest] = useState<{
    changes: PendingChange[];
    message: string;
  } | null>(null);
  
  // =========================================================================
  // SOCKET HANDLERS
  // =========================================================================
  
  useEffect(() => {
    if (!socket) return;
    
    const handleAgentMessage = (message: AgentMessage) => {
      console.log('Agent message:', message);
      
      switch (message.type) {
        case 'session_start':
          setSession({
            id: message.sessionId,
            userRequest: message.data.userRequest,
            workspaceRoot: message.data.workspaceRoot,
            phase: 'planning',
            plan: [],
            currentStepIndex: 0,
            actions: [],
            results: [],
            pendingChanges: [],
            reasoning: [],
            startTime: message.timestamp,
            filesModified: []
          });
          setIsRunning(true);
          break;
          
        case 'plan_created':
          setSession(prev => prev ? {
            ...prev,
            plan: message.data.plan,
            phase: 'executing'
          } : null);
          break;
          
        case 'action_start':
          setSession(prev => prev ? {
            ...prev,
            actions: [...prev.actions, message.data.action],
            currentStepIndex: message.data.stepIndex
          } : null);
          // Auto-expand active action
          setExpandedActions(prev => new Set([...prev, message.data.action.id]));
          break;
          
        case 'action_complete':
          setSession(prev => {
            if (!prev) return null;
            return {
              ...prev,
              actions: prev.actions.map(a => 
                a.id === message.data.actionId 
                  ? { ...a, status: 'completed', result: message.data.result }
                  : a
              )
            };
          });
          break;
          
        case 'reasoning':
          setSession(prev => prev ? {
            ...prev,
            reasoning: [...prev.reasoning, message.data]
          } : null);
          break;
          
        case 'diff_preview':
          setDiffView({
            path: message.data.path,
            original: message.data.originalContent,
            modified: message.data.newContent
          });
          setActiveTab('diff');
          break;
          
        case 'approval_required':
          setApprovalRequest({
            changes: message.data.changes,
            message: message.data.message
          });
          setSession(prev => prev ? { ...prev, phase: 'awaiting_approval' } : null);
          break;
          
        case 'session_complete':
          setSession(prev => prev ? {
            ...prev,
            phase: 'complete',
            endTime: message.timestamp,
            filesModified: message.data.filesModified
          } : null);
          setIsRunning(false);
          break;
          
        case 'session_error':
          setSession(prev => prev ? {
            ...prev,
            phase: 'error',
            error: message.data.error,
            endTime: message.timestamp
          } : null);
          setIsRunning(false);
          break;
      }
    };
    
    socket.on('agent_coding_message', handleAgentMessage);
    
    return () => {
      socket.off('agent_coding_message', handleAgentMessage);
    };
  }, [socket]);
  
  // =========================================================================
  // ACTIONS
  // =========================================================================
  
  const startSession = useCallback(() => {
    if (!inputValue.trim() || !socket) return;
    
    socket.emit('agent_coding_start', {
      request: inputValue.trim(),
      workspaceRoot
    });
    
    setInputValue('');
  }, [inputValue, socket, workspaceRoot]);
  
  const stopSession = useCallback(() => {
    if (!socket || !session) return;
    socket.emit('agent_coding_stop', { sessionId: session.id });
    setIsRunning(false);
  }, [socket, session]);
  
  const handleApprove = useCallback((changeIds?: string[]) => {
    if (!socket || !session) return;
    socket.emit('agent_coding_approve', {
      sessionId: session.id,
      changeIds,
      approveAll: !changeIds
    });
    setApprovalRequest(null);
  }, [socket, session]);
  
  const handleReject = useCallback((reason: string) => {
    if (!socket || !session) return;
    socket.emit('agent_coding_reject', {
      sessionId: session.id,
      reason
    });
    setApprovalRequest(null);
  }, [socket, session]);
  
  const toggleActionExpand = (actionId: string) => {
    setExpandedActions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(actionId)) {
        newSet.delete(actionId);
      } else {
        newSet.add(actionId);
      }
      return newSet;
    });
  };
  
  const getLanguageFromPath = (path: string): string => {
    const ext = path.split('.').pop()?.toLowerCase() || '';
    const langMap: Record<string, string> = {
      'ts': 'typescript',
      'tsx': 'typescript',
      'js': 'javascript',
      'jsx': 'javascript',
      'py': 'python',
      'rs': 'rust',
      'json': 'json',
      'md': 'markdown',
      'css': 'css',
      'scss': 'scss',
      'html': 'html',
      'yaml': 'yaml',
      'yml': 'yaml',
      'toml': 'toml',
      'sql': 'sql',
      'sh': 'shell',
      'bash': 'shell',
    };
    return langMap[ext] || 'plaintext';
  };
  
  // =========================================================================
  // RENDER
  // =========================================================================
  
  return (
    <div className="h-full flex flex-col bg-gray-950">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-white/10">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">🤖</span>
          <h2 className="text-lg font-semibold text-white">Agentic Coding</h2>
          {session && (
            <span className={`px-2 py-1 text-xs rounded ${
              session.phase === 'complete' ? 'bg-green-500/20 text-green-300' :
              session.phase === 'error' ? 'bg-red-500/20 text-red-300' :
              session.phase === 'awaiting_approval' ? 'bg-yellow-500/20 text-yellow-300' :
              'bg-blue-500/20 text-blue-300'
            }`}>
              {session.phase}
            </span>
          )}
        </div>
        
        {/* Input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !isRunning && startSession()}
            placeholder="Describe what you want to build or change..."
            disabled={isRunning}
            className="flex-1 px-4 py-2 bg-gray-900 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
          />
          {isRunning ? (
            <button
              onClick={stopSession}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center gap-2"
            >
              <span>Stop</span>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <rect x="5" y="5" width="10" height="10" rx="1" />
              </svg>
            </button>
          ) : (
            <button
              onClick={startSession}
              disabled={!inputValue.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <span>Run</span>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" />
              </svg>
            </button>
          )}
        </div>
      </div>
      
      {/* Tabs */}
      <div className="flex-shrink-0 flex border-b border-white/10">
        {(['actions', 'reasoning', 'diff'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === tab
                ? 'text-blue-400 border-b-2 border-blue-400'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {tab === 'actions' && `Actions${session ? ` (${session.actions.length})` : ''}`}
            {tab === 'reasoning' && `Reasoning${session ? ` (${session.reasoning.length})` : ''}`}
            {tab === 'diff' && 'Diff View'}
          </button>
        ))}
      </div>
      
      {/* Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        {!session ? (
          <div className="h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <span className="text-4xl mb-4 block">🚀</span>
              <p>Enter a coding task to get started</p>
              <p className="text-sm text-gray-600 mt-2">
                Examples: "Add a dark mode toggle", "Refactor this component", "Fix the bug in..."
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* Actions Tab */}
            {activeTab === 'actions' && (
              <div className="h-full overflow-y-auto p-4 space-y-2">
                {/* Plan */}
                {session.plan.length > 0 && (
                  <div className="mb-4 p-3 bg-purple-900/20 border border-purple-500/30 rounded-lg">
                    <h3 className="text-sm font-medium text-purple-300 mb-2">📋 Plan</h3>
                    <ol className="space-y-1">
                      {session.plan.map((step, i) => (
                        <li 
                          key={i}
                          className={`text-sm flex items-center gap-2 ${
                            i === session.currentStepIndex 
                              ? 'text-white' 
                              : step.status === 'completed' 
                                ? 'text-gray-500 line-through' 
                                : 'text-gray-400'
                          }`}
                        >
                          <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs ${
                            i === session.currentStepIndex 
                              ? 'bg-blue-500 text-white' 
                              : step.status === 'completed'
                                ? 'bg-green-500/20 text-green-300'
                                : 'bg-gray-700 text-gray-400'
                          }`}>
                            {step.status === 'completed' ? '✓' : i + 1}
                          </span>
                          {step.description}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
                
                {/* Actions */}
                {session.actions.map((action, i) => (
                  <ActionStep
                    key={action.id}
                    action={action}
                    index={i}
                    isActive={action.status === 'executing'}
                    isExpanded={expandedActions.has(action.id)}
                    onToggle={() => toggleActionExpand(action.id)}
                    onViewDiff={() => {
                      if ((action as any).result?.diff) {
                        // Would need to fetch original content here
                        setActiveTab('diff');
                      }
                    }}
                  />
                ))}
                
                {/* Session Summary */}
                {session.phase === 'complete' && (
                  <div className="mt-4 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                    <h3 className="text-sm font-medium text-green-300 mb-2">✅ Completed</h3>
                    <p className="text-sm text-gray-300 mb-2">
                      Modified {session.filesModified.length} files in {
                        ((session.endTime || Date.now()/1000) - session.startTime).toFixed(1)
                      }s
                    </p>
                    {session.filesModified.length > 0 && (
                      <ul className="text-xs text-gray-400">
                        {session.filesModified.map(f => (
                          <li key={f} className="flex items-center gap-1">
                            <span>📄</span>
                            <button 
                              onClick={() => onFileOpen?.(f)}
                              className="hover:text-white hover:underline"
                            >
                              {f}
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
                
                {/* Error */}
                {session.phase === 'error' && (
                  <div className="mt-4 p-4 bg-red-900/20 border border-red-500/30 rounded-lg">
                    <h3 className="text-sm font-medium text-red-300 mb-2">❌ Error</h3>
                    <p className="text-sm text-gray-300">{session.error}</p>
                  </div>
                )}
              </div>
            )}
            
            {/* Reasoning Tab */}
            {activeTab === 'reasoning' && (
              <ReasoningPanel reasoning={session.reasoning} />
            )}
            
            {/* Diff Tab */}
            {activeTab === 'diff' && (
              diffView ? (
                <DiffViewer
                  originalContent={diffView.original}
                  modifiedContent={diffView.modified}
                  path={diffView.path}
                  language={getLanguageFromPath(diffView.path)}
                />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <span className="text-4xl mb-4 block">📝</span>
                    <p>No diff to display</p>
                    <p className="text-sm text-gray-600 mt-2">
                      Diffs will appear here when files are modified
                    </p>
                  </div>
                </div>
              )
            )}
          </>
        )}
      </div>
      
      {/* Approval Modal */}
      {approvalRequest && (
        <ApprovalModal
          changes={approvalRequest.changes}
          message={approvalRequest.message}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      )}
    </div>
  );
};

export default AgentCodingPanel;

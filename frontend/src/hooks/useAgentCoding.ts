import { useState, useCallback, useEffect, useRef } from 'react';
import type {
  AgentCodingSession,
  AnyAgentAction,
  ReasoningStep,
  PendingChange,
  AgentMessage,
  AgentPhase
} from '../types/agentCoding';

interface UseAgentCodingOptions {
  socket: any;
  workspaceRoot: string;
  onSessionComplete?: (session: AgentCodingSession) => void;
  onFileModified?: (path: string) => void;
}

interface UseAgentCodingReturn {
  // State
  session: AgentCodingSession | null;
  isRunning: boolean;
  approvalRequest: { changes: PendingChange[]; message: string } | null;
  
  // Actions
  startSession: (request: string) => void;
  stopSession: () => void;
  approveChanges: (changeIds?: string[], approveAll?: boolean) => void;
  rejectChanges: (reason?: string) => void;
  
  // Helpers
  getActiveAction: () => AnyAgentAction | null;
  getProgress: () => { current: number; total: number };
}

export function useAgentCoding({
  socket,
  workspaceRoot,
  onSessionComplete,
  onFileModified
}: UseAgentCodingOptions): UseAgentCodingReturn {
  const [session, setSession] = useState<AgentCodingSession | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [approvalRequest, setApprovalRequest] = useState<{
    changes: PendingChange[];
    message: string;
  } | null>(null);
  
  const sessionRef = useRef(session);
  sessionRef.current = session;
  
  // Socket message handler
  useEffect(() => {
    if (!socket) return;
    
    const handleMessage = (message: AgentMessage) => {
      console.log('[AgentCoding] Message:', message.type, message.data);
      
      switch (message.type) {
        case 'session_start':
          setSession({
            id: message.sessionId,
            userRequest: message.data.userRequest,
            workspaceRoot: message.data.workspaceRoot,
            phase: 'planning' as AgentPhase,
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
          setApprovalRequest(null);
          break;
        
        case 'plan_created':
          setSession(prev => prev ? {
            ...prev,
            plan: message.data.plan,
            phase: 'executing' as AgentPhase
          } : null);
          break;
        
        case 'step_start':
          setSession(prev => prev ? {
            ...prev,
            currentStepIndex: message.data.stepIndex,
            plan: prev.plan.map((step, i) => 
              i === message.data.stepIndex 
                ? { ...step, status: 'active' } 
                : step
            )
          } : null);
          break;
        
        case 'action_start':
          setSession(prev => {
            if (!prev) return null;
            return {
              ...prev,
              actions: [...prev.actions, message.data.action as AnyAgentAction]
            };
          });
          break;
        
        case 'action_progress':
          // Update action with progress info (e.g., streaming content)
          setSession(prev => {
            if (!prev) return null;
            return {
              ...prev,
              actions: prev.actions.map(a =>
                a.id === message.data.actionId
                  ? { ...a, ...message.data.progress }
                  : a
              )
            };
          });
          break;
        
        case 'action_complete':
          setSession(prev => {
            if (!prev) return null;
            
            const updatedActions: AnyAgentAction[] = prev.actions.map(a => {
              if (a.id !== message.data.actionId) return a;
              const nextStatus = message.data.result.success ? 'completed' as const : 'failed' as const;
              return {
                ...a,
                status: nextStatus,
                result: message.data.result
              } as AnyAgentAction;
            });
            
            // Track modified files
            const action = prev.actions.find(a => a.id === message.data.actionId);
            const newFilesModified = [...prev.filesModified];
            if (action && message.data.result.success) {
              const modifyingActions = ['write_file', 'edit_file', 'create_file', 'delete_file'];
              if (modifyingActions.includes(action.type)) {
                const params = (action as any).params || {};
                const path = params.path || params.newPath || params.oldPath;
                if (path && !newFilesModified.includes(path)) {
                  newFilesModified.push(path);
                  onFileModified?.(path);
                }
              }
            }
            
            return {
              ...prev,
              actions: updatedActions,
              filesModified: newFilesModified
            };
          });
          break;
        
        case 'reasoning':
          setSession(prev => prev ? {
            ...prev,
            reasoning: [...prev.reasoning, message.data as ReasoningStep]
          } : null);
          break;
        
        case 'approval_required':
          setApprovalRequest({
            changes: message.data.changes,
            message: message.data.message
          });
          setSession(prev => prev ? {
            ...prev,
            phase: 'awaiting_approval' as AgentPhase,
            pendingChanges: message.data.changes
          } : null);
          break;
        
        case 'approval_received':
          setApprovalRequest(null);
          if (message.data.approved) {
            setSession(prev => prev ? {
              ...prev,
              phase: 'executing' as AgentPhase,
              pendingChanges: prev.pendingChanges.map(c => ({ ...c, approved: true }))
            } : null);
          }
          break;
        
        case 'step_complete':
          setSession(prev => prev ? {
            ...prev,
            plan: prev.plan.map((step, i) =>
              i === message.data.stepIndex
                ? { ...step, status: 'completed' }
                : step
            )
          } : null);
          break;
        
        case 'file_preview':
        case 'diff_preview':
          // These are handled by the UI component directly
          break;
        
        case 'session_complete':
          setSession(prev => {
            if (!prev) return null;
            const completed = {
              ...prev,
              phase: 'complete' as AgentPhase,
              endTime: message.timestamp,
              filesModified: message.data.filesModified || prev.filesModified
            };
            onSessionComplete?.(completed);
            return completed;
          });
          setIsRunning(false);
          break;
        
        case 'session_error':
          setSession(prev => prev ? {
            ...prev,
            phase: 'error' as AgentPhase,
            error: message.data.error,
            endTime: message.timestamp
          } : null);
          setIsRunning(false);
          setApprovalRequest(null);
          break;
      }
    };
    
    socket.on('agent_coding_message', handleMessage);
    
    return () => {
      socket.off('agent_coding_message', handleMessage);
    };
  }, [socket, onSessionComplete, onFileModified]);
  
  // Actions
  const startSession = useCallback((request: string) => {
    if (!socket || !request.trim()) return;
    
    socket.emit('agent_coding_start', {
      request: request.trim(),
      workspaceRoot
    });
  }, [socket, workspaceRoot]);
  
  const stopSession = useCallback(() => {
    if (!socket || !session) return;
    
    socket.emit('agent_coding_stop', {
      sessionId: session.id
    });
    
    setIsRunning(false);
  }, [socket, session]);
  
  const approveChanges = useCallback((changeIds?: string[], approveAll?: boolean) => {
    if (!socket || !session) return;
    
    socket.emit('agent_coding_approve', {
      sessionId: session.id,
      changeIds,
      approveAll: approveAll ?? !changeIds
    });
  }, [socket, session]);
  
  const rejectChanges = useCallback((reason?: string) => {
    if (!socket || !session) return;
    
    socket.emit('agent_coding_reject', {
      sessionId: session.id,
      reason: reason || ''
    });
  }, [socket, session]);
  
  // Helpers
  const getActiveAction = useCallback((): AnyAgentAction | null => {
    if (!session) return null;
    return session.actions.find(a => a.status === 'executing') || null;
  }, [session]);
  
  const getProgress = useCallback(() => {
    if (!session) return { current: 0, total: 0 };
    
    const completed = session.plan.filter(s => s.status === 'completed').length;
    return {
      current: completed,
      total: session.plan.length
    };
  }, [session]);
  
  return {
    session,
    isRunning,
    approvalRequest,
    startSession,
    stopSession,
    approveChanges,
    rejectChanges,
    getActiveAction,
    getProgress
  };
}

export default useAgentCoding;

/**
 * Agentic Coding System - Type Definitions
 * 
 * This defines the action schema for ReAct-style coding operations.
 * Nova (the orchestrator) uses these to perform multi-step coding tasks.
 */

// ============================================================================
// AGENT ACTIONS - What the agent can do
// ============================================================================

export type AgentActionType = 
  | 'read_file'
  | 'write_file'
  | 'edit_file'
  | 'create_file'
  | 'delete_file'
  | 'rename_file'
  | 'list_directory'
  | 'search_files'
  | 'search_in_files'
  | 'run_command'
  | 'run_python'
  | 'run_javascript'
  | 'apply_diff'
  | 'request_approval'
  | 'complete'
  | 'abort';

// Base action interface
export interface AgentAction {
  id: string;                    // Unique action ID
  type: AgentActionType;
  reasoning: string;             // Why the agent is taking this action
  timestamp: number;
  status: 'pending' | 'executing' | 'completed' | 'failed' | 'awaiting_approval';
  params?: Record<string, any>;   // Common params bag for runtime data
  result?: any;                  // Result payload from backend
}

// Specific action types
export interface ReadFileAction extends AgentAction {
  type: 'read_file';
  params: {
    path: string;
    startLine?: number;
    endLine?: number;
  };
}

export interface WriteFileAction extends AgentAction {
  type: 'write_file';
  params: {
    path: string;
    content: string;
    createIfMissing?: boolean;
  };
}

export interface EditFileAction extends AgentAction {
  type: 'edit_file';
  params: {
    path: string;
    edits: FileEdit[];
  };
}

export interface FileEdit {
  oldText: string;       // Exact text to find (must be unique in file)
  newText: string;       // Text to replace with
  description?: string;  // Human-readable description
}

export interface CreateFileAction extends AgentAction {
  type: 'create_file';
  params: {
    path: string;
    content: string;
  };
}

export interface DeleteFileAction extends AgentAction {
  type: 'delete_file';
  params: {
    path: string;
  };
}

export interface RenameFileAction extends AgentAction {
  type: 'rename_file';
  params: {
    oldPath: string;
    newPath: string;
  };
}

export interface ListDirectoryAction extends AgentAction {
  type: 'list_directory';
  params: {
    path: string;
    recursive?: boolean;
    maxDepth?: number;
  };
}

export interface SearchFilesAction extends AgentAction {
  type: 'search_files';
  params: {
    pattern: string;     // Glob pattern or filename
    directory?: string;
    excludePatterns?: string[];
  };
}

export interface SearchInFilesAction extends AgentAction {
  type: 'search_in_files';
  params: {
    query: string;       // Text or regex to search
    directory?: string;
    filePattern?: string;
    isRegex?: boolean;
  };
}

export interface RunCommandAction extends AgentAction {
  type: 'run_command';
  params: {
    command: string;
    cwd?: string;
    timeout?: number;
  };
}

export interface ApplyDiffAction extends AgentAction {
  type: 'apply_diff';
  params: {
    path: string;
    diff: UnifiedDiff;
  };
}

export interface RequestApprovalAction extends AgentAction {
  type: 'request_approval';
  params: {
    message: string;
    changes: PendingChange[];
  };
}

export interface CompleteAction extends AgentAction {
  type: 'complete';
  params: {
    summary: string;
    filesModified: string[];
  };
}

export interface AbortAction extends AgentAction {
  type: 'abort';
  params: {
    reason: string;
  };
}

// Union type for all actions
export type AnyAgentAction = 
  | ReadFileAction
  | WriteFileAction
  | EditFileAction
  | CreateFileAction
  | DeleteFileAction
  | RenameFileAction
  | ListDirectoryAction
  | SearchFilesAction
  | SearchInFilesAction
  | RunCommandAction
  | ApplyDiffAction
  | RequestApprovalAction
  | CompleteAction
  | AbortAction;

// ============================================================================
// ACTION RESULTS - What comes back from actions
// ============================================================================

export interface ActionResult {
  actionId: string;
  success: boolean;
  timestamp: number;
}

export interface ReadFileResult extends ActionResult {
  content?: string;
  lines?: number;
  error?: string;
}

export interface WriteFileResult extends ActionResult {
  bytesWritten?: number;
  error?: string;
}

export interface EditFileResult extends ActionResult {
  editsApplied: number;
  diff?: string;         // Unified diff of changes
  error?: string;
}

export interface ListDirectoryResult extends ActionResult {
  entries?: DirectoryEntry[];
  error?: string;
}

export interface DirectoryEntry {
  name: string;
  path: string;
  type: 'file' | 'directory';
  size?: number;
  modified?: string;
}

export interface SearchResult extends ActionResult {
  matches?: SearchMatch[];
  error?: string;
}

export interface SearchMatch {
  path: string;
  line?: number;
  column?: number;
  preview?: string;
  matchedText?: string;
}

export interface CommandResult extends ActionResult {
  stdout?: string;
  stderr?: string;
  exitCode?: number;
  error?: string;
}

// ============================================================================
// DIFF REPRESENTATION
// ============================================================================

export interface UnifiedDiff {
  oldPath: string;
  newPath: string;
  hunks: DiffHunk[];
}

export interface DiffHunk {
  oldStart: number;
  oldLines: number;
  newStart: number;
  newLines: number;
  lines: DiffLine[];
}

export interface DiffLine {
  type: 'context' | 'add' | 'remove';
  content: string;
  oldLineNumber?: number;
  newLineNumber?: number;
}

// ============================================================================
// PENDING CHANGES (for approval flow)
// ============================================================================

export interface PendingChange {
  id: string;
  type: 'create' | 'modify' | 'delete' | 'rename';
  path: string;
  newPath?: string;      // For renames
  diff?: UnifiedDiff;
  newContent?: string;   // For creates
  approved: boolean;
}

// ============================================================================
// AGENT SESSION STATE
// ============================================================================

export type AgentPhase = 
  | 'idle'
  | 'planning'
  | 'executing'
  | 'awaiting_approval'
  | 'reflecting'
  | 'complete'
  | 'error';

export interface AgentCodingSession {
  id: string;
  userRequest: string;           // Original user request
  phase: AgentPhase;
  plan: AgentPlanStep[];
  currentStepIndex: number;
  actions: AnyAgentAction[];
  results: ActionResult[];
  pendingChanges: PendingChange[];
  reasoning: ReasoningStep[];
  startTime: number;
  endTime?: number;
  error?: string;
  workspaceRoot: string;
  filesModified: string[];
}

export interface AgentPlanStep {
  index: number;
  description: string;
  status: 'pending' | 'active' | 'completed' | 'skipped';
  actions: string[];             // Action IDs associated with this step
}

export interface ReasoningStep {
  timestamp: number;
  phase: AgentPhase;
  thought: string;
  observation?: string;          // What the agent observed
  conclusion?: string;           // What the agent concluded
}

// ============================================================================
// AGENT MESSAGES (for streaming updates to UI)
// ============================================================================

export type AgentMessageType = 
  | 'session_start'
  | 'planning_start'
  | 'plan_created'
  | 'step_start'
  | 'action_start'
  | 'action_progress'
  | 'action_complete'
  | 'reasoning'
  | 'approval_required'
  | 'approval_received'
  | 'step_complete'
  | 'session_complete'
  | 'session_error'
  | 'file_preview'
  | 'diff_preview';

export interface AgentMessage {
  type: AgentMessageType;
  sessionId: string;
  timestamp: number;
  data: any;
}

export interface SessionStartMessage extends AgentMessage {
  type: 'session_start';
  data: {
    userRequest: string;
    workspaceRoot: string;
  };
}

export interface PlanCreatedMessage extends AgentMessage {
  type: 'plan_created';
  data: {
    plan: AgentPlanStep[];
    reasoning: string;
  };
}

export interface ActionStartMessage extends AgentMessage {
  type: 'action_start';
  data: {
    action: AnyAgentAction;
    stepIndex: number;
  };
}

export interface ActionCompleteMessage extends AgentMessage {
  type: 'action_complete';
  data: {
    actionId: string;
    result: ActionResult;
  };
}

export interface ReasoningMessage extends AgentMessage {
  type: 'reasoning';
  data: ReasoningStep;
}

export interface ApprovalRequiredMessage extends AgentMessage {
  type: 'approval_required';
  data: {
    changes: PendingChange[];
    message: string;
  };
}

export interface DiffPreviewMessage extends AgentMessage {
  type: 'diff_preview';
  data: {
    path: string;
    diff: UnifiedDiff;
    originalContent: string;
    newContent: string;
  };
}

export interface SessionCompleteMessage extends AgentMessage {
  type: 'session_complete';
  data: {
    summary: string;
    filesModified: string[];
    duration: number;
  };
}

// ============================================================================
// TOOL DEFINITIONS FOR NOVA
// ============================================================================

export const CODING_TOOLS_SCHEMA = [
  {
    type: 'function',
    function: {
      name: 'agent_read_file',
      description: 'Read the contents of a file. Use this to examine code before making changes.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Path to the file relative to workspace root' },
          startLine: { type: 'integer', description: 'Optional: Start reading from this line (1-indexed)' },
          endLine: { type: 'integer', description: 'Optional: Stop reading at this line' }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_write_file',
      description: 'Write content to a file. Use this to create or completely replace a file.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Path to the file' },
          content: { type: 'string', description: 'The complete content to write' }
        },
        required: ['path', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_edit_file',
      description: 'Make surgical edits to a file by finding and replacing exact text. Each edit must match unique text in the file.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Path to the file' },
          edits: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                oldText: { type: 'string', description: 'Exact text to find (must be unique in file)' },
                newText: { type: 'string', description: 'Text to replace with' },
                description: { type: 'string', description: 'Human-readable description of the change' }
              },
              required: ['oldText', 'newText']
            },
            description: 'Array of edits to apply'
          }
        },
        required: ['path', 'edits']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_create_file',
      description: 'Create a new file with the given content.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Path where the file should be created' },
          content: { type: 'string', description: 'Content for the new file' }
        },
        required: ['path', 'content']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_delete_file',
      description: 'Delete a file from the filesystem.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Path to the file to delete' }
        },
        required: ['path']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_list_directory',
      description: 'List files and directories in a path.',
      parameters: {
        type: 'object',
        properties: {
          path: { type: 'string', description: 'Directory path to list', default: '.' },
          recursive: { type: 'boolean', description: 'Whether to list recursively', default: false },
          maxDepth: { type: 'integer', description: 'Maximum recursion depth', default: 2 }
        }
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_search_files',
      description: 'Search for files by name pattern.',
      parameters: {
        type: 'object',
        properties: {
          pattern: { type: 'string', description: 'Filename pattern to search for (e.g., "*.tsx", "Component")' },
          directory: { type: 'string', description: 'Directory to search in', default: '.' },
          excludePatterns: { type: 'array', items: { type: 'string' }, description: 'Patterns to exclude (e.g., ["node_modules", "dist"])' }
        },
        required: ['pattern']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_search_in_files',
      description: 'Search for text content within files.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Text or regex to search for' },
          directory: { type: 'string', description: 'Directory to search in', default: '.' },
          filePattern: { type: 'string', description: 'Only search in files matching this pattern' },
          isRegex: { type: 'boolean', description: 'Whether query is a regex', default: false }
        },
        required: ['query']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_run_command',
      description: 'Run a shell command. Use sparingly and only for necessary operations like npm install, git status, etc.',
      parameters: {
        type: 'object',
        properties: {
          command: { type: 'string', description: 'The command to run' },
          cwd: { type: 'string', description: 'Working directory for the command' },
          timeout: { type: 'integer', description: 'Timeout in seconds', default: 30 }
        },
        required: ['command']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_complete',
      description: 'Signal that the coding task is complete.',
      parameters: {
        type: 'object',
        properties: {
          summary: { type: 'string', description: 'Summary of changes made' },
          filesModified: { type: 'array', items: { type: 'string' }, description: 'List of files that were modified' }
        },
        required: ['summary', 'filesModified']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'agent_request_approval',
      description: 'Request user approval before applying changes. Use this for destructive operations or significant changes.',
      parameters: {
        type: 'object',
        properties: {
          message: { type: 'string', description: 'Message explaining what approval is needed for' },
          changes: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                type: { type: 'string', enum: ['create', 'modify', 'delete', 'rename'] },
                path: { type: 'string' },
                description: { type: 'string' }
              }
            }
          }
        },
        required: ['message', 'changes']
      }
    }
  }
];

export default {
  CODING_TOOLS_SCHEMA
};

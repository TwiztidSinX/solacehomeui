// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::collections::HashMap;
use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{BufRead, BufReader};
use tauri::{
  Manager, Size, Position, LogicalSize, LogicalPosition, WebviewUrl, AppHandle, Runtime,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::tray::{TrayIconBuilder, MouseButton, MouseButtonState};
use tauri::State;
use tauri::async_runtime::Mutex;
use url::Url;
use walkdir::WalkDir;
use glob::Pattern;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .plugin(tauri_plugin_shell::init())
    .plugin(tauri_plugin_fs::init())
    .plugin(tauri_plugin_dialog::init())
    .plugin(tauri_plugin_notification::init())
    .setup(|app| {
      // Initialize logging
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      // Start Python backend
      start_python_backend();

      // Setup system tray
      let _tray = TrayIconBuilder::new()
        .tooltip("SolaceOS - AI Operating System")
        .on_tray_icon_event(move |tray, event| {
          match event {
            tauri::tray::TrayIconEvent::Click {
              button,
              button_state,
              ..
            } => {
              if button == MouseButton::Left && button_state == MouseButtonState::Up {
                if let Some(window) = tray.app_handle().get_webview_window("main") {
                  let _ = window.show();
                  let _ = window.set_focus();
                }
              }
            }
            _ => {}
          }
        })
        .build(app)?;

      Ok(())
    })
    .manage(EmbeddedWebviews::default())
    .manage(AgentCodingState::default())
    .invoke_handler(tauri::generate_handler![
      // Embedded webview commands
      create_embedded_webview,
      update_embedded_webview,
      remove_embedded_webview,
      agent_execute_js,
      agent_get_dom,
      // Agent coding commands
      agent_read_file,
      agent_write_file,
      agent_edit_file,
      agent_create_file,
      agent_delete_file,
      agent_rename_file,
      agent_list_directory,
      agent_search_files,
      agent_search_in_files,
      agent_run_command,
      agent_get_workspace_info
    ])
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

fn start_python_backend() {
  // Start the Python Flask backend server via startup script
  std::thread::spawn(|| {
    // Get the project root directory
    let backend_dir = if cfg!(debug_assertions) {
      // In development, use parent of src-tauri (project root)
      std::env::current_dir()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
    } else {
      // In production, use executable's directory
      std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|p| p.to_path_buf()))
    };

    log::info!("Starting Python backend from directory: {:?}", backend_dir);

    #[cfg(target_os = "windows")]
    let result = Command::new("cmd")
      .args(&["/C", "start_backend.bat"])
      .current_dir(backend_dir.clone().unwrap_or_else(|| std::path::PathBuf::from(".")))
      .spawn();

    #[cfg(not(target_os = "windows"))]
    let result = Command::new("sh")
      .arg("start_backend.sh")
      .current_dir(backend_dir.clone().unwrap_or_else(|| std::path::PathBuf::from(".")))
      .spawn();

    match result {
      Ok(child) => {
        log::info!("Backend startup script launched with PID: {:?}", child.id());
      }
      Err(e) => {
        log::error!("Failed to start backend: {}", e);
        log::error!("Make sure start_backend.bat exists in the project root");
      }
    }
  });
}

// --- Embedded webview management ("Ghost Panel") ---

#[derive(Default)]
struct EmbeddedWebviews {
  map: Mutex<HashMap<String, ()>>, // value unused, we track existence via window label
}

fn get_main_window<R: Runtime>(app: &AppHandle<R>) -> Result<tauri::WebviewWindow<R>, String> {
  app
    .get_webview_window("main")
    .ok_or_else(|| "Main window not found. Is the app initialized?".into())
}

#[tauri::command]
async fn create_embedded_webview<R: Runtime>(
  app: AppHandle<R>,
  state: State<'_, EmbeddedWebviews>,
  label: String,
  url: String,
  x: f64,
  y: f64,
  width: f64,
  height: f64,
) -> Result<(), String> {
  let mut map = state.map.lock().await;

  // If it exists, just update position/size and navigate
  if app.get_webview_window(&label).is_some() {
    drop(map);
    return update_embedded_webview(app, label, url, x, y, width, height).await;
  }

  let _main = get_main_window(&app)?;

  tauri::WebviewWindowBuilder::new(
    &app,
    &label,
    WebviewUrl::External(
      Url::parse(&url)
        .map_err(|_| format!("Invalid URL: {}", url))?
    ),
  )
  .title("SolaceOS Browser")
  .decorations(false)
  .resizable(true)
  .transparent(true)
  .visible(true)
  .skip_taskbar(true)
  .always_on_top(true)
  .inner_size(width, height)
  .position(x, y)
  .build()
  .map_err(|e| format!("Failed to build embedded webview: {}", e))?;

  map.insert(label, ());
  Ok(())
}

#[tauri::command]
async fn update_embedded_webview<R: Runtime>(
  app: AppHandle<R>,
  label: String,
  url: String,
  x: f64,
  y: f64,
  width: f64,
  height: f64,
) -> Result<(), String> {
  if let Some(window) = app.get_webview_window(&label) {
    window
      .set_position(Position::Logical(LogicalPosition { x, y }))
      .map_err(|e| e.to_string())?;
    window
      .set_size(Size::Logical(LogicalSize { width, height }))
      .map_err(|e| e.to_string())?;
    let parsed = Url::parse(&url).map_err(|e| format!("Invalid URL: {e}"))?;
    window
      .navigate(parsed)
      .map_err(|e| format!("Failed to navigate: {}", e))?;
    Ok(())
  } else {
    Err("Embedded webview not found".into())
  }
}

#[tauri::command]
async fn remove_embedded_webview<R: Runtime>(
  app: AppHandle<R>,
  state: State<'_, EmbeddedWebviews>,
  label: String,
) -> Result<(), String> {
  if let Some(window) = app.get_webview_window(&label) {
    window.close().map_err(|e| e.to_string())?;
  }
  let mut map = state.map.lock().await;
  map.remove(&label);
  Ok(())
}

#[tauri::command]
async fn agent_execute_js<R: Runtime>(
  app: AppHandle<R>,
  label: String,
  script: String,
) -> Result<(), String> {
  if let Some(window) = app.get_webview_window(&label) {
    window
      .eval(&script)
      .map_err(|e| format!("Eval failed: {}", e))?;
    Ok(())
  } else {
    Err("Embedded webview not found".into())
  }
}

#[tauri::command]
async fn agent_get_dom<R: Runtime>(
  app: AppHandle<R>,
  label: String,
) -> Result<Value, String> {
  if app.get_webview_window(&label).is_none() {
    return Err("Embedded webview not found".into());
  }
  let placeholder = serde_json::json!({
    "status": "not_implemented",
    "message": "DOM snapshot not available in this build; use injected JS to postMessage results."
  });
  Ok(placeholder)
}

// =============================================================================
// AGENT CODING COMMANDS
// =============================================================================

#[allow(dead_code)]
#[derive(Default)]
struct AgentCodingState {
  workspace_root: Mutex<Option<PathBuf>>,
}

// Response types for agent commands
#[derive(Serialize)]
struct ReadFileResult {
  success: bool,
  content: Option<String>,
  total_lines: Option<usize>,
  lines_read: Option<usize>,
  error: Option<String>,
}

#[derive(Serialize)]
struct WriteFileResult {
  success: bool,
  bytes_written: Option<usize>,
  error: Option<String>,
}

#[derive(Serialize)]
struct EditFileResult {
  success: bool,
  edits_applied: usize,
  edits_requested: usize,
  errors: Vec<String>,
}

#[derive(Serialize)]
struct DirectoryEntry {
  name: String,
  path: String,
  entry_type: String, // "file" or "directory"
  size: Option<u64>,
}

#[derive(Serialize)]
struct ListDirectoryResult {
  success: bool,
  entries: Vec<DirectoryEntry>,
  count: usize,
  error: Option<String>,
}

#[derive(Serialize)]
struct SearchMatch {
  path: String,
  line: Option<usize>,
  column: Option<usize>,
  preview: Option<String>,
  matched_text: Option<String>,
}

#[derive(Serialize)]
struct SearchResult {
  success: bool,
  matches: Vec<SearchMatch>,
  total_matches: usize,
  error: Option<String>,
}

#[derive(Serialize)]
struct CommandResult {
  success: bool,
  exit_code: Option<i32>,
  stdout: Option<String>,
  stderr: Option<String>,
  error: Option<String>,
}

#[derive(Serialize)]
struct WorkspaceInfo {
  root: String,
  name: String,
  has_package_json: bool,
  has_cargo_toml: bool,
  has_pyproject: bool,
}

// Edit request structure
#[allow(dead_code)]
#[derive(Deserialize)]
struct FileEdit {
  old_text: String,
  new_text: String,
  description: Option<String>,
}

/// Resolve a path relative to workspace root
fn resolve_path(workspace_root: &Path, path: &str) -> PathBuf {
  let p = Path::new(path);
  if p.is_absolute() {
    p.to_path_buf()
  } else {
    workspace_root.join(p)
  }
}

/// Check if a path should be excluded (node_modules, .git, etc.)
fn should_exclude(path: &Path) -> bool {
  let excluded = ["node_modules", ".git", "__pycache__", "target", "dist", "build", ".next", "venv", ".venv"];
  path.components().any(|c| {
    if let std::path::Component::Normal(name) = c {
      excluded.iter().any(|e| name == std::ffi::OsStr::new(e))
    } else {
      false
    }
  })
}

#[tauri::command]
async fn agent_read_file(
  workspace_root: String,
  path: String,
  start_line: Option<usize>,
  end_line: Option<usize>,
) -> ReadFileResult {
  let root = PathBuf::from(&workspace_root);
  let full_path = resolve_path(&root, &path);
  
  if !full_path.exists() {
    return ReadFileResult {
      success: false,
      content: None,
      total_lines: None,
      lines_read: None,
      error: Some(format!("File not found: {}", path)),
    };
  }
  
  match fs::read_to_string(&full_path) {
    Ok(content) => {
      let lines: Vec<&str> = content.lines().collect();
      let total_lines = lines.len();
      
      let start = start_line.unwrap_or(1).saturating_sub(1);
      let end = end_line.unwrap_or(total_lines);
      
      let selected: Vec<&str> = lines.into_iter()
        .skip(start)
        .take(end - start)
        .collect();
      
      ReadFileResult {
        success: true,
        content: Some(selected.join("\n")),
        total_lines: Some(total_lines),
        lines_read: Some(selected.len()),
        error: None,
      }
    }
    Err(e) => ReadFileResult {
      success: false,
      content: None,
      total_lines: None,
      lines_read: None,
      error: Some(e.to_string()),
    },
  }
}

#[tauri::command]
async fn agent_write_file(
  workspace_root: String,
  path: String,
  content: String,
) -> WriteFileResult {
  let root = PathBuf::from(&workspace_root);
  let full_path = resolve_path(&root, &path);
  
  // Create parent directories if needed
  if let Some(parent) = full_path.parent() {
    if let Err(e) = fs::create_dir_all(parent) {
      return WriteFileResult {
        success: false,
        bytes_written: None,
        error: Some(format!("Failed to create directories: {}", e)),
      };
    }
  }
  
  match fs::write(&full_path, &content) {
    Ok(()) => WriteFileResult {
      success: true,
      bytes_written: Some(content.len()),
      error: None,
    },
    Err(e) => WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(e.to_string()),
    },
  }
}

#[tauri::command]
async fn agent_edit_file(
  workspace_root: String,
  path: String,
  edits: Vec<FileEdit>,
) -> EditFileResult {
  let root = PathBuf::from(&workspace_root);
  let full_path = resolve_path(&root, &path);
  
  let content = match fs::read_to_string(&full_path) {
    Ok(c) => c,
    Err(e) => {
      return EditFileResult {
        success: false,
        edits_applied: 0,
        edits_requested: edits.len(),
        errors: vec![format!("Failed to read file: {}", e)],
      };
    }
  };
  
  let mut modified = content.clone();
  let mut applied = 0;
  let mut errors = Vec::new();
  
  for edit in &edits {
    if !modified.contains(&edit.old_text) {
      errors.push(format!("Could not find text: {}...", 
        edit.old_text.chars().take(50).collect::<String>()));
      continue;
    }
    
    // Check for uniqueness
    let count = modified.matches(&edit.old_text).count();
    if count > 1 {
      errors.push(format!("Text not unique ({} occurrences): {}...", 
        count, edit.old_text.chars().take(50).collect::<String>()));
      continue;
    }
    
    modified = modified.replacen(&edit.old_text, &edit.new_text, 1);
    applied += 1;
  }
  
  if applied > 0 {
    if let Err(e) = fs::write(&full_path, &modified) {
      return EditFileResult {
        success: false,
        edits_applied: 0,
        edits_requested: edits.len(),
        errors: vec![format!("Failed to write file: {}", e)],
      };
    }
  }
  
  EditFileResult {
    success: applied > 0 && errors.is_empty(),
    edits_applied: applied,
    edits_requested: edits.len(),
    errors,
  }
}

#[tauri::command]
async fn agent_create_file(
  workspace_root: String,
  path: String,
  content: String,
) -> WriteFileResult {
  let root = PathBuf::from(&workspace_root);
  let full_path = resolve_path(&root, &path);
  
  if full_path.exists() {
    return WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(format!("File already exists: {}", path)),
    };
  }
  
  // Create parent directories if needed
  if let Some(parent) = full_path.parent() {
    if let Err(e) = fs::create_dir_all(parent) {
      return WriteFileResult {
        success: false,
        bytes_written: None,
        error: Some(format!("Failed to create directories: {}", e)),
      };
    }
  }
  
  match fs::write(&full_path, &content) {
    Ok(()) => WriteFileResult {
      success: true,
      bytes_written: Some(content.len()),
      error: None,
    },
    Err(e) => WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(e.to_string()),
    },
  }
}

#[tauri::command]
async fn agent_delete_file(
  workspace_root: String,
  path: String,
) -> WriteFileResult {
  let root = PathBuf::from(&workspace_root);
  let full_path = resolve_path(&root, &path);
  
  if !full_path.exists() {
    return WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(format!("File not found: {}", path)),
    };
  }
  
  match fs::remove_file(&full_path) {
    Ok(()) => WriteFileResult {
      success: true,
      bytes_written: None,
      error: None,
    },
    Err(e) => WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(e.to_string()),
    },
  }
}

#[tauri::command]
async fn agent_rename_file(
  workspace_root: String,
  old_path: String,
  new_path: String,
) -> WriteFileResult {
  let root = PathBuf::from(&workspace_root);
  let old_full = resolve_path(&root, &old_path);
  let new_full = resolve_path(&root, &new_path);
  
  if !old_full.exists() {
    return WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(format!("Source file not found: {}", old_path)),
    };
  }
  
  if new_full.exists() {
    return WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(format!("Destination already exists: {}", new_path)),
    };
  }
  
  // Create parent directories for destination if needed
  if let Some(parent) = new_full.parent() {
    if let Err(e) = fs::create_dir_all(parent) {
      return WriteFileResult {
        success: false,
        bytes_written: None,
        error: Some(format!("Failed to create directories: {}", e)),
      };
    }
  }
  
  match fs::rename(&old_full, &new_full) {
    Ok(()) => WriteFileResult {
      success: true,
      bytes_written: None,
      error: None,
    },
    Err(e) => WriteFileResult {
      success: false,
      bytes_written: None,
      error: Some(e.to_string()),
    },
  }
}

#[tauri::command]
async fn agent_list_directory(
  workspace_root: String,
  path: String,
  recursive: Option<bool>,
  max_depth: Option<usize>,
) -> ListDirectoryResult {
  let root = PathBuf::from(&workspace_root);
  let full_path = resolve_path(&root, &path);
  
  if !full_path.exists() {
    return ListDirectoryResult {
      success: false,
      entries: Vec::new(),
      count: 0,
      error: Some(format!("Directory not found: {}", path)),
    };
  }
  
  let recursive = recursive.unwrap_or(false);
  let max_depth = max_depth.unwrap_or(2);
  let mut entries = Vec::new();
  
  if recursive {
    for entry in WalkDir::new(&full_path)
      .max_depth(max_depth)
      .into_iter()
      .filter_map(|e| e.ok())
    {
      let entry_path = entry.path();
      
      // Skip the root itself
      if entry_path == full_path {
        continue;
      }
      
      // Skip excluded directories
      if should_exclude(entry_path) {
        continue;
      }
      
      let rel_path = entry_path.strip_prefix(&full_path)
        .unwrap_or(entry_path)
        .to_string_lossy()
        .to_string();
      
      let entry_type = if entry_path.is_dir() { "directory" } else { "file" };
      let size = if entry_path.is_file() {
        entry_path.metadata().ok().map(|m| m.len())
      } else {
        None
      };
      
      entries.push(DirectoryEntry {
        name: entry.file_name().to_string_lossy().to_string(),
        path: rel_path,
        entry_type: entry_type.to_string(),
        size,
      });
    }
  } else {
    match fs::read_dir(&full_path) {
      Ok(read_dir) => {
        for entry in read_dir.filter_map(|e| e.ok()) {
          let entry_path = entry.path();
          
          // Skip excluded directories
          if should_exclude(&entry_path) {
            continue;
          }
          
          let name = entry.file_name().to_string_lossy().to_string();
          let entry_type = if entry_path.is_dir() { "directory" } else { "file" };
          let size = if entry_path.is_file() {
            entry_path.metadata().ok().map(|m| m.len())
          } else {
            None
          };
          
          entries.push(DirectoryEntry {
            name: name.clone(),
            path: name,
            entry_type: entry_type.to_string(),
            size,
          });
        }
      }
      Err(e) => {
        return ListDirectoryResult {
          success: false,
          entries: Vec::new(),
          count: 0,
          error: Some(e.to_string()),
        };
      }
    }
  }
  
  let count = entries.len();
  ListDirectoryResult {
    success: true,
    entries,
    count,
    error: None,
  }
}

#[tauri::command]
async fn agent_search_files(
  workspace_root: String,
  pattern: String,
  directory: Option<String>,
  exclude_patterns: Option<Vec<String>>,
) -> SearchResult {
  let root = PathBuf::from(&workspace_root);
  let search_dir = resolve_path(&root, &directory.unwrap_or_else(|| ".".to_string()));
  let excludes: Vec<&str> = exclude_patterns.as_ref()
    .map(|v| v.iter().map(|s| s.as_str()).collect())
    .unwrap_or_else(|| vec!["node_modules", ".git", "__pycache__"]);
  
  let mut matches = Vec::new();
  let pattern_lower = pattern.to_lowercase();
  
  // Try to parse as glob pattern
  let glob_pattern = Pattern::new(&pattern).ok();
  
  for entry in WalkDir::new(&search_dir)
    .into_iter()
    .filter_map(|e| e.ok())
  {
    let entry_path = entry.path();
    
    // Skip excluded directories
    if excludes.iter().any(|ex| {
      entry_path.components().any(|c| {
        if let std::path::Component::Normal(name) = c {
          name == std::ffi::OsStr::new(ex)
        } else {
          false
        }
      })
    }) {
      continue;
    }
    
    if entry_path.is_file() {
      let file_name = entry.file_name().to_string_lossy();
      
      let is_match = if let Some(ref glob) = glob_pattern {
        glob.matches(&file_name)
      } else {
        file_name.to_lowercase().contains(&pattern_lower)
      };
      
      if is_match {
        let rel_path = entry_path.strip_prefix(&search_dir)
          .unwrap_or(entry_path)
          .to_string_lossy()
          .to_string();
        
        matches.push(SearchMatch {
          path: rel_path,
          line: None,
          column: None,
          preview: None,
          matched_text: None,
        });
        
        // Limit results
        if matches.len() >= 100 {
          break;
        }
      }
    }
  }
  
  let total = matches.len();
  SearchResult {
    success: true,
    matches,
    total_matches: total,
    error: None,
  }
}

#[tauri::command]
async fn agent_search_in_files(
  workspace_root: String,
  query: String,
  directory: Option<String>,
  file_pattern: Option<String>,
  is_regex: Option<bool>,
) -> SearchResult {
  let root = PathBuf::from(&workspace_root);
  let search_dir = resolve_path(&root, &directory.unwrap_or_else(|| ".".to_string()));
  let is_regex = is_regex.unwrap_or(false);
  
  let regex = if is_regex {
    match regex::Regex::new(&query) {
      Ok(r) => Some(r),
      Err(e) => {
        return SearchResult {
          success: false,
          matches: Vec::new(),
          total_matches: 0,
          error: Some(format!("Invalid regex: {}", e)),
        };
      }
    }
  } else {
    None
  };
  
  let file_glob = file_pattern.as_ref().and_then(|p| Pattern::new(p).ok());
  let mut matches = Vec::new();
  
  for entry in WalkDir::new(&search_dir)
    .into_iter()
    .filter_map(|e| e.ok())
  {
    let entry_path = entry.path();
    
    // Skip excluded directories
    if should_exclude(entry_path) {
      continue;
    }
    
    if !entry_path.is_file() {
      continue;
    }
    
    // Check file pattern
    if let Some(ref glob) = file_glob {
      if !glob.matches(&entry.file_name().to_string_lossy()) {
        continue;
      }
    }
    
    // Read and search file
    let file = match fs::File::open(entry_path) {
      Ok(f) => f,
      Err(_) => continue,
    };
    
    let reader = BufReader::new(file);
    
    for (line_num, line) in reader.lines().enumerate() {
      let line = match line {
        Ok(l) => l,
        Err(_) => continue,
      };
      
      let found = if let Some(ref re) = regex {
        re.find(&line).map(|m| (m.start(), m.as_str().to_string()))
      } else if line.contains(&query) {
        line.find(&query).map(|pos| (pos, query.clone()))
      } else {
        None
      };
      
      if let Some((col, matched)) = found {
        let rel_path = entry_path.strip_prefix(&search_dir)
          .unwrap_or(entry_path)
          .to_string_lossy()
          .to_string();
        
        matches.push(SearchMatch {
          path: rel_path,
          line: Some(line_num + 1),
          column: Some(col + 1),
          preview: Some(line.chars().take(100).collect()),
          matched_text: Some(matched),
        });
        
        // Limit results
        if matches.len() >= 100 {
          break;
        }
      }
    }
    
    if matches.len() >= 100 {
      break;
    }
  }
  
  let total = matches.len();
  SearchResult {
    success: true,
    matches,
    total_matches: total,
    error: None,
  }
}

#[tauri::command]
async fn agent_run_command(
  workspace_root: String,
  command: String,
  cwd: Option<String>,
  timeout: Option<u64>,
) -> CommandResult {
  let root = PathBuf::from(&workspace_root);
  let working_dir = cwd.map(|c| resolve_path(&root, &c)).unwrap_or(root);
  let _timeout_secs = timeout.unwrap_or(30);
  
  #[cfg(target_os = "windows")]
  let output = Command::new("cmd")
    .args(&["/C", &command])
    .current_dir(&working_dir)
    .output();
  
  #[cfg(not(target_os = "windows"))]
  let output = Command::new("sh")
    .args(&["-c", &command])
    .current_dir(&working_dir)
    .output();
  
  match output {
    Ok(output) => {
      let stdout = String::from_utf8_lossy(&output.stdout);
      let stderr = String::from_utf8_lossy(&output.stderr);
      
      CommandResult {
        success: output.status.success(),
        exit_code: output.status.code(),
        stdout: Some(stdout.chars().take(5000).collect()),
        stderr: Some(stderr.chars().take(5000).collect()),
        error: None,
      }
    }
    Err(e) => CommandResult {
      success: false,
      exit_code: None,
      stdout: None,
      stderr: None,
      error: Some(e.to_string()),
    },
  }
}

#[tauri::command]
async fn agent_get_workspace_info(
  workspace_root: String,
) -> WorkspaceInfo {
  let root = PathBuf::from(&workspace_root);
  let name = root.file_name()
    .map(|n| n.to_string_lossy().to_string())
    .unwrap_or_else(|| "workspace".to_string());
  
  WorkspaceInfo {
    root: workspace_root,
    name,
    has_package_json: root.join("package.json").exists(),
    has_cargo_toml: root.join("Cargo.toml").exists(),
    has_pyproject: root.join("pyproject.toml").exists() || root.join("setup.py").exists(),
  }
}

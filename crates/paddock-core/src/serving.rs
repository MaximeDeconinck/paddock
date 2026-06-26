//! Registry of HTTP servers spawned by `paddock serve`, plus live Ollama
//! discovery. Consumed by `paddock tray` (and future UIs).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::PaddockError;
use crate::catalog::RuntimeKind;
use crate::hardware::SystemProbe;
use crate::runtime::ServePlan;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingRecord {
    pub pid: u32,
    pub runtime: RuntimeKind,
    pub endpoint: String,
    pub openai_url: String,
    pub model_ref: String,
    pub ready_path: String,
    pub started_at: i64,
    /// Resolved context window the server was launched with (0 for legacy records).
    #[serde(default)]
    pub ctx: u32,
    /// Log file for detached spawned children; None for foreground / Ollama.
    #[serde(default)]
    pub log_path: Option<PathBuf>,
    /// Port for spawned servers (None for the Ollama daemon).
    #[serde(default)]
    pub port: Option<u16>,
}

pub struct Registry {
    dir: PathBuf,
}

/// Default registry dir; `PADDOCK_SERVING_DIR` overrides (tests).
pub fn default_serving_dir() -> PathBuf {
    if let Ok(p) = std::env::var("PADDOCK_SERVING_DIR") {
        return PathBuf::from(p);
    }
    crate::paths::app_support_dir().join("serving")
}

impl Registry {
    pub fn at(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }

    pub fn open_default() -> Self {
        Self::at(default_serving_dir())
    }

    fn path_for(&self, pid: u32) -> PathBuf {
        self.dir.join(format!("{pid}.json"))
    }

    pub fn register(&self, r: &ServingRecord) -> Result<(), PaddockError> {
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| PaddockError::Other(format!("cannot create {:?}: {e}", self.dir)))?;
        let json = serde_json::to_vec_pretty(r).map_err(|e| PaddockError::Other(e.to_string()))?;
        let tmp = self.dir.join(format!("{}.json.tmp", r.pid));
        std::fs::write(&tmp, json)
            .map_err(|e| PaddockError::Other(format!("cannot write serving record: {e}")))?;
        std::fs::rename(&tmp, self.path_for(r.pid))
            .map_err(|e| PaddockError::Other(format!("cannot finalize serving record: {e}")))
    }

    pub fn unregister(&self, pid: u32) -> Result<(), PaddockError> {
        match std::fs::remove_file(self.path_for(pid)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(PaddockError::Other(format!(
                "cannot remove serving record: {e}"
            ))),
        }
    }

    /// Live records only. Deletes a file only when its JSON is unparseable
    /// (and its stem is a u32) or its PID is dead. An alive PID whose ready
    /// probe fails is filtered from the result but the file is kept (transient
    /// busy server). Files whose stem isn't a u32 are never touched.
    pub fn list_live(&self, probe: &dyn SystemProbe) -> Vec<ServingRecord> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut live = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let stem_is_pid = path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.parse::<u32>().is_ok());
            if !stem_is_pid {
                continue;
            }
            let record = std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<ServingRecord>(&bytes).ok());
            match record {
                None => {
                    let _ = std::fs::remove_file(&path);
                }
                Some(r) if !pid_alive(r.pid) => {
                    let _ = std::fs::remove_file(&path);
                }
                Some(r) => {
                    if probe
                        .http_get_local(&format!("{}{}", r.endpoint, r.ready_path))
                        .is_some()
                    {
                        live.push(r);
                    }
                }
            }
        }
        live.sort_by_key(|r| r.started_at);
        live
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub plan: ServePlan,
    pub last_served_at: i64,
}

/// Persistent log of spawned (llama.cpp/mlx) serves, so the TUI can offer them
/// for one-key relaunch. Self-contained: stores the full `ServePlan`.
pub struct History {
    dir: PathBuf,
}

impl History {
    pub fn at(dir: impl AsRef<Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }
    pub fn open_default() -> Self {
        Self::at(default_serving_dir())
    }
    fn path(&self) -> PathBuf {
        self.dir.join("history.json")
    }

    pub fn load(&self) -> Vec<HistoryEntry> {
        std::fs::read(self.path())
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    /// Upsert by `plan.model_ref`, stamping `last_served_at`. Best-effort: a
    /// write failure is ignored (history is a convenience, not load-bearing).
    pub fn record(&self, plan: &ServePlan, now: i64) {
        let mut entries = self.load();
        entries.retain(|e| e.plan.model_ref != plan.model_ref);
        entries.push(HistoryEntry {
            plan: plan.clone(),
            last_served_at: now,
        });
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let Ok(json) = serde_json::to_vec_pretty(&entries) else {
            return;
        };
        let tmp = self.dir.join("history.json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, self.path());
        }
    }
}

/// `kill(pid, 0)` liveness probe (signal 0 = existence check, no signal sent).
/// kill→-1/EPERM (alive process owned by another user) reads as dead here -
/// acceptable, we only track our own children.
fn pid_alive(pid: u32) -> bool {
    if pid == 0 || pid > i32::MAX as u32 {
        return false;
    }
    // SAFETY: kill with signal 0 only checks existence/permission.
    unsafe { libc_kill(pid as i32, 0) == 0 }
}

// Tiny extern to avoid adding the libc crate for one syscall.
unsafe extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

/// Result of resolving a `stop`/`logs` target against live records.
pub enum RecordMatch<'a> {
    /// One or more records to act on (also the `all` case).
    Matched(Vec<&'a ServingRecord>),
    /// A name substring hit several models - caller lists and aborts.
    Ambiguous(Vec<&'a ServingRecord>),
    NotFound,
}

impl<'a> RecordMatch<'a> {
    pub fn matched(&self) -> &[&'a ServingRecord] {
        match self {
            RecordMatch::Matched(v) => v,
            _ => &[],
        }
    }
}

/// Resolve a target: `all` → every record; all-digits → exact PID; otherwise a
/// case-insensitive substring of `model_ref` (Ambiguous if >1 distinct model).
pub fn match_records<'a>(records: &'a [ServingRecord], target: &str) -> RecordMatch<'a> {
    if target == "all" {
        return if records.is_empty() {
            RecordMatch::NotFound
        } else {
            RecordMatch::Matched(records.iter().collect())
        };
    }
    if let Ok(pid) = target.parse::<u32>() {
        return match records.iter().find(|r| r.pid == pid) {
            Some(r) => RecordMatch::Matched(vec![r]),
            None => RecordMatch::NotFound,
        };
    }
    let needle = target.to_lowercase();
    let hits: Vec<&ServingRecord> = records
        .iter()
        .filter(|r| r.model_ref.to_lowercase().contains(&needle))
        .collect();
    match hits.len() {
        0 => RecordMatch::NotFound,
        1 => RecordMatch::Matched(hits),
        _ => RecordMatch::Ambiguous(hits),
    }
}

/// Send SIGTERM to a pid. Best-effort; a dead pid is a no-op.
pub fn terminate(pid: u32) {
    if pid == 0 || pid > i32::MAX as u32 {
        return;
    }
    // SAFETY: kill with SIGTERM (15) signals an existing process.
    unsafe { libc_kill(pid as i32, 15) };
}

/// First free local TCP port at or above `start` (scans up to `start + 50`).
/// Returns None if the whole range is taken. Best-effort: a TOCTOU window
/// remains between this check and the server actually binding the port.
pub fn free_port(start: u16) -> Option<u16> {
    (start..=start.saturating_add(50))
        .find(|&p| std::net::TcpListener::bind(("127.0.0.1", p)).is_ok())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub name: String,
    pub size_bytes: u64,
}

const OLLAMA_BASE: &str = "http://127.0.0.1:11434";
const OLLAMA_OPENAI_URL: &str = "http://127.0.0.1:11434/v1/chat/completions";
const OLLAMA_PS_URL: &str = "http://127.0.0.1:11434/api/ps";

/// Models currently loaded in the local Ollama daemon, None when unreachable.
pub fn ollama_loaded_models(probe: &dyn SystemProbe) -> Option<Vec<LoadedModel>> {
    let body = probe.http_get_local(OLLAMA_PS_URL)?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    Some(
        v["models"]
            .as_array()?
            .iter()
            .filter_map(|m| {
                Some(LoadedModel {
                    name: m["name"].as_str()?.to_string(),
                    size_bytes: m["size"].as_u64().unwrap_or(0),
                })
            })
            .collect(),
    )
}

/// How to stop a running server shown in a UI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopHandle {
    /// A paddock-spawned child: SIGTERM the pid then unregister.
    Pid(u32),
    /// An Ollama-loaded model: `ollama stop <model>` (the daemon keeps running).
    OllamaModel(String),
}

/// A unified "running server" row for UIs: paddock-spawned servers from the
/// registry plus Ollama-loaded models from `/api/ps`. Fields that don't apply
/// to a source are `None` (Ollama has no ctx/start-time/pid; paddock-spawned
/// has no size).
#[derive(Debug, Clone)]
pub struct ServerRow {
    pub model: String,
    pub runtime: RuntimeKind,
    pub endpoint: String,
    pub openai_url: String,
    pub ctx: Option<u32>,
    pub started_at: Option<i64>,
    pub stop: StopHandle,
}

/// All running servers a UI should show: paddock-spawned (llama.cpp/mlx) from
/// `registry.list_live`, followed by Ollama-loaded models from `/api/ps`.
pub fn list_all_servers(registry: &Registry, probe: &dyn SystemProbe) -> Vec<ServerRow> {
    let mut rows: Vec<ServerRow> = registry
        .list_live(probe)
        .into_iter()
        .map(|r| ServerRow {
            model: r.model_ref,
            runtime: r.runtime,
            endpoint: r.endpoint,
            openai_url: r.openai_url,
            // 0 is the "not applicable" sentinel (mlx-lm runs without a paddock
            // ctx flag) - surface it as None so the UI shows "-", not "0".
            ctx: (r.ctx != 0).then_some(r.ctx),
            started_at: Some(r.started_at),
            stop: StopHandle::Pid(r.pid),
        })
        .collect();
    if let Some(models) = ollama_loaded_models(probe) {
        for m in models {
            rows.push(ServerRow {
                model: m.name.clone(),
                runtime: RuntimeKind::Ollama,
                endpoint: OLLAMA_BASE.to_string(),
                openai_url: OLLAMA_OPENAI_URL.to_string(),
                ctx: None,
                started_at: None,
                stop: StopHandle::OllamaModel(m.name),
            });
        }
    }
    rows
}

const OLLAMA_GENERATE_URL: &str = "http://127.0.0.1:11434/api/generate";
/// How long a freshly served model stays loaded without traffic. Long enough
/// to bridge the gap between `paddock serve` and the user's first request.
const WARM_UP_KEEP_ALIVE: &str = "30m";

/// Load `model_ref` into the local Ollama daemon's memory so the first real
/// request doesn't pay the cold start (and so the model shows up in
/// `/api/ps` - and the tray - right away). A prompt-less `/api/generate`
/// is Ollama's documented "just load it" call. Returns false when the
/// daemon is unreachable or refuses; callers treat this as best-effort.
pub fn warm_up_ollama(probe: &dyn SystemProbe, model_ref: &str) -> bool {
    let body = serde_json::json!({
        "model": model_ref,
        "keep_alive": WARM_UP_KEEP_ALIVE,
    })
    .to_string();
    probe.http_post_local(OLLAMA_GENERATE_URL, &body).is_some()
}

#[cfg(test)]
mod free_port_tests {
    use super::*;

    #[test]
    fn free_port_skips_an_occupied_port() {
        // Hold a listener on a high port, then free_port(that) must skip it.
        let held = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let taken = held.local_addr().unwrap().port();
        let got = free_port(taken).expect("a free port above the taken one");
        assert_ne!(got, taken);
        assert!(got >= taken);
        // and the returned port is actually bindable
        std::net::TcpListener::bind(("127.0.0.1", got)).unwrap();
    }
}

#[cfg(test)]
mod record_tests {
    use super::*;

    #[test]
    fn record_roundtrips_new_fields() {
        let r = ServingRecord {
            pid: 42,
            runtime: RuntimeKind::LlamaCpp,
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "repo:Q4_K_M".into(),
            ready_path: "/health".into(),
            started_at: 1000,
            ctx: 32768,
            log_path: Some(std::path::PathBuf::from("/tmp/42.log")),
            port: Some(8080),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: ServingRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back.ctx, 32768);
        assert_eq!(back.log_path, Some(std::path::PathBuf::from("/tmp/42.log")));
        assert_eq!(back.port, Some(8080));
    }

    #[test]
    fn old_record_without_new_fields_still_deserializes() {
        let old = r#"{
            "pid": 7, "runtime": "ollama",
            "endpoint": "http://127.0.0.1:11434",
            "openai_url": "http://127.0.0.1:11434/v1/chat/completions",
            "model_ref": "llama3.2:1b", "ready_path": "/api/version",
            "started_at": 5
        }"#;
        let r: ServingRecord = serde_json::from_str(old).unwrap();
        assert_eq!(r.ctx, 0);
        assert_eq!(r.log_path, None);
        assert_eq!(r.port, None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MockProbe;

    #[test]
    fn warm_up_posts_model_and_keep_alive() {
        let mut probe = MockProbe::default();
        probe.posts.insert(
            "http://127.0.0.1:11434/api/generate".into(),
            r#"{"done":true}"#.into(),
        );
        assert!(warm_up_ollama(&probe, "lfm2.5-thinking:1.2b-q8_0"));
        let bodies = probe.post_bodies.lock().unwrap();
        assert_eq!(bodies.len(), 1);
        let (url, body) = &bodies[0];
        assert_eq!(url, "http://127.0.0.1:11434/api/generate");
        let v: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["model"], "lfm2.5-thinking:1.2b-q8_0");
        assert_eq!(v["keep_alive"], "30m");
        // No prompt: load-only, never generates tokens.
        assert!(v.get("prompt").is_none());
    }

    #[test]
    fn warm_up_daemon_unreachable_is_false() {
        let probe = MockProbe::default();
        assert!(!warm_up_ollama(&probe, "llama3.2:1b"));
    }

    fn record(pid: u32) -> ServingRecord {
        ServingRecord {
            pid,
            runtime: crate::catalog::RuntimeKind::LlamaCpp,
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "repo:Q4_K_M".into(),
            ready_path: "/health".into(),
            started_at: 1_770_000_000,
            ctx: 0,
            log_path: None,
            port: None,
        }
    }

    #[test]
    fn register_list_unregister_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::at(dir.path());
        // our own PID is alive; probe answers the health check
        let mut probe = MockProbe::default();
        probe
            .http
            .insert("http://127.0.0.1:8080/health".into(), "ok".into());
        let r = record(std::process::id());
        reg.register(&r).unwrap();
        // atomic register: no .tmp lingers, exactly one file
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
        let live = reg.list_live(&probe);
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].model_ref, "repo:Q4_K_M");
        reg.unregister(r.pid).unwrap();
        assert!(reg.list_live(&probe).is_empty());
    }

    #[test]
    fn dead_pid_is_reaped() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::at(dir.path());
        let mut probe = MockProbe::default();
        probe
            .http
            .insert("http://127.0.0.1:8080/health".into(), "ok".into());
        reg.register(&record(4_000_000_000)).unwrap(); // PID far beyond pid_max
        assert!(reg.list_live(&probe).is_empty());
        // file physically gone
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn unreachable_endpoint_is_filtered_not_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::at(dir.path());
        let probe = MockProbe::default(); // no HTTP answers
        let pid = std::process::id();
        reg.register(&record(pid)).unwrap();
        assert!(reg.list_live(&probe).is_empty());
        // alive PID, transient probe failure: file kept
        assert!(dir.path().join(format!("{pid}.json")).exists());
    }

    #[test]
    fn foreign_json_files_untouched() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.json"), b"{not json").unwrap();
        let reg = Registry::at(dir.path());
        assert!(reg.list_live(&MockProbe::default()).is_empty());
        assert!(dir.path().join("notes.json").exists());
    }

    #[test]
    fn corrupt_record_file_is_ignored_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("999.json"), b"{not json").unwrap();
        let reg = Registry::at(dir.path());
        assert!(reg.list_live(&MockProbe::default()).is_empty());
    }

    #[test]
    fn parses_ollama_ps() {
        let mut probe = MockProbe::default();
        probe.http.insert(
            "http://127.0.0.1:11434/api/ps".into(),
            r#"{"models":[{"name":"qwen3:30b-a3b","model":"qwen3:30b-a3b","size":19327352832,"expires_at":"2026-06-10T12:00:00Z"},{"name":"llama3.2:1b","size":1400000000}]}"#.into(),
        );
        let models = ollama_loaded_models(&probe).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "qwen3:30b-a3b");
        assert_eq!(models[0].size_bytes, 19_327_352_832);
    }

    #[test]
    fn ollama_ps_unreachable_is_none() {
        assert!(ollama_loaded_models(&MockProbe::default()).is_none());
    }
}

#[cfg(test)]
mod list_all_tests {
    use super::*;
    use crate::hardware::MockProbe;

    #[test]
    fn merges_ollama_loaded_models() {
        let dir = std::env::temp_dir().join(format!("paddock-test-{}", std::process::id()));
        let registry = Registry::at(&dir); // empty registry (no files)
        let mut probe = MockProbe::default();
        probe.http.insert(
            "http://127.0.0.1:11434/api/ps".to_string(),
            r#"{"models":[{"name":"gemma:12b","size":18000000000}]}"#.to_string(),
        );
        let rows = list_all_servers(&registry, &probe);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "gemma:12b");
        assert_eq!(rows[0].runtime, RuntimeKind::Ollama);
        assert_eq!(rows[0].ctx, None);
        assert_eq!(rows[0].stop, StopHandle::OllamaModel("gemma:12b".to_string()));
        assert_eq!(rows[0].endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn empty_when_nothing_running() {
        let dir = std::env::temp_dir().join(format!("paddock-test-empty-{}", std::process::id()));
        let registry = Registry::at(&dir);
        let probe = MockProbe::default(); // no /api/ps fixture -> ollama_loaded_models None
        assert!(list_all_servers(&registry, &probe).is_empty());
    }
}

#[cfg(test)]
mod match_tests {
    use super::*;

    fn rec(pid: u32, model: &str) -> ServingRecord {
        ServingRecord {
            pid,
            runtime: RuntimeKind::LlamaCpp,
            endpoint: "e".into(),
            openai_url: "o".into(),
            model_ref: model.into(),
            ready_path: "/health".into(),
            started_at: 0,
            ctx: 8192,
            log_path: None,
            port: Some(8080),
        }
    }

    #[test]
    fn match_all_returns_everything() {
        let rs = vec![rec(1, "a"), rec(2, "b")];
        assert_eq!(match_records(&rs, "all").matched().len(), 2);
    }

    #[test]
    fn match_by_pid() {
        let rs = vec![rec(10, "a"), rec(20, "b")];
        let m = match_records(&rs, "20");
        assert_eq!(m.matched().len(), 1);
        assert_eq!(m.matched()[0].pid, 20);
    }

    #[test]
    fn match_by_model_substring() {
        let rs = vec![rec(1, "qwen3-35b"), rec(2, "llama3-8b")];
        let m = match_records(&rs, "qwen");
        assert_eq!(m.matched().len(), 1);
        assert_eq!(m.matched()[0].pid, 1);
    }

    #[test]
    fn ambiguous_substring_lists_candidates() {
        let rs = vec![rec(1, "qwen3-35b"), rec(2, "qwen3-8b")];
        assert!(matches!(
            match_records(&rs, "qwen"),
            RecordMatch::Ambiguous(_)
        ));
    }

    #[test]
    fn no_match_is_not_found() {
        let rs = vec![rec(1, "a")];
        assert!(matches!(match_records(&rs, "zzz"), RecordMatch::NotFound));
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;
    use crate::catalog::RuntimeKind;

    fn plan(model: &str, port: u16) -> crate::runtime::ServePlan {
        crate::runtime::ServePlan {
            server_argv: Some(vec![
                "mlx_lm.server".into(),
                "--model".into(),
                model.into(),
                "--port".into(),
                port.to_string(),
            ]),
            pre_steps: vec![],
            endpoint: format!("http://127.0.0.1:{port}"),
            openai_url: format!("http://127.0.0.1:{port}/v1/chat/completions"),
            model_ref: model.into(),
            ready_path: "/v1/models".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::MlxLm,
            ctx: 0,
            port: Some(port),
        }
    }

    #[test]
    fn record_then_load_roundtrips() {
        let dir = std::env::temp_dir().join(format!("paddock-hist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let h = History::at(&dir);
        h.record(&plan("mlx-community/A", 8080), 1000);
        let loaded = h.load();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].plan.model_ref, "mlx-community/A");
        assert_eq!(loaded[0].last_served_at, 1000);
    }

    #[test]
    fn record_upserts_by_model_ref() {
        let dir = std::env::temp_dir().join(format!("paddock-hist-up-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let h = History::at(&dir);
        h.record(&plan("mlx-community/A", 8080), 1000);
        h.record(&plan("mlx-community/A", 8081), 2000); // same model_ref, later
        h.record(&plan("mlx-community/B", 8082), 1500);
        let loaded = h.load();
        assert_eq!(loaded.len(), 2, "A upserted, B added");
        let a = loaded
            .iter()
            .find(|e| e.plan.model_ref == "mlx-community/A")
            .unwrap();
        assert_eq!(a.last_served_at, 2000);
    }
}

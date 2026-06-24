//! Registry of HTTP servers spawned by `paddock serve`, plus live Ollama
//! discovery. Consumed by `paddock tray` (and future UIs).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::PaddockError;
use crate::catalog::RuntimeKind;
use crate::hardware::SystemProbe;

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

/// `kill(pid, 0)` liveness probe (signal 0 = existence check, no signal sent).
/// kill→-1/EPERM (alive process owned by another user) reads as dead here —
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModel {
    pub name: String,
    pub size_bytes: u64,
}

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

const OLLAMA_GENERATE_URL: &str = "http://127.0.0.1:11434/api/generate";
/// How long a freshly served model stays loaded without traffic. Long enough
/// to bridge the gap between `paddock serve` and the user's first request.
const WARM_UP_KEEP_ALIVE: &str = "30m";

/// Load `model_ref` into the local Ollama daemon's memory so the first real
/// request doesn't pay the cold start (and so the model shows up in
/// `/api/ps` — and the tray — right away). A prompt-less `/api/generate`
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

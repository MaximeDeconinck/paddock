//! Registry of HTTP servers spawned by `tetro serve`, plus live Ollama
//! discovery. Consumed by `tetro tray` (and future UIs).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog::RuntimeKind;
use crate::hardware::SystemProbe;
use crate::TetroError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServingRecord {
    pub pid: u32,
    pub runtime: RuntimeKind,
    pub endpoint: String,
    pub openai_url: String,
    pub model_ref: String,
    pub ready_path: String,
    pub started_at: i64,
}

pub struct Registry {
    dir: PathBuf,
}

/// Default registry dir; `TETRO_SERVING_DIR` overrides (tests).
pub fn default_serving_dir() -> PathBuf {
    if let Ok(p) = std::env::var("TETRO_SERVING_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join("Library/Application Support/tetro/serving")
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

    pub fn register(&self, r: &ServingRecord) -> Result<(), TetroError> {
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| TetroError::Other(format!("cannot create {:?}: {e}", self.dir)))?;
        let json = serde_json::to_vec_pretty(r).map_err(|e| TetroError::Other(e.to_string()))?;
        std::fs::write(self.path_for(r.pid), json)
            .map_err(|e| TetroError::Other(format!("cannot write serving record: {e}")))
    }

    pub fn unregister(&self, pid: u32) -> Result<(), TetroError> {
        match std::fs::remove_file(self.path_for(pid)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(TetroError::Other(format!(
                "cannot remove serving record: {e}"
            ))),
        }
    }

    /// Live records only; stale files (dead PID, unreachable endpoint,
    /// unparseable JSON) are deleted as a side effect.
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
            let record = std::fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<ServingRecord>(&bytes).ok());
            let alive = record.as_ref().is_some_and(|r| {
                pid_alive(r.pid)
                    && probe
                        .http_get_local(&format!("{}{}", r.endpoint, r.ready_path))
                        .is_some()
            });
            match (record, alive) {
                (Some(r), true) => live.push(r),
                _ => {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        live.sort_by_key(|r| r.started_at);
        live
    }
}

/// `kill(pid, 0)` liveness probe (signal 0 = existence check, no signal sent).
fn pid_alive(pid: u32) -> bool {
    if pid == 0 || pid > i32::MAX as u32 {
        return false;
    }
    // SAFETY: kill with signal 0 only checks existence/permission.
    unsafe { libc_kill(pid as i32, 0) == 0 }
}

// Tiny extern to avoid adding the libc crate for one syscall.
extern "C" {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MockProbe;

    fn record(pid: u32) -> ServingRecord {
        ServingRecord {
            pid,
            runtime: crate::catalog::RuntimeKind::LlamaCpp,
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "repo:Q4_K_M".into(),
            ready_path: "/health".into(),
            started_at: 1_770_000_000,
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
    fn unreachable_endpoint_is_reaped() {
        let dir = tempfile::tempdir().unwrap();
        let reg = Registry::at(dir.path());
        let probe = MockProbe::default(); // no HTTP answers
        reg.register(&record(std::process::id())).unwrap();
        assert!(reg.list_live(&probe).is_empty());
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

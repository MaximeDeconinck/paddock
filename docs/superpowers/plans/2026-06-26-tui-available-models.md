# TUI Available Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show locally-available-but-not-running models (Ollama installed + a paddock served-history for llama.cpp/mlx) in a grey "AVAILABLE" group on the TUI servers tab, launchable with `enter`.

**Architecture:** A `History` store (mirrors `Registry`) persists each spawned serve as a self-contained `ServePlan`. `list_available` merges Ollama `/api/tags` with the cache-checked history, dedup'd against the running set. The TUI servers tab navigates both groups; `enter` on an available row reuses the existing `Action::Serve` flow.

**Tech Stack:** Rust (edition 2024), serde/serde_json, ratatui/crossterm, `std::net`/`MockProbe` for tests.

---

## File structure

- `crates/paddock-core/src/runtime.rs` — add `Deserialize` to `ServePlan`/`InstallPlan`. (Modify)
- `crates/paddock-core/src/serving.rs` — `History` store, `ollama_installed_models`, `AvailableRow`, `list_available`, cache check. (Modify)
- `crates/paddock-core/src/paths.rs` — `hf_cache_dir()` helper. (Modify)
- `crates/paddock/src/main.rs` — record each spawned serve into `History` inside `serve_with_plan`. (Modify)
- `crates/paddock/src/tui/servers_task.rs` — produce a `ServersSnapshot { running, available }`. (Modify)
- `crates/paddock/src/tui/state.rs` — `available` field, combined navigation, `enter` launch, `set_snapshot`. (Modify)
- `crates/paddock/src/tui/mod.rs` — drain `ServersSnapshot`, set both groups, inline-serve refresh rebuilds both. (Modify)
- `crates/paddock/src/tui/draw.rs` — render the AVAILABLE group + footer hint. (Modify)
- `README.md` — document the available group. (Modify)

---

## Task 1: History store (+ ServePlan/InstallPlan Deserialize)

**Files:**
- Modify: `crates/paddock-core/src/runtime.rs` (derives)
- Modify: `crates/paddock-core/src/serving.rs` (History)
- Test: both inline `#[cfg(test)]`

- [ ] **Step 1: Add `Deserialize` so a `ServePlan` round-trips through JSON**

In `runtime.rs`, change the derive on BOTH `InstallPlan` (line ~11) and `ServePlan` (line ~149) from `#[derive(Debug, Clone, Serialize)]` to:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
```

And update the import at the top of `runtime.rs`: `use serde::Serialize;` becomes `use serde::{Deserialize, Serialize};`. (`RuntimeKind` already derives both.)

- [ ] **Step 2: Write the failing History tests**

Add to the `serving.rs` test area (a new module):

```rust
#[cfg(test)]
mod history_tests {
    use super::*;
    use crate::catalog::RuntimeKind;

    fn plan(model: &str, port: u16) -> crate::runtime::ServePlan {
        crate::runtime::ServePlan {
            server_argv: Some(vec![
                "mlx_lm.server".into(), "--model".into(), model.into(),
                "--port".into(), port.to_string(),
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
        let a = loaded.iter().find(|e| e.plan.model_ref == "mlx-community/A").unwrap();
        assert_eq!(a.last_served_at, 2000);
    }
}
```

- [ ] **Step 3:** `cargo test -p paddock-core history_tests` → FAIL (`History` missing).

- [ ] **Step 4: Implement `History`** in `serving.rs` (near `Registry`). Mirror `Registry`'s dir + atomic-write style:

```rust
use crate::runtime::ServePlan;

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
        Self { dir: dir.as_ref().to_path_buf() }
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
        entries.push(HistoryEntry { plan: plan.clone(), last_served_at: now });
        if std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let Ok(json) = serde_json::to_vec_pretty(&entries) else { return };
        let tmp = self.dir.join("history.json.tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, self.path());
        }
    }
}
```

(`PathBuf`/`Path` are already imported in serving.rs; `serde::{Serialize, Deserialize}` too.)

- [ ] **Step 5:** `cargo test -p paddock-core history_tests` → PASS (2). `cargo build` clean.

- [ ] **Step 6: Commit**

```bash
git add crates/paddock-core/src/runtime.rs crates/paddock-core/src/serving.rs
git commit -m "feat(core): served-history store for relaunchable spawned servers"
```

---

## Task 2: Ollama installed models via `/api/tags`

**Files:**
- Modify: `crates/paddock-core/src/serving.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod installed_tests {
    use super::*;
    use crate::hardware::MockProbe;

    #[test]
    fn parses_ollama_tags() {
        let mut probe = MockProbe::default();
        probe.http.insert(
            "http://127.0.0.1:11434/api/tags".to_string(),
            r#"{"models":[{"name":"gemma4:26b","size":18000000000},{"name":"lfm2.5:latest","size":5200000000}]}"#.to_string(),
        );
        let got = ollama_installed_models(&probe).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "gemma4:26b");
        assert_eq!(got[0].size_bytes, 18000000000);
    }

    #[test]
    fn none_when_daemon_down() {
        assert!(ollama_installed_models(&MockProbe::default()).is_none());
    }
}
```

- [ ] **Step 2:** `cargo test -p paddock-core installed_tests` → FAIL.

- [ ] **Step 3: Implement** in `serving.rs`. Mirror `ollama_loaded_models` exactly, hitting `/api/tags`:

```rust
const OLLAMA_TAGS_URL: &str = "http://127.0.0.1:11434/api/tags";

/// Models installed in the local Ollama daemon (`/api/tags`), None when
/// unreachable. Reuses `LoadedModel { name, size_bytes }`.
pub fn ollama_installed_models(probe: &dyn SystemProbe) -> Option<Vec<LoadedModel>> {
    let body = probe.http_get_local(OLLAMA_TAGS_URL)?;
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
```

- [ ] **Step 4:** `cargo test -p paddock-core installed_tests` → PASS (2).

- [ ] **Step 5: Commit**

```bash
git add crates/paddock-core/src/serving.rs
git commit -m "feat(core): list installed Ollama models via /api/tags"
```

---

## Task 3: HF cache dir helper + `AvailableRow` + `list_available`

**Files:**
- Modify: `crates/paddock-core/src/paths.rs`
- Modify: `crates/paddock-core/src/serving.rs`
- Test: inline in both

- [ ] **Step 1: Add `hf_cache_dir()` to `paths.rs` with a test**

Append to `paths.rs`:

```rust
/// HuggingFace hub cache root (`~/.cache/huggingface/hub`), where mlx and
/// `-hf` GGUF models land. `HF_HOME` overrides the `~/.cache/huggingface` base.
pub fn hf_cache_dir() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/huggingface/hub")
}
```

Add to the `paths.rs` test module:

```rust
    #[test]
    fn hf_home_overrides_cache_dir() {
        // SAFETY: single-threaded test; set+read+remove the env var.
        unsafe { std::env::set_var("HF_HOME", "/tmp/xyzhf") };
        assert_eq!(hf_cache_dir(), std::path::PathBuf::from("/tmp/xyzhf/hub"));
        unsafe { std::env::remove_var("HF_HOME") };
    }
```

(edition 2024 makes `set_var`/`remove_var` `unsafe`; match whatever the existing paths tests do — if they don't touch env, the `unsafe` blocks above are correct for 2024.)

Run `cargo test -p paddock-core hf_home_overrides` → fails, then passes after adding the fn.

- [ ] **Step 2: Write the failing `list_available` test** in `serving.rs`:

```rust
#[cfg(test)]
mod available_tests {
    use super::*;
    use crate::hardware::MockProbe;

    #[test]
    fn merges_ollama_installed_and_dedups_running() {
        let dir = std::env::temp_dir().join(format!("paddock-avail-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let history = History::at(&dir); // empty history for this case
        let mut probe = MockProbe::default();
        probe.http.insert(
            "http://127.0.0.1:11434/api/tags".to_string(),
            r#"{"models":[{"name":"gemma4:26b","size":17000000000},{"name":"lfm2.5:latest","size":5200000000}]}"#.to_string(),
        );
        // gemma is already running -> must be excluded from available.
        let running = vec![ServerRow {
            model: "gemma4:26b".into(),
            runtime: RuntimeKind::Ollama,
            endpoint: "http://127.0.0.1:11434".into(),
            openai_url: "http://127.0.0.1:11434/v1/chat/completions".into(),
            ctx: None,
            started_at: None,
            stop: StopHandle::OllamaModel("gemma4:26b".into()),
        }];
        let avail = list_available(&history, &probe, &running);
        let names: Vec<&str> = avail.iter().map(|a| a.model.as_str()).collect();
        assert!(names.contains(&"lfm2.5:latest"));
        assert!(!names.contains(&"gemma4:26b"), "running model excluded");
    }
}
```

- [ ] **Step 3:** `cargo test -p paddock-core available_tests` → FAIL (`list_available`/`AvailableRow` missing).

- [ ] **Step 4: Implement `AvailableRow` + `list_available`** in `serving.rs`.

```rust
/// A locally-available model not currently running, for the servers tab's grey
/// group. `enter` serves `plan`. Display fields are raw (the TUI formats them).
#[derive(Debug, Clone)]
pub struct AvailableRow {
    pub model: String,
    pub runtime: RuntimeKind,
    /// Some for Ollama-installed rows (disk size).
    pub size_bytes: Option<u64>,
    /// Some for history rows (unix seconds of last serve).
    pub last_served_at: Option<i64>,
    pub plan: ServePlan,
}

/// True if the HF hub cache still holds the repo for `model_ref`
/// (`{org}/{name}[:quant]`). One stat, not a scan.
fn hf_cache_has(model_ref: &str) -> bool {
    let repo = model_ref.split(':').next().unwrap_or(model_ref);
    let dir = format!("models--{}", repo.replace('/', "--"));
    crate::paths::hf_cache_dir().join(dir).exists()
}

/// A minimal Ollama `ServePlan` for an installed tag (no catalog needed). The
/// daemon serves it; `serve_with_plan` only warms it (no registry entry, no
/// free-port step since `server_argv` is None).
fn ollama_serve_plan(name: &str) -> ServePlan {
    ServePlan {
        server_argv: None,
        pre_steps: vec![],
        endpoint: "http://127.0.0.1:11434".to_string(),
        openai_url: "http://127.0.0.1:11434/v1/chat/completions".to_string(),
        model_ref: name.to_string(),
        ready_path: "/api/version".to_string(),
        install: None,
        port_ignored: false,
        runtime: RuntimeKind::Ollama,
        ctx: 0,
        port: None,
    }
}

/// Locally-available models not in `running`, for the grey group.
pub fn list_available(
    history: &History,
    probe: &dyn SystemProbe,
    running: &[ServerRow],
) -> Vec<AvailableRow> {
    let is_running = |model: &str| running.iter().any(|r| r.model == model);

    let mut rows = Vec::new();

    // Ollama installed (authoritative), minus the loaded ones.
    if let Some(installed) = ollama_installed_models(probe) {
        for m in installed {
            if is_running(&m.name) {
                continue;
            }
            rows.push(AvailableRow {
                model: m.name.clone(),
                runtime: RuntimeKind::Ollama,
                size_bytes: Some(m.size_bytes),
                last_served_at: None,
                plan: ollama_serve_plan(&m.name),
            });
        }
    }

    // llama.cpp/mlx from history, minus running, minus evicted-from-cache.
    let mut hist = history.load();
    hist.sort_by_key(|e| std::cmp::Reverse(e.last_served_at));
    for e in hist {
        if is_running(&e.plan.model_ref) || !hf_cache_has(&e.plan.model_ref) {
            continue;
        }
        rows.push(AvailableRow {
            model: e.plan.model_ref.clone(),
            runtime: e.plan.runtime,
            size_bytes: None,
            last_served_at: Some(e.last_served_at),
            plan: e.plan,
        });
    }

    rows
}
```

- [ ] **Step 5:** `cargo test -p paddock-core available_tests hf_home_overrides` → PASS. `cargo build` clean.

- [ ] **Step 6: Commit**

```bash
git add crates/paddock-core/src/paths.rs crates/paddock-core/src/serving.rs
git commit -m "feat(core): list_available (ollama installed + cache-checked history)"
```

---

## Task 4: Record each spawned serve into history

**Files:**
- Modify: `crates/paddock/src/main.rs` (`serve_with_plan`)

- [ ] **Step 1: Record after the free-port step**

In `serve_with_plan`, right AFTER the free-port reallocation block and BEFORE the child is spawned, add (only spawned servers are relaunchable):

```rust
    // Remember spawned (llama.cpp/mlx) serves so the TUI can offer one-key
    // relaunch. Best-effort; Ollama is covered by /api/tags, not recorded.
    if plan.server_argv.is_some() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        paddock_core::serving::History::open_default().record(&plan, now);
    }
```

- [ ] **Step 2: Build + existing tests**

Run: `cargo build && cargo test -p paddock`
Expected: clean/green. (No new unit test here; behavior is exercised by the live smoke test in Task 8. `record` itself is unit-tested in Task 1.)

- [ ] **Step 3: Commit**

```bash
git add crates/paddock/src/main.rs
git commit -m "feat(serve): record spawned serves into history"
```

---

## Task 5: `ServersSnapshot` + background task + event-loop drain

**Files:**
- Modify: `crates/paddock/src/tui/servers_task.rs`
- Modify: `crates/paddock/src/tui/mod.rs`

- [ ] **Step 1: Produce both groups in the task**

Replace `servers_task.rs` body so the channel carries a snapshot of both groups:

```rust
//! Background servers refresh: a detached thread re-probes the serving state on
//! its own interval and sends a snapshot (running + available) over an mpsc
//! channel the event loop drains. Off the UI thread because the probes block.
//! Exits when the receiver is dropped (the next `send` errors).

use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

use paddock_core::hardware::RealSystemProbe;
use paddock_core::serving::{AvailableRow, History, Registry, ServerRow, list_all_servers, list_available};

const REFRESH_EVERY: Duration = Duration::from_secs(2);

pub struct ServersSnapshot {
    pub running: Vec<ServerRow>,
    pub available: Vec<AvailableRow>,
}

pub fn spawn_servers_refresh() -> Receiver<ServersSnapshot> {
    let (tx, rx) = channel();
    std::thread::spawn(move || {
        loop {
            let registry = Registry::open_default();
            let history = History::open_default();
            let running = list_all_servers(&registry, &RealSystemProbe);
            let available = list_available(&history, &RealSystemProbe, &running);
            if tx.send(ServersSnapshot { running, available }).is_err() {
                break;
            }
            std::thread::sleep(REFRESH_EVERY);
        }
    });
    rx
}
```

- [ ] **Step 2: Update the event loop**

In `mod.rs`:
- Change the `servers_rx` param type to `&std::sync::mpsc::Receiver<servers_task::ServersSnapshot>`.
- The drain (currently keeps the latest `Vec<ServerRow>` and calls `state.set_servers`) becomes:

```rust
        let mut latest = None;
        while let Ok(snapshot) = servers_rx.try_recv() {
            latest = Some(snapshot);
        }
        if let Some(s) = latest {
            state.set_snapshot(s.running, s.available);
        }
```

- The inline-serve immediate refresh (in the `Action::Serve` arm) currently calls `list_all_servers` + `set_servers`. Replace with a both-groups refresh:

```rust
                let registry = paddock_core::serving::Registry::open_default();
                let history = paddock_core::serving::History::open_default();
                let probe = paddock_core::hardware::RealSystemProbe;
                let running = paddock_core::serving::list_all_servers(&registry, &probe);
                let available = paddock_core::serving::list_available(&history, &probe, &running);
                state.set_snapshot(running, available);
```

(`set_snapshot` is added in Task 6. Until then the binary will not compile — implement Task 6 immediately after, or temporarily keep `set_servers`. Recommended: do Tasks 5 and 6 back-to-back, building only after both.)

- [ ] **Step 3: Build after Task 6** (this task alone leaves `set_snapshot` undefined). Commit together with Task 6, OR commit Task 5 with a temporary `state.set_servers(s.running)` then finalize in Task 6. Prefer the back-to-back approach:

```bash
# after Task 6 compiles:
git add crates/paddock/src/tui/servers_task.rs crates/paddock/src/tui/mod.rs
git commit -m "feat(tui): background snapshot carries running + available"
```

---

## Task 6: State — available group, combined navigation, enter launch

**Files:**
- Modify: `crates/paddock/src/tui/state.rs`
- Test: inline `#[cfg(test)]`

- [ ] **Step 1: Write the failing tests**

The test module has `srv(pid, model, port)` building a running `ServerRow`. Add an available-row helper and tests:

```rust
    use paddock_core::serving::AvailableRow;

    fn avail(model: &str) -> AvailableRow {
        AvailableRow {
            model: model.into(),
            runtime: RuntimeKind::MlxLm,
            size_bytes: None,
            last_served_at: Some(0),
            plan: paddock_core::runtime::ServePlan {
                server_argv: Some(vec!["mlx_lm.server".into(), "--model".into(), model.into(), "--port".into(), "8080".into()]),
                pre_steps: vec![],
                endpoint: "http://127.0.0.1:8080".into(),
                openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
                model_ref: model.into(),
                ready_path: "/v1/models".into(),
                install: None,
                port_ignored: false,
                runtime: RuntimeKind::MlxLm,
                ctx: 0,
                port: Some(8080),
            },
        }
    }

    #[test]
    fn navigation_spans_running_then_available() {
        let mut s = state();
        s.set_snapshot(vec![srv(1, "run-a", 8080)], vec![avail("avail-b"), avail("avail-c")]);
        s.tab = Tab::Servers;
        assert_eq!(s.server_selected, 0); // run-a
        s.handle_key(key(KeyCode::Down)); // avail-b
        s.handle_key(key(KeyCode::Down)); // avail-c
        assert_eq!(s.server_selected, 2);
        s.handle_key(key(KeyCode::Down)); // clamped
        assert_eq!(s.server_selected, 2);
    }

    #[test]
    fn enter_on_available_serves_its_plan() {
        let mut s = state();
        s.set_snapshot(vec![], vec![avail("avail-b")]);
        s.tab = Tab::Servers;
        match s.handle_key(key(KeyCode::Enter)) {
            Action::Serve(p) => assert_eq!(p.model_ref, "avail-b"),
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn enter_on_running_is_noop() {
        let mut s = state();
        s.set_snapshot(vec![srv(1, "run-a", 8080)], vec![]);
        s.tab = Tab::Servers;
        assert!(matches!(s.handle_key(key(KeyCode::Enter)), Action::None));
    }

    #[test]
    fn x_on_available_is_noop() {
        let mut s = state();
        s.set_snapshot(vec![], vec![avail("avail-b")]);
        s.tab = Tab::Servers;
        assert!(matches!(s.handle_key(key(KeyCode::Char('x'))), Action::None));
    }
```

- [ ] **Step 2:** `cargo test -p paddock navigation_spans enter_on_available` → FAIL.

- [ ] **Step 3: Implement.** Add the field, `set_snapshot`, a cursor resolver, and rewrite the Servers-tab key arm.

Add field to `TuiState` (after `servers`):

```rust
    /// Locally-available (not running) models shown greyed below the running ones.
    pub available: Vec<AvailableRow>,
```

Initialize in `new` (after `servers: Vec::new(),`): `available: Vec::new(),`. Import `AvailableRow`: extend the `use paddock_core::serving::{...}` line with `AvailableRow`.

Replace `set_servers` with `set_snapshot` (and update its callers — only the event loop, handled in Task 5):

```rust
    /// Replace both groups. Cursor preserved by identity (running by stop
    /// handle, available by model) then clamped to the combined length.
    pub fn set_snapshot(&mut self, running: Vec<ServerRow>, available: Vec<AvailableRow>) {
        let prev = self.selected_combined_key();
        self.servers = running;
        self.available = available;
        let len = self.servers.len() + self.available.len();
        self.server_selected = prev
            .and_then(|k| self.combined_keys().position(|c| c == k))
            .unwrap_or(0)
            .min(len.saturating_sub(1));
    }

    /// Stable identity string per combined row (running: "r:<pid-or-model>",
    /// available: "a:<model>") for cursor preservation across refreshes.
    fn combined_keys(&self) -> impl Iterator<Item = String> + '_ {
        let running = self.servers.iter().map(|r| format!("r:{}", r.model));
        let avail = self.available.iter().map(|a| format!("a:{}", a.model));
        running.chain(avail)
    }
    fn selected_combined_key(&self) -> Option<String> {
        self.combined_keys().nth(self.server_selected)
    }
```

Add a cursor resolver enum + helper:

```rust
    /// What the servers-tab cursor points at.
    fn selected_row(&self) -> SelectedRow<'_> {
        let n = self.servers.len();
        if self.server_selected < n {
            SelectedRow::Running(&self.servers[self.server_selected])
        } else if let Some(a) = self.available.get(self.server_selected - n) {
            SelectedRow::Available(a)
        } else {
            SelectedRow::None
        }
    }
```

Define the enum near `Tab` (top of state.rs):

```rust
enum SelectedRow<'a> {
    Running(&'a ServerRow),
    Available(&'a AvailableRow),
    None,
}
```

Rewrite the `Tab::Servers` key arm of `handle_key`:

```rust
                    Tab::Servers => match key.code {
                        K::Char('q') => return Action::Quit,
                        K::Up => {
                            self.server_selected = self.server_selected.saturating_sub(1)
                        }
                        K::Down => {
                            let len = self.servers.len() + self.available.len();
                            self.server_selected =
                                (self.server_selected + 1).min(len.saturating_sub(1));
                        }
                        K::Enter => {
                            if let SelectedRow::Available(a) = self.selected_row() {
                                return Action::Serve(a.plan.clone());
                            }
                        }
                        K::Char('x') => {
                            if let SelectedRow::Running(r) = self.selected_row() {
                                return Action::StopServer(r.stop.clone());
                            }
                        }
                        K::Char('c') => {
                            if let SelectedRow::Running(r) = self.selected_row() {
                                return Action::CopyEndpoint(r.openai_url.clone());
                            }
                        }
                        _ => {}
                    },
```

NOTE: `ServerRow` needs `model` (it has it). `remove_selected` (used by the StopServer event-loop arm) still operates on `self.servers` by index — but now the cursor may point into the available range. Update `remove_selected` to only remove when the cursor is in the running range:

```rust
    pub fn remove_selected(&mut self) {
        if self.server_selected < self.servers.len() {
            self.servers.remove(self.server_selected);
        }
        let len = self.servers.len() + self.available.len();
        self.server_selected = self.server_selected.min(len.saturating_sub(1));
    }
```

- [ ] **Step 4:** Build the whole binary now (Task 5 + 6 together). `cargo test -p paddock navigation_spans enter_on_available enter_on_running x_on_available` → PASS. Update any test that referenced the old `set_servers` (e.g. `set_servers_preserves_selection_by_handle`, `remove_selected_*`) to use `set_snapshot(running, vec![])`. Run the full `cargo test -p paddock` and fix any remaining `set_servers` references.

- [ ] **Step 5: Commit (with Task 5)**

```bash
git add crates/paddock/src/tui/state.rs crates/paddock/src/tui/servers_task.rs crates/paddock/src/tui/mod.rs
git commit -m "feat(tui): available group in state, combined nav, enter to launch"
```

---

## Task 7: Render the AVAILABLE group

**Files:**
- Modify: `crates/paddock/src/tui/draw.rs`

- [ ] **Step 1: Render both sections in `draw_servers`**

Currently `draw_servers` renders a single table from `state.servers`. Change it to render the running table, then (if `state.available` is non-empty) a blank line, an `AVAILABLE` header, and a grey table, with the highlight following `state.server_selected` across the combined index. Because ratatui tables manage their own selection, render the two groups as ONE table whose rows are running-then-available, so a single `TableState` highlight maps directly to `server_selected`:

```rust
fn draw_servers(frame: &mut Frame, area: Rect, state: &TuiState) {
    if state.servers.is_empty() && state.available.is_empty() {
        let msg = Paragraph::new("no servers running · press s on a model to serve one")
            .style(Style::new().fg(Color::DarkGray));
        frame.render_widget(msg, area);
        return;
    }
    let header = Row::new(["MODEL", "RUNTIME", "ENDPOINT / DETAIL", "CTX", "UPTIME", "PID"])
        .style(Style::new().fg(Color::DarkGray).add_modifier(Modifier::BOLD));

    let mut rows: Vec<Row> = Vec::new();
    for r in &state.servers {
        let pid = match &r.stop {
            paddock_core::serving::StopHandle::Pid(p) => p.to_string(),
            _ => "-".into(),
        };
        rows.push(Row::new(vec![
            Cell::from(r.model.clone()),
            Cell::from(crate::output::runtime_label(r.runtime)),
            Cell::from(crate::output::truncate(&r.endpoint, 26)),
            Cell::from(r.ctx.map(|c| c.to_string()).unwrap_or_else(|| "-".into())),
            Cell::from(r.started_at.map(crate::output::uptime_label).unwrap_or_else(|| "-".into())),
            Cell::from(pid),
        ]).style(Style::new().fg(Color::Gray)));
    }
    for a in &state.available {
        let detail = match (a.size_bytes, a.last_served_at) {
            (Some(sz), _) => format!("{} (installed)", crate::output::gib(sz)),
            (_, Some(ts)) => format!("served {} ago", crate::output::uptime_label(ts)),
            _ => String::new(),
        };
        rows.push(Row::new(vec![
            Cell::from(a.model.clone()),
            Cell::from(crate::output::runtime_label(a.runtime)),
            Cell::from(detail),
            Cell::from("-"),
            Cell::from("-"),
            Cell::from("-"),
        ]).style(Style::new().fg(Color::DarkGray)));
    }

    let table = Table::new(
        rows,
        [
            Constraint::Min(20),
            Constraint::Length(10),
            Constraint::Length(26),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
        ],
    )
    .header(header)
    .row_highlight_style(Style::new().fg(Color::White).bg(ACCENT_DEEP));
    let mut ts = TableState::default().with_selected(Some(state.server_selected));
    frame.render_stateful_widget(table, area, &mut ts);
}
```

(Running rows are `Gray`, available rows `DarkGray` so they read as dimmer. The selected row, running or available, is highlighted via `server_selected`. `gib`/`runtime_label`/`truncate`/`uptime_label` are all `pub` in `crate::output`.)

- [ ] **Step 2: Footer hint** — the servers-tab footer currently reads `↑↓ move · x stop · c copy endpoint · tab models · q quit`. Add `enter launch`:

Find that literal in `draw_footer` and change it to:

```rust
        let line = "↑↓ move · enter launch · x stop · c copy endpoint · tab models · q quit";
```

- [ ] **Step 3: Build + clippy**

Run: `cargo build && cargo test -p paddock && cargo clippy --workspace`
Expected: green/clean. No em-dash introduced (use `-`/`·`).

- [ ] **Step 4: Commit**

```bash
git add crates/paddock/src/tui/draw.rs
git commit -m "feat(tui): render the AVAILABLE group greyed in the servers tab"
```

---

## Task 8: Docs + live smoke test

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the available group**

In the README TUI paragraph (search "Press `Tab` to switch to the servers view"), append a sentence:

```text
Below the running servers, the tab also lists models available locally but not yet loaded - your installed Ollama models and any llama.cpp/mlx model paddock has served before - greyed out; press `enter` on one to launch it.
```

(No em-dash; the ` - ` hyphens above are fine.)

- [ ] **Step 2: Commit docs**

```bash
git add README.md
git commit -m "docs: document the available-models group in the TUI servers tab"
```

- [ ] **Step 3: Live smoke test (manual)**

```bash
cargo run   # opens TUI, Tab to servers
```
Verify:
1. With Ollama running and models installed (`ollama ls`), the AVAILABLE group lists them greyed with `{size} (installed)`.
2. An mlx/llama.cpp model served earlier appears greyed with `served {ago} ago` (serve one first via the models tab if history is empty).
3. Arrow keys move across running then available; the highlight is continuous.
4. `enter` on a greyed model launches it (suspend/resume), and it moves to the running group on the next refresh.
5. `enter` on a running row does nothing; `x` stops a running row; `x`/`c` on a greyed row do nothing.
6. A currently-running model does NOT also appear in available (dedup).

- [ ] **Step 4: Final verification**

Run: `cargo build && cargo test && cargo clippy --workspace`
Expected: all green/clean.

---

## Notes for the implementer

- The state machine stays pure and unit-tested (Tasks 1-3, 6). The background task, history IO, and rendering are integration concerns verified by the live smoke test (Task 8).
- Reuse, don't reinvent: `enter` on an available row returns the existing `Action::Serve(plan)`; the event loop already handles suspend/resume, free-port, switch-to-servers, and refresh.
- Do NOT scan the HF cache for never-served models, add disk-eviction, or cap history size (explicitly out of scope).
- Em-dash `—` is banned project-wide; use `-` or `·`.

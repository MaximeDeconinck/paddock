# tetro serve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `tetro serve <model>` starts (or reuses) the right runtime server and prints a ready-to-use HTTP endpoint (native + OpenAI-compatible) for the chosen model.

**Architecture:** Same split as `run`: tetro-core gains a pure `plan_serve` (data only — argv, pre-steps, endpoint URLs, readiness path, install proposal); the binary owns spawning, readiness polling, printing and lifecycle. Unlike `run` (exec/replace), `serve` SPAWNS the server as a child, polls readiness, runs pre-steps (e.g. `ollama pull`), prints the endpoint block, then waits on the child (Ctrl-C stops serving). When Ollama is already running, nothing is spawned: pull + print + exit 0.

**Tech Stack:** existing deps only (no new crates). Readiness polling reuses `RealSystemProbe::http_get_local`.

**Design decisions (validated against product intent):**
- Foreground model: the server is a child of tetro, terminal stays attached, Ctrl-C stops it. No daemon/PID management in this version. Exception: Ollama already running → tetro just configures and prints, exits 0 (Ollama is the daemon).
- Per-runtime endpoints: Ollama fixed `http://127.0.0.1:11434` (native + `/v1` OpenAI-compat); llama.cpp `llama-server` and `mlx_lm.server` on `--port` (default 8080), OpenAI-compat `/v1/chat/completions`.
- `serve` finally uses llama.cpp: for HF GGUF models without Ollama, `llama-server -hf {repo}:{quant}` downloads and serves directly.
- `--json` prints the ServePlan and performs zero side effects (no spawn, no pull) — same contract as `run --json`.
- Install confirmation reuses the existing single confirm path (never auto-install).

**Facts to verify during implementation (WebSearch, adjust code+tests if they differ):**
1. `mlx_lm.server --model {repo} --port {port}` exact command name/flags and default port (ml-explore/mlx-lm README).
2. `llama-server -hf {repo}:{quant}` flag syntax (short form of `--hf-repo`) and `/health` endpoint (ggml-org/llama.cpp server README).
3. Ollama OpenAI-compat path `/v1/chat/completions` (ollama.com/blog/openai-compatibility or docs).

---

### Task 1: tetro-core — plan_serve

**Files:**
- Modify: `crates/tetro-core/src/runtime.rs`

- [ ] **Step 1.1: failing tests** (append to the existing tests module in runtime.rs; reuse the existing `hf_model()`/`variant()`/`with_ollama()` helpers)

```rust
    fn ollama_running() -> RuntimesStatus {
        RuntimesStatus {
            ollama: RuntimeStatus { installed: true, version: Some("0.30.6".into()), running: true },
            ..Default::default()
        }
    }

    fn ollama_stopped() -> RuntimesStatus {
        RuntimesStatus {
            ollama: RuntimeStatus { installed: true, version: None, running: false },
            ..Default::default()
        }
    }

    fn llama_cpp_only() -> RuntimesStatus {
        RuntimesStatus {
            llama_cpp: RuntimeStatus { installed: true, version: None, running: false },
            ..Default::default()
        }
    }

    #[test]
    fn serve_hf_with_running_ollama_pulls_and_reuses_daemon() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_running(), None).unwrap();
        assert!(p.server_argv.is_none());
        assert_eq!(
            p.pre_steps,
            vec![vec![
                "ollama".to_string(),
                "pull".to_string(),
                "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_string()
            ]]
        );
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
        assert_eq!(p.openai_url, "http://127.0.0.1:11434/v1/chat/completions");
        assert_eq!(p.model_ref, "hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M");
        assert_eq!(p.ready_path, "/api/version");
        assert!(p.install.is_none());
    }

    #[test]
    fn serve_hf_with_stopped_ollama_boots_the_daemon() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &ollama_stopped(), None).unwrap();
        assert_eq!(p.server_argv, Some(vec!["ollama".to_string(), "serve".to_string()]));
        // pull still listed: the binary runs pre_steps AFTER readiness
        assert_eq!(p.pre_steps.len(), 1);
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn serve_hf_without_ollama_falls_back_to_llama_server() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec![
                "llama-server".to_string(),
                "-hf".to_string(),
                "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M".to_string(),
                "--port".to_string(),
                "8080".to_string(),
            ])
        );
        assert!(p.pre_steps.is_empty());
        assert_eq!(p.endpoint, "http://127.0.0.1:8080");
        assert_eq!(p.ready_path, "/health");
        assert!(p.install.is_none());
    }

    #[test]
    fn serve_port_override_applies_to_non_ollama_servers() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &llama_cpp_only(), Some(9999)).unwrap();
        assert_eq!(p.endpoint, "http://127.0.0.1:9999");
        assert!(p.server_argv.unwrap().contains(&"9999".to_string()));
        // ollama ignores --port (fixed daemon port)
        let p2 = plan_serve(&m, &m.variants[0], &ollama_running(), Some(9999)).unwrap();
        assert_eq!(p2.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn serve_mlx_model() {
        let mut m = hf_model();
        m.source = Source::Mlx;
        m.repo = Some("mlx-community/Llama-3.1-8B-Instruct-4bit".into());
        m.variants = vec![variant("MLX_4BIT", vec![RuntimeKind::MlxLm])];
        let rt = RuntimesStatus {
            mlx: RuntimeStatus { installed: true, version: Some("0.24.0".into()), running: false },
            ..Default::default()
        };
        let p = plan_serve(&m, &m.variants[0], &rt, None).unwrap();
        assert_eq!(
            p.server_argv,
            Some(vec![
                "mlx_lm.server".to_string(),
                "--model".to_string(),
                "mlx-community/Llama-3.1-8B-Instruct-4bit".to_string(),
                "--port".to_string(),
                "8080".to_string(),
            ])
        );
        assert_eq!(p.openai_url, "http://127.0.0.1:8080/v1/chat/completions");
        assert_eq!(p.model_ref, "mlx-community/Llama-3.1-8B-Instruct-4bit");
        assert_eq!(p.ready_path, "/v1/models");
    }

    #[test]
    fn serve_without_any_runtime_proposes_install() {
        let m = hf_model();
        let p = plan_serve(&m, &m.variants[0], &RuntimesStatus::default(), None).unwrap();
        let install = p.install.expect("must propose install");
        assert_eq!(install.argv, vec!["brew", "install", "ollama"]);
        // plan still describes what will happen post-install (ollama path)
        assert_eq!(p.endpoint, "http://127.0.0.1:11434");
    }

    #[test]
    fn serve_rejects_repo_less_and_bad_quants_like_run() {
        let mut m = hf_model();
        m.repo = None;
        assert!(plan_serve(&m, &m.variants[0], &ollama_running(), None).is_err());
        let m2 = hf_model();
        let bad = variant("MLX_4BIT", vec![RuntimeKind::MlxLm]);
        assert!(plan_serve(&m2, &bad, &ollama_running(), None).is_err());
    }
```

- [ ] **Step 1.2: run, red.** `cargo test -p tetro-core serve`

- [ ] **Step 1.3: implement in runtime.rs**

```rust
/// Default port for llama-server / mlx_lm.server (Ollama's daemon port is fixed).
pub const DEFAULT_SERVE_PORT: u16 = 8080;
const OLLAMA_ENDPOINT: &str = "http://127.0.0.1:11434";

/// Everything the binary needs to start serving and print a usable endpoint.
/// Pure data: no spawn, no IO here.
#[derive(Debug, Clone, Serialize)]
pub struct ServePlan {
    /// Server process to spawn as a foreground child; None when an already
    /// running daemon (Ollama) will do the serving.
    pub server_argv: Option<Vec<String>>,
    /// Commands run to completion after readiness, before printing the
    /// endpoint (e.g. `ollama pull`). Each is a full argv.
    pub pre_steps: Vec<Vec<String>>,
    /// Base URL, e.g. http://127.0.0.1:11434
    pub endpoint: String,
    /// OpenAI-compatible chat completions URL.
    pub openai_url: String,
    /// Value to put in the `model` field of API payloads.
    pub model_ref: String,
    /// Path polled (GET, 2xx) to detect server readiness, e.g. "/api/version".
    pub ready_path: String,
    /// Same contract as RunPlan: the binary MUST confirm before installing.
    pub install: Option<InstallPlan>,
}

/// Serve strategy mirrors `plan_run`, plus a llama.cpp fallback for HF GGUF
/// (llama-server can download from HF directly via `-hf repo:quant`).
/// Verified commands: `llama-server -hf` (llama.cpp server README),
/// `mlx_lm.server --model --port` (ml-explore/mlx-lm README, 2026-06).
pub fn plan_serve(
    model: &CatalogModel,
    variant: &CatalogVariant,
    rt: &RuntimesStatus,
    port: Option<u16>,
) -> Result<ServePlan, TetroError> {
    let port = port.unwrap_or(DEFAULT_SERVE_PORT);
    let local = format!("http://127.0.0.1:{port}");
    match model.source {
        Source::Mlx => {
            let repo = model.repo.clone().ok_or_else(|| no_repo(&model.name))?;
            Ok(ServePlan {
                server_argv: Some(s(&[
                    "mlx_lm.server", "--model", &repo, "--port", &port.to_string(),
                ])),
                pre_steps: vec![],
                endpoint: local.clone(),
                openai_url: format!("{local}/v1/chat/completions"),
                model_ref: repo,
                ready_path: "/v1/models".to_string(),
                install: (!rt.mlx.installed).then(|| InstallPlan {
                    kind: RuntimeKind::MlxLm,
                    argv: s(&["uv", "tool", "install", "mlx-lm"]),
                }),
            })
        }
        Source::Ollama | Source::HuggingFace => {
            let model_ref = match model.source {
                Source::Ollama => model.name.clone(),
                _ => {
                    let repo = model.repo.clone().ok_or_else(|| no_repo(&model.name))?;
                    validate_gguf_quant(model, variant)?; // factor out of plan_run
                    format!("hf.co/{repo}:{}", variant.quant)
                }
            };
            // Prefer Ollama; fall back to llama-server for HF GGUF when only
            // llama.cpp is installed.
            if !rt.ollama.installed && rt.llama_cpp.installed && model.source == Source::HuggingFace
            {
                let repo = model.repo.clone().ok_or_else(|| no_repo(&model.name))?;
                let hf_ref = format!("{repo}:{}", variant.quant);
                return Ok(ServePlan {
                    server_argv: Some(s(&[
                        "llama-server", "-hf", &hf_ref, "--port", &port.to_string(),
                    ])),
                    pre_steps: vec![],
                    endpoint: local.clone(),
                    openai_url: format!("{local}/v1/chat/completions"),
                    model_ref: hf_ref,
                    ready_path: "/health".to_string(),
                    install: None,
                });
            }
            Ok(ServePlan {
                server_argv: (!rt.ollama.running).then(|| s(&["ollama", "serve"])),
                pre_steps: vec![vec![
                    "ollama".to_string(), "pull".to_string(), model_ref.clone(),
                ]],
                endpoint: OLLAMA_ENDPOINT.to_string(),
                openai_url: format!("{OLLAMA_ENDPOINT}/v1/chat/completions"),
                model_ref,
                ready_path: "/api/version".to_string(),
                install: (!rt.ollama.installed).then(ollama_install),
            })
        }
    }
}
```

Refactor note: extract the existing repo-less / bad-quant validation from `plan_run` into private helpers (`no_repo(name) -> TetroError`, `validate_gguf_quant(model, variant) -> Result<()>`) used by both planners — plan_run behavior unchanged (its tests prove it).

- [ ] **Step 1.4: green + gates.** `cargo test -p tetro-core && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt`
- [ ] **Step 1.5: commit** `feat(core): serve planning — endpoint, readiness and pre-steps per runtime` + trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

### Task 2: CLI — tetro serve

**Files:**
- Modify: `crates/tetro/src/cli.rs` (Serve subcommand)
- Modify: `crates/tetro/src/main.rs` (serve_model + child lifecycle)
- Modify: `crates/tetro/src/output.rs` (endpoint block printer)
- Test: `crates/tetro/tests/cli.rs`

- [ ] **Step 2.1: cli.rs — add subcommand**

```rust
    /// Serve a model over HTTP and print the endpoint (OpenAI-compatible)
    Serve {
        model: String,
        /// Port for llama.cpp / mlx servers (Ollama always uses 11434)
        #[arg(long)]
        port: Option<u16>,
    },
```

- [ ] **Step 2.2: integration tests first (tests/cli.rs)**

```rust
#[test]
fn serve_unknown_model_fails_actionably() {
    let (mut cmd, _dir) = tetro();
    cmd.args(["serve", "definitely-not-a-model"])
        .assert()
        .failure()
        .stderr(predicates::str::contains("tetro sync"));
}

#[test]
fn serve_json_prints_plan_without_side_effects() {
    let (mut cmd, dir) = tetro();
    seed_one_model(dir.path()); // helper below
    let out = cmd.args(["serve", "fake-model", "--json"]).assert().success();
    let v: serde_json::Value = serde_json::from_slice(&out.get_output().stdout).unwrap();
    assert!(v["endpoint"].as_str().unwrap().starts_with("http://127.0.0.1:"));
    assert!(v["openai_url"].as_str().unwrap().ends_with("/v1/chat/completions"));
    assert!(v["model_ref"].is_string());
}
```

`seed_one_model(dir)`: open `Db` at the test's TETRO_DB_PATH and upsert one Ollama-source model ("fake-model", llama-arch numbers from the curated entries, one Q4_K_M variant) using `tetro_core::catalog::{db::Db, CatalogModel, CatalogVariant, RuntimeKind, Source}` — tetro-core is already a dependency. Note: the serve plan for an Ollama-source model never needs the network in --json mode.

- [ ] **Step 2.3: main.rs — serve_model**

```rust
fn serve_model(app: &App, query: &str, port: Option<u16>, json: bool) -> Result<()> {
    let db = app.open_db()?;
    let (model, mvs) = /* same find_model + best_variant resolution as run_model — factor the
                          lookup into a shared helper resolve_model(&db, app, query)
                          returning (CatalogModel, variant_idx) to avoid duplication */;
    let plan = tetro_core::runtime::plan_serve(&model, &model.variants[idx], &app.profile.runtimes, port)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }

    if let Some(install) = &plan.install {
        confirm_and_install(install)?; // existing single confirm path
    }

    // 1. spawn server child if needed
    let mut child = match &plan.server_argv {
        Some(argv) => {
            eprintln!("$ {}", argv.join(" "));
            Some(spawn_checked(argv)?)
        }
        None => None,
    };
    // 2. poll readiness (15s timeout, 250ms interval) via RealSystemProbe.http_get_local
    wait_ready(&plan, child.as_mut())?;
    // 3. pre-steps to completion (e.g. ollama pull — streams progress to the tty)
    for step in &plan.pre_steps {
        eprintln!("$ {}", step.join(" "));
        run_checked(step)?;
    }
    // 4. print the endpoint block (stdout — it's the product output)
    output::print_endpoint(&plan);
    // 5. lifecycle
    match child {
        Some(mut c) => {
            eprintln!("serving — press Ctrl-C to stop");
            let status = c.wait()?;
            if !status.success() {
                anyhow::bail!("server exited with {status}");
            }
            Ok(())
        }
        None => Ok(()), // ollama daemon keeps running, our job is done
    }
}
```

Helpers: `spawn_checked(argv) -> Result<std::process::Child>` (Command::spawn, actionable "not found in PATH" error); `run_checked(argv) -> Result<()>` (status() success check); `wait_ready(plan, child)` polls `RealSystemProbe.http_get_local(format!("{}{}", plan.endpoint, plan.ready_path))` every 250 ms up to 15 s — if the child exits early, bail with its status ("server failed to start — run it manually to see the error"); on timeout, kill child and bail actionably. While polling, if `server_argv` was None (daemon case) readiness should succeed immediately; treat failure as "ollama daemon not reachable on 11434 — is it running?".

`output::print_endpoint(plan)`:
```
endpoint    http://127.0.0.1:11434
openai      http://127.0.0.1:11434/v1/chat/completions
model       hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M
try it      curl -s http://127.0.0.1:11434/v1/chat/completions \
              -d '{"model":"<model_ref>","messages":[{"role":"user","content":"hello"}]}'
```
(single quotes in the curl line are literal — no shell execution here, it's display only).

Wire `Command::Serve` in main(). The `run_model`/`serve_model` shared lookup refactor (resolve_model) must keep all five existing run-path tests green.

- [ ] **Step 2.4: gates + manual smoke**

`cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt`
Manual (Ollama is running on this machine): `cargo run -q -- serve llama3.2:1b` → should pull (fast, 1.3 GB) then print the endpoint block and exit 0; `curl http://127.0.0.1:11434/v1/chat/completions -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"say hi"}]}'` must answer. `cargo run -q -- serve llama3.2:1b --json` → plan JSON, no pull triggered. Report outputs.

- [ ] **Step 2.5: commit** `feat(cli): serve subcommand — endpoint with readiness wait and ollama pull` + trailer.

---

### Task 3: TUI key + README

**Files:**
- Modify: `crates/tetro/src/tui/state.rs` (s key → Action::Serve)
- Modify: `crates/tetro/src/tui/mod.rs` (Serve action: restore, then serve flow)
- Modify: `crates/tetro/src/tui/draw.rs` (footer hint + detail popup shows serve hint)
- Modify: `README.md`

- [ ] **Step 3.1: state — `Action::Serve(ServePlan)`**, key `s` in List and Detail modes, built via `plan_serve(... , None)` analogous to `run_selected` (same error-to-footer handling). Test: `s_returns_serve_action_with_endpoint` (assert endpoint non-empty, ollama argv/pre_steps shape) + ctrl-c test untouched.
- [ ] **Step 3.2: mod.rs** — on Action::Serve: `ratatui::restore()`, then call the same serve lifecycle as the CLI (factor `serve_with_plan(plan) -> Result<()>` out of `serve_model` into a `pub(crate)` fn in main.rs; serve_model = resolve + plan + serve_with_plan).
- [ ] **Step 3.3: draw** — footer hint becomes `↑↓ move · enter detail · x run · s serve · / search · g/c/r/h use-case · q quit`; detail popup adds the serve command/endpoint line under the run command.
- [ ] **Step 3.4: README** — Usage section: `tetro serve` with example output block (capture real output), note on foreground vs Ollama-daemon behavior, `--port`, `--json`. TUI key table adds `s`.
- [ ] **Step 3.5: gates + commit** `feat(tui): s key serves the selected model; document serve` + trailer.

---

### Task 1bis: tetro-core — serving registry + Ollama /api/ps

(Design: docs/superpowers/specs/2026-06-10-tetro-tray-design.md. Runs after Task 1, before Task 2 — `serve_model` consumes the registry from day one.)

**Files:**
- Create: `crates/tetro-core/src/serving.rs`
- Modify: `crates/tetro-core/src/lib.rs` (`pub mod serving;`)
- Modify: `crates/tetro-core/src/hardware/runtimes.rs` or new fn in serving.rs (ollama_loaded_models)

- [ ] **Step 1bis.1: failing tests**

```rust
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
        probe.http.insert("http://127.0.0.1:8080/health".into(), "ok".into());
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
        probe.http.insert("http://127.0.0.1:8080/health".into(), "ok".into());
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
```

- [ ] **Step 1bis.2: implement serving.rs**

```rust
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
        Self { dir: dir.as_ref().to_path_buf() }
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
            Err(e) => Err(TetroError::Other(format!("cannot remove serving record: {e}"))),
        }
    }

    /// Live records only; stale files (dead PID, unreachable endpoint,
    /// unparseable JSON) are deleted as a side effect.
    pub fn list_live(&self, probe: &dyn SystemProbe) -> Vec<ServingRecord> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return Vec::new() };
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
                    && probe.http_get_local(&format!("{}{}", r.endpoint, r.ready_path)).is_some()
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
```

Note: `pid_alive` permission case — `kill` returns -1/EPERM for alive-but-other-user processes; for our own children EPERM won't occur; acceptable. If clippy objects to the inline extern, adding the `libc` crate to tetro-core is an acceptable substitute (document the choice).

- [ ] **Step 1bis.3: green + gates + commit** `feat(core): serving registry and ollama /api/ps discovery` + trailer.

**Task 2 amendment (registry wiring):** in `serve_model` step 5 (lifecycle): after readiness and pre-steps, when a child was spawned, `Registry::open_default().register(&record)` with the child's PID and the plan's fields, then wait; on exit (normal or error) `unregister(pid)`. Use a small RAII guard struct so unregister also runs on `?` early-returns. SIGINT: default terminal behavior kills tetro and the child together; the next `list_live` reaps the stale file — guard is best-effort, document this.

---

### Task 4: tetro tray (menu bar)

(Design: docs/superpowers/specs/2026-06-10-tetro-tray-design.md. After Task 3.)

**Files:**
- Modify: `crates/tetro/Cargo.toml` (macOS-only deps: tray-icon, tao, arboard)
- Modify: `crates/tetro/src/cli.rs` (Tray subcommand)
- Create: `crates/tetro/src/tray/mod.rs` (event loop + tray-icon rendering)
- Create: `crates/tetro/src/tray/menu_model.rs` (pure menu construction + tests)
- Modify: `crates/tetro/src/main.rs` (wire subcommand)
- Modify: `README.md` (tray section)

- [ ] **Step 4.1: deps**

```toml
[target.'cfg(target_os = "macos")'.dependencies]
tray-icon = "0.19"
tao = "0.30"
arboard = "3"
```
(Versions: check latest compatible on crates.io during implementation; tray-icon and tao must agree on their event-loop integration — follow tray-icon's README example for macOS.)

- [ ] **Step 4.2: menu_model.rs — pure model + failing tests first**

```rust
//! Pure construction of the tray menu content. No tray-icon types here so it
//! is unit-testable and portable to the future Tauri app.

use tetro_core::hardware::RuntimesStatus;
use tetro_core::serving::{LoadedModel, ServingRecord};

#[derive(Debug, PartialEq)]
pub enum MenuEntry {
    /// Disabled section header, e.g. "Ollama — 127.0.0.1:11434".
    Header(String),
    /// Clickable model row; activating copies `copy_url`.
    Model { label: String, copy_url: String },
    /// Disabled informational line.
    Info(String),
    Separator,
}

pub struct MenuModel {
    pub entries: Vec<MenuEntry>,
}

pub fn build_menu_model(
    records: &[ServingRecord],
    ollama_ps: Option<&[LoadedModel]>, // None = unreachable; ollama_installed gates display
    runtimes: &RuntimesStatus,
) -> MenuModel
```

Rules (from the spec — implement exactly):
1. Ollama section first, only when `runtimes.ollama.installed`:
   - `/api/ps` reachable with ≥1 model: Header "Ollama — 127.0.0.1:11434" + one Model row per loaded model, label `"{name} — {size GiB}"`, copy_url `http://127.0.0.1:11434/v1/chat/completions`. Skip any registry record that is an Ollama runtime (dedup: the daemon section already covers it).
   - reachable with 0 models: Header + Info "aucun modèle chargé".
   - unreachable (`None`): Header "Ollama — injoignable" (as Info, no models).
2. One section per non-Ollama registry record: Header `"{runtime} — {endpoint host:port}"` (runtime label "llama-server"/"mlx-lm"), Model row with `model_ref` label and the record's `openai_url`.
3. Empty state: no sections at all → Info "Aucun serveur actif". Additionally, when none of the three runtimes is installed → Info "Aucun runtime installé — lance `tetro run` pour installer".
4. Separator before the footer (footer items Refresh/Quit are added by the rendering layer, not the model).

Tests (write first, red): `ollama_with_loaded_models`, `ollama_unreachable_shows_injoignable`, `ollama_not_installed_no_section`, `registry_records_get_sections`, `ollama_registry_record_deduped`, `empty_all` (both Info lines when nothing installed), size formatting via `output::gib`.

- [ ] **Step 4.3: tray/mod.rs — rendering layer (macOS only)**

`#[cfg(target_os = "macos")] pub fn run() -> anyhow::Result<()>`:
- Build tao event loop; create `tray_icon::TrayIcon` with an embedded template icon (generate a tiny monochrome 22×22 PNG asset `crates/tetro/assets/tray-icon.png` — a simple "t" glyph or 2×2 block pattern is fine; `tray_icon::Icon::from_rgba`; set `.with_icon_as_template(true)` so it adapts to dark/light menu bar).
- Refresh closure: `RealSystemProbe` → `detect_runtimes` (cheap subset: only ollama matters for /api/ps gating — reuse full detect, acceptable), `Registry::open_default().list_live(&probe)`, `ollama_loaded_models(&probe)` → `build_menu_model` → rebuild `tray_icon::menu::Menu` (Header/Info → disabled MenuItem, Model → enabled MenuItem with id→copy_url map, Separator → PredefinedMenuItem::separator) + footer "Rafraîchir" / "Quitter tetro tray".
- Event loop: `MenuEvent::receiver()` polled via tao's event loop (follow tray-icon README pattern: `tao::event_loop::EventLoopBuilder`, `ControlFlow::WaitUntil(now + 5s)` → refresh on each wakeup); on Model click → `arboard::Clipboard::new()?.set_text(copy_url)` (errors → eprintln, non-fatal); Quitter → exit loop.
- `#[cfg(not(target_os = "macos"))] pub fn run()` → `anyhow::bail!("tetro tray is macOS-only for now")`.

cli.rs: `/// Menu bar status item showing active serve endpoints` `Tray`. main.rs: `Some(Command::Tray) => tray::run()?`.

- [ ] **Step 4.4: gates + manual smoke**

`cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt`
Manual (REQUIRED, human has a GUI session): `cargo run -q -- tray` → icon appears top-right; with Ollama running and a model loaded (`ollama ps` non-empty — load one via a quick chat request if empty), menu shows the section + model; click copies the OpenAI URL (`pbpaste` to verify); Quitter exits cleanly. Report what was verified; DONE_WITH_CONCERNS if no GUI access.

- [ ] **Step 4.5: README** — "Menu bar" subsection: `tetro tray`, screenshot placeholder, what it shows (hybrid discovery), click-to-copy, macOS-only, v1 limits (no stop action, no login item).

- [ ] **Step 4.6: commit** `feat(tray): macOS menu bar item with live serve endpoints` + trailer.

---

## Self-review notes

- Spec coverage: endpoint per runtime (Ollama daemon reuse incl. pull, ollama serve boot, llama-server -hf fallback, mlx_lm.server), OpenAI-compat URL always printed, --json pure, --port where meaningful, install confirmation unchanged, TUI parity (s key), README.
- Type consistency: `ServePlan.server_argv: Option<Vec<String>>` (None = daemon reuse); `pre_steps: Vec<Vec<String>>`; shared helpers `no_repo`/`validate_gguf_quant` extracted from plan_run; `resolve_model` shared by run/serve; `serve_with_plan` shared by CLI/TUI.
- Deliberate v1 simplifications: no `--host` (loopback only — safe default); mlx_lm.server has no auth — loopback mitigates; Ollama port not overridable (daemon owns it). The "no daemon management" simplification is amended by Task 1bis: a passive PID registry (reaped lazily), still no active daemon supervision.
- Tray addendum (2026-06-10, spec docs/superpowers/specs/2026-06-10-tetro-tray-design.md): execution order is Task 1 → Task 1bis → Task 2 (with registry wiring amendment) → Task 3 → Task 4. Type consistency: `ServingRecord` fields mirror `ServePlan` (endpoint/openai_url/model_ref/ready_path); `build_menu_model` consumes `&[ServingRecord]`, `Option<&[LoadedModel]>`, `&RuntimesStatus`; tray rendering layer owns only tray-icon/tao/arboard types.

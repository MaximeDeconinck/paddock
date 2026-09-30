# paddock mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `paddock mcp` exposes provisioning (scan, fit, recommend, serve, ps, stop) over the Model Context Protocol on stdio, on top of a serve/stop lifecycle extracted from `main.rs` into a `lifecycle` module that returns values instead of printing and exiting.

**Architecture:** `crates/paddock/src/lifecycle.rs` owns model resolution, the serve lifecycle (install policy, port fallback, spawn, readiness wait with an optional timeout, pre-steps, warm-up, registry) and server stopping; it reports progress through a `Progress` trait and fails with a `LifecycleError` enum whose `Display` reproduces today's CLI wording. `main.rs` (CLI) and `tui/mod.rs` become thin adapters that render `LifecycleError` and keep the interactive prompts. `crates/paddock/src/mcp.rs` is a third adapter: an `rmcp` 3.5 server with six tools, each running the blocking lifecycle code under `spawn_blocking`, returning JSON in both `content` and `structured_content`, and mapping `LifecycleError` to `{ code, message, ... }` error bodies.

**Tech Stack:** Rust 2024 (rust-version 1.88, let-chains are used), `rmcp` 3.5 (`server`, `transport-io`, `macros`), `schemars` 1, `thiserror` 2 (workspace), `tokio` (`rt-multi-thread`), `assert_cmd` for the integration test. `paddock-core` is not modified.

**Spec:** `docs/superpowers/specs/2026-09-28-mcp-server-design.md`

## Global Constraints

- Provisioning only: exactly six tools, `paddock_scan`, `paddock_fit`, `paddock_recommend`, `paddock_serve`, `paddock_ps`, `paddock_stop`. No sync, logs, run, chat, run_agent.
- stdio transport only. Nothing but JSON-RPC may be written to stdout while `paddock mcp` runs; diagnostics go to stderr.
- `rmcp = "3.5"` with `default-features = false, features = ["server", "transport-io", "macros"]`; `schemars = "1"`. No hand-rolled JSON-RPC.
- Same binary, `Command::Mcp` subcommand. `--json` / `--cli` are ignored with it. `paddock-core` untouched.
- Lifecycle extracted, not duplicated: CLI, TUI and MCP call the same `lifecycle::serve` / `lifecycle::stop`.
- `paddock_serve` blocks until ready with `timeout_secs` (default 600). On timeout the detached server keeps running, the registry entry is written first, and the tool returns `status: "starting"` with the pid.
- No install from MCP: `InstallPolicy::Refuse` returns `no_runtime` with the install command; the agent must ask the human.
- No `stop all` from MCP: `target == "all"` returns `invalid_target` before touching the registry.
- Every `LifecycleError` `Display` reproduces today's CLI wording exactly. The existing 14 `assert_cmd` tests in `crates/paddock/tests/cli.rs` and the 54 TUI tests pass with no edits.
- No em-dash anywhere (code, docs, output). Use `-` or ` · `.
- Refinement over the spec (approved as an implementation detail, not a re-litigation): `LifecycleError::NotReady` also carries `plan: Box<ServePlan>` so the MCP adapter can report the (possibly port-shifted) endpoint in the `starting` result; `lifecycle::stop` is split into `resolve_servers` + `stop_records` so the CLI can prompt between matching and stopping; the TUI keeps calling `crate::serve_with_plan(plan, false)`, which is now a thin adapter over `lifecycle::serve` that prints the same lines as before (identical terminal output, one adapter instead of two).

## Review Focus

Inputs the spec implies but no task's tests would otherwise exercise, most likely to bite first. Each line names the owning task, and that task carries the test.

1. `paddock_serve` with `port: 0` (Task 6): `plan_serve` already rejects port 0; the MCP layer must surface it as `code: "internal"` with the message, not panic. Test in Task 6 (`error_body_other_is_internal`).
2. `paddock_fit` with `limit: 0` (Task 6): returns `[]`, not an error, mirroring `fit -n 0`. Test in Task 7 (`fit_limit_zero_is_empty_array`).
3. `paddock_stop` with a numeric target that is not running (Task 6): `no_server_match` with `running: []`. Test in Task 7 (`stop_unknown_pid_is_no_server_match`).
4. Detached readiness timeout leaves the child alive and registered (Task 2 + Task 3): the whole point of `status: "starting"`. Test in Task 2 (`wait_ready_timeout_leaves_child_alive`).
5. Unknown tool name or malformed arguments (Task 7): rmcp answers with a JSON-RPC error, not a crash; the process must keep answering later requests. Test in Task 7 (`unknown_tool_does_not_kill_session`).

---

## File Structure

- `crates/paddock/src/lifecycle.rs` (new): `Progress`, `StderrProgress`, `SilentProgress`, `LifecycleError`, `InstallPolicy`, `ServeMode`, `ServeOutcome`, `Stopped`, `find_model`/`Lookup`/`resolve_quant`/`resolve_in`/`resolve_model`/`resolved_ctx` (moved from `main.rs`), `readiness_deadline`, `wait_ready`, `spawn_checked`, `spawn_detached`, `run_checked`, `build_record`, `register_detached`, `RegistryGuard`, `serve`, `resolve_servers`, `stop_records`, `stop`. Unit tests for error display, resolution, readiness.
- `crates/paddock/src/main.rs`: loses everything above; keeps `main`, `fit`, `run_model`, `serve_model`, `serve_with_plan` (thin), `stop_servers` (prompt + `lifecycle`), `show_logs`, `bench_server`, `launch`, `exec`, `confirm_and_install`, `find_in_path`, `installer_missing_hint`, plus a new `cli_fail` mapper.
- `crates/paddock/src/output.rs`: `FitRow` / `RecommendRow` become `pub` with `pub fn fit_rows` / `pub fn recommend_rows` builders; the two `print_*_json` functions call them.
- `crates/paddock/src/cli.rs`: `Command::Mcp`.
- `crates/paddock/src/mcp.rs` (new): `PaddockMcp`, input structs, six tools, `error_body`, `run`.
- `crates/paddock/Cargo.toml`: `rmcp`, `schemars`, `thiserror`.
- `crates/paddock/tests/mcp.rs` (new): stdio integration tests.
- `README.md`: `paddock mcp` section, subcommand count, roadmap.

---

## Task 1: `lifecycle` module: types, errors, model resolution

**Files:**
- Create: `crates/paddock/src/lifecycle.rs`
- Modify: `crates/paddock/src/main.rs` (remove `Lookup`, `find_model`, `resolve_quant`, `resolved_ctx`, `resolve_model` and their tests; add `mod lifecycle;`)
- Modify: `crates/paddock/Cargo.toml` (add `thiserror.workspace = true`)

**Interfaces:**
- Produces:
  - `pub trait Progress { fn note(&self, msg: &str); fn command(&self, argv: &[String]); fn quiet(&self) -> bool { false } }`
  - `pub struct StderrProgress;` `pub struct SilentProgress;` (`quiet()` returns `true`)
  - `pub enum LifecycleError { ModelNotFound(String), Ambiguous { query, candidates: Vec<String> }, NoFit { model, ram_bytes: u64 }, UnknownQuant { label, available: Vec<String> }, NoRuntime { install: InstallPlan }, InstallDeclined { command: String }, ServerExited { status: String, argv: Vec<String>, log_path: Option<PathBuf> }, NotReady { pid: Option<u32>, log_path: Option<PathBuf>, plan: Box<ServePlan> }, OllamaUnreachable, NoServerMatch { target, running: Vec<String> }, AmbiguousServer { target, candidates: Vec<(String, u32)> }, CatalogEmpty, Other(anyhow::Error) }`
  - `pub enum InstallPolicy<'a> { Ask(&'a dyn Fn(&InstallPlan) -> Result<(), LifecycleError>), Refuse }`
  - `pub enum ServeMode { Foreground, Detached }`
  - `pub struct ServeOutcome { pub plan: ServePlan, pub pid: Option<u32>, pub log_path: Option<PathBuf>, pub child: Option<std::process::Child> }`
  - `pub struct Stopped { pub model_ref: String, pub pid: u32 }`
  - `pub fn resolve_in(models: &[CatalogModel], budget: &MemoryBudget, query: &str, quant: Option<&str>) -> Result<(CatalogModel, usize), LifecycleError>`
  - `pub fn resolve_model(app: &App, query: &str, quant: Option<&str>) -> Result<(CatalogModel, usize), LifecycleError>`
  - `pub fn resolved_ctx(app: &App, model: &CatalogModel, idx: usize, ctx: Option<u32>) -> u32`

- [ ] **Step 1: Add `thiserror` to the binary crate**

In `crates/paddock/Cargo.toml`, under `[dependencies]`, after `serde_json.workspace = true`:

```toml
thiserror.workspace = true
```

- [ ] **Step 2: Write the failing tests**

Create `crates/paddock/src/lifecycle.rs` with only the test module for now (the implementation comes in Step 4). The `find_model` / `resolve_quant` tests are moved verbatim from `main.rs`; the display and resolution tests are new.

```rust
//! Serve / stop lifecycle shared by the CLI, the TUI and the MCP server.
//! Everything here returns values: no printing, no stdin, no process::exit.
//! Adapters render `LifecycleError` and report `Progress` their own way.

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::{CatalogVariant, RuntimeKind, Source};
    use paddock_core::estimate::ModelVariant;

    fn model(name: &str) -> CatalogModel {
        CatalogModel {
            id: 0,
            name: name.to_string(),
            family: None,
            source: Source::HuggingFace,
            repo: None,
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            architecture: None,
            context_max: 8192,
            released_at: None,
            released_approx: false,
            variants: vec![],
        }
    }

    /// One-variant Ollama model small enough to fit any budget used below.
    fn small_ollama_model(name: &str) -> CatalogModel {
        let mut m = model(name);
        m.source = Source::Ollama;
        m.params_total = 1_000_000_000;
        m.params_active = 1_000_000_000;
        m.variants = vec![CatalogVariant {
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            file_size_bytes: None,
            layers: 16,
            kv_heads: 8,
            head_dim: 64,
            embedding_dim: 2048,
            runtime_compat: vec![RuntimeKind::Ollama],
            source_tag: None,
        }];
        m
    }

    fn budget_gib(gib: u64) -> MemoryBudget {
        let bytes = gib * 1024 * 1024 * 1024;
        MemoryBudget {
            gpu_effective_bytes: bytes,
            ram_total_bytes: bytes,
        }
    }

    #[test]
    fn exact_match_wins_over_contains() {
        let models = vec![model("Llama3"), model("Llama3-70B")];
        match find_model(&models, "Llama3") {
            Lookup::Found(m) => assert_eq!(m.name, "Llama3"),
            _ => panic!("expected exact match"),
        }
    }

    #[test]
    fn case_insensitive_exact_match_beats_ambiguous_contains() {
        let models = vec![model("Llama3"), model("Llama3-70B")];
        match find_model(&models, "llama3") {
            Lookup::Found(m) => assert_eq!(m.name, "Llama3"),
            _ => panic!("expected case-insensitive exact match"),
        }
    }

    #[test]
    fn contains_still_resolves_unique_substring() {
        let models = vec![model("Llama3-70B"), model("Qwen2.5-Coder")];
        match find_model(&models, "qwen") {
            Lookup::Found(m) => assert_eq!(m.name, "Qwen2.5-Coder"),
            _ => panic!("expected contains match"),
        }
    }

    #[test]
    fn ambiguous_when_no_exact_and_multiple_contains() {
        let models = vec![model("Llama3-8B"), model("Llama3-70B")];
        match find_model(&models, "llama3") {
            Lookup::Ambiguous(names) => assert_eq!(names.len(), 2),
            _ => panic!("expected ambiguous"),
        }
    }

    #[test]
    fn not_found_when_nothing_matches() {
        let models = vec![model("Llama3")];
        assert!(matches!(find_model(&models, "mistral"), Lookup::NotFound));
    }

    fn mv(quant: &str, bpw: f64) -> ModelVariant {
        ModelVariant {
            model_name: "test".into(),
            quant: quant.into(),
            bpw,
            params_total: 8_000_000_000,
            params_active: 8_000_000_000,
            layers: 32,
            kv_heads: 8,
            head_dim: 128,
            embedding_dim: 4096,
            context_max: 8192,
        }
    }

    #[test]
    fn resolve_quant_exact_and_case_insensitive() {
        let vs = vec![mv("Q8_0", 8.5), mv("Q4_K_M", 4.83), mv("Q2_K", 3.35)];
        assert_eq!(resolve_quant(&vs, "Q4_K_M").unwrap(), 1);
        assert_eq!(resolve_quant(&vs, "q4_k_m").unwrap(), 1);
    }

    #[test]
    fn resolve_quant_collision_picks_best_quality() {
        let vs = vec![mv("Q4_K_M", 4.5), mv("Q4_K_M", 5.0)];
        assert_eq!(resolve_quant(&vs, "Q4_K_M").unwrap(), 1);
    }

    #[test]
    fn resolve_quant_no_match_lists_available() {
        let vs = vec![mv("Q8_0", 8.5), mv("Q4_K_M", 4.83)];
        let err = resolve_quant(&vs, "Q3_K_M").unwrap_err();
        match &err {
            LifecycleError::UnknownQuant { label, available } => {
                assert_eq!(label, "Q3_K_M");
                assert_eq!(available, &["Q8_0", "Q4_K_M"]);
            }
            other => panic!("expected UnknownQuant, got {other}"),
        }
        assert_eq!(
            err.to_string(),
            "no quant `Q3_K_M` for this model; available: Q8_0, Q4_K_M"
        );
    }

    #[test]
    fn resolve_in_empty_catalog_is_catalog_empty() {
        let err = resolve_in(&[], &budget_gib(16), "anything", None).unwrap_err();
        assert!(matches!(err, LifecycleError::CatalogEmpty));
        assert_eq!(err.to_string(), "catalog is empty - run `paddock sync` first");
    }

    #[test]
    fn resolve_in_unknown_model_is_model_not_found() {
        let models = vec![small_ollama_model("fake-model")];
        let err = resolve_in(&models, &budget_gib(16), "nope", None).unwrap_err();
        assert!(matches!(err, LifecycleError::ModelNotFound(ref q) if q == "nope"));
        assert!(err.to_string().contains("paddock sync"));
    }

    #[test]
    fn resolve_in_ambiguous_lists_candidates() {
        let models = vec![small_ollama_model("llama3-8b"), small_ollama_model("llama3-70b")];
        let err = resolve_in(&models, &budget_gib(16), "llama3", None).unwrap_err();
        match &err {
            LifecycleError::Ambiguous { query, candidates } => {
                assert_eq!(query, "llama3");
                assert_eq!(candidates, &["llama3-8b", "llama3-70b"]);
            }
            other => panic!("expected Ambiguous, got {other}"),
        }
        assert_eq!(
            err.to_string(),
            "model name `llama3` is ambiguous - candidates:\n  llama3-8b\n  llama3-70b"
        );
    }

    #[test]
    fn resolve_in_picks_best_fitting_variant() {
        let models = vec![small_ollama_model("fake-model")];
        let (m, idx) = resolve_in(&models, &budget_gib(16), "fake-model", None).unwrap();
        assert_eq!(m.name, "fake-model");
        assert_eq!(idx, 0);
    }

    #[test]
    fn resolve_in_no_fit_reports_ram() {
        // 1 GiB budget cannot hold a 1B-parameter Q4 model plus its KV cache.
        let models = vec![small_ollama_model("fake-model")];
        let err = resolve_in(&models, &budget_gib(1), "fake-model", None).unwrap_err();
        match &err {
            LifecycleError::NoFit { model, ram_bytes } => {
                assert_eq!(model, "fake-model");
                assert_eq!(*ram_bytes, 1024 * 1024 * 1024);
            }
            other => panic!("expected NoFit, got {other}"),
        }
        assert_eq!(
            err.to_string(),
            "no quantization of `fake-model` fits this machine (1.0 GiB RAM); try a smaller model from `paddock fit`"
        );
    }

    #[test]
    fn explicit_quant_bypasses_fit_check() {
        let models = vec![small_ollama_model("fake-model")];
        let (_, idx) = resolve_in(&models, &budget_gib(1), "fake-model", Some("q4_k_m")).unwrap();
        assert_eq!(idx, 0);
    }

    #[test]
    fn error_display_matches_cli_wording() {
        let e = LifecycleError::NoServerMatch {
            target: "nope".into(),
            running: vec![],
        };
        assert_eq!(e.to_string(), "no running server matches `nope`");

        let e = LifecycleError::NoServerMatch {
            target: "nope".into(),
            running: vec!["a".into(), "b".into()],
        };
        assert_eq!(e.to_string(), "no running server matches `nope`\nrunning: a, b");

        let e = LifecycleError::AmbiguousServer {
            target: "qwen".into(),
            candidates: vec![("qwen3-8b".into(), 1), ("qwen3-4b".into(), 2)],
        };
        assert_eq!(
            e.to_string(),
            "`qwen` matches several servers - be specific:\n  qwen3-8b (pid 1)\n  qwen3-4b (pid 2)"
        );

        let e = LifecycleError::OllamaUnreachable;
        assert_eq!(e.to_string(), "ollama daemon not reachable on 11434 - is it running?");

        let e = LifecycleError::ServerExited {
            status: "exit status: 1".into(),
            argv: vec!["llama-server".into(), "-hf".into(), "x".into()],
            log_path: None,
        };
        assert_eq!(
            e.to_string(),
            "server exited with exit status: 1 before becoming ready - run `llama-server -hf x` manually to see the error"
        );

        let e = LifecycleError::NoRuntime {
            install: InstallPlan {
                kind: RuntimeKind::LlamaCpp,
                argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
            },
        };
        assert_eq!(
            e.to_string(),
            "required runtime is not installed - run `brew install llama.cpp` then retry"
        );

        let e = LifecycleError::InstallDeclined {
            command: "brew install llama.cpp".into(),
        };
        assert_eq!(
            e.to_string(),
            "install declined - nothing launched. Run `brew install llama.cpp` yourself, then retry."
        );
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Add `mod lifecycle;` to the top of `crates/paddock/src/main.rs` (after `mod cli;`), then:

Run: `cargo test -p paddock --bin paddock lifecycle`
Expected: compile error (`find_model`, `LifecycleError`, `resolve_in`, ... not found).

- [ ] **Step 4: Write the implementation**

Prepend to `crates/paddock/src/lifecycle.rs`, above the test module:

```rust
use std::path::PathBuf;

use paddock_core::catalog::CatalogModel;
use paddock_core::estimate::{MemoryBudget, ModelVariant};
use paddock_core::runtime::{InstallPlan, ServePlan};
use paddock_core::score::best_variant;

use crate::app::App;

/// How the lifecycle reports what it is doing. The CLI/TUI print to stderr
/// (byte-for-byte today's messages); the MCP server discards everything
/// because stdout is the protocol channel and stderr is for internal errors.
pub trait Progress {
    /// Informational line: "downloading/loading model - this can take a while",
    /// "port 8080 is busy - serving on 8081 instead", heartbeats.
    fn note(&self, msg: &str);
    /// A command about to run ("$ ollama pull ...").
    fn command(&self, argv: &[String]);
    /// True when child commands (pre-steps, `ollama stop`) must not inherit
    /// this process's stdout/stderr.
    fn quiet(&self) -> bool {
        false
    }
}

pub struct StderrProgress;

impl Progress for StderrProgress {
    fn note(&self, msg: &str) {
        eprintln!("{msg}");
    }
    fn command(&self, argv: &[String]) {
        eprintln!("$ {}", argv.join(" "));
    }
}

pub struct SilentProgress;

impl Progress for SilentProgress {
    fn note(&self, _msg: &str) {}
    fn command(&self, _argv: &[String]) {}
    fn quiet(&self) -> bool {
        true
    }
}

/// Every way serve/stop can fail. `Display` reproduces the CLI wording so the
/// CLI adapter can print it as-is; the MCP adapter maps variants to codes.
#[derive(Debug, thiserror::Error)]
pub enum LifecycleError {
    #[error(
        "model `{0}` not found in catalog. Run `paddock sync` or check `paddock fit` for names."
    )]
    ModelNotFound(String),
    #[error("model name `{query}` is ambiguous - candidates:{}", candidates.iter().map(|c| format!("\n  {c}")).collect::<String>())]
    Ambiguous {
        query: String,
        candidates: Vec<String>,
    },
    #[error(
        "no quantization of `{model}` fits this machine ({} RAM); try a smaller model from `paddock fit`",
        crate::output::gib(*ram_bytes)
    )]
    NoFit { model: String, ram_bytes: u64 },
    #[error("no quant `{label}` for this model; available: {}", available.join(", "))]
    UnknownQuant {
        label: String,
        available: Vec<String>,
    },
    #[error("required runtime is not installed - run `{}` then retry", install.argv.join(" "))]
    NoRuntime { install: InstallPlan },
    #[error("install declined - nothing launched. Run `{command}` yourself, then retry.")]
    InstallDeclined { command: String },
    #[error(
        "server exited with {status} before becoming ready - run `{}` manually to see the error",
        argv.join(" ")
    )]
    ServerExited {
        status: String,
        argv: Vec<String>,
        log_path: Option<PathBuf>,
    },
    /// Readiness timeout on a detached child. The server is still running and
    /// (for llama.cpp/mlx) already registered, so `paddock ps` can see it.
    #[error(
        "server not ready yet (pid {}) - it keeps running; check `paddock ps`",
        pid.map(|p| p.to_string()).unwrap_or_else(|| "?".into())
    )]
    NotReady {
        pid: Option<u32>,
        log_path: Option<PathBuf>,
        plan: Box<ServePlan>,
    },
    #[error("ollama daemon not reachable on 11434 - is it running?")]
    OllamaUnreachable,
    #[error("no running server matches `{target}`{}", if running.is_empty() { String::new() } else { format!("\nrunning: {}", running.join(", ")) })]
    NoServerMatch { target: String, running: Vec<String> },
    #[error("`{target}` matches several servers - be specific:{}", candidates.iter().map(|(m, p)| format!("\n  {m} (pid {p})")).collect::<String>())]
    AmbiguousServer {
        target: String,
        candidates: Vec<(String, u32)>,
    },
    #[error("catalog is empty - run `paddock sync` first")]
    CatalogEmpty,
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl From<paddock_core::PaddockError> for LifecycleError {
    fn from(e: paddock_core::PaddockError) -> Self {
        LifecycleError::Other(e.into())
    }
}

/// What to do when the plan needs a runtime that is not installed.
pub enum InstallPolicy<'a> {
    /// Ask the human (CLI/TUI): the callback prompts and installs, or returns
    /// `InstallDeclined`.
    Ask(&'a dyn Fn(&InstallPlan) -> Result<(), LifecycleError>),
    /// Never install (MCP): return `NoRuntime` before touching anything.
    Refuse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServeMode {
    /// Child inherits the tty; the caller waits on `ServeOutcome.child`.
    Foreground,
    /// Child is detached (own session, log file) and registered; it outlives
    /// this process.
    Detached,
}

pub struct ServeOutcome {
    /// Final plan (the port may have moved).
    pub plan: ServePlan,
    /// Spawned child pid; None when the already-running Ollama daemon serves.
    pub pid: Option<u32>,
    /// Detached log file, when a child was spawned detached.
    pub log_path: Option<PathBuf>,
    /// Foreground only: the child handle for the caller to wait on.
    pub child: Option<std::process::Child>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stopped {
    pub model_ref: String,
    pub pid: u32,
}

pub(crate) enum Lookup<'a> {
    Found(&'a CatalogModel),
    Ambiguous(Vec<&'a str>),
    NotFound,
}

/// Exact name match first, then case-insensitive exact, then
/// case-insensitive contains.
pub(crate) fn find_model<'a>(models: &'a [CatalogModel], query: &str) -> Lookup<'a> {
    if let Some(m) = models.iter().find(|m| m.name == query) {
        return Lookup::Found(m);
    }
    let q = query.to_lowercase();
    if let Some(m) = models.iter().find(|m| m.name.to_lowercase() == q) {
        return Lookup::Found(m);
    }
    let matches: Vec<&CatalogModel> = models
        .iter()
        .filter(|m| m.name.to_lowercase().contains(&q))
        .collect();
    match matches.as_slice() {
        [] => Lookup::NotFound,
        [one] => Lookup::Found(one),
        many => Lookup::Ambiguous(many.iter().map(|m| m.name.as_str()).collect()),
    }
}

/// Index into `variants` of the variant whose quant label equals `label`
/// (case-insensitive). On a label shared by several variants, returns the
/// best-quality one (first in `variants_by_quality` order).
pub(crate) fn resolve_quant(variants: &[ModelVariant], label: &str) -> Result<usize, LifecycleError> {
    let order = paddock_core::score::variants_by_quality(variants);
    if let Some(&idx) = order
        .iter()
        .find(|&&i| variants[i].quant.eq_ignore_ascii_case(label))
    {
        return Ok(idx);
    }
    Err(LifecycleError::UnknownQuant {
        label: label.to_string(),
        available: order.iter().map(|&i| variants[i].quant.clone()).collect(),
    })
}

/// Catalog lookup + best-fitting variant pick over an in-memory model list.
/// Returns the model and the index into `model.variants` of the chosen quant.
pub fn resolve_in(
    models: &[CatalogModel],
    budget: &MemoryBudget,
    query: &str,
    quant: Option<&str>,
) -> Result<(CatalogModel, usize), LifecycleError> {
    if models.is_empty() {
        return Err(LifecycleError::CatalogEmpty);
    }
    let model = match find_model(models, query) {
        Lookup::Found(m) => m.clone(),
        Lookup::Ambiguous(names) => {
            return Err(LifecycleError::Ambiguous {
                query: query.to_string(),
                candidates: names.into_iter().map(str::to_string).collect(),
            });
        }
        Lookup::NotFound => return Err(LifecycleError::ModelNotFound(query.to_string())),
    };

    let mvs: Vec<_> = model
        .variants
        .iter()
        .map(|v| model.to_model_variant(v))
        .collect();

    // Explicit quant launches that variant even if it does not fit; the
    // verdict is informational (consistent with the TUI quant picker).
    if let Some(label) = quant {
        let idx = resolve_quant(&mvs, label)?;
        return Ok((model, idx));
    }

    let Some(best) = best_variant(&mvs, budget) else {
        return Err(LifecycleError::NoFit {
            model: model.name.clone(),
            ram_bytes: budget.ram_total_bytes,
        });
    };
    // Pointer identity, not quant-label equality: two variants can share the
    // same quant string, and `best` borrows from `mvs` (same order as
    // `model.variants`).
    let best_idx = mvs
        .iter()
        .position(|v| std::ptr::eq(v, best))
        .expect("best_variant borrows from mvs");
    Ok((model, best_idx))
}

/// `resolve_in` over the on-disk catalog. Shared by `run`, `serve` and MCP.
pub fn resolve_model(
    app: &App,
    query: &str,
    quant: Option<&str>,
) -> Result<(CatalogModel, usize), LifecycleError> {
    let db = app.open_db()?;
    let models = db
        .list_models()
        .map_err(|e| LifecycleError::Other(anyhow::Error::from(e).context("reading catalog")))?;
    resolve_in(&models, &app.budget, query, quant)
}

/// Resolve the launch context for a chosen model variant: explicit `--ctx`
/// wins, otherwise auto-size against this machine's memory budget.
pub fn resolved_ctx(app: &App, model: &CatalogModel, idx: usize, ctx: Option<u32>) -> u32 {
    let mv = model.to_model_variant(&model.variants[idx]);
    paddock_core::estimate::resolve_ctx(ctx, &mv, &app.budget, model.context_max)
}
```

Now edit `crates/paddock/src/main.rs`:

1. Delete `enum Lookup`, `fn find_model`, `fn resolve_quant`, `fn resolved_ctx`, `fn resolve_model` and, inside `mod tests`, the `model`, `mv` helpers and the seven tests that use them (`exact_match_wins_over_contains` through `resolve_quant_no_match_lists_available`). Keep `spawned_child_waits_without_deadline` and `daemon_probe_keeps_short_deadline` for now (Task 2 moves them).
2. Add at the top: `use crate::lifecycle::{LifecycleError, resolve_model, resolved_ctx};`
3. Add the CLI error renderer, right after `fn main`:

```rust
/// CLI rendering of lifecycle errors. The interactive-disambiguation cases
/// print exactly today's multi-line message and exit 1; everything else
/// propagates through anyhow (Rust prints `Error: <Display>`, as `bail!` did).
fn cli_fail(e: LifecycleError) -> anyhow::Error {
    match e {
        LifecycleError::Ambiguous { .. }
        | LifecycleError::AmbiguousServer { .. }
        | LifecycleError::NoServerMatch { .. }
        | LifecycleError::InstallDeclined { .. } => {
            eprintln!("{e}");
            std::process::exit(1);
        }
        other => other.into(),
    }
}
```

4. In `run_model` and `serve_model`, replace `resolve_model(app, query, quant.as_deref())?` with `resolve_model(app, query, quant.as_deref()).map_err(cli_fail)?`. The `resolved_ctx` calls stay as-is (same signature, now imported from `lifecycle`).
5. Remove the now-unused imports from `main.rs` (`ModelVariant`, `best_variant`, `PaddockError` if nothing else uses them; let the compiler tell you).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p paddock`
Expected: all lifecycle tests pass; `tests/cli.rs` still passes (14 tests); the TUI tests still pass. `cargo build -p paddock` has no warnings.

- [ ] **Step 6: Commit**

```bash
git add crates/paddock/Cargo.toml crates/paddock/src/lifecycle.rs crates/paddock/src/main.rs
git commit -m "refactor(paddock): extract model resolution and LifecycleError into lifecycle module"
```

---

## Task 2: Readiness wait with an optional timeout, spawn helpers, registry helpers

**Files:**
- Modify: `crates/paddock/src/lifecycle.rs`
- Modify: `crates/paddock/src/main.rs` (remove `readiness_deadline`, `wait_ready`, `spawn_checked`, `spawn_detached`, `run_checked`, `build_record`, `register_detached`, `RegistryGuard`, the libc `setsid` extern block, and the two deadline tests)

**Interfaces:**
- Produces (all in `lifecycle`):
  - `pub(crate) fn readiness_deadline(child_spawned: bool, requested: Option<Duration>) -> Option<Duration>`
  - `pub(crate) fn wait_ready(probe: &dyn SystemProbe, plan: &ServePlan, child: Option<&mut std::process::Child>, timeout: Option<Duration>, progress: &dyn Progress) -> Result<(), LifecycleError>`
  - `pub(crate) fn spawn_checked(argv: &[String]) -> Result<std::process::Child, LifecycleError>`
  - `pub(crate) fn spawn_detached(argv: &[String], log_path: &Path) -> Result<std::process::Child, LifecycleError>`
  - `pub(crate) fn run_checked(argv: &[String], progress: &dyn Progress) -> Result<(), LifecycleError>`
  - `pub(crate) fn build_record(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) -> ServingRecord`
  - `pub(crate) fn register_detached(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>)`
  - `pub struct RegistryGuard` with `pub fn register(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) -> Self` and `Drop` that unregisters.

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests` in `lifecycle.rs`:

```rust
    use paddock_core::hardware::{MockProbe, SystemProbe};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Probe that answers the readiness GET only from the Nth call on.
    struct CountingProbe {
        calls: AtomicUsize,
        answer_from: usize,
    }

    impl SystemProbe for CountingProbe {
        fn sysctl_string(&self, _: &str) -> Option<String> {
            None
        }
        fn sysctl_u64(&self, _: &str) -> Option<u64> {
            None
        }
        fn gpu_recommended_working_set(&self) -> Option<u64> {
            None
        }
        fn which(&self, _: &str) -> Option<String> {
            None
        }
        fn run_command(&self, _: &str, _: &[&str]) -> Option<String> {
            None
        }
        fn http_get_local(&self, _: &str) -> Option<String> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            (n >= self.answer_from).then(|| "ok".to_string())
        }
        fn http_post_local(&self, _: &str, _: &str) -> Option<String> {
            None
        }
    }

    fn sleeping_child() -> std::process::Child {
        std::process::Command::new("sleep")
            .arg("30")
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("sleep is available")
    }

    fn spawned_plan() -> ServePlan {
        ServePlan {
            server_argv: Some(vec!["llama-server".into(), "--port".into(), "8080".into()]),
            pre_steps: vec![],
            endpoint: "http://127.0.0.1:8080".into(),
            openai_url: "http://127.0.0.1:8080/v1/chat/completions".into(),
            model_ref: "x".into(),
            ready_path: "/health".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::LlamaCpp,
            ctx: 4096,
            port: Some(8080),
        }
    }

    #[test]
    fn spawned_child_waits_without_deadline_by_default() {
        // First-run model downloads can take tens of minutes; the CLI passes
        // None and any fixed cap would kill a healthy child mid-download.
        assert_eq!(readiness_deadline(true, None), None);
        assert_eq!(
            readiness_deadline(true, Some(Duration::from_secs(600))),
            Some(Duration::from_secs(600))
        );
    }

    #[test]
    fn daemon_probe_keeps_short_deadline() {
        assert_eq!(readiness_deadline(false, None), Some(Duration::from_secs(3)));
        assert_eq!(
            readiness_deadline(false, Some(Duration::from_secs(600))),
            Some(Duration::from_secs(3))
        );
    }

    #[test]
    fn wait_ready_timeout_leaves_child_alive() {
        let probe = MockProbe::default(); // never answers
        let mut child = sleeping_child();
        let pid = child.id();
        let err = wait_ready(
            &probe,
            &spawned_plan(),
            Some(&mut child),
            Some(Duration::from_millis(300)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::NotReady { pid: Some(p), log_path: None, .. } => assert_eq!(p, pid),
            other => panic!("expected NotReady, got {other}"),
        }
        assert!(child.try_wait().unwrap().is_none(), "child must still be running");
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn wait_ready_returns_once_probe_answers() {
        let probe = CountingProbe {
            calls: AtomicUsize::new(0),
            answer_from: 3,
        };
        let mut child = sleeping_child();
        wait_ready(
            &probe,
            &spawned_plan(),
            Some(&mut child),
            Some(Duration::from_secs(10)),
            &SilentProgress,
        )
        .unwrap();
        assert_eq!(probe.calls.load(Ordering::SeqCst), 3);
        let _ = child.kill();
        let _ = child.wait();
    }

    #[test]
    fn wait_ready_daemon_timeout_is_ollama_unreachable() {
        let probe = MockProbe::default();
        let err = wait_ready(
            &probe,
            &spawned_plan(),
            None,
            Some(Duration::ZERO),
            &SilentProgress,
        )
        .unwrap_err();
        assert!(matches!(err, LifecycleError::OllamaUnreachable));
    }

    #[test]
    fn wait_ready_child_exit_is_server_exited() {
        let probe = MockProbe::default();
        let mut child = std::process::Command::new("false").spawn().unwrap();
        let err = wait_ready(
            &probe,
            &spawned_plan(),
            Some(&mut child),
            Some(Duration::from_secs(5)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::ServerExited { argv, .. } => {
                assert_eq!(argv[0], "llama-server");
            }
            other => panic!("expected ServerExited, got {other}"),
        }
    }

    #[test]
    fn run_checked_quiet_does_not_fail_on_success() {
        run_checked(&["true".into()], &SilentProgress).unwrap();
        let err = run_checked(&["false".into()], &SilentProgress).unwrap_err();
        assert!(err.to_string().contains("`false` failed"));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p paddock --bin paddock lifecycle`
Expected: compile error (`readiness_deadline`, `wait_ready`, `run_checked` not found in `lifecycle`).

- [ ] **Step 3: Move and adapt the helpers**

Add to the `use` block at the top of `lifecycle.rs`:

```rust
use std::path::Path;
use std::time::{Duration, Instant};

use paddock_core::hardware::SystemProbe;
use paddock_core::serving::{Registry, ServingRecord};
```

Then add, above `#[cfg(test)]`:

```rust
/// How long to wait for readiness. A spawned child gets whatever the caller
/// asked for (the CLI asks for None: runtimes like `llama-server -hf` may be
/// downloading a multi-GB model on first run, and any fixed cap conflates
/// "still downloading" with "hung"). Without a child the Ollama daemon is
/// expected up already, so refusal should be near-instant.
pub(crate) fn readiness_deadline(child_spawned: bool, requested: Option<Duration>) -> Option<Duration> {
    if child_spawned {
        requested
    } else {
        Some(Duration::from_secs(3))
    }
}

/// Poll `{endpoint}{ready_path}` until it answers 2xx. Each iteration blocks
/// at most ~800 ms (300 ms connect + 500 ms read in `http_get_local`, plus a
/// 250 ms sleep), so Ctrl-C feels instant. A notice after 5 s and a heartbeat
/// every 60 s go through `progress`. On `timeout`: with a child, `NotReady`
/// (the child is NOT killed here, the caller decides); without, the daemon
/// case, `OllamaUnreachable`.
pub(crate) fn wait_ready(
    probe: &dyn SystemProbe,
    plan: &ServePlan,
    mut child: Option<&mut std::process::Child>,
    timeout: Option<Duration>,
    progress: &dyn Progress,
) -> Result<(), LifecycleError> {
    let url = format!("{}{}", plan.endpoint, plan.ready_path);
    let start = Instant::now();
    let mut notified = false;
    let mut next_heartbeat = Duration::from_secs(60);
    loop {
        if probe.http_get_local(&url).is_some() {
            return Ok(());
        }
        if let Some(c) = child.as_deref_mut()
            && let Some(status) = c.try_wait().map_err(anyhow::Error::from)?
        {
            return Err(LifecycleError::ServerExited {
                status: status.to_string(),
                argv: plan.server_argv.clone().unwrap_or_default(),
                log_path: None,
            });
        }
        if let Some(deadline) = timeout
            && start.elapsed() >= deadline
        {
            return Err(match child.as_deref() {
                Some(c) => LifecycleError::NotReady {
                    pid: Some(c.id()),
                    log_path: None,
                    plan: Box::new(plan.clone()),
                },
                None => LifecycleError::OllamaUnreachable,
            });
        }
        if !notified && start.elapsed() >= Duration::from_secs(5) {
            progress.note("downloading/loading model - this can take a while");
            notified = true;
        }
        if start.elapsed() >= next_heartbeat {
            progress.note(&format!(
                "still waiting for {} - Ctrl-C to stop",
                plan.endpoint
            ));
            next_heartbeat += Duration::from_secs(60);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Spawn a server child on the tty, with an actionable error when the binary
/// is missing.
pub(crate) fn spawn_checked(argv: &[String]) -> Result<std::process::Child, LifecycleError> {
    std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .spawn()
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]).into())
}

// libc-free, matching serving.rs style: detach into a new session.
unsafe extern "C" {
    #[link_name = "setsid"]
    fn libc_setsid() -> i32;
}

/// Spawn a server child detached from the controlling terminal, with stdout +
/// stderr captured to `log_path`. Dropping the handle does NOT kill it.
pub(crate) fn spawn_detached(argv: &[String], log_path: &Path) -> Result<std::process::Child, LifecycleError> {
    use std::os::unix::process::CommandExt;

    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("cannot create log dir {parent:?}: {e}"))?;
    }
    let log = std::fs::File::create(log_path)
        .map_err(|e| anyhow::anyhow!("cannot create log file {log_path:?}: {e}"))?;
    let log_err = log.try_clone().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut cmd = std::process::Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log_err);
    // SAFETY: setsid only creates a new session; async-signal-safe, no allocation.
    unsafe {
        cmd.pre_exec(|| {
            libc_setsid();
            Ok(())
        });
    }
    cmd.spawn()
        .map_err(|e| anyhow::anyhow!("failed to start {}: {e}. Is it in PATH?", argv[0]).into())
}

/// Run a pre-step to completion and fail on non-zero exit. Output streams to
/// the tty unless `progress.quiet()` (MCP: stdout is the protocol channel).
pub(crate) fn run_checked(argv: &[String], progress: &dyn Progress) -> Result<(), LifecycleError> {
    use std::process::Stdio;
    let cmd = argv.join(" ");
    let stdio = || if progress.quiet() { Stdio::null() } else { Stdio::inherit() };
    let status = std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(Stdio::null())
        .stdout(stdio())
        .stderr(stdio())
        .status()
        .map_err(|e| anyhow::anyhow!("running `{cmd}`: {e}"))?;
    if !status.success() {
        return Err(anyhow::anyhow!("`{cmd}` failed ({status}); fix it and retry").into());
    }
    Ok(())
}

/// Build a serving registry record for a running server.
pub(crate) fn build_record(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) -> ServingRecord {
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    ServingRecord {
        pid,
        runtime: plan.runtime,
        endpoint: plan.endpoint.clone(),
        openai_url: plan.openai_url.clone(),
        model_ref: plan.model_ref.clone(),
        ready_path: plan.ready_path.clone(),
        started_at,
        ctx: plan.ctx,
        log_path,
        port: plan.port,
    }
}

/// Register a detached child server. Unlike `RegistryGuard`, this does NOT
/// unregister on drop - the server must survive this process exiting.
pub(crate) fn register_detached(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) {
    let record = build_record(plan, pid, log_path);
    if let Err(e) = Registry::open_default().register(&record) {
        eprintln!("warning: could not record serving state: {e}");
    }
}

/// RAII wrapper around the serving registry for foreground children:
/// best-effort register on creation, unregister on drop.
pub struct RegistryGuard {
    registry: Registry,
    pid: u32,
}

impl RegistryGuard {
    pub fn register(plan: &ServePlan, pid: u32, log_path: Option<PathBuf>) -> Self {
        let registry = Registry::open_default();
        let record = build_record(plan, pid, log_path);
        if let Err(e) = registry.register(&record) {
            eprintln!("warning: could not record serving state: {e}");
        }
        Self { registry, pid }
    }
}

impl Drop for RegistryGuard {
    fn drop(&mut self) {
        let _ = self.registry.unregister(self.pid);
    }
}
```

Note: `run_checked` sets `.stdin(Stdio::null())` in both modes. Today it inherits stdin; no pre-step (`ollama pull`, `ollama stop`) reads stdin, and inheriting it under MCP would let a child consume protocol bytes.

Now in `main.rs`: delete `fn readiness_deadline`, `fn wait_ready`, `fn spawn_checked`, the `unsafe extern "C"` block, `fn spawn_detached`, `fn run_checked`, `fn build_record`, `fn register_detached`, `struct RegistryGuard` + impls, and the two tests `spawned_child_waits_without_deadline` / `daemon_probe_keeps_short_deadline`. `serve_with_plan` and `stop_servers` still reference these; make them compile for now by importing: `use crate::lifecycle::{RegistryGuard, register_detached, run_checked, spawn_checked, spawn_detached, wait_ready, StderrProgress};` and:

- in `serve_with_plan`, replace `wait_ready(&plan, child.as_mut())` with `wait_ready(&RealSystemProbe, &plan, child.as_mut(), readiness_deadline_for(child.is_some()), &StderrProgress).map_err(anyhow::Error::from)` where a temporary local closure `let readiness_deadline_for = |spawned: bool| crate::lifecycle::readiness_deadline(spawned, None);` is defined above it; replace `run_checked(step)?` with `run_checked(step, &StderrProgress)?`; the `spawn_*` calls get `?` on a `LifecycleError` which converts into anyhow via `?` automatically (thiserror implements `std::error::Error`).
- in `stop_servers`, replace `run_checked(&[...])` with `run_checked(&[...], &StderrProgress)`.

(Task 3 and Task 4 replace these bodies entirely; this step only keeps the build green.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p paddock`
Expected: all pass, including the new `wait_ready_*` tests (the timeout one takes ~0.3 s). `cargo build -p paddock` has no warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/lifecycle.rs crates/paddock/src/main.rs
git commit -m "refactor(paddock): move readiness wait, spawn and registry helpers into lifecycle"
```

---

## Task 3: `lifecycle::serve` and the CLI/TUI adapter

**Files:**
- Modify: `crates/paddock/src/lifecycle.rs`
- Modify: `crates/paddock/src/main.rs` (`serve_with_plan`, `confirm_and_install`, `launch`)
- Modify: `crates/paddock/src/tui/mod.rs` (no code change required; verify the call still compiles)

**Interfaces:**
- Consumes: `wait_ready`, `spawn_checked`, `spawn_detached`, `run_checked`, `register_detached`, `readiness_deadline` (Task 2).
- Produces: `pub fn serve(plan: ServePlan, mode: ServeMode, policy: InstallPolicy<'_>, ready_timeout: Option<Duration>, progress: &dyn Progress) -> Result<ServeOutcome, LifecycleError>`; `main.rs`: `pub(crate) fn confirm_and_install(install: &InstallPlan) -> Result<(), LifecycleError>`.

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests` in `lifecycle.rs`:

```rust
    #[test]
    fn serve_refuse_policy_returns_no_runtime_before_spawning() {
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["definitely-not-a-binary-xyz".into()]);
        plan.install = Some(InstallPlan {
            kind: RuntimeKind::LlamaCpp,
            argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
        });
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(1)),
            &SilentProgress,
        )
        .unwrap_err();
        match err {
            LifecycleError::NoRuntime { install } => assert_eq!(install.argv[0], "brew"),
            other => panic!("expected NoRuntime, got {other}"),
        }
    }

    #[test]
    fn serve_ask_policy_propagates_install_declined() {
        let mut plan = spawned_plan();
        plan.install = Some(InstallPlan {
            kind: RuntimeKind::LlamaCpp,
            argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
        });
        let decline = |i: &InstallPlan| {
            Err(LifecycleError::InstallDeclined {
                command: i.argv.join(" "),
            })
        };
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Ask(&decline),
            None,
            &SilentProgress,
        )
        .unwrap_err();
        assert!(matches!(err, LifecycleError::InstallDeclined { .. }));
    }

    #[test]
    fn serve_missing_binary_is_actionable() {
        // Isolated serving dir so the history/registry writes never touch the
        // real one.
        let dir = tempfile::tempdir().unwrap();
        // SAFETY: tests in this module that read PADDOCK_SERVING_DIR run in
        // this process only; the var is scoped to this test's lifetime.
        unsafe { std::env::set_var("PADDOCK_SERVING_DIR", dir.path()) };
        let mut plan = spawned_plan();
        plan.server_argv = Some(vec!["definitely-not-a-binary-xyz".into(), "--port".into(), "8080".into()]);
        let err = serve(
            plan,
            ServeMode::Detached,
            InstallPolicy::Refuse,
            Some(Duration::from_secs(1)),
            &SilentProgress,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("failed to start definitely-not-a-binary-xyz"),
            "got: {err}"
        );
        unsafe { std::env::remove_var("PADDOCK_SERVING_DIR") };
    }
```

Check how `default_serving_dir` reads its override before relying on `PADDOCK_SERVING_DIR` here: `grep -n "PADDOCK_SERVING_DIR" crates/paddock-core/src/serving.rs`. It is the same variable `tests/cli.rs` sets, so it exists.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p paddock --bin paddock lifecycle::tests::serve_`
Expected: compile error (`serve` not found).

- [ ] **Step 3: Implement `serve`**

Add to the `use` block in `lifecycle.rs`:

```rust
use paddock_core::catalog::RuntimeKind;
use paddock_core::hardware::RealSystemProbe;
use paddock_core::serving::History;
```

Add above `#[cfg(test)]`:

```rust
/// Full serve lifecycle: install policy, port fallback, history record, spawn
/// (foreground or detached), readiness wait, pre-steps, Ollama warm-up,
/// registry entry. Returns as soon as the endpoint is ready. In `Foreground`
/// mode the child handle is returned for the caller to wait on; in `Detached`
/// mode the registry entry has already been written.
///
/// On a readiness timeout with a detached child, the child is left running
/// and registered, and `NotReady { pid, log_path, plan }` is returned so the
/// caller can report "starting". Every other failure kills the child.
pub fn serve(
    mut plan: ServePlan,
    mode: ServeMode,
    policy: InstallPolicy<'_>,
    ready_timeout: Option<Duration>,
    progress: &dyn Progress,
) -> Result<ServeOutcome, LifecycleError> {
    if plan.port_ignored {
        progress.note("warning: --port is ignored for the Ollama daemon (fixed 11434)");
    }
    if let Some(install) = &plan.install {
        match policy {
            InstallPolicy::Ask(confirm) => confirm(install)?,
            InstallPolicy::Refuse => {
                return Err(LifecycleError::NoRuntime {
                    install: install.clone(),
                });
            }
        }
    }

    // Spawned servers (llama.cpp/mlx) default to 8080; pick a free port so
    // concurrent servers don't collide (and so the readiness probe can't be
    // answered by a different process already on the port). The Ollama daemon
    // has a fixed port and no server_argv, so it's untouched.
    if plan.server_argv.is_some()
        && let Some(requested) = plan.port
        && let Some(free) = paddock_core::serving::free_port(requested)
        && free != requested
    {
        progress.note(&format!("port {requested} is busy - serving on {free} instead"));
        plan = plan.with_port(free);
    }

    // Remember spawned (llama.cpp/mlx) serves so the TUI can offer one-key
    // relaunch. Best-effort; Ollama is covered by /api/tags, not recorded.
    if plan.server_argv.is_some() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        History::open_default().record(&plan, now);
    }

    let log_dir = paddock_core::serving::default_serving_dir().join("logs");
    let mut log_path: Option<PathBuf> = None;
    let mut child = match &plan.server_argv {
        Some(argv) => {
            progress.command(argv);
            match mode {
                ServeMode::Foreground => Some(spawn_checked(argv)?),
                ServeMode::Detached => {
                    // Spawn to a per-invocation placeholder (our own pid makes
                    // it unique across concurrent serves), then rename to
                    // <child-pid>.log once the child pid is known.
                    let tmp_log = log_dir.join(format!("pending-{}.log", std::process::id()));
                    let c = spawn_detached(argv, &tmp_log)?;
                    let final_log = log_dir.join(format!("{}.log", c.id()));
                    let actual = match std::fs::rename(&tmp_log, &final_log) {
                        Ok(()) => final_log,
                        Err(_) => tmp_log, // rename failed: the data is still at the pending path
                    };
                    log_path = Some(actual);
                    Some(c)
                }
            }
        }
        None => None,
    };

    // Readiness + pre-steps; never leave an orphaned server behind on failure,
    // except the detached-timeout case, which is the whole point of `NotReady`.
    let timeout = readiness_deadline(child.is_some(), ready_timeout);
    let prepared = wait_ready(&RealSystemProbe, &plan, child.as_mut(), timeout, progress).and_then(|()| {
        for step in &plan.pre_steps {
            progress.command(step);
            run_checked(step, progress)?;
        }
        Ok(())
    });
    match prepared {
        Ok(()) => {}
        Err(LifecycleError::NotReady { pid, .. }) if mode == ServeMode::Detached => {
            if plan.runtime != RuntimeKind::Ollama
                && let Some(pid) = pid
            {
                register_detached(&plan, pid, log_path.clone());
            }
            return Err(LifecycleError::NotReady {
                pid,
                log_path,
                plan: Box::new(plan),
            });
        }
        Err(e) => {
            if let Some(c) = child.as_mut() {
                let _ = c.kill();
                let _ = c.wait();
            }
            return Err(e);
        }
    }

    // Ollama loads a model only on its first request and the daemon outlives
    // us - warm it up now so "ready" means ready (and the model shows in
    // /api/ps + the tray). Best-effort: a failure leaves a working endpoint
    // that simply cold-starts on first use.
    if plan.runtime == RuntimeKind::Ollama {
        progress.note(&format!("loading {} into memory…", plan.model_ref));
        if !paddock_core::serving::warm_up_ollama(&RealSystemProbe, &plan.model_ref) {
            progress.note("warning: warm-up failed - the model will load on the first request");
        }
    }

    let pid = child.as_ref().map(|c| c.id());
    match mode {
        ServeMode::Foreground => Ok(ServeOutcome {
            plan,
            pid,
            log_path: None,
            child,
        }),
        ServeMode::Detached => {
            // Register WITHOUT the drop-guard so the child outlives us. A
            // cold-started Ollama daemon is managed by ollama itself.
            if let Some(pid) = pid
                && plan.runtime != RuntimeKind::Ollama
            {
                register_detached(&plan, pid, log_path.clone());
            }
            Ok(ServeOutcome {
                plan,
                pid,
                log_path,
                child: None,
            })
        }
    }
}
```

`ServeMode` needs `PartialEq` (already derived in Task 1).

- [ ] **Step 4: Rewrite the CLI adapter in `main.rs`**

Replace the whole `serve_with_plan` function with:

```rust
/// CLI/TUI adapter over `lifecycle::serve`: same terminal output as before
/// (endpoint block on stdout, progress on stderr), install confirmed on the
/// tty, no readiness timeout. Shared with the TUI `s` key.
pub(crate) fn serve_with_plan(plan: ServePlan, foreground: bool) -> Result<()> {
    use crate::lifecycle::{InstallPolicy, ServeMode, StderrProgress, serve};
    use paddock_core::catalog::RuntimeKind;

    let mode = if foreground {
        ServeMode::Foreground
    } else {
        ServeMode::Detached
    };
    let outcome = serve(
        plan,
        mode,
        InstallPolicy::Ask(&confirm_and_install),
        None,
        &StderrProgress,
    )
    .map_err(cli_fail)?;
    let plan = outcome.plan;
    output::print_endpoint(&plan);

    match outcome.child {
        Some(mut c) => {
            // Best-effort registry entry for tray/UIs; the guard unregisters
            // on every exit path including `?`. SIGINT kills paddock and the
            // child together (default tty behavior) without running Drop -
            // the stale file is reaped by the next `list_live`.
            let _guard = (plan.runtime != RuntimeKind::Ollama)
                .then(|| RegistryGuard::register(&plan, c.id(), None));
            eprintln!("serving - press Ctrl-C to stop");
            let status = c.wait()?;
            if !status.success() {
                bail!("server exited with {status}");
            }
            Ok(())
        }
        None => {
            match outcome.pid {
                // Cold-started the Ollama daemon; it serves in the background on
                // its fixed port. ollama ps / ollama stop manage it, not paddock.
                Some(_) if plan.runtime == RuntimeKind::Ollama => {
                    eprintln!("ollama daemon started in the background");
                }
                Some(pid) => {
                    eprintln!(
                        "serving in background · pid {pid} · paddock logs {}",
                        plan.model_ref
                    );
                }
                // Already-running Ollama daemon: nothing was spawned.
                None => {}
            }
            Ok(())
        }
    }
}
```

Change `confirm_and_install` to return `LifecycleError` instead of exiting on decline (the non-tty and missing-installer cases keep their `exit(1)` since they are CLI-only, untested paths and their wording is unchanged):

```rust
pub(crate) fn confirm_and_install(install: &InstallPlan) -> Result<(), LifecycleError> {
    use std::io::IsTerminal;

    let cmd = install
        .argv
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    if !std::io::stdin().is_terminal() {
        eprintln!(
            "required runtime is not installed and stdin is not a terminal - \
             re-run interactively to confirm install (`{cmd}`)."
        );
        std::process::exit(1);
    }
    eprint!("required runtime is not installed. install with `{cmd}`? [y/N] ");
    std::io::stderr().flush().map_err(anyhow::Error::from)?;
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(anyhow::Error::from)?;
    let answer = answer.trim().to_ascii_lowercase();
    if answer != "y" && answer != "yes" {
        return Err(LifecycleError::InstallDeclined { command: cmd });
    }
    // Check the installer binary exists before running it (avoid exec-ENOENT).
    let installer = &install.argv[0];
    if !find_in_path(installer) {
        eprintln!("{}", installer_missing_hint(installer));
        std::process::exit(1);
    }
    let status = std::process::Command::new(installer)
        .args(&install.argv[1..])
        .status()
        .with_context(|| format!("running `{cmd}`"))?;
    if !status.success() {
        bail!("`{cmd}` failed ({status}); fix the install and retry");
    }
    Ok(())
}
```

(`bail!` and `.with_context(...)?` produce `anyhow::Error`, which converts into `LifecycleError::Other` through the `#[from]`.)

In `launch`, change `confirm_and_install(install)?;` to `confirm_and_install(install).map_err(cli_fail)?;` so a declined install still prints today's line and exits 1.

Remove the temporary imports added in Task 2 Step 3 that are now unused (`register_detached`, `run_checked`, `spawn_checked`, `spawn_detached`, `wait_ready`); keep `RegistryGuard`. Remove `use std::io::Write;` only if nothing else uses it (`stop_servers` still does).

`tui/mod.rs` line 149 still calls `crate::serve_with_plan(plan, false)`. No edit needed.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p paddock`
Expected: all pass (lifecycle, cli.rs 14, TUI 54). `cargo build -p paddock` has no warnings.

- [ ] **Step 6: Manual smoke (requires Ollama installed)**

Run: `cargo run -p paddock -- serve llama3.2:1b`
Expected: exactly today's output (pull, `endpoint`, `openai`, `model`, `try it` lines). Then `cargo run -p paddock -- stop llama3.2:1b` or `ollama stop llama3.2:1b`.

- [ ] **Step 7: Commit**

```bash
git add crates/paddock/src/lifecycle.rs crates/paddock/src/main.rs
git commit -m "refactor(paddock): serve lifecycle returns values; CLI and TUI become adapters"
```

---

## Task 4: `lifecycle::stop` and the CLI adapter

**Files:**
- Modify: `crates/paddock/src/lifecycle.rs`
- Modify: `crates/paddock/src/main.rs` (`stop_servers`)

**Interfaces:**
- Produces:
  - `pub fn resolve_servers(target: &str) -> Result<Vec<ServingRecord>, LifecycleError>` (`"all"` still understood via `match_records`)
  - `pub fn stop_records(records: Vec<ServingRecord>, progress: &dyn Progress) -> Vec<Stopped>`
  - `pub fn stop(target: &str, progress: &dyn Progress) -> Result<Vec<Stopped>, LifecycleError>` = both.

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests` in `lifecycle.rs`:

```rust
    #[test]
    fn resolve_servers_unknown_target_lists_running() {
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("PADDOCK_SERVING_DIR", dir.path()) };
        let err = resolve_servers("nope").unwrap_err();
        match err {
            LifecycleError::NoServerMatch { target, running } => {
                assert_eq!(target, "nope");
                assert!(running.is_empty());
            }
            other => panic!("expected NoServerMatch, got {other}"),
        }
        unsafe { std::env::remove_var("PADDOCK_SERVING_DIR") };
    }

    #[test]
    fn stop_records_terminates_and_unregisters() {
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("PADDOCK_SERVING_DIR", dir.path()) };
        let mut child = sleeping_child();
        let pid = child.id();
        let plan = spawned_plan();
        register_detached(&plan, pid, None);
        let registry = Registry::open_default();
        assert_eq!(registry.list_live(&RealSystemProbe).len(), 1);

        let stopped = stop_records(registry.list_live(&RealSystemProbe), &SilentProgress);
        assert_eq!(
            stopped,
            vec![Stopped {
                model_ref: "x".into(),
                pid
            }]
        );
        // SIGTERM delivered: the child exits (reap it so the pid is not reused).
        let _ = child.wait();
        assert!(registry.list_live(&RealSystemProbe).is_empty());
        unsafe { std::env::remove_var("PADDOCK_SERVING_DIR") };
    }
```

These tests and `serve_missing_binary_is_actionable` / `resolve_servers_unknown_target_lists_running` all set the same env var, and cargo runs tests in parallel threads. Mark each of the four env-var tests with `#[serial]`? No new dependency: instead, guard them with a shared lock:

```rust
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
```

and start each of those four tests with `let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());`. Add the lock and the guard line to `serve_missing_binary_is_actionable` (Task 3) as well.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p paddock --bin paddock lifecycle::tests::stop_ lifecycle::tests::resolve_servers`
Expected: compile error (`resolve_servers`, `stop_records` not found).

- [ ] **Step 3: Implement**

Add above `#[cfg(test)]` in `lifecycle.rs`:

```rust
/// Resolve a stop target (model name substring, pid, or `all`) against the
/// live registry. `all` on an empty registry is `NoServerMatch`.
pub fn resolve_servers(target: &str) -> Result<Vec<ServingRecord>, LifecycleError> {
    use paddock_core::serving::{RecordMatch, match_records};

    let records = Registry::open_default().list_live(&RealSystemProbe);
    match match_records(&records, target) {
        RecordMatch::Matched(v) => Ok(v.into_iter().cloned().collect()),
        RecordMatch::Ambiguous(cands) => Err(LifecycleError::AmbiguousServer {
            target: target.to_string(),
            candidates: cands.iter().map(|r| (r.model_ref.clone(), r.pid)).collect(),
        }),
        RecordMatch::NotFound => Err(LifecycleError::NoServerMatch {
            target: target.to_string(),
            running: records.iter().map(|r| r.model_ref.clone()).collect(),
        }),
    }
}

/// Stop every record: `ollama stop` for Ollama-served models, SIGTERM for
/// paddock-spawned servers; unregister each. Best-effort per record.
pub fn stop_records(records: Vec<ServingRecord>, progress: &dyn Progress) -> Vec<Stopped> {
    use paddock_core::serving::terminate;

    let registry = Registry::open_default();
    let mut stopped = Vec::with_capacity(records.len());
    for r in records {
        if r.runtime == RuntimeKind::Ollama {
            let _ = run_checked(
                &["ollama".into(), "stop".into(), r.model_ref.clone()],
                progress,
            );
        } else {
            terminate(r.pid);
        }
        let _ = registry.unregister(r.pid);
        stopped.push(Stopped {
            model_ref: r.model_ref,
            pid: r.pid,
        });
    }
    stopped
}

/// `resolve_servers` + `stop_records`, for callers that need no confirmation
/// step in between (MCP rejects `all` before calling this).
pub fn stop(target: &str, progress: &dyn Progress) -> Result<Vec<Stopped>, LifecycleError> {
    let records = resolve_servers(target)?;
    Ok(stop_records(records, progress))
}
```

Replace `stop_servers` in `main.rs` with:

```rust
fn stop_servers(target: &str, yes: bool) -> Result<()> {
    use crate::lifecycle::{StderrProgress, resolve_servers, stop_records};

    let chosen = resolve_servers(target).map_err(cli_fail)?;

    if target == "all" && !yes {
        eprintln!("about to stop {} server(s):", chosen.len());
        for r in &chosen {
            eprintln!("  {} (pid {})", r.model_ref, r.pid);
        }
        eprint!("proceed? [y/N] ");
        std::io::stderr().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        if !matches!(line.trim(), "y" | "Y") {
            eprintln!("aborted");
            return Ok(());
        }
    }

    for s in stop_records(chosen, &StderrProgress) {
        println!("stopped {} (pid {})", s.model_ref, s.pid);
    }
    Ok(())
}
```

Remove the `run_checked`/`StderrProgress` temporary import from Task 2 if still present.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p paddock`
Expected: all pass, including `stop_unknown_target_errors` and `logs_unknown_target_errors` in `tests/cli.rs` (the latter still uses `show_logs`, untouched). No warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/lifecycle.rs crates/paddock/src/main.rs
git commit -m "refactor(paddock): stop lifecycle returns Stopped records; prompt stays in the CLI"
```

---

## Task 5: JSON row builders in `output.rs`

**Files:**
- Modify: `crates/paddock/src/output.rs` (`FitRow`, `print_fit_json`, `RecommendRow`, `print_recommendations_json`)

**Interfaces:**
- Produces: `pub struct FitRow<'a>` (fields as today, now `pub`), `pub fn fit_rows(rows: &[ScoredModel]) -> Vec<FitRow<'_>>`, `pub struct RecommendRow<'a>` (fields `pub`), `pub fn recommend_rows(rows: &[ScoredModel]) -> Vec<RecommendRow<'_>>`.

- [ ] **Step 1: Write the failing test**

`output.rs` has no test module today. Add at the end of the file:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source};
    use paddock_core::estimate::{DEFAULT_CONTEXT, MemoryBudget, estimate_memory, estimate_speed};
    use paddock_core::score::{UseCase, score_variant};

    fn scored() -> ScoredModel {
        let model = CatalogModel {
            id: 0,
            name: "fake-model".into(),
            family: Some("llama".into()),
            source: Source::Ollama,
            repo: None,
            params_total: 1_000_000_000,
            params_active: 1_000_000_000,
            architecture: Some("llama".into()),
            context_max: 8192,
            released_at: None,
            released_approx: false,
            variants: vec![CatalogVariant {
                quant: "Q4_K_M".into(),
                bpw: 4.83,
                file_size_bytes: None,
                layers: 16,
                kv_heads: 8,
                head_dim: 64,
                embedding_dim: 2048,
                runtime_compat: vec![RuntimeKind::Ollama],
                source_tag: None,
            }],
        };
        let budget = MemoryBudget {
            gpu_effective_bytes: 16 << 30,
            ram_total_bytes: 16 << 30,
        };
        let mv = model.to_model_variant(&model.variants[0]);
        let memory = estimate_memory(&mv, DEFAULT_CONTEXT, &budget);
        let speed = estimate_speed(&mv, 400.0, memory.kv_cache_bytes);
        let score = score_variant(&mv, &memory, &speed, UseCase::General, None);
        ScoredModel {
            model,
            variant_idx: 0,
            memory,
            speed,
            score,
        }
    }

    #[test]
    fn fit_rows_serialize_with_fit_json_keys() {
        let rows = vec![scored()];
        let v = serde_json::to_value(fit_rows(&rows)).unwrap();
        let first = &v[0];
        assert_eq!(first["name"], "fake-model");
        assert_eq!(first["quant"], "Q4_K_M");
        assert!(first["memory"]["total_bytes"].is_u64());
        assert!(first["speed"]["generation_tps"].is_number());
        assert!(first["score"]["total"].is_number());
    }

    #[test]
    fn recommend_rows_carry_justification() {
        let rows = vec![scored()];
        let v = serde_json::to_value(recommend_rows(&rows)).unwrap();
        assert_eq!(v[0]["model"], "fake-model");
        assert!(v[0]["justification"].as_str().unwrap().contains("tok/s"));
    }
}
```

If `estimate_speed`'s signature differs, check `grep -n "pub fn estimate_speed" crates/paddock-core/src/estimate.rs` and adapt the call; the bench plan recorded it as `estimate_speed(v, bandwidth_gbps, kv_cache_bytes)`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p paddock --bin paddock output::tests`
Expected: compile error (`fit_rows`, `recommend_rows` not found).

- [ ] **Step 3: Implement the builders**

Replace the `FitRow` / `print_fit_json` / `RecommendRow` / `print_recommendations_json` block in `output.rs` with:

```rust
/// One row of `fit --json` and of the MCP `paddock_fit` result.
#[derive(serde::Serialize)]
pub struct FitRow<'a> {
    pub name: &'a str,
    pub released_at: Option<i64>,
    pub released_approx: bool,
    pub quant: &'a str,
    pub memory: &'a paddock_core::estimate::MemoryEstimate,
    pub speed: &'a paddock_core::estimate::SpeedEstimate,
    pub score: &'a paddock_core::score::Score,
}

pub fn fit_rows(rows: &[ScoredModel]) -> Vec<FitRow<'_>> {
    rows.iter()
        .map(|r| FitRow {
            name: &r.model.name,
            released_at: r.model.released_at,
            released_approx: r.model.released_approx,
            quant: &r.model.variants[r.variant_idx].quant,
            memory: &r.memory,
            speed: &r.speed,
            score: &r.score,
        })
        .collect()
}

pub fn print_fit_json(rows: &[ScoredModel]) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(&fit_rows(rows))?);
    Ok(())
}
```

and, keeping `justification`, `context_label` and `print_recommendations` where they are:

```rust
/// One row of `recommend --json` and of the MCP `paddock_recommend` result.
#[derive(serde::Serialize)]
pub struct RecommendRow<'a> {
    pub model: &'a str,
    pub quant: &'a str,
    pub score: f64,
    pub justification: String,
}

pub fn recommend_rows(rows: &[ScoredModel]) -> Vec<RecommendRow<'_>> {
    rows.iter()
        .map(|r| RecommendRow {
            model: &r.model.name,
            quant: &r.model.variants[r.variant_idx].quant,
            score: r.score.total,
            justification: justification(r),
        })
        .collect()
}

pub fn print_recommendations_json(rows: &[ScoredModel]) -> anyhow::Result<()> {
    println!("{}", serde_json::to_string_pretty(&recommend_rows(rows))?);
    Ok(())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p paddock`
Expected: all pass, `fit_json_on_empty_catalog_is_empty_array` and `recommend_json_is_array_max_5` unchanged.

- [ ] **Step 5: Commit**

```bash
git add crates/paddock/src/output.rs
git commit -m "refactor(paddock): fit/recommend JSON rows as reusable builders"
```

---

## Task 6: `paddock mcp` server

**Files:**
- Modify: `crates/paddock/Cargo.toml` (add `rmcp`, `schemars`)
- Modify: `crates/paddock/src/cli.rs` (`Command::Mcp`)
- Modify: `crates/paddock/src/main.rs` (`mod mcp;`, dispatch)
- Create: `crates/paddock/src/mcp.rs`

**Interfaces:**
- Consumes: `lifecycle::{resolve_model, resolved_ctx, serve, stop, LifecycleError, InstallPolicy, ServeMode, SilentProgress}`, `output::{fit_rows, recommend_rows}`, `App`.
- Produces: `pub fn run(app: App) -> anyhow::Result<()>`; `pub(crate) fn error_body(e: &LifecycleError) -> serde_json::Value`; `pub(crate) const INSTRUCTIONS: &str`.

- [ ] **Step 1: Add the dependencies**

In `crates/paddock/Cargo.toml` `[dependencies]`:

```toml
rmcp = { version = "3.5", default-features = false, features = ["server", "transport-io", "macros"] }
schemars = "1"
```

`rmcp`'s `transport-io` feature enables `tokio/io-std` itself; the crate's existing `tokio` line (`rt-multi-thread`) is enough.

- [ ] **Step 2: Add the subcommand**

In `crates/paddock/src/cli.rs`, after the `Tray` variant:

```rust
    /// Serve paddock's provisioning tools over MCP on stdio (for coding agents)
    Mcp,
```

In `main.rs`: add `mod mcp;` after `mod lifecycle;`, and in the `match cli.command` add, after the `Tray` arm:

```rust
        Some(Command::Mcp) => mcp::run(app)?,
```

- [ ] **Step 3: Write the failing unit tests**

Create `crates/paddock/src/mcp.rs` with the test module first:

```rust
//! `paddock mcp`: the provisioning tools over the Model Context Protocol on
//! stdio. Third adapter over `lifecycle` (after the CLI and the TUI): no
//! prompts, no install, JSON in, JSON out. stdout is the protocol channel;
//! nothing else may write to it.

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_core::catalog::RuntimeKind;
    use paddock_core::runtime::InstallPlan;

    #[test]
    fn error_body_no_runtime_carries_install_command_and_hint() {
        let e = LifecycleError::NoRuntime {
            install: InstallPlan {
                kind: RuntimeKind::LlamaCpp,
                argv: vec!["brew".into(), "install".into(), "llama.cpp".into()],
            },
        };
        let v = error_body(&e);
        assert_eq!(v["code"], "no_runtime");
        assert_eq!(v["install_command"], "brew install llama.cpp");
        assert_eq!(v["hint"], "ask the user before installing");
        assert!(v["message"].as_str().unwrap().contains("brew install llama.cpp"));
    }

    #[test]
    fn error_body_codes_match_spec_table() {
        let cases: Vec<(LifecycleError, &str)> = vec![
            (LifecycleError::ModelNotFound("x".into()), "model_not_found"),
            (
                LifecycleError::Ambiguous {
                    query: "l".into(),
                    candidates: vec!["a".into(), "b".into()],
                },
                "ambiguous",
            ),
            (
                LifecycleError::NoFit {
                    model: "x".into(),
                    ram_bytes: 8 << 30,
                },
                "no_fit",
            ),
            (
                LifecycleError::UnknownQuant {
                    label: "Q9".into(),
                    available: vec!["Q4_K_M".into()],
                },
                "unknown_quant",
            ),
            (
                LifecycleError::ServerExited {
                    status: "exit status: 1".into(),
                    argv: vec!["llama-server".into()],
                    log_path: None,
                },
                "server_exited",
            ),
            (LifecycleError::OllamaUnreachable, "ollama_unreachable"),
            (
                LifecycleError::NoServerMatch {
                    target: "x".into(),
                    running: vec!["a".into()],
                },
                "no_server_match",
            ),
            (
                LifecycleError::AmbiguousServer {
                    target: "x".into(),
                    candidates: vec![("a".into(), 1)],
                },
                "ambiguous_server",
            ),
            (LifecycleError::CatalogEmpty, "catalog_empty"),
            (LifecycleError::Other(anyhow::anyhow!("boom")), "internal"),
        ];
        for (e, code) in cases {
            let v = error_body(&e);
            assert_eq!(v["code"], code, "{e}");
            assert!(v["message"].is_string());
        }
    }

    #[test]
    fn error_body_extra_fields() {
        let v = error_body(&LifecycleError::ModelNotFound("x".into()));
        assert_eq!(v["hint"], "call paddock_fit for names");

        let v = error_body(&LifecycleError::Ambiguous {
            query: "l".into(),
            candidates: vec!["a".into(), "b".into()],
        });
        assert_eq!(v["candidates"], serde_json::json!(["a", "b"]));

        let v = error_body(&LifecycleError::NoFit {
            model: "x".into(),
            ram_bytes: 8 << 30,
        });
        assert_eq!(v["ram_gib"], 8.0);

        let v = error_body(&LifecycleError::UnknownQuant {
            label: "Q9".into(),
            available: vec!["Q4_K_M".into()],
        });
        assert_eq!(v["available"], serde_json::json!(["Q4_K_M"]));

        let v = error_body(&LifecycleError::ServerExited {
            status: "exit status: 1".into(),
            argv: vec!["llama-server".into()],
            log_path: Some("/tmp/1.log".into()),
        });
        assert_eq!(v["argv"], serde_json::json!(["llama-server"]));
        assert_eq!(v["log_path"], "/tmp/1.log");

        let v = error_body(&LifecycleError::NoServerMatch {
            target: "x".into(),
            running: vec!["a".into()],
        });
        assert_eq!(v["running"], serde_json::json!(["a"]));

        let v = error_body(&LifecycleError::AmbiguousServer {
            target: "x".into(),
            candidates: vec![("a".into(), 1)],
        });
        assert_eq!(
            v["candidates"],
            serde_json::json!([{ "model_ref": "a", "pid": 1 }])
        );

        let v = error_body(&LifecycleError::CatalogEmpty);
        assert_eq!(v["hint"], "run paddock sync");
    }

    #[test]
    fn error_body_other_is_internal() {
        // Review Focus 1: plan_serve rejects port 0 with a PaddockError; it
        // reaches the agent as `internal` with the message, never a panic.
        let e: LifecycleError = paddock_core::PaddockError::Other(
            "port 0 is not supported; pick a fixed port so the endpoint is known upfront".into(),
        )
        .into();
        let v = error_body(&e);
        assert_eq!(v["code"], "internal");
        assert!(v["message"].as_str().unwrap().contains("port 0"));
    }

    #[test]
    fn serve_result_starting_carries_pid_and_endpoint() {
        let plan = paddock_core::runtime::ServePlan {
            server_argv: Some(vec!["llama-server".into()]),
            pre_steps: vec![],
            endpoint: "http://127.0.0.1:8081".into(),
            openai_url: "http://127.0.0.1:8081/v1/chat/completions".into(),
            model_ref: "x".into(),
            ready_path: "/health".into(),
            install: None,
            port_ignored: false,
            runtime: RuntimeKind::LlamaCpp,
            ctx: 4096,
            port: Some(8081),
        };
        let v = serve_result("starting", &plan, Some(42), Some(std::path::PathBuf::from("/tmp/42.log")));
        assert_eq!(v["status"], "starting");
        assert_eq!(v["endpoint"], "http://127.0.0.1:8081");
        assert_eq!(v["openai_url"], "http://127.0.0.1:8081/v1/chat/completions");
        assert_eq!(v["model_ref"], "x");
        assert_eq!(v["runtime"], "llama_cpp");
        assert_eq!(v["ctx"], 4096);
        assert_eq!(v["port"], 8081);
        assert_eq!(v["pid"], 42);
        assert_eq!(v["log_path"], "/tmp/42.log");
    }

    #[test]
    fn tool_list_has_six_tools_with_spec_names_and_annotations() {
        let router = PaddockMcp::tool_router();
        let mut names: Vec<String> = router
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "paddock_fit",
                "paddock_ps",
                "paddock_recommend",
                "paddock_scan",
                "paddock_serve",
                "paddock_stop",
            ]
        );
        for t in router.list_all() {
            let ann = t.annotations.clone().unwrap_or_default();
            match t.name.as_ref() {
                "paddock_serve" => assert_ne!(ann.read_only_hint, Some(true)),
                "paddock_stop" => assert_eq!(ann.destructive_hint, Some(true)),
                _ => assert_eq!(ann.read_only_hint, Some(true), "{}", t.name),
            }
            assert!(t.description.as_deref().unwrap_or("").len() > 20, "{}", t.name);
        }
    }
}
```

Check the exact `RuntimeKind` serde names (`grep -n "enum RuntimeKind" -A 6 crates/paddock-core/src/catalog/mod.rs`): the test assumes `snake_case`, so `LlamaCpp` serializes as `"llama_cpp"`. Check `ToolRouter::list_all` exists in rmcp 3.5: `grep -n "pub fn list_all" ~/.cargo/registry/src/*/rmcp-3.5.0/src/handler/server/router/tool.rs`. If the method is named differently, use what is there.

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test -p paddock --bin paddock mcp::tests`
Expected: compile error (`error_body`, `serve_result`, `PaddockMcp` not found).

- [ ] **Step 5: Implement the server**

Prepend to `crates/paddock/src/mcp.rs`, above the test module:

```rust
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use paddock_core::hardware::RealSystemProbe;
use paddock_core::runtime::{ServePlan, plan_serve};
use paddock_core::score::UseCase;
use paddock_core::serving::Registry;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Implementation, ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::App;
use crate::lifecycle::{
    InstallPolicy, LifecycleError, ServeMode, SilentProgress, resolve_model, resolved_ctx,
};
use crate::output;

const DEFAULT_SERVE_TIMEOUT_SECS: u64 = 600;

/// Shown to the agent at `initialize`. Written for the model, not the human.
pub(crate) const INSTRUCTIONS: &str = "\
paddock knows which LLMs fit this Apple Silicon Mac, how fast they run, and how to serve them.
Workflow: paddock_scan (hardware) -> paddock_fit or paddock_recommend (ranked models) -> \
paddock_serve (start one, get an endpoint) -> paddock_ps / paddock_stop (manage).
paddock_serve blocks until the server answers, up to timeout_secs (default 600); a first \
run may download a multi-GB model. status \"starting\" means the timeout elapsed but the \
server is still coming up: poll paddock_ps until it is listed, then use the endpoint.
openai_url speaks the OpenAI chat-completions protocol; put model_ref in the `model` field.
An error with code \"no_runtime\" means a runtime (ollama, llama.cpp, mlx-lm) is not installed: \
show install_command to the user and ask before anything is installed. paddock never installs \
on an agent's request.
Fit verdicts: \"fits\" = whole model in GPU memory; \"tune sysctl\" = fits after raising the \
Metal working-set limit; \"ram only\" = CPU offload, slow; \"no fit\" = do not try.";

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum UseCaseInput {
    General,
    Coding,
    Chat,
    Reasoning,
}

impl From<UseCaseInput> for UseCase {
    fn from(v: UseCaseInput) -> Self {
        match v {
            UseCaseInput::General => UseCase::General,
            UseCaseInput::Coding => UseCase::Coding,
            UseCaseInput::Chat => UseCase::Chat,
            UseCaseInput::Reasoning => UseCase::Reasoning,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FitInput {
    /// Scoring profile: general (default), coding, chat or reasoning.
    #[serde(default)]
    pub use_case: Option<UseCaseInput>,
    /// Max rows (default 10).
    #[serde(default)]
    pub limit: Option<usize>,
    /// Also list models that do not fit (default false).
    #[serde(default)]
    pub include_unfit: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecommendInput {
    /// Scoring profile: general (default), coding, chat or reasoning.
    #[serde(default)]
    pub use_case: Option<UseCaseInput>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ServeInput {
    /// Catalog model name, or a unique substring of one (see paddock_fit).
    pub model: String,
    /// Quantization label (e.g. Q4_K_M); default picks the best fitting one.
    #[serde(default)]
    pub quant: Option<String>,
    /// Context window in tokens (llama.cpp only); default auto-sizes to memory.
    #[serde(default)]
    pub ctx: Option<u32>,
    /// Port for llama.cpp / mlx servers (Ollama always uses 11434).
    #[serde(default)]
    pub port: Option<u16>,
    /// Seconds to wait for readiness before returning status "starting" (default 600).
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StopInput {
    /// Model name substring or pid of a running server. "all" is rejected.
    pub target: String,
}

/// MCP error body: `{ code, message, ...extra }`, one code per variant.
pub(crate) fn error_body(e: &LifecycleError) -> Value {
    let (code, extra) = match e {
        LifecycleError::ModelNotFound(_) => (
            "model_not_found",
            json!({ "hint": "call paddock_fit for names" }),
        ),
        LifecycleError::Ambiguous { candidates, .. } => {
            ("ambiguous", json!({ "candidates": candidates }))
        }
        LifecycleError::NoFit { ram_bytes, .. } => (
            "no_fit",
            json!({ "ram_gib": *ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0) }),
        ),
        LifecycleError::UnknownQuant { available, .. } => {
            ("unknown_quant", json!({ "available": available }))
        }
        LifecycleError::NoRuntime { install } => (
            "no_runtime",
            json!({
                "install_command": install.argv.join(" "),
                "hint": "ask the user before installing",
            }),
        ),
        LifecycleError::InstallDeclined { .. } => ("install_declined", json!({})),
        LifecycleError::ServerExited { argv, log_path, .. } => (
            "server_exited",
            json!({ "argv": argv, "log_path": log_path }),
        ),
        // Mapped to a `starting` result by `paddock_serve`; listed for completeness.
        LifecycleError::NotReady { pid, log_path, .. } => {
            ("not_ready", json!({ "pid": pid, "log_path": log_path }))
        }
        LifecycleError::OllamaUnreachable => ("ollama_unreachable", json!({})),
        LifecycleError::NoServerMatch { running, .. } => {
            ("no_server_match", json!({ "running": running }))
        }
        LifecycleError::AmbiguousServer { candidates, .. } => (
            "ambiguous_server",
            json!({
                "candidates": candidates
                    .iter()
                    .map(|(m, p)| json!({ "model_ref": m, "pid": p }))
                    .collect::<Vec<_>>()
            }),
        ),
        LifecycleError::CatalogEmpty => ("catalog_empty", json!({ "hint": "run paddock sync" })),
        LifecycleError::Other(_) => ("internal", json!({})),
    };
    let mut body = json!({ "code": code, "message": e.to_string() });
    if let (Some(dst), Some(src)) = (body.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }
    body
}

/// The `paddock_serve` result for both `ready` and `starting`.
pub(crate) fn serve_result(
    status: &str,
    plan: &ServePlan,
    pid: Option<u32>,
    log_path: Option<PathBuf>,
) -> Value {
    json!({
        "status": status,
        "endpoint": plan.endpoint,
        "openai_url": plan.openai_url,
        "model_ref": plan.model_ref,
        "runtime": plan.runtime,
        "ctx": plan.ctx,
        "port": plan.port,
        "pid": pid,
        "log_path": log_path,
    })
}

fn ok(value: Value) -> CallToolResult {
    CallToolResult::structured(value)
}

fn fail(e: &LifecycleError) -> CallToolResult {
    CallToolResult::structured_error(error_body(e))
}

/// Run blocking lifecycle code off the async executor and render the result.
/// A panic in the closure becomes an `internal` error instead of killing the
/// session.
async fn blocking<F>(f: F) -> CallToolResult
where
    F: FnOnce() -> Result<Value, LifecycleError> + Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(Ok(v)) => ok(v),
        Ok(Err(e)) => fail(&e),
        Err(join) => fail(&LifecycleError::Other(anyhow::anyhow!("tool panicked: {join}"))),
    }
}

#[derive(Clone)]
pub struct PaddockMcp {
    app: Arc<App>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl PaddockMcp {
    pub fn new(app: App) -> Self {
        Self {
            app: Arc::new(app),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "paddock_scan",
        description = "Hardware profile of this Mac: chip, RAM, GPU memory limit, memory bandwidth, and which runtimes (ollama, llama.cpp, mlx-lm) are installed or running. Call first to know what you are working with.",
        annotations(read_only_hint = true, idempotent_hint = true)
    )]
    async fn scan(&self) -> CallToolResult {
        match serde_json::to_value(&self.app.profile) {
            Ok(v) => ok(v),
            Err(e) => fail(&LifecycleError::Other(e.into())),
        }
    }

    #[tool(
        name = "paddock_fit",
        description = "Catalog models ranked for this machine: quant picked, memory estimate, generation tok/s estimate, fit verdict (fits / tune sysctl / ram only / no fit) and score. Empty array means the catalog is empty: the user must run `paddock sync`.",
        annotations(read_only_hint = true)
    )]
    async fn fit(&self, Parameters(input): Parameters<FitInput>) -> CallToolResult {
        let app = self.app.clone();
        blocking(move || {
            let db = app.open_db()?;
            let use_case = input.use_case.map(UseCase::from).unwrap_or(UseCase::General);
            let mut rows = app.scored_models(&db, use_case, input.include_unfit.unwrap_or(false))?;
            rows.truncate(input.limit.unwrap_or(10));
            Ok(serde_json::to_value(output::fit_rows(&rows)).map_err(anyhow::Error::from)?)
        })
        .await
    }

    #[tool(
        name = "paddock_recommend",
        description = "Top 5 models for this machine with a one-line justification each (fit headroom, speed tier, context). Use when you want a short answer instead of the full ranking.",
        annotations(read_only_hint = true)
    )]
    async fn recommend(&self, Parameters(input): Parameters<RecommendInput>) -> CallToolResult {
        let app = self.app.clone();
        blocking(move || {
            let db = app.open_db()?;
            let use_case = input.use_case.map(UseCase::from).unwrap_or(UseCase::General);
            let mut rows = app.scored_models(&db, use_case, false)?;
            rows.truncate(5);
            Ok(serde_json::to_value(output::recommend_rows(&rows)).map_err(anyhow::Error::from)?)
        })
        .await
    }

    #[tool(
        name = "paddock_serve",
        description = "Start serving a catalog model with the best available runtime and return an OpenAI-compatible endpoint. Picks the best fitting quant unless `quant` is given. Blocks until ready (up to timeout_secs, default 600; a first run may download the model). status \"starting\" means it is still loading: poll paddock_ps. Never installs a runtime: a no_runtime error carries the command for the user."
    )]
    async fn serve(&self, Parameters(input): Parameters<ServeInput>) -> CallToolResult {
        let app = self.app.clone();
        blocking(move || {
            let (model, idx) = resolve_model(&app, &input.model, input.quant.as_deref())?;
            let ctx = Some(resolved_ctx(&app, &model, idx, input.ctx));
            let plan = plan_serve(
                &model,
                &model.variants[idx],
                &app.profile.runtimes,
                input.port,
                ctx,
            )?;
            let timeout = Duration::from_secs(input.timeout_secs.unwrap_or(DEFAULT_SERVE_TIMEOUT_SECS));
            match crate::lifecycle::serve(
                plan,
                ServeMode::Detached,
                InstallPolicy::Refuse,
                Some(timeout),
                &SilentProgress,
            ) {
                Ok(outcome) => Ok(serve_result("ready", &outcome.plan, outcome.pid, outcome.log_path)),
                Err(LifecycleError::NotReady { pid, log_path, plan }) => {
                    Ok(serve_result("starting", &plan, pid, log_path))
                }
                Err(e) => Err(e),
            }
        })
        .await
    }

    #[tool(
        name = "paddock_ps",
        description = "Running servers (paddock-spawned llama.cpp / mlx servers and models loaded in the Ollama daemon) with endpoint, context and pid, plus models available locally but not running.",
        annotations(read_only_hint = true)
    )]
    async fn ps(&self) -> CallToolResult {
        blocking(move || {
            let probe = RealSystemProbe;
            let registry = Registry::open_default();
            let running = paddock_core::serving::list_all_servers(&registry, &probe);
            let history = paddock_core::serving::History::open_default();
            let available = paddock_core::serving::list_available(&history, &probe, &running);
            Ok(json!({ "running": running, "available": available }))
        })
        .await
    }

    #[tool(
        name = "paddock_stop",
        description = "Stop one running server by model name substring or pid. Ollama-loaded models are unloaded from the daemon; paddock-spawned servers are terminated. \"all\" is rejected: stop servers one at a time.",
        annotations(destructive_hint = true)
    )]
    async fn stop(&self, Parameters(input): Parameters<StopInput>) -> CallToolResult {
        if input.target == "all" {
            return CallToolResult::structured_error(json!({
                "code": "invalid_target",
                "message": "`all` is not accepted over MCP",
                "hint": "stop servers one at a time",
            }));
        }
        blocking(move || {
            let stopped = crate::lifecycle::stop(&input.target, &SilentProgress)?;
            Ok(json!({
                "stopped": stopped
                    .iter()
                    .map(|s| json!({ "model_ref": s.model_ref, "pid": s.pid }))
                    .collect::<Vec<_>>()
            }))
        })
        .await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for PaddockMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("paddock", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}

/// Block on a stdio MCP session until the client closes stdin.
pub fn run(app: App) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let service = PaddockMcp::new(app)
            .serve(rmcp::transport::stdio())
            .await?;
        service.waiting().await?;
        Ok(())
    })
}
```

Two things the prototype confirmed for rmcp 3.5: the handler attribute must name the router field (`#[tool_handler(router = self.tool_router)]`), and `ServerConfig` is `#[non_exhaustive]`, so it is built with `ServerConfig::new(...)` + `with_*`. `App` holds only owned data (`HardwareProfile`, `MemoryBudget`, `SpeedCalibration`) so `Arc<App>` is `Send + Sync`; the catalog `Db` is opened inside each `spawn_blocking` closure.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p paddock`
Expected: `mcp::tests` all pass, everything else unchanged. `cargo build -p paddock` has no warnings (if `InstallDeclined` or `NotReady` mapping arms trigger "unreachable pattern" or unused warnings, they do not: both are real variants).

- [ ] **Step 7: Commit**

```bash
git add crates/paddock/Cargo.toml Cargo.lock crates/paddock/src/cli.rs crates/paddock/src/main.rs crates/paddock/src/mcp.rs
git commit -m "feat(paddock): paddock mcp serves provisioning tools over stdio"
```

---

## Task 7: stdio integration tests

**Files:**
- Create: `crates/paddock/tests/mcp.rs`

**Interfaces:**
- Consumes: the `paddock mcp` binary; `PADDOCK_DB_PATH`, `PADDOCK_SERVING_DIR`, `PADDOCK_CALIBRATION_PATH` env overrides (same as `tests/cli.rs`).

- [ ] **Step 1: Write the tests**

```rust
//! `paddock mcp` over real stdio: spawn the binary, write JSON-RPC lines,
//! read the responses back. Responses can arrive out of order (tools run on
//! a thread pool), so everything is matched by request id.

use std::collections::HashMap;

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::TempDir;

fn paddock() -> (Command, TempDir) {
    let mut c = Command::cargo_bin("paddock").unwrap();
    let dir = tempfile::tempdir().unwrap();
    c.env("PADDOCK_DB_PATH", dir.path().join("catalog.db"));
    c.env("PADDOCK_SERVING_DIR", dir.path().join("serving"));
    c.env(
        "PADDOCK_CALIBRATION_PATH",
        dir.path().join("calibration.json"),
    );
    (c, dir)
}

fn seed_one_model(dir: &std::path::Path) {
    use paddock_core::catalog::{CatalogModel, CatalogVariant, RuntimeKind, Source, db::Db};
    let db = Db::open(dir.join("catalog.db")).unwrap();
    db.upsert_model(&CatalogModel {
        id: 0,
        name: "fake-model".into(),
        family: Some("llama".into()),
        source: Source::Ollama,
        repo: None,
        params_total: 1_000_000_000,
        params_active: 1_000_000_000,
        architecture: Some("llama".into()),
        context_max: 8192,
        released_at: None,
        released_approx: false,
        variants: vec![CatalogVariant {
            quant: "Q4_K_M".into(),
            bpw: 4.83,
            file_size_bytes: None,
            layers: 16,
            kv_heads: 8,
            head_dim: 64,
            embedding_dim: 2048,
            runtime_compat: vec![RuntimeKind::Ollama],
            source_tag: None,
        }],
    })
    .unwrap();
}

fn initialize() -> Vec<Value> {
    vec![
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2025-06-18","capabilities":{},
            "clientInfo":{"name":"paddock-test","version":"0"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
    ]
}

fn call(id: u64, tool: &str, args: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":tool,"arguments":args}})
}

/// Run one session: `initialize` + `initialized` + `requests`, stdin closed
/// after the last line. Returns responses keyed by id.
fn session(cmd: &mut Command, requests: Vec<Value>) -> HashMap<u64, Value> {
    let mut lines: Vec<String> = initialize().iter().map(Value::to_string).collect();
    lines.extend(requests.iter().map(Value::to_string));
    let stdin = lines.join("\n") + "\n";
    let out = cmd.arg("mcp").write_stdin(stdin).output().unwrap();
    assert!(
        out.status.success(),
        "paddock mcp exited with {}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("stdout is utf-8");
    let mut by_id = HashMap::new();
    for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("stdout line is not JSON-RPC ({e}): {line}"));
        if let Some(id) = v["id"].as_u64() {
            by_id.insert(id, v);
        }
    }
    by_id
}

fn structured(resp: &Value) -> &Value {
    &resp["result"]["structuredContent"]
}

#[test]
fn initialize_advertises_tools_and_instructions() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![]);
    let init = &r[&1]["result"];
    assert_eq!(init["serverInfo"]["name"], "paddock");
    assert!(init["capabilities"]["tools"].is_object());
    let instructions = init["instructions"].as_str().unwrap();
    assert!(instructions.contains("paddock_ps"));
    assert!(instructions.contains("no_runtime"));
}

#[test]
fn tools_list_has_six_tools_with_object_schemas() {
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![json!({"jsonrpc":"2.0","id":2,"method":"tools/list"})],
    );
    let tools = r[&2]["result"]["tools"].as_array().unwrap();
    let mut names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    names.sort();
    assert_eq!(
        names,
        ["paddock_fit", "paddock_ps", "paddock_recommend", "paddock_scan", "paddock_serve", "paddock_stop"]
    );
    for t in tools {
        assert_eq!(t["inputSchema"]["type"], "object", "{}", t["name"]);
        assert!(t["inputSchema"]["properties"].is_object(), "{}", t["name"]);
    }
    let serve = tools.iter().find(|t| t["name"] == "paddock_serve").unwrap();
    assert_eq!(serve["inputSchema"]["required"], json!(["model"]));
    let stop = tools.iter().find(|t| t["name"] == "paddock_stop").unwrap();
    assert_eq!(stop["annotations"]["destructiveHint"], true);
}

#[test]
fn scan_returns_hardware_profile() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_scan", json!({}))]);
    let v = structured(&r[&2]);
    assert!(v["chip_name"].is_string());
    assert!(v["ram_total_bytes"].is_u64());
    assert!(v["runtimes"]["ollama"]["installed"].is_boolean());
    // The text block carries the same JSON.
    let text = r[&2]["result"]["content"][0]["text"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(text).unwrap();
    assert_eq!(&parsed, v);
}

#[test]
fn ps_on_empty_registry_has_running_and_available() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_ps", json!({}))]);
    let v = structured(&r[&2]);
    assert_eq!(v["running"], json!([]));
    assert!(v["available"].is_array());
}

#[test]
fn fit_on_empty_catalog_is_empty_array() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_fit", json!({"use_case": "coding"}))]);
    assert_eq!(structured(&r[&2]), &json!([]));
}

#[test]
fn fit_limit_zero_is_empty_array() {
    // Review Focus 2: a zero limit is a valid "give me nothing", not an error.
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(&mut cmd, vec![call(2, "paddock_fit", json!({"limit": 0}))]);
    assert_eq!(r[&2]["result"]["isError"], false);
    assert_eq!(structured(&r[&2]), &json!([]));
}

#[test]
fn fit_and_recommend_list_seeded_model() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(
        &mut cmd,
        vec![
            call(2, "paddock_fit", json!({})),
            call(3, "paddock_recommend", json!({})),
        ],
    );
    assert_eq!(structured(&r[&2])[0]["name"], "fake-model");
    let rec = structured(&r[&3]);
    assert_eq!(rec[0]["model"], "fake-model");
    assert!(rec[0]["justification"].as_str().unwrap().contains("tok/s"));
}

#[test]
fn stop_all_is_rejected_before_touching_registry() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_stop", json!({"target": "all"}))]);
    assert_eq!(r[&2]["result"]["isError"], true);
    assert_eq!(structured(&r[&2])["code"], "invalid_target");
}

#[test]
fn stop_unknown_pid_is_no_server_match() {
    // Review Focus 3: a numeric target that is not running.
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_stop", json!({"target": "99999999"}))]);
    assert_eq!(r[&2]["result"]["isError"], true);
    let v = structured(&r[&2]);
    assert_eq!(v["code"], "no_server_match");
    assert_eq!(v["running"], json!([]));
}

#[test]
fn serve_on_empty_catalog_is_catalog_empty() {
    let (mut cmd, _dir) = paddock();
    let r = session(&mut cmd, vec![call(2, "paddock_serve", json!({"model": "x"}))]);
    assert_eq!(r[&2]["result"]["isError"], true);
    assert_eq!(structured(&r[&2])["code"], "catalog_empty");
}

#[test]
fn serve_unknown_model_is_model_not_found() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(&mut cmd, vec![call(2, "paddock_serve", json!({"model": "nope"}))]);
    let v = structured(&r[&2]);
    assert_eq!(v["code"], "model_not_found");
    assert_eq!(v["hint"], "call paddock_fit for names");
    // Zero side effects: nothing was spawned or recorded.
    assert!(!dir.path().join("serving").exists());
}

#[test]
fn serve_unknown_quant_lists_available() {
    let (mut cmd, dir) = paddock();
    seed_one_model(dir.path());
    let r = session(
        &mut cmd,
        vec![call(2, "paddock_serve", json!({"model": "fake-model", "quant": "Q9_X"}))],
    );
    let v = structured(&r[&2]);
    assert_eq!(v["code"], "unknown_quant");
    assert_eq!(v["available"], json!(["Q4_K_M"]));
}

#[test]
fn unknown_tool_does_not_kill_session() {
    // Review Focus 5: a bad call gets a JSON-RPC error; the next call still works.
    let (mut cmd, _dir) = paddock();
    let r = session(
        &mut cmd,
        vec![
            call(2, "paddock_nope", json!({})),
            call(3, "paddock_stop", json!({})), // missing required `target`
            call(4, "paddock_ps", json!({})),
        ],
    );
    assert!(r[&2].get("error").is_some() || r[&2]["result"]["isError"] == true, "{}", r[&2]);
    assert!(r[&3].get("error").is_some() || r[&3]["result"]["isError"] == true, "{}", r[&3]);
    assert_eq!(structured(&r[&4])["running"], json!([]));
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p paddock --test mcp`
Expected: all 13 pass. If `serve_unknown_model_is_model_not_found` fails on the `serving` dir assertion, `History::open_default()` or `Registry::open_default()` created the directory before resolution: that is a bug in Task 6's ordering (resolution happens before `lifecycle::serve`), fix in `mcp.rs`, not in the test.

- [ ] **Step 3: Run the whole suite**

Run: `cargo test --workspace`
Expected: everything green. Then `cargo build --release -p paddock` and note the binary size delta in the commit body if it is above 5 MB.

- [ ] **Step 4: Commit**

```bash
git add crates/paddock/tests/mcp.rs
git commit -m "test(paddock): paddock mcp stdio integration tests"
```

---

## Task 8: Docs and live smoke

**Files:**
- Modify: `README.md` (line 53 subcommand count; new section after `paddock bench`, before `paddock tray`; roadmap)

- [ ] **Step 1: Update the subcommand count**

Line 53: `Ten subcommands cover everything scriptable:` becomes `Eleven subcommands cover everything scriptable:`.

- [ ] **Step 2: Add the `paddock mcp` section**

Insert before `### \`paddock tray\`: menu bar (macOS)`:

````markdown
### `paddock mcp`: let coding agents provision models

`paddock mcp` exposes paddock over the [Model Context Protocol](https://modelcontextprotocol.io) on stdio, so an agent (Claude Code, Claude Desktop, Cursor, anything that can spawn a process) can ask "what can this machine run?", start a model, and get an OpenAI-compatible endpoint back, without parsing tables or driving a terminal.

```text
$ claude mcp add paddock -- paddock mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "paddock": { "command": "paddock", "args": ["mcp"] }
  }
}
```

Six tools, provisioning only:

| Tool | Does | Returns |
|---|---|---|
| `paddock_scan` | hardware profile | same JSON as `paddock scan --json` |
| `paddock_fit` | ranked models (`use_case`, `limit`, `include_unfit`) | same rows as `paddock fit --json` |
| `paddock_recommend` | top 5 with justifications | same rows as `paddock recommend --json` |
| `paddock_serve` | start a model (`model`, `quant?`, `ctx?`, `port?`, `timeout_secs?`) | `{ status, endpoint, openai_url, model_ref, runtime, ctx, port, pid, log_path }` |
| `paddock_ps` | running + available servers | same as `paddock ps --json` |
| `paddock_stop` | stop one server by name or pid | `{ stopped: [{ model_ref, pid }] }` |

Two contracts the agent is told about at connect time:

- `paddock_serve` blocks until the server answers, up to `timeout_secs` (default 600, because a first run may download the model). If the timeout elapses the server keeps running detached and the tool returns `status: "starting"` with the pid; the agent polls `paddock_ps`.
- paddock never installs a runtime on an agent's request. A missing runtime is an error with `code: "no_runtime"` and the `install_command`, for the agent to show you.

Other errors carry a `code` too (`model_not_found`, `ambiguous` with `candidates`, `no_fit`, `unknown_quant` with `available`, `no_server_match` with `running`, `catalog_empty`, ...). `stop` refuses `all` over MCP: one server at a time. `sync`, `logs` and `run` are not exposed (slow and networked, or interactive).
````

- [ ] **Step 3: Update the roadmap**

Replace the line `- **MCP server**: let coding agents ask "what can this machine run?"` with:

```markdown
- **MCP v2**: `paddock_chat` (relay a completion to a served endpoint), `paddock_run_agent` (spawn a local agent runner against it and return its diff), HTTP transport, `paddock://hardware` and `paddock://catalog` resources
```

- [ ] **Step 4: Em-dash check**

Run: `grep -rn "$(printf '\xe2\x80\x94')" README.md crates/paddock/src crates/paddock/tests docs/superpowers/plans/2026-09-29-mcp-server.md`
Expected: no output.

- [ ] **Step 5: Live smoke with Claude Code**

```bash
cargo install --path crates/paddock --locked
claude mcp add paddock -- paddock mcp
claude mcp list
```

Expected: `paddock` listed as connected. In a Claude Code session ask "what can this machine run?" and confirm `paddock_fit` is called; ask it to serve the smallest Ollama model and confirm `paddock_serve` returns `status: "ready"` and `paddock ps` shows it; ask it to stop it. Then `claude mcp remove paddock` if you do not want to keep it.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: paddock mcp usage, tool table and error contracts"
```

---

## Self-review

**Spec coverage.**
- `lifecycle` module with `Progress`, `StderrProgress`, `SilentProgress`, `InstallPolicy`, `ServeMode`, `LifecycleError` (all 13 variants), `ServeOutcome` (incl. `child`), `Stopped`, `resolve_model`, `serve`, `stop`: Tasks 1-4.
- `resolve_model` maps ambiguity and empty catalog to variants: Task 1.
- `serve` differences (`InstallPolicy::Ask` through the CLI closure / `Refuse` returns `NoRuntime` first; `wait_ready` with explicit `Option<Duration>`; CLI passes `None`; daemon keeps 3 s and maps to `OllamaUnreachable`; detached timeout registers first, leaves the child, returns `NotReady`; foreground timeout kills; returns as soon as ready in both modes with `child` for foreground): Tasks 2-3.
- `stop` minus prompt, `"all"` understood, prompt in `main.rs`, MCP rejects `"all"`: Tasks 4 and 6.
- `Display` reproduces CLI wording, 14 CLI tests + 54 TUI tests untouched: Tasks 1-4 (each verify step runs the full suite).
- `Command::Mcp`, `PaddockMcp { app: Arc<App> }`, `App::load()` once, DB reopened per call, `get_info` instructions with the four contracts, `spawn_blocking`, `structured` results with text + `structured_content`, `structured_error` bodies, stdout reserved: Task 6.
- Six tools with the spec's inputs, outputs and annotations: Task 6 (unit test on names/annotations) and Task 7 (schemas over the wire).
- Error code table incl. `invalid_target` and `NotReady` as a `starting` result: Task 6.
- CLI/TUI adapters: Task 3 (`serve_with_plan`, `confirm_and_install`, `launch`), Task 4 (`stop_servers`); TUI unchanged by design (documented refinement).
- `rmcp` + `schemars` deps, README section with `claude mcp add`, Claude Desktop JSON, tool list, `starting` / `no_runtime` contracts, roadmap replaced by v2 items: Tasks 6 and 8.
- Testing section: lifecycle unit tests for error mapping and `wait_ready` (timeout leaves child alive; answers on the third poll): Tasks 1-2; MCP `assert_cmd` test with `initialize`, `tools/list` (six tools, schemas parse), `tools/call paddock_ps` on an empty registry, `paddock_stop` with `all`: Task 7.

**Placeholder scan.** No TBD/TODO; every code step has full code; the three "check with grep" notes (env var name, `estimate_speed` signature, `RuntimeKind` serde names, `ToolRouter::list_all`) name the exact command and the fallback.

**Type consistency.** `LifecycleError::NotReady { pid, log_path, plan: Box<ServePlan> }` is used identically in Tasks 2, 3 and 6. `InstallPolicy::Ask(&confirm_and_install)` requires `confirm_and_install: fn(&InstallPlan) -> Result<(), LifecycleError>` (Task 3) and `Progress::quiet` (Task 1) is what `run_checked` (Task 2) reads. `output::fit_rows` / `recommend_rows` (Task 5) are the names `mcp.rs` (Task 6) calls. `Stopped` derives `PartialEq` for the Task 4 assertion.

**Review Focus.** Items 1-5 each have a named test: `error_body_other_is_internal` (Task 6), `fit_limit_zero_is_empty_array` (Task 7), `stop_unknown_pid_is_no_server_match` (Task 7), `wait_ready_timeout_leaves_child_alive` (Task 2), `unknown_tool_does_not_kill_session` (Task 7).
